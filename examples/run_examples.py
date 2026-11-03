#!/usr/bin/env python3
"""Run the ReRDMFT examples (strictly one at a time) and print [ OK ] / [FAIL] per check.

    examples/run_examples.py                      run the fast examples (slow ones are skipped by default)
    examples/run_examples.py -i lih_gnof_c4_full_optimization co-sto-3g
    examples/run_examples.py -e co-cc-pvdz h2     run everything except these
    examples/run_examples.py --all                also run the slow examples (> 60 s, flagged in reference.yaml)
    examples/run_examples.py --update [-i ...]    (re)write the reference values from the current run
    examples/run_examples.py -l                   list the examples and exit

Checks per example (reference values live in reference.yaml):
  * the run exits with status 0 and finishes within the timeout;
  * no line of the output contains "[FAIL]" (the program's own internal checks);
  * every "<label> ... energy: <value> Hartree" line printed by the program matches the
    reference (same label, same occurrence number) within the tolerance (default 1e-6 Ha);
  * the number of such energy lines is unchanged; the geminal occupations n_p (count, sum, max, min per block);
  * optional hand-written 'external' entries (PySCF/literature values from the input comments);
  * printed h-onebody+Vee+N-N and electronic+nuclear sums equal the printed total energy.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent
EXE = ROOT / "rerdmft"
REFERENCE = EXAMPLES / "reference.yaml"
SLOW_SECONDS = 60.0
DEFAULT_TOL = 1.0e-6
DEFAULT_TIMEOUT = 3600.0

ENERGY_RE = re.compile(r"^\s*([^:\n]*[Ee]nerg[^:\n]*?|h-onebody|Vee):\s+(-?\d+\.\d+)\s+Hartree")
# Tracked quantities (by label): SCF totals, final/optimized totals of the functional or CAS runs,
# UMP2/UCCSD correlation + total energies, and the final energy split h-onebody / Vee.
TRACKED_RE = re.compile(r"HF energy|DHF \(4C\) energy|^Final total energy|^Optimized total energy|"
                        r"^UMP2|^UCCSD|^CCSD|^h-onebody|^Vee")
OCC_RE = re.compile(r"\bn = (-?\d+\.\d+)")
OCC_TOL = 1.0e-4  # occupations are printed with 5 decimals
OK = "[ \033[92mOK\033[0m ]"
FAIL = "[\033[91mFAIL\033[0m]"


def extract_energies(text):
    """Ordered [(key, value)] with key = '<label>#<occurrence>'."""
    seen, out = {}, []
    for line in text.splitlines():
        m = ENERGY_RE.match(line)
        if not m:
            continue
        label = " ".join(m.group(1).split())
        seen[label] = seen.get(label, 0) + 1
        out.append((f"{label}#{seen[label]}", float(m.group(2))))
    return out


def key_energies(energies):
    """The tracked subset (see TRACKED_RE); electronic and N-N pieces are covered by consistency()."""
    return [(k, v) for k, v in energies if TRACKED_RE.search(k)]


def occupation_stats(text):
    """Per printed block of geminal occupations n_p: count, sum, max, min (key 'occupations#k/<stat>')."""
    blocks, cur, gap = [], [], 0
    for line in text.splitlines():
        m = OCC_RE.search(line)
        if m:
            cur.append(float(m.group(1)))
            gap = 0
        elif cur:
            gap += 1
            if gap > 2:
                blocks.append(cur)
                cur, gap = [], 0
    if cur:
        blocks.append(cur)
    out = {}
    for k, b in enumerate(blocks, 1):
        out.update({f"occupations#{k}/count": float(len(b)), f"occupations#{k}/sum": sum(b),
                    f"occupations#{k}/max": max(b), f"occupations#{k}/min": min(b)})
    return out


VAL = r"(-?\d+\.\d+)\s+Hartree"
LINE = lambda label: re.compile(r"^\s*" + label + r"[^:\n]*:\s+" + VAL)
ELEC, NUC, TOT = LINE("Electronic energy"), LINE("Nuclear repulsion energy"), LINE(r"[^:\n]*[Tt]otal[^:\n]*?energy")
HONE, VEE, NN = LINE("h-onebody"), LINE("Vee"), LINE("N-N")


def consistency(text, tol=1e-7):
    """Physics self-consistency of the printed numbers; returns a list of problems."""
    bad, elec, nuc, last_tot, comp = [], None, None, None, {}
    if re.search(r"energy[^:\n]*:\s+(nan|-?inf)", text, re.I):
        bad.append("non-finite energy printed")
    for line in text.splitlines():
        for rx, name in ((ELEC, "elec"), (NUC, "nuc"), (HONE, "h"), (VEE, "vee"), (NN, "nn")):
            m = rx.match(line)
            if m:
                v = float(m.group(1))
                if name == "elec": elec = v
                elif name == "nuc": nuc = v
                else: comp[name] = v
        m = TOT.match(line)
        if m:
            last_tot = float(m.group(1))
            comp = {}
        if len(comp) == 3 and last_tot is not None:
            d = comp["h"] + comp["vee"] + comp["nn"] - last_tot
            if abs(d) > tol:
                bad.append(f"h-onebody+Vee+N-N differs from the total energy {last_tot:.8f} by {d:+.1e}")
            comp = {}
    for m in re.finditer(r"Electronic energy:\s+" + VAL + r"\s*\n\s*Nuclear repulsion energy:\s+" + VAL
                         + r"\s*\n\s*Total[^:\n]*energy:\s+" + VAL, text):
        e, n, t = map(float, m.groups())
        if abs(e + n - t) > tol:
            bad.append(f"electronic+nuclear differs from total by {e + n - t:+.1e}")
    return bad


def run_example(inp, timeout):
    """Run one example; return (returncode or None on timeout, stdout text, seconds)."""
    t0 = time.time()
    try:
        p = subprocess.run([str(EXE), inp.name], cwd=EXAMPLES, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout)
        return p.returncode, p.stdout, time.time() - t0
    except subprocess.TimeoutExpired as e:
        txt = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, txt, time.time() - t0


def check(label, ok, detail=""):
    print(f"  {OK if ok else FAIL} {label}{('  ' + detail) if detail else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-i", "--input", nargs="+", default=[], help="only these examples (name or name.inp)")
    ap.add_argument("-e", "--exclude", nargs="+", default=[], help="all examples but these")
    ap.add_argument("-a", "--all", action="store_true", help="also run the examples recorded as slow (> 60 s)")
    ap.add_argument("--update", action="store_true", help="write reference values from this run")
    ap.add_argument("-t", "--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per example")
    ap.add_argument("-l", "--list", action="store_true", help="list examples and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the output of failing examples")
    args = ap.parse_args()

    norm = lambda n: Path(n).name[:-4] if n.endswith(".inp") else Path(n).name
    names = sorted(p.stem for p in EXAMPLES.glob("*.inp"))
    if args.input:
        wanted = [norm(n) for n in args.input]
        missing = [n for n in wanted if n not in names]
        if missing:
            sys.exit(f"unknown example(s): {', '.join(missing)}")
        names = [n for n in names if n in wanted]
    excluded = {norm(n) for n in args.exclude}
    names = [n for n in names if n not in excluded]

    ref = yaml.safe_load(REFERENCE.read_text()) if REFERENCE.exists() else {}
    ref = ref or {}
    if not args.all and not args.input:  # slow examples only on request (--all, or named with -i)
        skipped = [n for n in names if ref.get(n, {}).get("slow", False)]
        names = [n for n in names if n not in skipped]
        if skipped and not args.list:
            print(f"skipping {len(skipped)} slow example(s) (> {SLOW_SECONDS:.0f} s); use --all to include them")
    if args.list:
        print("\n".join(names))
        return 0
    if not EXE.exists():
        sys.exit(f"{EXE} not found: run make first")

    n_ok = n_fail = n_noref = 0
    failed, t_all = [], time.time()
    for k, name in enumerate(names, 1):
        print(f"\n[{k}/{len(names)}] {name}.inp", flush=True)
        rc, text, secs = run_example(EXAMPLES / f"{name}.inp", args.timeout)
        energies = key_energies(extract_energies(text))
        entry = ref.get(name)
        good = True

        good &= check("run completed", rc == 0,
                      f"(timeout after {secs:.0f} s)" if rc is None else f"(exit {rc}, {secs:.1f} s)")
        bad_lines = [l.strip() for l in text.splitlines() if "[FAIL]" in l]
        good &= check("no internal [FAIL] check", not bad_lines, bad_lines[0][:90] if bad_lines else "")

        bad = consistency(text)
        good &= check("energy self-consistency (h+Vee+N-N, elec+nuc = total)", not bad, bad[0] if bad else "")

        if args.update:
            if good:
                ref[name] = dict(ref.get(name) or {})  # keeps hand-written "external" entries
                ref[name].update({"slow": secs > SLOW_SECONDS, "tolerance": DEFAULT_TOL,
                                  "energies": {key: val for key, val in energies},
                                  "occupations": occupation_stats(text)})
                print(f"  reference written: {len(energies)} energies")
            else:
                print("  reference NOT written (run did not pass)")
        elif entry is None:
            n_noref += 1
            print("  NO REFERENCE (run with --update)")
        else:
            tol = float(entry.get("tolerance", DEFAULT_TOL))
            refs = entry.get("energies", {})
            got = dict(energies)
            good &= check("number of energies", len(got) == len(refs), f"({len(got)} vs ref {len(refs)})")
            occ_ref, occ = entry.get("occupations") or {}, occupation_stats(text)
            good &= check("number of occupation values", len(occ) == len(occ_ref), f"({len(occ)} vs ref {len(occ_ref)})")
            for key, rv in occ_ref.items():
                if key in occ:
                    d = occ[key] - rv
                    good &= check(key, abs(d) < OCC_TOL, f"{occ[key]:16.8f} ref {rv:16.8f} diff {d:+.2e}")
                else:
                    good &= check(key, False, "NOT FOUND")
            for key, ext in (entry.get("external") or {}).items():
                if key not in got:
                    good &= check(f"{key} vs {ext['source']}", False, "NOT FOUND")
                else:
                    d = got[key] - ext["value"]
                    good &= check(f"{key} vs {ext['source']}", abs(d) < ext["tolerance"],
                                  f"{got[key]:16.10f} ref {ext['value']:16.10f} diff {d:+.2e}")
            for key, rv in refs.items():
                if key not in got:
                    good &= check(key, False, "NOT FOUND")
                else:
                    d = got[key] - rv
                    good &= check(key, abs(d) < tol, f"{got[key]:16.10f} ref {rv:16.10f} diff {d:+.2e}")
        if not args.update and entry is not None:
            n_ok, n_fail = (n_ok + 1, n_fail) if good else (n_ok, n_fail + 1)
            if not good:
                failed.append(name)
                if args.verbose:
                    print(text)
        elif not good:
            failed.append(name)
        if args.update:  # save after every example so an interrupted sweep keeps its progress
            REFERENCE.write_text(yaml.safe_dump(ref, sort_keys=True, width=120))

    print("\n" + "=" * 60)
    print(f"{len(names)} examples in {(time.time() - t_all) / 60:.1f} min: "
          f"{n_ok} passed, {n_fail} failed, {n_noref} without reference")
    if failed:
        print("failed: " + " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
