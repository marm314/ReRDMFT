#!/usr/bin/env python3
"""Reads a ReRDMFT input file and downloads its BASIS from the Basis Set Exchange
(basissetexchange.org, the modern successor to the old EMSL Basis Set Exchange) in
Gaussian94 format -- the same text format ReRDMFT's own basis-set reader
(AO_basis/BasisSet.cpp) expects.

Usage:
    python3 download_basis.py <input_file.inp>

What it does:
    1. Reads the BASIS keyword's value from the input file and strips its extension
       (e.g. "BASIS cc-pVDZ.gbs" -> the name "cc-pVDZ" to query). If that name isn't
       recognized, it also tries progressively stripping a leading "<word>-" segment
       and re-querying -- so this project's own "<molecule>-<basis>.gbs" naming
       convention works too (e.g. "lih-6-31g.gbs" tries "lih-6-31g" then "6-31g";
       "co-cc-pvdz.gbs" tries "co-cc-pvdz", then "cc-pvdz", then "pvdz"), without
       guessing where a basis name's OWN internal hyphens end: every candidate is
       checked against Basis Set Exchange itself (see basis_name_candidates /
       download_basis), so a wrong guess can't silently succeed.
    2. Reads the element symbols out of the GEOMETRY ... END block.
    3. Downloads the basis for exactly those elements from Basis Set Exchange's REST
       API, in Gaussian94 format.
    4. Writes the result to <BASIS keyword's original value> in the SAME DIRECTORY as
       the input file -- so a run of `rerdmft <input_file.inp>` from that directory
       finds it exactly where the BASIS keyword says to look.

Comment convention matches ReRDMFT's own input parser (Input/Input.cpp's
stripComment): '#' or '!' start a comment anywhere on the line.
"""

import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BSE_API_URL = "https://www.basissetexchange.org/api/basis/{name}/format/gaussian94/"


def strip_comment(line: str) -> str:
    """Mirrors Input/Input.cpp's stripComment: '#' or '!' start a comment anywhere."""
    for marker in ("#", "!"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line.strip()


def read_input_lines(path: Path):
    with path.open() as f:
        for raw_line in f:
            stripped = strip_comment(raw_line)
            if stripped:
                yield stripped


def find_basis_value(lines) -> str:
    for line in lines:
        parts = line.split(None, 1)
        if parts and parts[0].upper() == "BASIS":
            if len(parts) < 2 or not parts[1].strip():
                raise ValueError("BASIS keyword found but has no value")
            return parts[1].strip().split()[0]
    raise ValueError("no BASIS keyword found in the input file")


def find_geometry_elements(lines):
    elements = []
    seen = set()
    in_geometry = False
    for line in lines:
        if not in_geometry:
            if line.split(None, 1)[0].upper() == "GEOMETRY":
                in_geometry = True
            continue
        if line.upper() == "END":
            break
        symbol = line.split()[0]
        # Normalize to the usual "Symbol"/"Element" capitalization BSE expects
        # (e.g. "h" -> "H", "NA" -> "Na"), tolerant of an optional trailing index/label
        # some geometry conventions append (e.g. "H1", "O2") by stripping trailing digits.
        symbol = re.sub(r"\d+$", "", symbol).capitalize()
        if symbol not in seen:
            seen.add(symbol)
            elements.append(symbol)
    if not in_geometry:
        raise ValueError("no GEOMETRY block found in the input file")
    if not elements:
        raise ValueError("GEOMETRY block has no atoms")
    return elements


def basis_name_candidates(basis_value: str):
    """Candidate Basis Set Exchange names to try, in order: the whole file stem first
    (covers plain names like "6-31G.gbs" or "cc-pVDZ.gbs" directly), then progressively
    with leading hyphen-separated segments stripped off, covering this project's own
    "<molecule>-<basis>.gbs" naming convention (e.g. "lih-6-31g.gbs" -> "6-31g",
    "co-cc-pvdz.gbs" -> "cc-pvdz") without having to guess where a basis name's OWN
    internal hyphens end -- each candidate is verified against Basis Set Exchange itself
    (download_basis), never just assumed, so a wrong guess can't silently succeed.
    """
    stem = Path(basis_value).stem
    if not stem:
        return [basis_value]
    segments = stem.split("-")
    return ["-".join(segments[i:]) for i in range(len(segments))]


def _fetch(name: str, elements) -> str:
    url = BSE_API_URL.format(name=urllib.parse.quote(name, safe="")) + "?" + urllib.parse.urlencode(
        {"elements": ",".join(elements)}
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def download_basis(candidates, elements) -> tuple:
    """Tries each candidate name in turn; returns (text, name) for the first one Basis
    Set Exchange accepts. A 404 (unrecognized name) moves on to the next candidate; any
    other failure (network, timeout, non-404 HTTP error) stops immediately, since
    retrying a different name would not fix it.
    """
    tried = []
    for name in candidates:
        tried.append(name)
        try:
            return _fetch(name, elements), name
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise RuntimeError(f"Basis Set Exchange returned HTTP {e.code} for basis '{name}'") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"could not reach Basis Set Exchange: {e}") from e
    raise RuntimeError(
        f"none of these names were recognized by Basis Set Exchange: {tried}. Check the "
        f"basis name against https://www.basissetexchange.org -- it must match one of its "
        f"listed names (case-insensitive, e.g. 'cc-pVDZ', '6-31G', 'sto-3g')."
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file.inp>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"Error: input file '{input_path}' not found", file=sys.stderr)
        return 1

    try:
        lines = list(read_input_lines(input_path))
        basis_value = find_basis_value(lines)
        elements = find_geometry_elements(lines)
        candidates = basis_name_candidates(basis_value)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"BASIS value:   {basis_value}")
    print(f"Elements:      {', '.join(elements)}")
    sys.stdout.flush()

    try:
        text, basis_name = download_basis(candidates, elements)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Basis set:     {basis_name}")
    out_path = input_path.parent / basis_value
    existed = out_path.exists()
    out_path.write_text(text)
    print(f"{'Overwrote' if existed else 'Wrote'} {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
