# ReRDMFT

Relativistic Reduced Density Matrix Functional Theory, implemented in
C++17. Includes a 4-component Dirac-Hartree-Fock SCF (restricted
kinetic balance spinor basis, `C4_DHF`) and a standard nonrelativistic
Hartree-Fock SCF (`NON_REL`) for comparison/validation.

## Prerequisites

- A C++17 compiler (`g++`; the Makefile assumes GCC-style flags).
- [LIBCINT](https://github.com/sunqm/libcint), built as a static
  library (`libcint.a`) with its headers in an `include/` directory
  next to it -- exactly what libcint's own CMake build produces, e.g.:

  ```sh
  git clone https://github.com/sunqm/libcint.git
  cd libcint && mkdir build && cd build
  cmake .. && make
  ```

  This gives you `libcint/build/libcint.a` and `libcint/build/include/`.
- LAPACKE, LAPACK, and BLAS (e.g. on Debian/Ubuntu:
  `sudo apt install liblapacke-dev liblapack-dev libblas-dev`).
- OpenMP support in your compiler (bundled with GCC; used to
  parallelize two-electron integral construction and the Fock-matrix
  builds).

## Building

The `LIBCINT` variable must point at your compiled `libcint.a`; the
Makefile has no default for it and errors out if it is unset:

```sh
make LIBCINT=/path/to/libcint/build/libcint.a
```

If your libcint headers live somewhere other than
`$(dirname LIBCINT)/include`, also set `LIBCINT_INC`:

```sh
make LIBCINT=/path/to/libcint.a LIBCINT_INC=/path/to/libcint/headers
```

This produces the `rerdmft` binary in the repository root.

### Persisting the LIBCINT path

To avoid retyping `LIBCINT=...` on every build, create a git-ignored
`Makefile.local` in the repository root with:

```make
LIBCINT := /path/to/libcint/build/libcint.a
```

Plain `make` then picks it up automatically.

### Cleaning

```sh
make clean
```

## Running

```sh
./rerdmft examples/water.inp
```

See `examples/*.inp` for sample input files (basis sets, geometries,
and the `C4_SPINOR` / `NON_RELATIVISTIC` flags controlling which SCF
path(s) run).

### Live progress

The result report is printed at the end of the run, so a long calculation would otherwise show nothing while it
works. While it runs, `./rerdmft examples/name.inp` writes a live log to `examples/name.live` (next to the input
file, truncated at the start of every run, flushed after every line): each phase with its time, every SCF
iteration, the Cholesky decomposition (every 200 vectors), every NEO Newton step (energy, gradient, trust radius,
number of Hessian products) and every `FULL_OPTIMIZATION` macro-iteration, prefixed with the elapsed wall time and
the method (`NON_REL`, `X2C_HF`, `C4_DHF`). Follow it with `tail -f examples/name.live`.

## Input file keywords

Keywords are case-insensitive, one per line, optionally followed by
`=` before the value (e.g. `DEBUG TRUE`, `DEBUG = TRUE`, and
`DEBUG=TRUE` are all equivalent). Lines starting with `#` (or a `#`
anywhere on a line) are comments.

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `NELEC` (or `NELECTRONS`) | int | *required* | Number of electrons. |
| `BASIS` | string | *required* | Gaussian basis set file name. `src/Utils/download_basis.py <input_file.inp>` fetches it for you: it reads this keyword and the `GEOMETRY` block, downloads the matching basis from [Basis Set Exchange](https://www.basissetexchange.org) in Gaussian94 format for exactly the elements present, and writes it next to the input file under this exact name -- accepts a plain basis name (`BASIS 6-31G.gbs`) or this project's own `<molecule>-<basis>.gbs` convention (`BASIS lih-6-31g.gbs`), trying candidate names against Basis Set Exchange itself rather than guessing. |
| `UNIT_LENGTH` | string | `ANGS` | Units of the `GEOMETRY` block's coordinates: `ANGS` (Angstrom, the default, converted to Bohr internally via the CODATA Bohr radius) or `BOHR`/`AU` (already atomic units, no conversion). Applied after the whole input file is parsed, so it may appear before or after `GEOMETRY`. Template: `examples/lih_gnof_unit_length_bohr.inp` (same system and converged energy as `lih_gnof_full_optimization.inp`, geometry given in Bohr instead of Angstrom). |
| `GEOMETRY` ... `END` | block | *required* | Molecular geometry as `<symbol> <x> <y> <z>` lines, one atom per line, coordinates in the units `UNIT_LENGTH` says (Angstrom by default), converted to Bohr internally. |
| `NON_RELATIVISTIC` | bool | `FALSE` | Run the standard nonrelativistic Hartree-Fock SCF (`NON_REL`). |
| `CARTESIAN` | bool | `TRUE` | Only affects `NON_RELATIVISTIC`: `TRUE` (the default, bit-identical to this project's long-standing behavior) runs it in the raw Cartesian large-component AO basis libcint produces; `FALSE` runs its entire one-/two-electron pipeline in the spherical large-component basis instead (`Utils/SphericalTransform.h`, the same transform `X2C`/`C4_SPINOR` always use, see below), removing `l >= 2` Cartesian contaminant combinations from its results too. `X2C`/`C4_SPINOR` are unaffected by this keyword -- they always use the spherical basis, no option, matching DIRAC. |
| `PRECONDITION_SMALL_OVERLAP` | bool | `FALSE` | Diagonal (Jacobi) preconditioning of the RKB Small-component overlap before its `S^-1/2` (`Linear_Algebra/LinearAlgebra.h`'s `inverseSqrtHermitian`), only relevant to `X2C`/`C4_SPINOR`. Added while the Small-component basis was still uKB (pooled, heavy-element near-duplicate exponents made the overlap genuinely near-singular); with true analytic RKB (one Small partner per Large function, no pooling) plus the spherical/LOWGEN large-component reduction, a basis-only diagnostic on Xe/dyall.v2z shows the overlap is no longer anywhere near singular either way (smallest eigenvalue ~0.35, nine orders of magnitude above the LOWGEN threshold), and DIRAC's own `LOWGEN` (`dirone.F`) diagonalizes its overlaps raw, with no such rescaling at all -- hence the default is now `FALSE`. Kept as a keyword (not removed) in case a more extreme basis than `dyall.v2z` ever needs it. |
| `C4_SPINOR` | bool | `FALSE` | Run the 4-component Dirac-Hartree-Fock SCF (`C4_DHF`), building the RKB two-electron Coulomb tensor. Opt-in since both time and memory cost scale steeply with basis size. Both this SCF and the `X2C` one are Kramers-restricted for an even `NELEC`: every iteration's density is projected onto its time-reversal-even part (spin-orbit mixing of the spinors is kept; only the magnetization is removed), so they cannot drift into a lower-energy Kramers-broken solution at unstable geometries (e.g. stretched LiH with `CHOLESKY TRUE`). The output reports the largest element removed (~0 when the iteration stayed symmetric by itself). |
| `DEBUG` | bool | `FALSE` | Print detailed basis/matrix diagnostics, plus internal cross-checks (efficient-vs-RDMFT-ansatz gradient formulas, finite-difference gradient/Hessian tests) for whichever of `NON_RELATIVISTIC`/`C4_SPINOR` is on. Every test that needs the two-electron integrals as a DENSE tensor (the RDMFT-ansatz gradient test, the Kramers/spin structure tests of the integrals, exact-vs-Cholesky rotation, the Hessian-diagonal finite difference, dense-vs-Cholesky comparisons of the Fock matrix and MO integrals) runs only with `DEBUG TRUE`; the dense tensors are built for it on demand. |
| `HESSIAN_MEAN_FIELD` | bool | `FALSE` | For every method that is on (`NON_RELATIVISTIC`, `X2C`, `C4_SPINOR`), build the full dense real-step orbital-rotation Hessian of the converged HF/DHF solution (integer occupations, Hartree/exchange formulas) and diagonalize it completely, reporting the numbers of negative, near-zero and positive eigenvalues: a genuine minimum is expected for `NON_REL` and `X2C`, a saddle for `C4_SPINOR` (negative-energy branch included). O(n^5)/O(n^6), needs the dense two-electron tensor, opt-in. |
| `FCIDUMP` | bool | `FALSE` | Only for `NON_RELATIVISTIC` (the only case currently supported), and requires `FUNCTIONAL`. After that method's RDMFT functional evaluation (occupation optimization, and `FULL_OPTIMIZATION`'s orbital rotation too if that ran), writes an `FCIDUMP` file (`Utils/Fcidump.h`, the MOLPRO/PySCF convention: `&FCI` namelist header, then one integral per line as `VALUE I J K L`) of the one- and two-electron MO integrals in the spin-up channel, in the SAME orbital order the RDMFT calculation itself used (real orbitals, `MS2=0`; spin-restricted, so spin-down is identical) -- for handing off to an external CI/DMRG/FCI code. `ORBSYM` is all `1` (no point-group symmetry tracked). Works with `CHOLESKY` TRUE or FALSE (the two-electron integrals come from the AO Cholesky vectors in the former case, the exact AO tensor in the latter -- converged energies agree exactly either way, individual integral elements to the Cholesky threshold). Templates: `examples/lih_gnof_fcidump.inp`, `examples/lih_gnof_fcidump_cholesky.inp` (`CHOLESKY TRUE` counterpart). |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (atomic units): larger probes the nonrelativistic limit, smaller exaggerates relativistic effects. |
| `MIXING` | double, in `(0, 1]` | `0.4` | Linear density-mixing weight for the SCF loops, used only with `DIIS FALSE`. |
| `DIIS` | bool | `TRUE` | Use Pulay DIIS (`Utils/DIIS.h`, commutator error `F P S - S P F`) instead of linear density mixing in all three SCF loops. Typically 5-10x fewer iterations. Converges to *a* stationary point, not always the one mixing finds -- stretched LiH (`examples/lih_X2C_gnof_full_optimization_dissociated.inp`) sets `DIIS FALSE` for that reason. |
| `DIIS_SIZE` | int (>= 2) | `5` | Number of (error, Fock) pairs kept by `DIIS TRUE`. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum number of `C4_DHF` SCF cycles before giving up. |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` Hartree | `C4_DHF` SCF energy-change convergence threshold (OR'd with `DENSITY_TOLERANCE`). |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold. |
| `CHOLESKY` | bool | `FALSE` | Hold the two-electron integrals as Cholesky vectors and never build an n^4 object. The real AO Coulomb matrix is decomposed ONCE (`Utils/Cholesky_Decomposition.h`: NON_REL/X2C decompose the AO integrals, C4_SPINOR the combined {Large-Large} u {Small-Small} pair matrix, `C4_DHF/RkbCholesky.h`); the SCF Fock matrices, the MO-basis integrals (`Utils/AoCholesky.h`) and `FULL_OPTIMIZATION` all work on those vectors, and the AO integrals are not used again (they are rebuilt only under `DEBUG`, for the dense-vs-Cholesky checks). Every decomposition is verified against the integrals and retried with a smaller pivot batch if it misses `100*CHOLESKY_THRESHOLD + 1e-9`. With `ORBITAL_OPTIMIZER NEO` the Hessian-vector product is a finite difference of the gradient on the rotated vectors, O(N_chol n^3) with no dense cache. Without `CHOLESKY` the integrals are held as unique-element stores (`Utils/SymmetricEri.h`: about n^4/8 real or n^4/4 complex numbers, the rest rebuilt by symmetry) and transformed in slabs, never as a dense n^4 tensor. |
| `CHOLESKY_THRESHOLD` | double (> 0) | `1e-10` | Residual-diagonal cutoff for the decomposition; looser = fewer vectors (faster, less accurate), tighter = more (slower, more exact). Only with `CHOLESKY TRUE`. |
| `ON_DEMAND_ERI` | bool | `TRUE` | Only with `CHOLESKY TRUE`: how the pivoted decomposition reads its Coulomb pair matrix (`Utils/Cholesky_Decomposition.h`'s `PairMatrixSource`). `TRUE`: every quadruplet -- `(LL\|LL)`/`(LL\|SS)`/`(SS\|SS)` for `C4_SPINOR` (`C4_DHF/RkbCholesky.cpp`'s `OnDemandUnionCoulombPairs`), the single-basis `(LL\|LL)` for `NON_RELATIVISTIC`/`X2C` (`Utils/AoCholesky.cpp`'s analogous class) -- is evaluated ON DEMAND, as the decomposition needs it; no packed/dense AO tensor is ever materialized, so memory stays O(n^2) (the Schwarz diagonals) regardless of basis size. `FALSE`: the classic route, build the full packed tensor once, then decompose it -- faster whenever that tensor fits in memory. Defaulting to `TRUE` makes a packed-tensor `std::bad_alloc` (hit on Xe/dyall.v2z, whose `(SS\|SS)` alone needs ~85 GB) structurally impossible by default; `examples/` sets this `FALSE` on every `CHOLESKY TRUE` input except `lih_gnof_c4_full_optimization_cholesky.inp`, which keeps the default to demonstrate the on-demand path (validated there to be bit-identical to `FALSE`). |
| `X_LIN_DEP_THRS_L` | double (> 0) | `1e-6` | DIRAC's own STOL(1): the large-component LOWGEN safety net's threshold (`Utils/LinearAlgebra.h`'s `canonicalOrthogonalize`) -- below this eigenvalue, a direction in the large-component overlap is genuinely DROPPED (the orthonormalizing matrix comes back with fewer columns than rows) rather than kept-and-inverted. Used identically by `NON_RELATIVISTIC` (on its own `s_large_cart`, `CARTESIAN TRUE` or `FALSE`) and by `X2C`/`C4_SPINOR` (on their own spherical large-component overlap, beyond whatever the deterministic Cartesian-to-spherical reduction already removes) -- one keyword, same value, all three call sites. A no-op (bit-identical to the plain symmetric-orthogonalization result) for every basis in `examples/`, which never actually crosses this threshold. |
| `X_LIN_DEP_THRS_S` | double (> 0) | `1e-8` | DIRAC's own STOL(2): the RKB small-component overlap's own LOWGEN threshold (`canonicalOrthogonalizeHermitian`) -- only meaningful for `X2C`/`C4_SPINOR` (the only methods with a small component). Tighter than `X_LIN_DEP_THRS_L` by convention (DIRAC's own choice; see `doc/RKB.tex`'s empirical note on why the small side is generally better-conditioned once the large side's own reduction is in effect). A genuine drop here throws (the asymmetric-dimension case, DIRAC's `NESH != NPSH`, that this project does not yet thread through the rest of the pipeline) rather than proceeding unsafely. |
| `FUNCTIONAL` | string | *(none)* | Selects the density matrix functional to evaluate on the converged orbitals: a JK-only functional (`Occ_opt/JK_only.h`) -- `SD`, `MBB`/`MULLER`, `BBC2`, `CA`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER` -- a Piris natural orbital functional (`Occ_opt/PNOFs.h`) -- `PNOF5`, `PNOF7`, `PNOF7S`, `GNOF` -- or `PCCD` (`Occ_opt/pCCD.h`, Kramers-restricted pCCD: t-/z-amplitude equations solved at fixed orbitals, then, with `FULL_OPTIMIZATION TRUE`, orbitals re-optimized exactly like PNOF/JK_only via the same generalized Fock/gradient machinery). Unset: the whole RDMFT evaluation step below is skipped. |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | Initial fractional occupations for a JK-only `FUNCTIONAL`. `PROPORTIONAL`: aufbau redistributed into an interior box. `FERMI_DIRAC`: smeared at `TEMPERATURE`. (PNOF functionals build their own guess.) |
| `JK_FROZEN_PAIRS` | int (>= 0) | `0` | Only with a JK-only `FUNCTIONAL`. Freezes the `2*JK_FROZEN_PAIRS` LOWEST-energy spin-orbitals/spinors at EXACTLY occupation 1 (never an SQP variable), mirroring PNOF's own frozen core. Counted in pairs (spin/Kramers partners) so no separate evenness check is ever needed. |
| `JK_ACTIVE_PAIRS` | int (>= 1) | all remaining | Only with a JK-only `FUNCTIONAL`. The NEXT `2*JK_ACTIVE_PAIRS` spin-orbitals/spinors by energy (above `JK_FROZEN_PAIRS`) are the fractional-occupation SQP window, `sum(n) = NELEC - 2*JK_FROZEN_PAIRS`; everything above that is deep virtual, pinned at exactly 0. `NELEC - 2*JK_FROZEN_PAIRS` must be strictly between 0 and `2*JK_ACTIVE_PAIRS`, else a clear error. |
| `TEMPERATURE` | double (> 0) | `1000` (Kelvin) | Smearing temperature, only used with `OCCUPATION_INIT FERMI_DIRAC`. |
| `PNOF_SUBSPACES` | int (>= 1) | `1` | Only with a PNOF `FUNCTIONAL`. Number of independent coupling subspaces built outward from HOMO (`Occ_opt/Orb_subspaces.h`); throws if it exceeds the occupied pairs available. |
| `PNOF_COUPLING` | int (>= 2) | `2` | Only with a PNOF `FUNCTIONAL`. Size of each subspace in pairs: 1 occupied + (`PNOF_COUPLING`-1) unoccupied. `2` is plain HOMO-LUMO pairing. |
| `SQP_PNOF_OCC` | bool | `FALSE` | Only with a PNOF `FUNCTIONAL`. `FALSE`: optimize via `Utils/LBFGS.h` over unconstrained gamma angles (DoNOF's own approach). `TRUE`: optimize via `Utils/SQP.h` over occupations directly, with explicit box+equality constraints. Both agree to full precision when both converge. |
| `PCCD_FROZEN_PAIRS` | int (>= 0) | `0` | Only with `FUNCTIONAL PCCD`. Freezes the `2*PCCD_FROZEN_PAIRS` LOWEST-energy spin-orbitals/spinors at EXACTLY occupation 1 (never a t-/z-amplitude variable) -- the frozen-core approximation, with no separate formula needed: a frozen pair is simply one with no amplitude at all (`x=0` identically), which the pCCD RDM formulas already reduce to correctly at that limit. Same pairs-not-electrons counting convention as `JK_FROZEN_PAIRS`. |
| `PCCD_ACTIVE_PAIRS` | int (>= 1) | all remaining | Only with `FUNCTIONAL PCCD`. The NEXT `2*PCCD_ACTIVE_PAIRS` spin-orbitals/spinors by energy (above `PCCD_FROZEN_PAIRS`) become the t-/z-amplitude window, split into occupied/virtual pairs at `NELEC - 2*PCCD_FROZEN_PAIRS` electrons; everything above that is deep virtual, excluded entirely (pCCD never even builds coefficients for it). |
| `PCCD_AMPLITUDE_SOLVER` | string | `NEWTON` | Only with `FUNCTIONAL PCCD`. `NEWTON`: exact Newton-Raphson, rebuilding and directly solving the analytic Jacobian every iteration (quadratically convergent; the z-equation is linear in z, so it converges in one iteration for free once t is converged). `LBFGS`: `Utils/LBFGS.h` minimizing 0.5\*\|\|residual\|\|^2 with the exact chain-rule gradient, no Jacobian ever solved. |
| `FULL_OPTIMIZATION` | bool | `FALSE` | After occupation optimization (needs `FUNCTIONAL`), macro-iterate to convergence: an orbital-rotation step (`ORBITAL_OPTIMIZER`) at fixed occupations, then occupation re-optimization at the new orbitals, until `|E-E_old| < MACRO_ENERGY_TOLERANCE`. Works for `NON_REL` (real spin-orbitals), `X2C` (complex, Kramers-restricted rotations), and `C4_SPINOR` (complex, Kramers-restricted rotations *restricted to the positive-energy spinors only* -- see below). Validation checks gate the loop, and a final test verifies the optimized orbitals keep the expected symmetry (Kramers pairing for X2C/C4_SPINOR, spin symmetry for NON_REL, pair-symmetric energy for PNOF). |
| `MAX_MACRO_ITERATIONS` | int | `1000` | Maximum number of macro-iterations of `FULL_OPTIMIZATION`. |
| `MACRO_ENERGY_TOLERANCE` | float | `1e-9` | Energy convergence threshold of the macro-iteration loop. |
| `ORBITAL_GRADIENT_TOLERANCE` | float | `1e-5` | ADAM's/NEO's orbital-gradient convergence threshold (max gradient entry). |
| `ORBITAL_OPTIMIZER` | string | `ADAM` | Which method drives `FULL_OPTIMIZATION`'s orbital-rotation step. `ADAM`: DoNOF's own first-order optimizer (`Utils/ADAM.h`). `NEO`: `Utils/NEO.h`'s matrix-free, second-order Newton method targeting the ground state, using a row-based Hessian-vector product (no dense Hessian formed); also works with `CHOLESKY TRUE` and `C4_SPINOR`. Converges in far fewer macro-iterations than ADAM and usually matches its energy to 1e-6-1e-9. On some PNOF/GNOF NON_REL systems NEO can land on a different stationary point; with `CHECK_HESS_NEO TRUE` a post-loop Hessian check (lowest 3 eigenvalues, Davidson) detects a saddle and automatically escapes it (perturb along the negative-curvature eigenvector, retry up to 3 times), but a residual gap to a genuine alternate minimum is reported rather than silently fixed -- compare against `ADAM` as a routine cross-check. `ADAM_NEO`: a hybrid -- every macro-iteration re-decides its driver from the PREVIOUS one's own `|dE|`: ADAM while it is above `ADAM_NEO_SWITCH_TOLERANCE`, NEO once it drops to or below that, and back to ADAM whenever it grows above it again (iteration 1 always starts on ADAM, no previous `|dE|` yet). With `DEBUG TRUE` the orbital-rotation Hessian is additionally diagonalized densely at the end (for any optimizer) and its negative eigenvalues are counted. Templates: `examples/*_neo_full_optimization.inp`, `examples/lih_gnof_adam_neo_full_optimization.inp`. |
| `NEO_MAX_ITERATIONS` | int | `100` | Only with `ORBITAL_OPTIMIZER NEO` or `ADAM_NEO`: the hard cap on Newton steps *one macro-iteration's* NEO-driven orbital-rotation descent may take. NEO always runs each descent to its own full `ORBITAL_GRADIENT_TOLERANCE` convergence rather than an ADAM-style growing budget -- cutting it short hands the occupation re-optimization a not-actually-stationary point, which on some PNOF/GNOF systems locks the macro loop into a measurably worse final answer with no way back (see `Full_opt/FullOptimization.h`'s own comment). Lower it only to bound run time on a system already known to be slow; the default is generous since NEO is quadratically convergent and rarely needs more than a few tens of steps. |
| `ADAM_NEO_SWITCH_TOLERANCE` | double | `1e-4` | Only with `ORBITAL_OPTIMIZER ADAM_NEO`: the `|dE|` threshold (Hartree) between two macro-iterations that switches the driver from ADAM to NEO (once `|dE|` falls to or below this) and back to ADAM (once it rises above it again). Re-evaluated every macro-iteration, so the run can switch back and forth many times. Switching which parametrization drives the Kramers-/spin-restricted rotation subspace (ADAM's "representative" embedding vs. NEO's "isometric" one, `Utils/KramersRestriction.h`) is exact by construction, and this is checked directly on the data at every switch: a "switching orbital optimizer: ..." log line reports the Kramers-pair (X2C, C4_SPINOR) or spin (NON_REL) deviation of the integrals being handed off (the one-electron part always; the two-electron part under `DEBUG TRUE`, which needs the dense tensor). |
| `CHECK_HESS_NEO` | bool | `FALSE` | Only with `ORBITAL_OPTIMIZER NEO` or `ADAM_NEO`: after a macro-iteration loop that ended on NEO converges (or hits `MAX_MACRO_ITERATIONS`), performs the post-loop Hessian check described under `ORBITAL_OPTIMIZER` -- a matrix-free block Davidson diagonalization (`Utils/NEO.h`'s `neoLowestHessianEigenpairs`) of the lowest 3 orbital-rotation Hessian eigenvalues, confirming a genuine minimum or, on a saddle, automatically escaping and re-optimizing (up to 3 attempts). **Off by default**, which means NEO's gradient-only stopping rule is trusted as-is: if it reports "converged" at a point that is actually a shallow saddle (see `examples/lih_pnof5_neo_saddle_escape.inp`'s own header, a real case ~2e-5 Hartree above the true minimum), nothing here will detect or fix that -- the run keeps that point silently, exactly as NEO always did before this keyword existed. Turn it on when you cannot already cross-check the NEO energy against `ADAM` on the same system, or whenever you have any reason to suspect a saddle; the cost is extra Hessian-vector products beyond the Newton steps already taken. Independent of `DEBUG TRUE`'s own dense diagonalization check (exact, but only affordable for small systems), which still runs on its own when requested regardless of this keyword. |
| `FIXED_OCCUPANCIES` | bool | `FALSE` | Only meaningful with `FULL_OPTIMIZATION TRUE` (`NON_RELATIVISTIC`, `X2C` or `C4_SPINOR` alike -- the same macro-iteration loop, `Full_opt/FullOptimization.cpp`, drives all three, including `FULL_OPTIMIZATION_4C_NEG`'s min-max stage). The occupation-number optimization that runs ONCE at the starting HF/DHF orbitals, before the macro loop, is unaffected. When `TRUE`, the macro loop itself never re-optimizes occupations again: each macro-iteration only re-optimizes the orbitals at those fixed occupations (logged as `occupations-fixed`), turning `FULL_OPTIMIZATION` into a pure orbital optimization at fixed occupation numbers instead of its usual alternation of the two. For `FUNCTIONAL PCCD` this also freezes the full 2-RDM, not just the occupations: `model.optimize_occupations` (skipped entirely when this is `TRUE`) is the SAME call that re-solves the t-/z-amplitudes and refreshes the cached pair-transfer/density-density matrices, so skipping it leaves the amplitudes -- and therefore `D_pq`/`Q_pq`, not only their diagonal `n_p` -- exactly as they were before the macro loop, verified directly (identical printed occupation numbers before/after a 22-macro-iteration orbital-only run on Ne/cc-pVDZ). Template: `examples/lih_gnof_fixed_occupancies.inp`. |
| `READ_OCCUPANCIES` | bool | `FALSE` | Requires `FUNCTIONAL`. Before the occupation-number optimization that normally runs once at the starting HF/DHF orbitals, reads a plain-text file named `OCC.in` in the working directory (one line per geminal/pair: `<index> <occupation>`) and uses those occupations DIRECTLY -- the optimizer (SQP/LBFGS for PNOF, SQP for JK_only) does not run at all for that stage. For PNOF, `index` is the SUBSPACE number (`0..PNOF_SUBSPACES-1`): the file needs exactly `PNOF_SUBSPACES` groups of `PNOF_COUPLING` lines each, in file order within a group (first line of a group is that subspace's principal geminal), summing to 1 per subspace (renormalized if not exact); core (frozen) geminals are NOT listed. For JK_only functionals `index` is ignored -- only the occupation column matters, one line per Kramers/spin-tied active pair, in the same order the active window itself uses, summing to `NELEC - 2*JK_FROZEN_PAIRS`. If `READ_RESTART` is also `TRUE`, orbitals still come from the RESTART file, but these occupations override the RESTART file's own. `FULL_OPTIMIZATION`'s own macro loop (if it runs) is unaffected by this keyword alone -- combine with `FIXED_OCCUPANCIES TRUE` to also keep it from re-optimizing them. Template: `examples/lih_gnof_read_occupancies.inp` (with its companion `examples/OCC.in`). |
| `READ_RESTART` | bool | `FALSE` | Requires a `FUNCTIONAL`. `TRUE` skips the HF/DHF SCF of every requested method (`NON_RELATIVISTIC`, `X2C`, `C4_SPINOR`) and starts the functional calculation from `RESTART.NON_REL` / `RESTART.X2C_HF` / `RESTART.4C` of an earlier run, possibly at another geometry (potential-energy scans) -- see *Restarting from a previous run* below. |
| `FULL_OPTIMIZATION_4C_NEG` | bool | `FALSE` | Only meaningful for `C4_SPINOR` + `FULL_OPTIMIZATION` (any `FUNCTIONAL`, `CHOLESKY` TRUE or FALSE). After the positive-energy-only optimization has converged, runs the genuine **min-max** stage: orbital rotations now include the positive <-> negative-energy pairs, driven by NEO to a saddle point (whatever `ORBITAL_OPTIMIZER` says), alternating with a full re-minimization of the occupation numbers -- see below. Skipped, with a message, if the first stage did not converge. Validated for `FUNCTIONAL PCCD` too (fixed: the amplitude warm-up that pre-populates pCCD's frozen 2-RDM cache used to solve against the UNROTATED starting integrals instead of the no-pair minimum's own rotated ones -- a basis mismatch that made NEO's very first trust-region step wildly oversized and the whole search diverge catastrophically; confirmed on LiH/6-31G and Ne/cc-pVDZ before the fix, both converge cleanly to the expected O(1/c^2) correction now -- see `examples/ne_pccd_c4_neg.inp`). |
| `X2C` | bool | `FALSE` | Print the one-electron X2C decoupling report and run the approximate X2C-HF SCF (see below), between the `NON_RELATIVISTIC` and `C4_SPINOR` reports. Independent of `C4_SPINOR` (the RKB Hamiltonian it needs is always built). With `DEBUG`, adds extra cross-checks. |

## X2C decoupling and X2C-HF

Setting `X2C` runs everything below, built from
`X2C_DHF/X2C_decoupling.h`, `X2C_DHF/X2C_hamiltonian.h`, and
`X2C_DHF/X2C_HF.h`, and prints it **between** the `NON_RELATIVISTIC`
and `C4_SPINOR` (4-component DHF) final reports, regardless of whether
either of those keywords is itself set (X2C sits conceptually between
the nonrelativistic and exact 4-component treatments). With `DEBUG`
also set, extra detail is added throughout (called out per step
below); the underlying computation itself is unaffected by `DEBUG`.

Both this and `C4_SPINOR` share one upstream piece: `X_full =
diag(X_Large, X_Large, X_Small)`, built the way DIRAC itself builds its
own RKB basis, rather than the numerical uKB-projection scheme earlier
versions of this project used:

- **Analytic restricted kinetic balance**: each Small-component AO is
  built directly, in closed form, as `sigma.p` applied to its own
  Large-component partner (`d/dx[x^lx y^ly z^lz exp(-a r^2)]`'s
  lowering/raising pieces, `RKB/RkbDerivativeTerms.h`) -- exactly ONE
  RKB Small partner per Large function, by construction, with no
  redundant unrestricted-kinetic-balance (uKB) basis to pool or project
  down from (`RKB/RkbTransformation.h`'s `rkbCoefficients` is now pure
  bookkeeping of these closed-form weights, not a numerical `C = M S^+`
  solve).
- **Spherical Large-component basis**: `X2C`/`C4_SPINOR` always work in
  a spherical (real-solid-harmonic) Large-component basis, not the raw
  Cartesian one libcint produces -- `l >= 2` Cartesian shells (6
  Cartesian `d`'s for 5 genuine spherical harmonics, 10 `f` for 7, ...)
  are reduced to their non-redundant spherical combinations
  (`Utils/SphericalTransform.h`, derived from first principles: a
  Laplacian null-space computation, not a transcribed coefficient
  table). This transform is applied to `rkb_coefficients`' own Large-
  orbital rows *before* the RKB projection above happens, so "exactly
  one RKB Small partner per Large function" automatically lands at the
  new, smaller spherical dimension -- Small's own dimension tracks
  Large's exactly, with no separate bookkeeping needed anywhere
  downstream. `NON_RELATIVISTIC` is unaffected unless `CARTESIAN FALSE`
  opts it into the same treatment (see above).
- **LOWGEN safety net**: on top of the exact spherical reduction,
  `Linear_Algebra/LinearAlgebra.h`'s `canonicalOrthogonalize`/
  `canonicalOrthogonalizeHermitian` (DIRAC's own `LOWGEN`, with its
  `STOL(1)=1e-6`/`STOL(2)=1e-8`-style thresholds) catches any further,
  *genuine* near-linear-dependence -- e.g. near-duplicate exponents
  across different shells of a heavy element's uncontracted basis --
  that the deterministic spherical step does not remove. On the Large
  side this is folded into the SAME transform used above (so it can
  never desync Large from Small); it is a no-op (bit-identical results)
  for every basis this project has been tested on. If the Small side
  ever needs an *independent* reduction beyond what tracking Large
  already gives it -- the genuine `n_positive != n_negative` case DIRAC
  itself accepts (`NESH != NPSH`) -- this project does not yet thread
  that asymmetric dimension through the rest of the pipeline
  (`FullOptimization`'s no-pair slicing, RESTART, ...), so it stops with
  a clear, actionable error rather than silently mismatching dimensions.

1. **Decoupling**: diagonalizing the orthonormalized 4-component
   Hamiltonian `H_RKB_ortho = X_full^dagger H_RKB X_full` block-
   diagonalizes the Dirac equation into positive-/negative-energy
   branches -- this diagonalization *is* the one-electron X2C
   transformation. Its eigenvalues are printed in two columns (adjacent
   Kramers pairs side by side), along with the Kramers-pair splitting
   and the eigenvector-partner-deviation check (both always printed,
   confirming Kramers' theorem holds, never hidden behind `DEBUG`).
   With `DEBUG`, this also prints a check that `C_tmp = X_full * U`
   genuinely solves the *original* generalized eigenvalue problem
   `H_RKB * C_tmp = S_full * C_tmp * E`.
2. **Exact X2C Hamiltonian**: eliminating the small component from the
   positive-energy block (via the decoupling matrix `R = C_S * C_L^-1`
   and the exact renormalization metric `Lambda = S_LL + R^dagger S_SS
   R`) gives a Hermitian Hamiltonian acting purely on the large-
   component space. Diagonalizing it reproduces the positive-energy
   spectrum from step 1 *exactly* (no approximation is introduced
   anywhere in the construction) -- the no-pair approximation is simply
   using this Hamiltonian alone, since the negative-energy branch never
   appears in it.
3. **Approximate X2C Hamiltonian**: the same one-electron effective
   Hamiltonian, but orthogonalized with only the plain large-component
   overlap (`X_Large`) instead of the exact renormalization metric --
   a common simplification. Its eigenvalues are close to, but do not
   exactly reproduce, the true spectrum (the deviation is largest for
   the most relativistic, deepest-lying orbitals); Kramers symmetry is
   still exact.
4. **Approximate X2C-HF SCF**: a full self-consistent Hartree-Fock
   treatment using the exact X2C Hamiltonian from step 2 as a *fixed*
   one-electron core (no picture-change correction as the density
   changes) and the ordinary non-relativistic two-electron Coulomb
   integrals over the Large-component basis (no two-electron
   picture-change correction either). Because the density is a general
   complex Hermitian matrix (spin-orbit coupling in the core
   Hamiltonian mixes alpha/beta already at the one-electron level), the
   Fock build includes genuine opposite-spin exchange. Every iteration
   orthogonalizes the Fock matrix with only the plain large-component
   overlap (step 3's approach, not step 2's exact metric), diagonalizes
   it to get eigenvectors `U`, and builds the density matrix from
   `C = X_Large * U` -- so this is a deliberately approximate SCF,
   confirmed to reduce exactly to ordinary `NON_RELATIVISTIC` HF in the
   `SPEED_OF_LIGHT -> infinity` limit, but its converged energy at
   realistic `SPEED_OF_LIGHT` can fall *below* the exact 4-component DHF
   energy (the usual variational bound does not apply once the metric is
   no longer exact) -- a known consequence of skipping both
   picture-change corrections, not a bug. The converged orbital energies
   are printed the same way as step 1, with both the Kramers-pair
   splitting and the eigenvector-partner-deviation check always shown
   (never hidden behind `DEBUG`, matching `C4_SPINOR`'s own
   `Fock_ortho` Kramers check). With `DEBUG`, this also prints each
   iteration's orbital energies and, at convergence, a check that
   `C = X_Large * U` genuinely solves `F * C = S_Large * C * E` in the
   original (non-orthogonal) Large-component AO basis.
5. **Gradient/Hessian test suite**: after the SCF converges, `h_x2c`
   and the Large-component spin-orbital two-electron integrals are
   transformed into the converged X2C-HF MO basis
   (`X2C_DHF/X2C_MoTransform.h`), and the SAME `Hessian_opt` test suite
   `C4_SPINOR` runs for DHF is run here too: the RDMFT-ansatz gradient
   (always on); under `DEBUG`, the X2C-HF-specific efficient gradient
   (`X2C_DHF/X2C_OrbitalGradient.h`) and a finite-difference gradient/Hessian
   check;
   and, under `HESSIAN_MEAN_FIELD`, the full orbital-rotation Hessian
   diagonalization. Unlike `C4_DHF`'s saddle point, X2C-HF's converged
   solution is a genuine **minimum** (no negative eigenvalues) -- X2C's
   own decoupling already eliminated the negative-energy branch, so
   there is no downhill rotation direction left for the occupied
   spinors to admit. Confirmed on water/STO-3G (91x91 Hessian: 0
   negative, 40 positive, min eigenvalue `~-1e-11`) and CO/STO-3G
   (190x190: same result) -- and all three independently-derived
   gradient formulas (RDMFT-ansatz, efficient, general) agree to
   machine precision at both.

See `examples/water_X2C.inp` for a plain worked example (now including
`HESSIAN_MEAN_FIELD TRUE`), or `examples/water_X2C_debug.inp` for the same run
with `DEBUG TRUE` (all the extra cross-checks described above, plus
`HESSIAN_MEAN_FIELD`).

## RDMFT functional evaluation

Setting `FUNCTIONAL` runs an additional step after each requested SCF
(`NON_RELATIVISTIC`/`C4_SPINOR`/`X2C`'s own X2C-HF) converges, using its
orbitals and one-/two-electron integrals as a **fixed** background (no
orbital reoptimization):

1. Generate initial fractional occupation numbers via `OCCUPATION_INIT`
   (`PROPORTIONAL` by default, or `FERMI_DIRAC` at `TEMPERATURE`).
2. Evaluate `FUNCTIONAL`'s energy on those occupations
   (`Occ_opt/JK_only.h` + `Hessian_opt/HartreeExchangeGradient.h`'s
   `hartreeExchangeEnergy`).
3. Optimize the occupation numbers further via a sequential quadratic
   programming solver (`Occ_opt/SQP.h`), minimizing that same energy
   subject to `sum(n_p) = NELEC` and `0 < n_p < 1`. The SQP solves in a
   REDUCED space with one variable per Kramers (or, for `NON_RELATIVISTIC`,
   spin) pair, both partners tied to it -- so partners always get EXACTLY
   the same occupation by construction, not merely a close numerical
   coincidence -- and reports the optimized occupations, their sum, and
   the optimized energy.

For `C4_SPINOR`, the negative-energy (Dirac sea) branch is excluded
from both steps entirely (pinned at exactly zero occupation, never an
optimization variable), preserving the no-pair approximation; for `X2C`
(like `NON_RELATIVISTIC`), every spinor competes for occupation, since
X2C's own decoupling already eliminated that branch entirely. Results
are printed after the corresponding SCF's own energy, gradient, and
(if requested) Hessian diagnostics -- optimized occupation numbers are
listed at fixed 5-decimal precision, in two columns for `C4_SPINOR` and
`X2C` (adjacent Kramers pairs side by side, printed identical by
construction) and one column otherwise (`NON_RELATIVISTIC`'s own spin
pairs, not printed side by side, but tied via the same SQP mechanism).

See `examples/water_muller.inp` and `examples/co-sto-3g_muller.inp` for
worked `C4_SPINOR`/`NON_RELATIVISTIC` examples, or
`examples/water_X2C_muller.inp` for the `X2C` case (`FUNCTIONAL MULLER`
throughout).

## PNOF functionals and geminal occupation-number optimization

Setting `FUNCTIONAL` to `PNOF5`, `PNOF7`, `PNOF7S`, or `GNOF` (the Piris
natural orbital functionals, `Occ_opt/PNOFs.h`) instead of a JK-only
name runs a different, subspace-based occupation-number optimization on
the same converged `NON_REL`/`X2C`/`C4_DHF` orbitals:

1. Partition the occupied/unoccupied Kramers (or, for `NON_REL`,
   spin) pairs into `PNOF_SUBSPACES` independent coupling subspaces
   (`Occ_opt/Orb_subspaces.h`), each with one occupied ("principal")
   pair and `PNOF_COUPLING - 1` unoccupied pairs; every other occupied
   pair is deep core (pinned at n=1, interacting HF-like with
   everything) and every other unoccupied pair is deep virtual (n=0,
   excluded entirely).
2. Build a feasible initial guess for every subspace via the same
   trigonometric ("gamma") occupation-number parameterization as
   `standalone_donof`'s reference DoNOF code (`Occ_opt/PNOFs.h`'s
   `pnofSubspaceOccupationsFromGammas`/`pnofDefaultGuessGammas`):
   independent angles map to occupations that automatically sum to 1
   within a subspace for ANY angle, so DoNOF's own default guess (every
   angle = pi/4, giving each principal pair n=0.75 and each successive
   unoccupied pair half of what remains) is reused directly rather than
   inventing a new ad hoc starting point.
3. Optimize the GEMINAL occupations (one variable per Kramers/spin pair
   -- both members of a pair always share the same occupation, by
   construction, not by a separate constraint) via ONE of two methods,
   selected by `SQP_PNOF_OCC`:
   - `SQP_PNOF_OCC FALSE` (the **default**): `Utils/LBFGS.h` directly
     over the UNCONSTRAINED gamma angles from step 2
     (`Occ_opt/PNOFs.h`'s `pnofSubspaceOccupationsFromGammasWithGradient`)
     -- the same approach `standalone_donof`'s reference DoNOF code
     itself uses (`m_optocc.F90` optimizes directly over `GAMMAs`).
     Since gamma guarantees `sum(n)=1` and `0<n<1` for ANY real angle,
     this needs no box/equality-constrained QP subproblem at all -- the
     chain rule turns `pnofOccupationGradient`'s geminal-indexed
     gradient into a gamma-indexed one via each subspace's own Jacobian
     (`docc(i)/dgamma(k)`, block-diagonal across subspaces).
   - `SQP_PNOF_OCC TRUE`: `Utils/SQP.h` over the occupations directly,
     subject to `sum(n) = 1` per subspace (2 electrons per subspace at
     full pairing) and `0 < n_p < 1` as explicit box+equality
     constraints.
   Only one of the two runs per calculation; both solve the same
   problem and agree to full displayed precision whenever both converge
   cleanly.
4. `PNOF5`/`PNOF7`/`PNOF7S` use a fully analytic gradient AND (for
   `SQP_PNOF_OCC TRUE`) Hessian; `GNOF`'s inter-subspace coupling term
   additionally depends on each pair's own subspace principal
   occupation, so its `SQP_PNOF_OCC TRUE` Hessian is built via central
   finite differences of the (still fully analytic) gradient instead
   (`pnofOccupationHessianFD`) -- `SQP_PNOF_OCC FALSE`/`LBFGS.h` never
   needs a Hessian at all.

The default `PNOF_SUBSPACES 1`, `PNOF_COUPLING 2` (plain HOMO-LUMO
pairing) converges cleanly for `NON_REL`, `X2C`, and `C4_DHF` alike,
under either `SQP_PNOF_OCC` setting. Larger `PNOF_SUBSPACES` (multiple
simultaneous per-subspace equality constraints for `SQP.h`, or a larger
unconstrained problem for `LBFGS.h`) can hit either optimizer's
iteration cap without its `converged` flag ever firing, even when the
returned occupations already satisfy the correct KKT stationarity
condition (for `SQP.h`, verified as the projected gradient within each
subspace's null space being zero) -- a known limitation of each solver's
own convergence check on this harder landscape, not of the PNOF
energy/gradient/Hessian themselves (each independently validated against
finite differences to machine precision); switching `SQP_PNOF_OCC` to
compare the OTHER method's result is a useful cross-check in that case,
since both have been observed to converge cleanly on cases where the
other's flag did not fire.

See `examples/water_gnof.inp` and `examples/co-sto-3g_gnof.inp` for
worked `C4_SPINOR`/`NON_RELATIVISTIC` examples, or
`examples/water_X2C_gnof.inp` for the `X2C` case (`FUNCTIONAL GNOF`,
`PNOF_COUPLING 2` throughout) -- direct PNOF counterparts of the MULLER
examples above.

## pCCD functional

Setting `FUNCTIONAL` to `PCCD` (`Occ_opt/pCCD.h`) runs Kramers-restricted
pair-coupled-cluster doubles (generalization of Henderson, Bulik, Stein,
Scuseria, J. Chem. Phys. **141**, 244104 (2014) to a spin-with/Kramers-
restricted spinor basis -- derivation in `doc/kr_pccd.tex`) instead of a
JK-only or PNOF occupation-number optimization:

1. Partition the occupied/unoccupied Kramers (or, for `NON_REL`, spin)
   pairs into frozen-core (`PCCD_FROZEN_PAIRS`, pinned at n=1), active
   occupied, and active virtual (`PCCD_ACTIVE_PAIRS`, or all remaining)
   windows -- the same pairs-counted convention `JK_FROZEN_PAIRS`/
   `JK_ACTIVE_PAIRS` use.
2. Solve the t-/z-amplitude residue equations at the FIXED starting
   orbitals (`PCCD_AMPLITUDE_SOLVER`: `NEWTON`, the default, or `LBFGS`)
   and build the pair-level 1-/2-RDM (occupation numbers `n_p`, pair-
   transfer `D_pq`, density-density `Q_pq`).
3. With `FULL_OPTIMIZATION TRUE`, `Hessian_opt/PccdFock.h` unfolds that
   RDM into the SAME `two_rdm_h`/`two_rdm_x`/`two_rdm_l1`/`two_rdm_l2`
   ansatz PNOF's own `PnofFock.h` uses (the pair-transfer `D_pq` plays
   exactly the role PNOF's `Pi_pq` does), so `ADAM`/`NEO` drive pCCD's
   orbital rotations through the identical generalized Fock/Hessian
   machinery -- no new, slower optimizer path. Each macro-iteration
   re-solves the amplitudes at the newly rotated orbitals (pCCD's own
   "occupation re-optimization" step) before the next rotation.

Validated end to end on NON_REL/X2C/C4_SPINOR alike, both the ordinary,
minimization-only `FULL_OPTIMIZATION` and the `FULL_OPTIMIZATION_4C_NEG`
min-max saddle stage (see that row above for the warm-up basis-mismatch bug
this surfaced and its fix).
`examples/ne_pccd_full_optimization.inp` (`NON_RELATIVISTIC`) matches the
literature oo-pCCD/cc-pVDZ energy for the Ne atom (-128.559674 Hartree) to
~1e-7 Hartree; `examples/ne_pccd_x2c.inp` and `examples/ne_pccd_c4.inp` are
its `X2C`/`C4_SPINOR` counterparts (`CHOLESKY TRUE`, since `C4_SPINOR`'s RKB
dimension there is sizable), and `examples/ne_pccd_c4_neg.inp` adds the
min-max stage on top of `ne_pccd_c4.inp` (converges to the same energy,
`-6.689e-09` Hartree below the no-pair minimum -- the expected O(1/c^2)
scale) -- the pCCD correlation energy recovered on top
of the HF/DHF reference agrees between `NON_RELATIVISTIC` and exact
4-component to ~4e-6 Hartree. `ne_pccd_x2c.inp`'s own header flags a
separate, pre-existing issue it surfaced: `X2C` 's approximate (one-electron
picture-change only) treatment has a much larger error for a basis with d
(or higher) functions than for the s/p-only systems it was previously
exercised on -- see that file's own comment for the numbers.

`READ_RESTART`/the `RESTART.*` files work for `PCCD` too (NON_REL/X2C/C4_SPINOR alike), including the
t-/z-amplitudes themselves as the warm start for the next run's amplitude solve -- see *Restart file*
below.

## FULL_OPTIMIZATION for C4_SPINOR: positive-energy-only orbital rotations

Relativistic SCF is not a plain minimization over the full 4-component spinor space: it is a
min-max problem (Talman, *Phys. Rev. Lett.* 57, 1091 (1986); Saue, *"Relativistic Hamiltonians
for chemistry: A primer,"* ChemPhysChem 12, 3077 (2011)) -- a minimum over rotations *among*
positive-energy spinors, but a maximum over rotations that mix an occupied positive-energy
spinor into the negative-energy (Dirac sea) branch, since admitting that character would let the
energy decrease without bound (variational collapse). `FULL_OPTIMIZATION`'s orbital-rotation step
therefore never explores that direction for `C4_SPINOR`: every pair touching a negative-energy
index is dropped from the parameter space entirely (not merely left at zero gradient), for both
`ORBITAL_OPTIMIZER ADAM` and `NEO`. Since `κ` (the rotation generator) then has an exact
block-diagonal structure (zero coupling to the negative branch), `exp(κ)` is exactly block-diagonal
too: the negative-energy spinors stay bit-for-bit unchanged throughout the whole macro loop, and
orthonormality with the (untouched) negative branch is preserved exactly. This turns C4_SPINOR's
own orbital optimization into an ordinary minimization, same as `NON_REL`/`X2C` -- `NEO`'s default
`target_order = 0` is then the physically correct target, confirmable with `CHECK_HESS_NEO TRUE`'s
post-loop Hessian check (`[PASS] the point is a genuine minimum`); `DEBUG TRUE` adds the dense Hessian
diagonalization. The positive-energy branch is itself
Kramers-paired the same way X2C's spinors are, so it is Kramers-restricted here too.

`CHOLESKY TRUE` for C4_SPINOR: the RKB integrals come from one Cholesky decomposition of the real AO
Coulomb matrix over the {Large-Large} u {Small-Small} pairs (`C4_DHF/RkbCholesky.h`), each vector projected
into the RKB spinor basis; the DHF SCF and the MO-basis vectors follow from them. The MO integrals span an
enormous dynamic range (the negative branch's diagonal is order -2mc^2, ~1e4 Hartree), but since that
branch never meets a nonzero occupation in the no-pair treatment, the MO vectors are restricted to the
positive-energy block and recompressed (eigen-decomposition of their Gram matrix, element error below
`CHOLESKY_THRESHOLD`) -- for LiH/6-31G 662 vectors become about 60.

See `examples/lih_gnof_c4_full_optimization.inp` (`ADAM`, dense integrals, all three SCF paths in
one run), `examples/lih_gnof_c4_neo_full_optimization.inp` (`NEO`), and
`examples/lih_gnof_c4_full_optimization_cholesky.inp` (`CHOLESKY TRUE`) -- all three converge to
the same C4_DHF energy to ~1e-9 Hartree.

### `FULL_OPTIMIZATION_4C_NEG`: the min-max stage

With `FULL_OPTIMIZATION_4C_NEG TRUE` a second stage follows the positive-energy-only minimization above
(only if it converged and passed its checks). It starts from that minimum's orbitals and occupations and
solves the genuine relativistic min-max problem (Talman 1986; Saue 2011): the energy is **minimized** over
the positive-energy rotations and **maximized** over the electron-positron ones (occupied positive-energy
spinor <-> negative-energy spinor), i.e. a saddle point.

* *Order of the saddle.* Every parameter of a rotation between a positive-energy spinor with occupation
  `> 1e-6` and a negative-energy spinor is a maximization direction (Kramers-reduced count when the
  Kramers restriction applies: 176 for LiH/6-31G with GNOF, 484 with MULLER). The log prints it.
* *Orbital step.* NEO (`Utils/NEO.h`) with a **dynamic** saddle order: at every Newton step the target
  eigenvector index is the number of leading Ritz roots that are electron-positron directions, because a fixed
  `target_order` only counts the negative-curvature directions *coupled to the gradient*, which symmetry makes
  far fewer than the Hessian index. The electron-positron sector (curvature `-4 c^2 n_i`, ~1e5 below everything
  else) is kept apart in the Davidson space by a sector partition, so the usual trust-radius machinery works.
  Cost: a few Hessian-vector products per Newton step, independent of the saddle order.
* *Occupation step.* The occupation numbers are fully re-minimized at the new orbitals (the negative-energy
  branch stays at zero occupation), macro-iterated like the first stage.
* *Checks.* The integrals rotated to the starting point must reproduce the first stage's energy. Verifying
  the type of the saddle runs only under `DEBUG TRUE`: the dense joint Hessian is built at the converged
  point from the dense two-electron tensor, contracted to the Kramers-reduced space, diagonalized exactly
  and its negative eigenvalues are counted (LiH/6-31G: 176 for GNOF, 484 for MULLER; CO/6-31G: 720, about
  2 minutes for the diagonalization step). It needs the dense tensor, so it is memory-bound for larger
  bases (CO/cc-pVDZ would need ~4 GB for the tensor alone).

For light systems the second stage moves the energy by ~1e-10 Hartree (the electron-positron gradient at the
no-pair minimum is tiny); it is a check that the no-pair minimum really is the min-max point and it is exact
for any `FUNCTIONAL`. See `examples/lih_gnof_c4_neg_full_optimization.inp` (GNOF, dense) and
`examples/lih_muller_c4_neg_full_optimization_cholesky.inp` (MULLER, `CHOLESKY TRUE`).

## Restart file

At the end of a `NON_RELATIVISTIC`, `X2C` and/or `C4_SPINOR` run with a `FUNCTIONAL`,
the final RDMFT state is written in **binary** to `RESTART.NON_REL`,
`RESTART.X2C_HF` and, for the 4-component path, `RESTART.4C` (the
positive-energy-only minimization; the min-max stage of `FULL_OPTIMIZATION_4C_NEG` writes no
file of its own, it is meant to be run on top of `RESTART.4C` at each geometry), in the working
directory. The format, writer and reader are in `Utils/Restart.h`. The files are read back
right after writing (an identity check) and by `READ_RESTART` (below).

Contents:

- the **occupation numbers** (full vector, one entry per spin-orbital /
  spinor of the MO basis) -- for the JK-only functionals this is the state;
- for the JK-only and **PCCD** functionals the resolved **occupation window** (`JK_FROZEN_PAIRS` /
  `PCCD_FROZEN_PAIRS` pairs pinned at occupation 1 and the number of active pairs above them; an absent
  `JK_ACTIVE_PAIRS`/`PCCD_ACTIVE_PAIRS` is stored as the number of pairs it resolved to);
- for the **PNOF** functionals also the **gamma angles** (subspace after
  subspace, `PNOF_COUPLING - 1` angles each, the trigonometric
  parameterization of `Occ_opt/PNOFs.h`), obtained from the final
  occupations, so the SQP and the L-BFGS branches write the same thing;
- for **PCCD** also the **t-/z-amplitudes** (flattened `[t; z]`, same convention `Full_opt/
  FullOptimization.cpp`'s `makePccdModel` uses for its in-memory warm start), from the final
  macro-iteration's own amplitude solve (or the fixed-orbital one if `FULL_OPTIMIZATION` did not
  run);
- the final **molecular-orbital coefficients** `C = C_scf * U_total` in the
  AO spin-orbital basis (`4C`: the 4-component RKB spinor basis
  [Large-alpha; Large-beta; Small; Small], columns in ascending energy with the
  negative-energy branch first and occupation 0): `U_total` is the accumulated `FULL_OPTIMIZATION`
  rotation (identity without it). `NON_REL`: `blockdiag(C, C)`, real,
  `2 n_AO x 2 n_MO`, ordered `[alpha, beta]`. `X2C`: the Kramers-fixed
  spinors, complex, `2 n_Large x 2 n_Large`. Column `j` is MO `j` and has
  occupation `occupations[j]`;
- the Large-basis fingerprint (`Utils/BasisFingerprint.h`), the number of
  electrons, the PNOF subspace/coupling/core counts, the final total
  energy and flags telling whether the orbitals come from
  `FULL_OPTIMIZATION` and whether the optimization converged.

Every file is read back right after it is written and the program prints
the checks: identity with what was written, `C^dagger S C = 1`,
`C^dagger h_AO C = U^dagger h_MO U` (which fixes the AO/MO layout and the
`C_new = C_old U` convention) and, for PNOF, that the gamma angles
regenerate the occupation numbers. The file layout is documented at the
top of `Utils/Restart.h`. Unit test: `make test_restart LIBCINT=...`.


### Restarting from a previous run (`READ_RESTART TRUE`)

For every requested method the SCF is skipped and the restart file supplies the state:

1. `RESTART.<method>` is read and checked against the run (method, `NELEC`, number and type of the
   coefficients). The basis *fingerprint* is not compared: it contains the atomic centers and would
   forbid the geometry changes this is meant for.
2. **Orthonormality.** `S_check = C_read^dagger S C_read` is formed with the overlap of the *current*
   geometry (`blockdiag(S, S)` for `NON_REL`/`X2C`, the RKB overlap for the 4-component basis). If it is
   not the identity, the orbitals are Loewdin-orthonormalized, `C = C_read S_check^(-1/2)` (then
   `C^dagger S C = S_check^(-1/2) S_check S_check^(-1/2) = 1`, the closest orthonormal set), and both
   deviations are printed.
3. **Kramers pairs (`X2C`, `C4_SPINOR`).** The pairing `Theta|2k> = |2k+1>` is re-established exactly
   after the Loewdin step (`Utils/KramersPairing.h`; the deviation before/after is printed) and the
   Kramers structure of `h` in the new basis is tested. `NON_REL` checks that alpha and beta orbitals are
   still identical.
4. **4-component positive-energy space.** The no-pair calculation is restricted to the positive-energy
   spinors, so that space has to be the one of the *current* geometry: the space read from the file is the
   old geometry's and carries a negative-energy admixture (~1e-4 for a 0.1 bohr step in LiH) that would
   lower the positive-only energy by ~1e-4 Hartree and give the min-max stage a large electron-positron
   gradient. When the orbitals had to be re-orthonormalized, one Fock build from the restart density (no SCF
   iteration) defines the split at the new geometry: the read positive spinors are projected onto its
   positive-energy eigenspace and re-orthonormalized, the negative-energy columns are its negative-energy
   eigenvectors. The admixture is printed. LiH/6-31G at 1.70 bohr from a 1.5949 bohr restart reproduces the
   from-scratch energies of `NON_REL`, `X2C` and `C4_SPINOR` (also with `FULL_OPTIMIZATION_4C_NEG`) to ~1e-8
   Hartree, dense and Cholesky, GNOF and MULLER.
5. **Occupations/amplitudes.** The occupation numbers (the gammas for PNOF, through the occupations)
   replace the smeared-orbital-energy guess as the starting point of the occupation optimization, which
   still runs at the fixed orbitals before `FULL_OPTIMIZATION`. A PNOF file that does not fit the current
   geminal structure falls back to the default guess with a note; a JK-only file whose frozen/deep-virtual
   positions disagree with the current `JK_FROZEN_PAIRS`/`JK_ACTIVE_PAIRS` window is refused. For **PCCD**,
   the file's `t`-/`z`-amplitudes seed the Newton/L-BFGS solve directly (instead of the usual cold `t = z =
   0` start) -- at an unchanged geometry this converges in 0 further iterations (the restored amplitudes
   are already the exact solution); a file whose amplitude-vector size does not fit this run's resolved
   `PCCD_FROZEN_PAIRS`/`PCCD_ACTIVE_PAIRS` window is refused with a clear error (same convention as
   JK-only's window check).

The SCF-based diagnostics (HF energies, the HF-occupation gradient/Hessian tests, `HESSIAN_*` and the DEBUG
checks against SCF quantities) do not exist in this mode and are not printed. The new results overwrite
`RESTART.*`, so a scan is a chain: run the first geometry, then edit the geometry, add `READ_RESTART TRUE`
and repeat -- see `examples/lih_gnof_read_restart.inp`.

## Contributors

- Dr. M. Rodriguez-Mayorga 

