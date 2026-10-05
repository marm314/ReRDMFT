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
- OpenMP support in your compiler (bundled with GCC).

## Building

```sh
make LIBCINT=/path/to/libcint/build/libcint.a
```

If your libcint headers live somewhere other than
`$(dirname LIBCINT)/include`, also set `LIBCINT_INC`. This produces
the `rerdmft` binary in the repository root.

To avoid retyping `LIBCINT=...`, create a git-ignored `Makefile.local`:

```make
LIBCINT := /path/to/libcint/build/libcint.a
```

`make clean` removes build artifacts.

## Running

```sh
./rerdmft examples/water.inp
```

See `examples/*.inp` for sample input files. While a run is in
progress, `examples/name.live` (truncated at the start of every run)
shows live progress -- each phase, every SCF iteration, Cholesky
decomposition progress, and every `FULL_OPTIMIZATION` macro-iteration.
Follow it with `tail -f examples/name.live`.

## Input file keywords

Keywords are case-insensitive, one per line, optionally followed by
`=` before the value (`DEBUG TRUE`, `DEBUG = TRUE`, `DEBUG=TRUE` are
all equivalent). Lines starting with `#` (or a `#` anywhere on a line)
are comments.

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `NELEC` (or `NELECTRONS`) | int | *required* | Number of electrons. |
| `BASIS` | string | *required* | Gaussian basis set file name. `src/Utils/download_basis.py <input_file.inp>` fetches it automatically from Basis Set Exchange for the elements in `GEOMETRY`. |
| `UNIT_LENGTH` | string | `ANGS` | Units of `GEOMETRY`'s coordinates: `ANGS` or `BOHR`/`AU`. |
| `GEOMETRY` ... `END` | block | *required* | Molecular geometry, `<symbol> <x> <y> <z>` lines, one atom per line. |
| `NON_RELATIVISTIC` | bool | `FALSE` | Run the standard nonrelativistic Hartree-Fock SCF (`NON_REL`). |
| `CARTESIAN` | bool | `TRUE` | Only `NON_RELATIVISTIC`: `TRUE` uses the raw Cartesian AO basis; `FALSE` uses the spherical basis (as `X2C`/`C4_SPINOR` always do). |
| `PRECONDITION_SMALL_OVERLAP` | bool | `FALSE` | Jacobi-precondition the RKB Small-component overlap before inverting it (`X2C`/`C4_SPINOR`). Rarely needed with the current analytic RKB basis; kept for extreme bases. |
| `C4_SPINOR` | bool | `FALSE` | Run the 4-component Dirac-Hartree-Fock SCF (`C4_DHF`). Kramers-restricted like `X2C`, for even `NELEC`. |
| `DEBUG` | bool | `FALSE` | Extra diagnostics: internal gradient cross-checks, finite-difference tests, dense-vs-Cholesky comparisons. Builds the dense two-electron tensor on demand. |
| `HESSIAN_MEAN_FIELD` | bool | `FALSE` | Build and fully diagonalize the dense orbital-rotation Hessian (minimum expected for `NON_REL`/`X2C`, saddle for `C4_SPINOR`). O(n^5)-O(n^6), needs the dense tensor. |
| `FCIDUMP` | bool | `FALSE` | `NON_RELATIVISTIC` + `FUNCTIONAL` only: write an `FCIDUMP` file of the MO integrals for an external CI/DMRG/FCI code. |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (a.u.), to probe the nonrelativistic limit or exaggerate relativistic effects. |
| `MIXING` | double, `(0,1]` | `0.4` | Linear density-mixing weight, used only with `DIIS FALSE`. |
| `DIIS` | bool | `TRUE` | Pulay DIIS instead of linear mixing in every SCF loop. |
| `DIIS_SIZE` | int (>= 2) | `5` | Number of (error, Fock) pairs kept by DIIS. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum `C4_DHF` SCF cycles. |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` | `C4_DHF` SCF energy-change convergence threshold. |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold. |
| `CHOLESKY` | bool | `FALSE` | Hold the two-electron integrals as Cholesky vectors instead of a dense/packed tensor -- SCF, MO integrals, and `FULL_OPTIMIZATION` all work from the vectors. |
| `CHOLESKY_THRESHOLD` | double (> 0) | `1e-10` | Residual-diagonal cutoff for the decomposition; looser = fewer vectors (faster, less accurate). |
| `ON_DEMAND_ERI` | bool | `TRUE` | With `CHOLESKY TRUE`: evaluate each AO quadruplet on demand during decomposition (memory O(n^2)) instead of building the full packed tensor first. |
| `SCF_DIRECT_4C` | bool | `FALSE` | `C4_SPINOR` only: run the SCF fully integral-direct (no two-electron representation held during SCF, O(n^2) memory). Implies the Cholesky route downstream. Not yet with `READ_RESTART`/`DEBUG`/`HESSIAN_MEAN_FIELD`. |
| `FUNCTIONAL_DIRECT_4C` | bool | `FALSE` | `C4_SPINOR` + (`CHOLESKY TRUE` or `SCF_DIRECT_4C TRUE`): build `FUNCTIONAL`/`FULL_OPTIMIZATION`'s positive-energy MO Cholesky vectors in one fused pass, without ever holding the full AO-basis RKB vector set. No effect on `FULL_OPTIMIZATION_4C_NEG`. Memory benefit confirmed on small systems only so far -- not yet validated at heavy-element scale. |
| `X_LIN_DEP_THRS_L` | double (> 0) | `1e-6` | Large-component linear-dependence threshold (LOWGEN-style safety net). |
| `X_LIN_DEP_THRS_S` | double (> 0) | `1e-8` | Small-component linear-dependence threshold (`X2C`/`C4_SPINOR` only). |
| `FUNCTIONAL` | string | *(none)* | Selects the RDMFT functional: JK-only (`SD`, `MULLER`/`MBB`, `BBC2`, `CA`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER`), PNOF (`PNOF5`, `PNOF7`, `PNOF7S`, `GNOF`), or `PCCD`. Unset skips the whole RDMFT step. |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | Initial occupations for a JK-only functional: `PROPORTIONAL` or `FERMI_DIRAC` (at `TEMPERATURE`). |
| `JK_FROZEN_PAIRS` | int (>= 0) | `0` | JK-only: freeze the lowest `2*JK_FROZEN_PAIRS` spin-orbitals/spinors at occupation 1. |
| `JK_ACTIVE_PAIRS` | int (>= 1) | all remaining | JK-only: size of the fractional-occupation window above the frozen core; above that is deep virtual (occupation 0). |
| `TEMPERATURE` | double (> 0) | `1000` K | Smearing temperature for `OCCUPATION_INIT FERMI_DIRAC`. |
| `PNOF_SUBSPACES` | int (>= 1) | `1` | PNOF: number of independent coupling subspaces. |
| `PNOF_COUPLING` | int (>= 2) | `2` | PNOF: pairs per subspace (1 occupied + the rest unoccupied). |
| `SQP_PNOF_OCC` | bool | `FALSE` | PNOF: `FALSE` optimizes unconstrained gamma angles via L-BFGS; `TRUE` optimizes occupations directly via SQP. |
| `PCCD_FROZEN_PAIRS` | int (>= 0) | `0` | pCCD: freeze the lowest pairs at occupation 1 (no amplitudes). |
| `PCCD_ACTIVE_PAIRS` | int (>= 1) | all remaining | pCCD: size of the active t-/z-amplitude window. |
| `PCCD_AMPLITUDE_SOLVER` | string | `NEWTON` | pCCD amplitude solver: `NEWTON` (exact Jacobian) or `LBFGS`. |
| `FULL_OPTIMIZATION` | bool | `FALSE` | After occupation optimization, macro-iterate orbital rotation + occupation re-optimization to convergence. Works for `NON_REL`, `X2C`, `C4_SPINOR`. |
| `MAX_MACRO_ITERATIONS` | int | `1000` | Macro-iteration cap. |
| `MACRO_ENERGY_TOLERANCE` | float | `1e-9` | Macro-loop energy convergence threshold. |
| `ORBITAL_GRADIENT_TOLERANCE` | float | `1e-5` | ADAM/NEO orbital-gradient convergence threshold. |
| `ORBITAL_OPTIMIZER` | string | `ADAM` | Orbital-rotation driver: `ADAM` (first-order), `NEO` (matrix-free Newton, fewer iterations), or `ADAM_NEO` (switches between them by `|dE|`). |
| `NEO_MAX_ITERATIONS` | int | `100` | Cap on Newton steps per macro-iteration for `NEO`/`ADAM_NEO`. |
| `ADAM_NEO_SWITCH_TOLERANCE` | double | `1e-4` | `|dE|` threshold that switches `ADAM_NEO` between drivers. |
| `CHECK_HESS_NEO` | bool | `FALSE` | After a NEO-driven macro loop converges, verify it's a genuine minimum (matrix-free Hessian check) and auto-escape a saddle if found. |
| `FIXED_OCCUPANCIES` | bool | `FALSE` | `FULL_OPTIMIZATION`: keep occupations (and, for pCCD, the full 2-RDM) fixed through the macro loop -- pure orbital optimization. |
| `READ_OCCUPANCIES` | bool | `FALSE` | Read starting occupations from `OCC.in` instead of optimizing them from scratch. |
| `READ_RESTART` | bool | `FALSE` | Skip the SCF and start the functional calculation from a previous run's `RESTART.*` file (see *Restart file* below). |
| `FULL_OPTIMIZATION_4C_NEG` | bool | `FALSE` | `C4_SPINOR` + `FULL_OPTIMIZATION`: after the positive-energy-only minimum converges, run the genuine relativistic min-max stage (see below). |
| `X2C` | bool | `FALSE` | Run the approximate X2C-HF SCF alongside `NON_RELATIVISTIC`/`C4_SPINOR` (see below). |

## X2C decoupling and X2C-HF

Setting `X2C` prints a one-electron X2C decoupling report and runs an
approximate X2C-HF SCF, between the `NON_RELATIVISTIC` and `C4_SPINOR`
reports, independent of whether either of those is itself set.

Both `X2C` and `C4_SPINOR` share the same relativistic machinery
(`X2C_DHF/`, `RKB/`): an analytic restricted-kinetic-balance small
component (one small-component function per large-component function,
built in closed form, no numerical projection), a spherical
large-component basis (`Utils/SphericalTransform.h`), and a DIRAC-style
LOWGEN safety net for any residual linear dependence
(`Linear_Algebra/LinearAlgebra.h`).

1. **Decoupling**: diagonalizing the orthonormalized 4-component
   Hamiltonian block-diagonalizes the Dirac equation into
   positive-/negative-energy branches. Eigenvalues, Kramers-pair
   splitting, and a Kramers-partner check are always printed.
2. **Exact X2C Hamiltonian**: eliminates the small component exactly,
   reproducing the positive-energy spectrum from step 1 with no
   approximation -- the no-pair approximation is simply using this
   Hamiltonian alone.
3. **Approximate X2C Hamiltonian**: the same Hamiltonian orthogonalized
   with only the plain large-component overlap -- a common
   simplification, close to but not identical to the exact spectrum.
4. **Approximate X2C-HF SCF**: full SCF using step 2's Hamiltonian as a
   *fixed* one-electron core (no picture-change correction as the
   density changes) and ordinary nonrelativistic two-electron
   integrals. Genuine opposite-spin exchange appears because the core
   already mixes spins. Can converge below the exact 4-component
   energy (the usual variational bound no longer applies once the
   metric isn't exact) -- expected, not a bug.
5. **Gradient/Hessian tests**: the same `Hessian_opt` test suite
   `C4_SPINOR` runs is run here too. Unlike `C4_DHF`'s saddle, X2C-HF's
   solution is a genuine minimum.

See `examples/water_X2C.inp` (plain) or `examples/water_X2C_debug.inp`
(`DEBUG TRUE`).

## RDMFT functional evaluation

Setting `FUNCTIONAL` runs an RDMFT evaluation after each requested SCF
converges, using its orbitals/integrals as a fixed background:

1. Generate initial fractional occupations (`OCCUPATION_INIT`).
2. Evaluate the functional's energy.
3. Optimize the occupations via SQP (`Occ_opt/SQP.h`), subject to
   `sum(n_p) = NELEC` and `0 < n_p < 1`, with both members of each
   Kramers/spin pair tied to one variable.

For `C4_SPINOR` the negative-energy branch is excluded entirely
(no-pair approximation); for `X2C` and `NON_RELATIVISTIC`, every
orbital competes for occupation.

See `examples/water_muller.inp`, `examples/co-sto-3g_muller.inp`, or
`examples/water_X2C_muller.inp`.

## PNOF functionals

`FUNCTIONAL PNOF5`/`PNOF7`/`PNOF7S`/`GNOF` (`Occ_opt/PNOFs.h`) runs a
subspace-based geminal occupation-number optimization instead of the
JK-only one:

1. Partition occupied/unoccupied pairs into `PNOF_SUBSPACES` coupling
   subspaces (`Occ_opt/Orb_subspaces.h`), each with one occupied
   ("principal") pair and `PNOF_COUPLING - 1` unoccupied pairs.
2. Build a feasible starting guess via the same trigonometric
   ("gamma angle") parameterization DoNOF uses.
3. Optimize via `SQP_PNOF_OCC`: `FALSE` (default) uses L-BFGS over the
   unconstrained gamma angles; `TRUE` uses SQP over the occupations
   directly with explicit box+equality constraints. Both agree to full
   precision when both converge.
4. `GNOF`'s inter-subspace coupling needs a finite-difference Hessian
   for the `SQP_PNOF_OCC TRUE` path; `PNOF5`/`7`/`7S` are fully
   analytic.

The default (`PNOF_SUBSPACES 1`, `PNOF_COUPLING 2`) converges cleanly
for all three SCF paths. Larger subspace counts can hit an optimizer's
iteration cap without its `converged` flag firing even at a correct
stationary point -- cross-check with the other `SQP_PNOF_OCC` setting
if that happens.

See `examples/water_gnof.inp`, `examples/co-sto-3g_gnof.inp`, or
`examples/water_X2C_gnof.inp`.

## pCCD functional

`FUNCTIONAL PCCD` (`Occ_opt/pCCD.h`) runs Kramers-restricted
pair-coupled-cluster doubles (Henderson, Bulik, Stein, Scuseria,
*J. Chem. Phys.* **141**, 244104 (2014), generalized to a spinor basis
-- derivation in `doc/kr_pccd.tex`):

1. Partition pairs into frozen-core (`PCCD_FROZEN_PAIRS`), active, and
   deep-virtual windows (same convention as `JK_FROZEN_PAIRS`).
2. Solve the t-/z-amplitude equations at fixed orbitals
   (`PCCD_AMPLITUDE_SOLVER`), building the pair-level 1-/2-RDM.
3. With `FULL_OPTIMIZATION TRUE`, the RDM feeds the same generalized
   Fock/Hessian machinery PNOF uses, so `ADAM`/`NEO` optimize pCCD
   orbitals through the identical code path; each macro-iteration
   re-solves the amplitudes at the new orbitals.

Validated end to end on `NON_REL`/`X2C`/`C4_SPINOR`, including the
`FULL_OPTIMIZATION_4C_NEG` min-max stage. See
`examples/ne_pccd_full_optimization.inp`, `examples/ne_pccd_x2c.inp`,
`examples/ne_pccd_c4.inp`, `examples/ne_pccd_c4_neg.inp`.
`READ_RESTART` works for pCCD too, including the amplitudes as a warm
start.

## FULL_OPTIMIZATION for C4_SPINOR: positive-energy-only orbital rotations

Relativistic SCF is a min-max problem, not a plain minimization
(Talman, *Phys. Rev. Lett.* 57, 1091 (1986); Saue, ChemPhysChem 12,
3077 (2011)): a minimum over rotations among positive-energy spinors,
but a maximum over rotations that mix in the negative-energy (Dirac
sea) branch. `FULL_OPTIMIZATION` therefore excludes every rotation
pair touching a negative-energy index from the parameter space
entirely (for both `ORBITAL_OPTIMIZER ADAM` and `NEO`), turning
`C4_SPINOR`'s own optimization into an ordinary minimization, just
like `NON_REL`/`X2C`.

`CHOLESKY TRUE` for `C4_SPINOR`: one decomposition of the real AO
Coulomb matrix over `{Large-Large} u {Small-Small}` pairs
(`C4_DHF/RkbCholesky.h`), projected into the RKB spinor basis; the SCF
and MO-basis vectors follow from it, recompressed to the positive-
energy block since the negative branch never has nonzero occupation.

See `examples/lih_gnof_c4_full_optimization.inp` (dense),
`examples/lih_gnof_c4_neo_full_optimization.inp` (NEO), or
`examples/lih_gnof_c4_full_optimization_cholesky.inp` (Cholesky) --
all converge to the same energy.

### `FULL_OPTIMIZATION_4C_NEG`: the min-max stage

With `FULL_OPTIMIZATION_4C_NEG TRUE`, after the positive-energy-only
minimum converges, a second stage solves the genuine min-max problem:
minimizing over positive-energy rotations, maximizing over
electron-positron rotations (occupied positive-energy <-> negative-
energy spinor), to a saddle point of the expected order. Driven by NEO
with a dynamic saddle order and a sector partition that keeps the
electron-positron curvature separate from ordinary Davidson search
directions; occupations are fully re-minimized at each new orbital set.
`DEBUG TRUE` adds an exact dense-Hessian check of the saddle order
(memory-bound for larger bases).

For light systems this moves the energy negligibly (~1e-10 Hartree) --
it's primarily a check that the no-pair minimum is also the min-max
point. See `examples/lih_gnof_c4_neg_full_optimization.inp` or
`examples/lih_muller_c4_neg_full_optimization_cholesky.inp`.

## Restart file

At the end of a run with a `FUNCTIONAL`, the final RDMFT state is
written in binary to `RESTART.NON_REL`, `RESTART.X2C_HF`, and/or
`RESTART.4C` (`Utils/Restart.h`), read back immediately as an identity
check. Contents: occupation numbers, the resolved frozen/active window
(JK-only/pCCD), gamma angles (PNOF), t-/z-amplitudes (pCCD), the final
MO coefficients, and metadata (basis fingerprint, electron count,
converged energy, convergence flags).

### `READ_RESTART TRUE`

Skips the SCF for every requested method and starts from the restart
file's state:

1. Read `RESTART.<method>` and check it against the run (method,
   `NELEC`, coefficient shape) -- not the geometry, so this also
   supports potential-energy scans.
2. Re-orthonormalize the coefficients against the *current* geometry's
   overlap (Loewdin) if they aren't already orthonormal.
3. Re-establish exact Kramers pairing (`X2C`/`C4_SPINOR`) after that
   step.
4. For `C4_SPINOR`, re-establish the positive-energy space at the
   current geometry (one Fock build from the restart density, no SCF
   iteration) -- the old geometry's positive space otherwise carries a
   small negative-energy admixture.
5. Occupations/amplitudes from the file seed the occupation
   optimization (still run once before `FULL_OPTIMIZATION`) instead of
   the usual cold start; a file that doesn't fit the current window
   (`JK_*`/`PCCD_*`) is refused with a clear error.

The new results overwrite `RESTART.*`, so a geometry scan is a chain:
run the first point, then edit the geometry, add `READ_RESTART TRUE`,
and repeat. See `examples/lih_gnof_read_restart.inp`.

## Contributors

- Dr. M. Rodriguez-Mayorga
