# ReRDMFT

Relativistic Reduced Density Matrix Functional Theory in C++17. Includes a
4-component Dirac-Hartree-Fock SCF (`C4_DHF`), an approximate X2C-HF SCF,
and a nonrelativistic Hartree-Fock SCF (`NON_REL`) for comparison.

## Build

Requirements: a C++17 compiler (GCC), OpenMP, LAPACKE/LAPACK/BLAS
(`sudo apt install liblapacke-dev liblapack-dev libblas-dev`), and a static
[LIBCINT](https://github.com/sunqm/libcint) (`libcint.a` with its `include/`
next to it; recipe at the end).

```sh
make LIBCINT=/path/to/libcint/build/libcint.a   # produces ./rerdmft
```

Set `LIBCINT_INC` if the headers are elsewhere, or put
`LIBCINT := /path/to/libcint.a` in a git-ignored `Makefile.local`.
`make clean` removes build artifacts.

## Run

```sh
./rerdmft examples/water.inp
tail -f examples/water.live      # live progress: phases, SCF and macro-iterations
```

## Input keywords

Case-insensitive, one per line, optional `=` (`DEBUG TRUE` = `DEBUG=TRUE`);
`#` starts a comment. See `examples/*.inp`.

### System and SCF

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `NELEC` (or `NELECTRONS`) | int | *required* | Number of electrons. |
| `BASIS` | string | *required* | Basis file; `src/Utils/download_basis.py <input.inp>` fetches it from Basis Set Exchange. |
| `UNIT_LENGTH` | string | `ANGS` | `ANGS` or `BOHR`/`AU`. |
| `GEOMETRY` ... `END` | block | *required* | `<symbol> <x> <y> <z>` per atom. |
| `NON_RELATIVISTIC` | bool | `FALSE` | Run the nonrelativistic HF SCF (`NON_REL`). |
| `X2C` | bool | `FALSE` | Run the X2C-HF SCF (see *X2C*). |
| `X2C_NORM` | bool | `TRUE` | `X2C`: renormalized X2C-1e core `R†hR`; `FALSE` uses the raw `h` with the large-component overlap. |
| `C4_SPINOR` | bool | `FALSE` | Run the 4-component DHF SCF (`C4_DHF`), Kramers-restricted, even `NELEC`. |
| `CARTESIAN` | bool | `TRUE` | `NON_REL` only: Cartesian (`TRUE`) or spherical AOs; `X2C`/`C4_SPINOR` are always spherical. |
| `SPEED_OF_LIGHT` | double | CODATA | Speed of light (a.u.). |
| `DIIS` | bool | `TRUE` | Pulay DIIS in every SCF loop; `FALSE` = linear density mixing. |
| `DIIS_SIZE` | int (>= 2) | `5` | DIIS vectors. |
| `MIXING` | double (0,1] | `0.4` | Mixing weight, `DIIS FALSE` only. |
| `MAX_ITERATIONS` | int | `100` | Maximum SCF cycles. |
| `ENERGY_TOLERANCE` | double | `1e-8` | SCF energy convergence. |
| `DENSITY_TOLERANCE` | double | `1e-6` | SCF density convergence. |
| `X_LIN_DEP_THRS_L` | double | `1e-6` | Large-component linear-dependence threshold. |
| `X_LIN_DEP_THRS_S` | double | `1e-8` | Small-component threshold (`X2C`/`C4_SPINOR`). |
| `PRECONDITION_SMALL_OVERLAP` | bool | `FALSE` | Jacobi-precondition the RKB Small overlap (rarely needed). |
| `DEBUG` | bool | `FALSE` | Extra cross-checks and dense-vs-Cholesky tests (builds the dense tensor). |
| `HESSIAN_MEAN_FIELD` | bool | `FALSE` | Build and diagonalize the dense orbital Hessian (needs the dense tensor). |

### Two-electron integrals

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `USE_RI` | bool | `TRUE` when an SCF method is active and nothing else picked a kernel | Resolution-of-identity 3-center tensor for the SCF and the whole functional stage (all functionals, ADAM/NEO, min-max); accuracy ~1e-7 to 1e-4 Ha. `FALSE` forces the dense/Cholesky path. |
| `AUX_BASIS_TYPE` | string | `PAUTO` | RI auxiliary basis recipe: `PAUTO` (all shell pairs) or `AUTO` (self-products). |
| `CHOLESKY` | bool | `FALSE` | Hold the integrals as Cholesky vectors (SCF, MO integrals, `FULL_OPTIMIZATION`). |
| `CHOLESKY_THRESHOLD` | double | `1e-10` | Decomposition cutoff. |
| `ON_DEMAND_ERI` | bool | `TRUE` | With `CHOLESKY`: evaluate AO quadruplets on demand (O(n^2) memory). |
| `SCF_DIRECT_4C` | bool | `TRUE` for `C4_SPINOR` unless `READ_RESTART`/`DEBUG`/`HESSIAN_MEAN_FIELD`/`USE_RI` | `C4_SPINOR`: integral-direct SCF, O(n^2) memory. |
| `FUNCTIONAL_POS_CHO_4C` | bool | `TRUE` for `C4_SPINOR` + `FUNCTIONAL` | With `CHOLESKY`: build the positive-energy MO Cholesky vectors in one fused pass (only acts when the full AO vector set is not otherwise built). |
| `FUNCTIONAL_DIRECT_4C` | bool | `FALSE` | `C4_SPINOR` + `FULL_OPTIMIZATION`: ADAM builds its Fock matrix directly from UKB AO integrals (see below). |

### UCCSD

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `UCCSD` | bool | `FALSE` | Complex spin-orbital CCSD (`src/CC`, `doc/uccsd.tex`) on the SCF reference; 4C correlates positive-energy spinors only. |
| `CC_RDM` | bool | `FALSE` | Solve Lambda; write `<input>_MO.dm1/.dm2` and `<input>_GAU.*` (see `doc/uccsd.tex`, `examples/lih_uccsd_rdm.inp`). |
| `CC_RDM_GAUSSIAN` | bool | `TRUE` | Also write the `_GAU` files. |
| `CC_RDM_MAXMEM` | float | `4` | Memory (GB) for the transform to primitives. |
| `CC_RDM_CNORM1` | bool | `FALSE` | Put primitive normalization into the 2-RDM (`$CNORM1`). |
| `CC_RDM_THRESHOLD` | float | `1e-10` | Elements below this are not written. |
| `CC_FROZEN_CORE` | int | `0` | Frozen occupied spinors (even). |
| `CC_MAX_ITERATIONS` | int | `100` | Amplitude/Lambda iterations. |
| `CC_CONVERGENCE` | float | `1e-8` | Energy-change threshold. |
| `CC_DIIS_SIZE` | int | `8` | DIIS vectors (< 2 disables). |
| `CC_LEVEL_SHIFT` | float | `0` | Denominator level shift (Hartree). |

### Functionals

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `FUNCTIONAL` | string | *(none)* | JK-only (`SD`, `MULLER`/`MBB`, `CA`, `RHFB`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER`), PNOF (`PNOF5`, `PNOF7`, `PNOF7S`, `GNOF`) or `PCCD`. Unset skips the RDMFT step. |
| `FUNCTIONAL_INIT` | string | `SCF` | Starting orbitals of the functional stage: converged `SCF` orbitals, or `CORE` (eigenvectors of the one-electron Hamiltonian; the SCF still runs and reports its energy). |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | JK-only initial occupations: `PROPORTIONAL` or `FERMI_DIRAC`. |
| `TEMPERATURE` | double | `1000` K | Smearing for `FERMI_DIRAC`. |
| `JK_FROZEN_PAIRS` | int | `0` | JK-only: lowest pairs frozen at occupation 1. |
| `JK_ACTIVE_PAIRS` | int | all remaining | JK-only: size of the fractional-occupation window; above it, occupation 0. |
| `PNOF_SUBSPACES` | int | `1` | PNOF: number of coupling subspaces. |
| `PNOF_COUPLING` | int (>= 2) | `2` | PNOF: pairs per subspace. |
| `SQP_PNOF_OCC` | bool | `FALSE` | PNOF: `FALSE` = L-BFGS over gamma angles, `TRUE` = SQP over occupations. |
| `PCCD_FROZEN_PAIRS` | int | `0` | pCCD: lowest pairs frozen at occupation 1. |
| `PCCD_ACTIVE_PAIRS` | int | all remaining | pCCD: size of the amplitude window. |
| `PCCD_AMPLITUDE_SOLVER` | string | `DIIS` | `DIIS` (UCCSD-style, up to 1000 iterations; falls back to `NEWTON`), `NEWTON`, `LBFGS` or `KRYLOV` (Newton-Krylov; stops if it fails). |
| `PCCD_AMPLITUDE_GUESS` | string | `ZERO` | pCCD starting t without restart amplitudes: `ZERO`, `MP2` (uncoupled first-order) or `PNOF` (from the occupations of a PNOF `READ_RESTART`). |
| `PCCD_DEBUG` | bool | `FALSE` | Print per-macro-iteration amplitude-solve diagnostics to the live file. |
| `READ_OCCUPANCIES` | bool | `FALSE` | Read starting occupations from `OCC.in`; for pCCD they set the starting t, z (see *pCCD functional*). |
| `READ_RESTART` | bool | `FALSE` | Skip the SCF and start from `RESTART.*` (see *Restart*). |
| `FCIDUMP` | bool | `FALSE` | `NON_RELATIVISTIC` + `FUNCTIONAL`: write an `FCIDUMP` of the MO integrals. |
| `AIMPAC` | bool | `FALSE` | Write AIMPAC `.wfn` files: `hf_<input>_<level>.wfn` (always the SCF density), `<input>.wfn` (functional natural orbitals), `cc_<input>_<level>.wfn` (UCCSD, with `CC_RDM`), and `core_<input>_<level>.wfn` (core-Hamiltonian state, with `FUNCTIONAL_INIT CORE`). |

### Orbital optimization

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `FULL_OPTIMIZATION` | bool | `FALSE` | Macro-iterate orbital rotation and occupation re-optimization (`NON_REL`, `X2C`, `C4_SPINOR`). |
| `ORBITAL_OPTIMIZER` | string | `ADAM` | `ADAM`, `NEO` (matrix-free Newton) or `ADAM_NEO` (switches by `|dE|`). |
| `MAX_MACRO_ITERATIONS` | int | `1000` | Macro-iteration cap. |
| `MACRO_ENERGY_TOLERANCE` | float | `1e-9` | Macro-loop energy convergence. |
| `ORBITAL_GRADIENT_TOLERANCE` | float | `1e-5` | Orbital-gradient convergence. |
| `NEO_MAX_ITERATIONS` | int | `20` | Newton steps per macro-iteration. |
| `ADAM_NEO_SWITCH_TOLERANCE` | double | `2e-3` | `|dE|` threshold for switching drivers. |
| `ADAM_NEO_INITIAL_ADAM_ITERATIONS` | int | `3` | Initial macro-iterations always run with ADAM. |
| `ADAM_NEO_MIN_ADAM_ITERATIONS` | int | `3` | Minimum ADAM macro-iterations after a switch from NEO. |
| `NEO_DAVIDSON_STAGNATION_WINDOW` | int | `5` | Davidson gives up if the residual has not improved by the factor below over this many iterations; `0` disables. |
| `NEO_DAVIDSON_STAGNATION_MIN_ITERATIONS` | int | `10` | Iterations before the rule may fire. |
| `NEO_DAVIDSON_STAGNATION_FACTOR` | double | `0.5` | Required residual improvement over the window. |
| `CHECK_HESS_NEO` | bool | `FALSE` | After a NEO/`ADAM_NEO` loop, verify a genuine minimum and escape a saddle. |
| `FIXED_OCCUPANCIES` | bool | `FALSE` | Keep occupations (and pCCD's 2-RDM) fixed: pure orbital optimization. |
| `FULL_OPTIMIZATION_4C_NEG` | bool | `FALSE` | `C4_SPINOR`: after the positive-energy minimum, run the min-max stage (see below). |

## Keyword compatibility

**Errors** (raised when the input is read; the `USE_RI` rows only when `USE_RI TRUE` is written explicitly):

| Combination | Why |
| --- | --- |
| `SCF_DIRECT_4C` + `READ_RESTART`/`DEBUG`/`HESSIAN_MEAN_FIELD` | The restart branch has its own Fock build; the others need a dense tensor. |
| `USE_RI` without `NON_RELATIVISTIC`/`X2C`/`C4_SPINOR` | No SCF to affect. |
| `USE_RI` + `DEBUG`/`HESSIAN_MEAN_FIELD` | They need a dense tensor. |
| `USE_RI` + explicit `SCF_DIRECT_4C` or `FUNCTIONAL_DIRECT_4C` | Mutually exclusive kernels. |
| `CC_RDM` without `UCCSD` | Built from the UCCSD amplitudes. |
| `UCCSD` without an SCF method, or + `READ_RESTART` | Needs a fresh SCF reference. |
| `FUNCTIONAL_INIT CORE` without `FUNCTIONAL`, or + `READ_RESTART`/`DEBUG`/`HESSIAN_MEAN_FIELD` | Needs a functional stage; the restart supplies the orbitals; the diagnostics validate HF orbitals. |
| `FCIDUMP` without `NON_RELATIVISTIC` + `FUNCTIONAL` | NON_REL only. |
| `READ_OCCUPANCIES` without `FUNCTIONAL` | Nothing to read into. |
| `FUNCTIONAL_DIRECT_4C` without `C4_SPINOR` + `FUNCTIONAL` + `FULL_OPTIMIZATION` | Replaces only that ADAM sub-loop. |
| `FUNCTIONAL_DIRECT_4C` + `FULL_OPTIMIZATION_4C_NEG` or `NEO`/`ADAM_NEO` | No UKB-direct Hessian exists. |

**Worth knowing** (everything else combines freely):

| Combination | Effect |
| --- | --- |
| `USE_RI` + `READ_RESTART`/`FULL_OPTIMIZATION_4C_NEG`/`NEO`/`ADAM_NEO` | Full effect; the RI tensor is rebuilt at the current geometry, never stored in the restart file. |
| `SCF_DIRECT_4C` + `CHOLESKY` | Independent; the post-SCF representation follows `CHOLESKY` alone. |
| `UCCSD` + `FUNCTIONAL` | CC runs first, always on the SCF orbitals (also with `FUNCTIONAL_INIT CORE`). |
| `FUNCTIONAL_POS_CHO_4C` + `CHOLESKY` + `SCF_DIRECT_4C` | Full effect: AO-basis `RkbCholesky` vectors are never built. |
| `FUNCTIONAL_POS_CHO_4C` + `READ_RESTART` + `CHOLESKY` | Full effect only if the geometry is unchanged and `FULL_OPTIMIZATION_4C_NEG` is off. |
| `FUNCTIONAL_POS_CHO_4C` otherwise (no `CHOLESKY`, SCF already needs `rkb_cholesky`, min-max, `DEBUG`) | No-op. |
| `ON_DEMAND_ERI`/`CHOLESKY_THRESHOLD` without `CHOLESKY`; `CARTESIAN` with `X2C`/`C4_SPINOR`; `X2C_NORM` without `X2C`; `CHECK_HESS_NEO` with `ADAM` | No-op. |

## X2C

`X2C` prints a one-electron X2C decoupling report and runs the X2C-HF SCF, independent of `NON_RELATIVISTIC`/`C4_SPINOR`. Both share an
analytic restricted-kinetic-balance small component, a spherical large basis
and a LOWGEN linear-dependence safety net.
Derivation of the decoupling and the renormalization: `doc/X2C.tex`.

1. Diagonalizing the orthonormalized 4-component Hamiltonian separates the
   positive- and negative-energy branches; eigenvalues, Kramers splitting and
   a partner check are printed.
2. The exact X2C Hamiltonian eliminates the small component exactly.
3. The X2C-HF core is the renormalized `R†hR` (`X2C_NORM TRUE`, matches PySCF's
   spinor X2C-1e), or the raw `h` orthogonalized with the large-component
   overlap only (`X2C_NORM FALSE`).
4. The X2C-HF SCF uses it as a fixed core with nonrelativistic two-electron
   integrals; with `X2C_NORM FALSE` it can converge below the 4-component energy.
5. The `Hessian_opt` gradient/Hessian tests run as for `C4_SPINOR`; X2C-HF is a
   genuine minimum.

Examples: `examples/water_X2C.inp`, `examples/water_X2C_debug.inp`.

## RDMFT functional evaluation

`FUNCTIONAL` runs after each SCF, with its orbitals and integrals as a fixed
background: initial occupations (`OCCUPATION_INIT`), the functional energy,
then occupations optimized by SQP with `sum(n_p) = NELEC`, `0 < n_p < 1`,
both members of each Kramers/spin pair tied together. `C4_SPINOR` excludes the
negative-energy branch (no-pair); `X2C` and `NON_RELATIVISTIC` let every
orbital compete.

`FUNCTIONAL_INIT CORE` starts the functional stage from the core-Hamiltonian
eigenvectors instead of the SCF orbitals; it can end in a different basin of the
orbital optimization. UCCSD is unaffected.

`Tr(2-RDM)` is printed next to N(N-1)/2: exact for MULLER, CA, CGA, PNOF and
pCCD; it deviates for GU, ML, MLSIC, POWER, RHFB, SD at fractional occupations
and for MULLER_AS.

Examples: `examples/water_muller.inp`, `examples/co-sto-3g_muller.inp`,
`examples/water_X2C_muller.inp`.

## PNOF functionals

`PNOF5`/`PNOF7`/`PNOF7S`/`GNOF` (`Occ_opt/PNOFs.h`): occupied and unoccupied
pairs are partitioned into `PNOF_SUBSPACES` coupling subspaces, each with one
principal pair and `PNOF_COUPLING - 1` unoccupied pairs, and occupations are
optimized from a feasible gamma-angle guess (`SQP_PNOF_OCC` selects L-BFGS or SQP;
both agree when converged). `GNOF`'s inter-subspace coupling needs a
finite-difference Hessian for SQP; the others are analytic. The default
(`1` subspace, coupling `2`) converges cleanly; larger settings may hit the
iteration cap at a correct point, so cross-check with the other `SQP_PNOF_OCC`.

Examples: `examples/water_gnof.inp`, `examples/co-sto-3g_gnof.inp`,
`examples/water_X2C_gnof.inp`.

## pCCD functional

`FUNCTIONAL PCCD` (`Occ_opt/pCCD.h`, derivation in `doc/kr_pccd.tex`) is
Kramers-restricted pair-CCD (Henderson et al., *J. Chem. Phys.* **141**, 244104
(2014)): pairs are split into frozen-core, active and deep-virtual windows, the
t-/z-amplitudes are solved at fixed orbitals, and with `FULL_OPTIMIZATION` the
resulting RDM feeds the same ADAM/NEO machinery as PNOF, re-solving the
amplitudes every macro-iteration (warm-started from the previous ones). Works on
`NON_REL`/`X2C`/`C4_SPINOR`, including the min-max stage, and with `READ_RESTART`
(amplitudes included). A restart file written by another functional (PNOF,
JK-only) is also accepted: its orbitals are used and t, z start from zero.

Amplitude solves use `DIIS` and switch to `NEWTON` if it diverges or reaches 1000
iterations; the first solve may further use Levenberg-Marquardt. A solve that
still fails is used unconverged and listed under *Issues*.

The energy is not variational in the orbitals, so the orbital optimization can
have several branches (e.g. stretched bonds). A cold start may land on a higher
one: follow the curve with a restart chain from a shorter bond length, or try
`FUNCTIONAL_INIT CORE`.

`READ_OCCUPANCIES` with pCCD: `OCC.in` lists the window occupations (occupied lowest..HOMO, then
virtuals; frozen pairs not listed), pair k couples HOMO-k with LUMO+k (n_occ + n_vir = 1), and the starting
t = -sqrt(n_a/n_i), z = -sqrt(n_i n_a) are then solved as usual. With the environment variable
`RERDMFT_PCCD_FIX_AMPLITUDES` they are never solved and stay fixed (a warning is printed).
A restart written with more frozen pairs than the new run (e.g. 4 -> 0) is embedded: the former core pairs
enter the window with t = z = 0. `PCCD_DEBUG` also prints ||R(t0)|| of the starting amplitudes.

Examples: `examples/ne_pccd_full_optimization.inp`, `ne_pccd_x2c.inp`,
`ne_pccd_c4.inp`, `ne_pccd_c4_neg.inp`.

## FULL_OPTIMIZATION for C4_SPINOR

The relativistic SCF is a min-max problem (Talman, *PRL* 57, 1091 (1986); Saue,
*ChemPhysChem* 12, 3077 (2011)): a minimum over positive-energy rotations, a
maximum over rotations mixing in the negative-energy branch. `FULL_OPTIMIZATION`
therefore excludes every rotation touching a negative-energy index (for `ADAM`
and `NEO`), which makes it an ordinary minimization like `NON_REL`/`X2C`.
`CHOLESKY` for `C4_SPINOR` decomposes the real AO Coulomb matrix over
Large-Large and Small-Small pairs (`C4_DHF/RkbCholesky.h`) and recompresses the
MO vectors to the positive-energy block.

Examples: `examples/lih_gnof_c4_full_optimization.inp` (dense),
`lih_gnof_c4_neo_full_optimization.inp` (NEO),
`lih_gnof_c4_full_optimization_cholesky.inp` (Cholesky); all give the same energy.

### FULL_OPTIMIZATION_4C_NEG

After the positive-energy minimum converges, a second stage finds the true
min-max point: minimize over positive-energy rotations, maximize over
electron-positron rotations, with NEO using a dynamic saddle order and a sector
partition; occupations are fully re-minimized each step. `DEBUG` adds an exact
dense-Hessian check. For light systems it moves the energy by ~1e-10 Ha, so it
mostly confirms the no-pair minimum. Examples:
`examples/lih_gnof_c4_neg_full_optimization.inp`,
`examples/lih_muller_c4_neg_full_optimization_cholesky.inp`.

### FUNCTIONAL_DIRECT_4C

ADAM builds its generalized Fock (and, through a double-counting trace identity,
its energy) directly from UKB AO integrals at the current orbitals
(`Hessian_opt/Ukb*Fock*.h`), instead of re-expressing a stored MO tensor in the
rotated basis at every step (O(n^5) dense, O(n_chol n^2) Cholesky). The final
numbers are identical. Cost per Fock build depends on the functional:

- `SD`/`MBB`/`CA`/`CGA`/`POWER`/`MULLER_AS`: O(1) UKB densities, the same speed as the stored path.
- `GU`/`ML`/`MLSIC`, `PNOF`, pCCD: O(n_mo) to O(n_total) densities (not separable).

It pays off where `CHOLESKY` and the dense tensor are infeasible (heavy elements,
big bases), but for PNOF/pCCD it is then still slow. Only `ORBITAL_OPTIMIZER ADAM`
is supported.

## Restart

A run with a `FUNCTIONAL` writes the final state to `RESTART.NON_REL`,
`RESTART.X2C` and/or `RESTART.4C` (`Utils/Restart.h`), read back as an identity
check; `FULL_OPTIMIZATION` rewrites it every 10 macro-iterations, and the min-max
result goes to `RESTART.4C_NEG` (write-only). It holds occupations, the frozen/
active window, gamma angles (PNOF), t-/z-amplitudes (pCCD), the MO coefficients
and metadata.

`READ_RESTART TRUE` skips the SCF and starts from that file:

1. Check method, `NELEC` and coefficient shape (not the geometry, so scans work).
2. Löwdin-orthonormalize against the current overlap if needed.
3. Re-establish exact Kramers pairing (`X2C`/`C4_SPINOR`).
4. `C4_SPINOR`: re-establish the positive-energy space with one Fock build.
5. Occupations/amplitudes seed the occupation optimization; a file that does not
   fit the current `JK_*`/`PCCD_*` window is refused.

A geometry scan is a chain: run the first point, then change the geometry, add
`READ_RESTART TRUE`, repeat (the new results overwrite `RESTART.*`; copy it per
point to keep a backup). Example: `examples/lih_gnof_read_restart.inp`.

## Environment variables

| Variable | Effect |
|---|---|
| `OMP_NUM_THREADS` | OpenMP threads (also the threaded BLAS). |
| `RERDMFT_PCCD_FIX_AMPLITUDES` | Any value, with `READ_OCCUPANCIES` + `FUNCTIONAL PCCD`: t, z built from `OCC.in` are never solved. |
| `RERDMFT_RI_HV_REFERENCE` | Any value: `USE_RI` products use the element-by-element reference code (same results, much slower). |
| `SWITCH_SIGN_ERI` | Any value: flips the sign of every electron-repulsion integral (diagnostic; dense integrals only). |

## Compiling LIBCINT

ReRDMFT links LIBCINT statically and only uses its Cartesian kernels. The
recipe below also serves [MOLGW](https://www.molgw.org/tuto_compilation/):

```sh
git clone https://github.com/sunqm/libcint.git
cd libcint && mkdir build && cd build
cmake -DWITH_RANGE_COULOMB=1 -DPYPZPX=1 -DBUILD_SHARED_LIBS=0 \
      -DENABLE_EXAMPLE=0 -DENABLE_TEST=0 -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Then use `libcint/build/libcint.a` (headers in `libcint/build/include/`).
`BUILD_SHARED_LIBS=0` is required; keep `WITH_CINT2_INTERFACE` on (default); add
`-DWITH_FORTRAN=0` if cmake asks for a Fortran compiler.

## Contributors

- Dr. M. Rodriguez-Mayorga
