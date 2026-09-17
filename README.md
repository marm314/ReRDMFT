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

## Input file keywords

Keywords are case-insensitive, one per line, optionally followed by
`=` before the value (e.g. `DEBUG TRUE`, `DEBUG = TRUE`, and
`DEBUG=TRUE` are all equivalent). Lines starting with `#` (or a `#`
anywhere on a line) are comments.

| Keyword | Type | Default | Meaning |
| --- | --- | --- | --- |
| `NELEC` (or `NELECTRONS`) | int | *required* | Number of electrons. |
| `BASIS` | string | *required* | Gaussian basis set file name. |
| `GEOMETRY` ... `END` | block | *required* | Molecular geometry as `<symbol> <x> <y> <z>` lines, one atom per line, coordinates in Angstrom (converted to Bohr internally). |
| `NON_RELATIVISTIC` | bool | `FALSE` | Run the standard nonrelativistic Hartree-Fock SCF (`NON_REL`). |
| `C4_SPINOR` | bool | `FALSE` | Run the 4-component Dirac-Hartree-Fock SCF (`C4_DHF`), building the RKB two-electron Coulomb tensor. Opt-in since both time and memory cost scale steeply with basis size. |
| `DEBUG` | bool | `FALSE` | Print detailed basis/matrix diagnostics, plus internal cross-checks (efficient-vs-general gradient/Hessian formulas, finite-difference gradient/Hessian tests) for whichever of `NON_RELATIVISTIC`/`C4_SPINOR` is on. |
| `VERBOSE` | int (>= 0) | `0` | Only meaningful with `DEBUG TRUE`. At `0`, skips the extra O(n^5) dense-2-RDM cross-check (`Hessian_opt/GeneralizedFock.h`'s fully general path); at `> 0`, also runs it. At `> 1` (`C4_SPINOR` only), additionally validates the mixed real/imaginary orbital-rotation Hessian block (`Hessian_opt/HartreeExchangeHessian.h`'s `hartreeExchangeHessianElementMixed`) against a genuine mixed-direction finite difference. All other `DEBUG` cross-checks are cheap and always run regardless of this setting. |
| `HESSIAN_NON_REL` | bool | `FALSE` | Build the FULL cheap (Hartree/exchange-ansatz) orbital-rotation Hessian for the converged `NON_REL` solution and diagonalize it, reporting whether it is a genuine minimum (no negative eigenvalues). Independent of `DEBUG`/`VERBOSE`; O(n^5) to build plus O(n^6) to diagonalize, so opt-in. |
| `HESSIAN_4C` | bool | `FALSE` | Same as `HESSIAN_NON_REL`, but for the converged `C4_DHF` solution -- spans the FULL RKB spinor space (including the negative-energy branch), where negative eigenvalues (a saddle point) are physically expected rather than a bug. |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (atomic units). A very large value probes the nonrelativistic limit; smaller values exaggerate relativistic effects. |
| `MIXING` | double, in `(0, 1]` | `0.4` | Linear density-matrix mixing weight for the `C4_DHF` SCF loop: the density fed into the next iteration is `mixing*P_new + (1-mixing)*P_current`. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum number of `C4_DHF` SCF cycles before giving up (the last cycle's results are still returned, with `converged = false`). |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` Hartree | `C4_DHF` SCF energy-change convergence threshold. Combined with `DENSITY_TOLERANCE` by OR: converged as soon as either is satisfied. |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold (see `ENERGY_TOLERANCE`). |
| `CACHE_INTEGRALS` | bool | `FALSE` | Cache the two-electron integral tensors to disk and reuse them on a later run with matching geometry+basis (see below). |
| `CACHE_DIR` | string | `.rerdmft_cache` | Directory (created if missing) used by `CACHE_INTEGRALS`. |
| `FUNCTIONAL` | string | *(none)* | Selects a JK-only density matrix functional approximation (`Occ_opt/JK_only.h`, Table 1 of Rodriguez-Mayorga et al., *Phys. Chem. Chem. Phys.* 2017) to evaluate on the converged `NON_REL`/`C4_DHF` orbitals: one of `SD`, `MBB` (or `MULLER`), `BBC2`, `CA`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER`. Setting this triggers the whole RDMFT functional evaluation described below; with no `FUNCTIONAL` keyword at all, that step is skipped entirely (it is not gated by `DEBUG`). |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | How to generate the initial fractional occupation numbers for `FUNCTIONAL` (see below). `PROPORTIONAL`: an aufbau (idempotent) reference redistributed proportionally into an interior box -- temperature-independent. `FERMI_DIRAC`: smeared at `TEMPERATURE` instead. Only meaningful when `FUNCTIONAL` is set. |
| `TEMPERATURE` | double (> 0) | `1000` (Kelvin) | Electronic temperature used to smear orbital energies into fractional Fermi-Dirac occupations. Only consumed when `OCCUPATION_INIT FERMI_DIRAC` is selected; ignored (but still validated) otherwise. |
| `X2C` | bool | `FALSE` | Print the one-electron X2C decoupling report AND run the approximate X2C-HF SCF, both described below, printed between the `NON_RELATIVISTIC` and `C4_SPINOR` final reports. Independent of `C4_SPINOR` -- the underlying one-electron RKB Hamiltonian is always built regardless (it also seeds `C4_SPINOR`'s own SCF initial guess); this keyword gates running/printing the X2C-specific steps only. Combined with `DEBUG`, additional cross-checks are printed within the X2C sections (see below). |

## X2C decoupling and X2C-HF

Setting `X2C` runs everything below, built from
`X2C_DHF/X2C_decoupling.h`, `X2C_DHF/X2C_hamiltonian.h`, and
`X2C_DHF/X2C_HF.h`, and prints it **between** the `NON_RELATIVISTIC`
and `C4_SPINOR` (4-component DHF) final reports, regardless of whether
either of those keywords is itself set (X2C sits conceptually between
the nonrelativistic and exact 4-component treatments). With `DEBUG`
also set, extra detail is added throughout (called out per step
below); the underlying computation itself is unaffected by `DEBUG`.

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

See `examples/water_X2C.inp` for a plain worked example, or
`examples/water_X2C_debug.inp` for the same run with `DEBUG TRUE` (all
the extra cross-checks described above, plus `HESSIAN_4C`).

## RDMFT functional evaluation

Setting `FUNCTIONAL` runs an additional step after each requested SCF
(`NON_RELATIVISTIC`/`C4_SPINOR`) converges, using its orbitals and
one-/two-electron integrals as a **fixed** background (no orbital
reoptimization):

1. Generate initial fractional occupation numbers via `OCCUPATION_INIT`
   (`PROPORTIONAL` by default, or `FERMI_DIRAC` at `TEMPERATURE`).
2. Evaluate `FUNCTIONAL`'s energy on those occupations
   (`Occ_opt/JK_only.h` + `Hessian_opt/HartreeExchangeGradient.h`'s
   `hartreeExchangeEnergy`).
3. Optimize the occupation numbers further via a sequential quadratic
   programming solver (`Occ_opt/SQP.h`), minimizing that same energy
   subject to `sum(n_p) = NELEC` and `0 < n_p < 1`, and report the
   optimized occupations, their sum, and the optimized energy.

For `C4_SPINOR`, the negative-energy (Dirac sea) branch is excluded
from both steps entirely (pinned at exactly zero occupation, never an
optimization variable), preserving the no-pair approximation. Results
are printed after the corresponding SCF's own energy, gradient, and
(if requested) Hessian diagnostics -- optimized occupation numbers are
listed at fixed 5-decimal precision, in two columns for `C4_SPINOR`
(adjacent Kramers pairs side by side, which should read as identical
values) and one column otherwise.

See `examples/water_muller.inp` and `examples/co-sto-3g_muller.inp`
for worked examples (`FUNCTIONAL MULLER`).

## Two-electron integral disk cache

The (expensive, O(N^4)) two-electron integral tensors depend only on the
molecular geometry and basis set -- NOT on `SPEED_OF_LIGHT` -- so runs
that share a geometry+basis but scan over the speed of light (e.g. a
`water-c1000.inp` / `water-c100000.inp` / `water-c10000000.inp` series)
would otherwise recompute bit-identical integrals every time. Setting

```
CACHE_INTEGRALS TRUE
```

in an input file caches each built tensor to disk (under `CACHE_DIR`,
default `.rerdmft_cache`, created automatically) keyed by a hash of the
actual basis functions, and reuses it on a later run with a matching
key instead of rebuilding via libcint. It is off by default, since it
writes files to disk; a cache from a different build of the code is
detected via an embedded format version and never reused. Files under
the cache directory are a same-machine binary format, not meant to be
inspected or shared.

## Contributors

- Dr. M. Rodriguez-Mayorga 

