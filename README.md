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
| `BASIS` | string | *required* | Gaussian basis set file name. |
| `GEOMETRY` ... `END` | block | *required* | Molecular geometry as `<symbol> <x> <y> <z>` lines, one atom per line, coordinates in Angstrom (converted to Bohr internally). |
| `NON_RELATIVISTIC` | bool | `FALSE` | Run the standard nonrelativistic Hartree-Fock SCF (`NON_REL`). |
| `C4_SPINOR` | bool | `FALSE` | Run the 4-component Dirac-Hartree-Fock SCF (`C4_DHF`), building the RKB two-electron Coulomb tensor. Opt-in since both time and memory cost scale steeply with basis size. Both this SCF and the `X2C` one are Kramers-restricted for an even `NELEC`: every iteration's density is projected onto its time-reversal-even part (spin-orbit mixing of the spinors is kept; only the magnetization is removed), so they cannot drift into a lower-energy Kramers-broken solution at unstable geometries (e.g. stretched LiH with `CHOLESKY TRUE`). The output reports the largest element removed (~0 when the iteration stayed symmetric by itself). |
| `DEBUG` | bool | `FALSE` | Print detailed basis/matrix diagnostics, plus internal cross-checks (efficient-vs-general gradient/Hessian formulas, finite-difference gradient/Hessian tests) for whichever of `NON_RELATIVISTIC`/`C4_SPINOR` is on. Every test that needs the two-electron integrals as a DENSE tensor (the RDMFT-ansatz gradient test, the Kramers/spin structure tests of the integrals, exact-vs-Cholesky rotation, the Hessian-diagonal finite difference, dense-vs-Cholesky comparisons of the Fock matrix and MO integrals) runs only with `DEBUG TRUE`; the dense tensors are built for it on demand. |
| `VERBOSE` | int (>= 0) | `0` | Only with `DEBUG TRUE`. `0` skips the extra dense-2-RDM cross-check; `>0` runs it; `>1` (`C4_SPINOR` only) also checks the mixed real/imaginary Hessian block against a finite difference. |
| `HESSIAN_NON_REL` | bool | `FALSE` | Build and diagonalize the full orbital-rotation Hessian of the converged `NON_REL` solution, reporting whether it is a genuine minimum. O(n^5)/O(n^6), opt-in. |
| `HESSIAN_4C` | bool | `FALSE` | Same as `HESSIAN_NON_REL` for the converged `C4_DHF` solution; negative eigenvalues (a saddle) are physically expected there (negative-energy branch included). |
| `HESSIAN_X2C` | bool | `FALSE` | Same as `HESSIAN_NON_REL` for the converged approximate X2C-HF solution; a genuine minimum is physically expected (no negative-energy branch). |
| `HESSIAN_FUNCTIONAL` | bool | `FALSE` | Only with `FUNCTIONAL`: after occupation optimization, build and diagonalize the full orbital-rotation Hessian of that functional at the optimized occupations (fixed orbitals), reporting eigenvalue counts and the orbital gradient. For complex spinors also builds the joint real+imaginary (`t`,`y`) Hessian/gradient. Examples: `examples/water_muller_hessian.inp`, `examples/lih_gnof_hessian.inp`. |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (atomic units): larger probes the nonrelativistic limit, smaller exaggerates relativistic effects. |
| `MIXING` | double, in `(0, 1]` | `0.4` | Linear density-mixing weight for the SCF loops, used only with `DIIS FALSE`. |
| `DIIS` | bool | `TRUE` | Use Pulay DIIS (`Utils/DIIS.h`, commutator error `F P S - S P F`) instead of linear density mixing in all three SCF loops. Typically 5-10x fewer iterations. Converges to *a* stationary point, not always the one mixing finds -- stretched LiH (`examples/lih_X2C_gnof_full_optimization_dissociated.inp`) sets `DIIS FALSE` for that reason. |
| `DIIS_SIZE` | int (>= 2) | `5` | Number of (error, Fock) pairs kept by `DIIS TRUE`. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum number of `C4_DHF` SCF cycles before giving up. |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` Hartree | `C4_DHF` SCF energy-change convergence threshold (OR'd with `DENSITY_TOLERANCE`). |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold. |
| `CHOLESKY` | bool | `FALSE` | Hold the two-electron integrals as Cholesky vectors and never build an n^4 object. The real AO Coulomb matrix is decomposed ONCE (`Utils/Cholesky_Decomposition.h`: NON_REL/X2C decompose the AO integrals, C4_SPINOR the combined {Large-Large} u {Small-Small} pair matrix, `C4_DHF/RkbCholesky.h`); the SCF Fock matrices, the MO-basis integrals (`Utils/AoCholesky.h`) and `FULL_OPTIMIZATION` all work on those vectors, and the AO integrals are not used again (they are rebuilt only under `DEBUG`, for the dense-vs-Cholesky checks). Every decomposition is verified against the integrals and retried with a smaller pivot batch if it misses `100*CHOLESKY_THRESHOLD + 1e-9`. With `ORBITAL_OPTIMIZER NEO` the Hessian-vector product is a finite difference of the gradient on the rotated vectors, O(N_chol n^3) with no dense cache. Without `CHOLESKY` the integrals are held as unique-element stores (`Utils/SymmetricEri.h`: about n^4/8 real or n^4/4 complex numbers, the rest rebuilt by symmetry) and transformed in slabs, never as a dense n^4 tensor. |
| `CHOLESKY_THRESHOLD` | double (> 0) | `1e-10` | Residual-diagonal cutoff for the decomposition; looser = fewer vectors (faster, less accurate), tighter = more (slower, more exact). Only with `CHOLESKY TRUE`. |
| `RESTART_FILE` | string | `RESTART` | Base name of the binary restart files written after a `NON_RELATIVISTIC`/`X2C`/`C4_SPINOR` run with a `FUNCTIONAL` (see *Restart file* below). `NONE` disables them. |
| `FUNCTIONAL` | string | *(none)* | Selects the density matrix functional to evaluate on the converged orbitals: a JK-only functional (`Occ_opt/JK_only.h`) -- `SD`, `MBB`/`MULLER`, `BBC2`, `CA`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER` -- or a Piris natural orbital functional (`Occ_opt/PNOFs.h`) -- `PNOF5`, `PNOF7`, `PNOF7S`, `GNOF`. Unset: the whole RDMFT evaluation step below is skipped. |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | Initial fractional occupations for a JK-only `FUNCTIONAL`. `PROPORTIONAL`: aufbau redistributed into an interior box. `FERMI_DIRAC`: smeared at `TEMPERATURE`. (PNOF functionals build their own guess.) |
| `JK_FROZEN_PAIRS` | int (>= 0) | `0` | Only with a JK-only `FUNCTIONAL`. Freezes the `2*JK_FROZEN_PAIRS` LOWEST-energy spin-orbitals/spinors at EXACTLY occupation 1 (never an SQP variable), mirroring PNOF's own frozen core. Counted in pairs (spin/Kramers partners) so no separate evenness check is ever needed. |
| `JK_ACTIVE_PAIRS` | int (>= 1) | all remaining | Only with a JK-only `FUNCTIONAL`. The NEXT `2*JK_ACTIVE_PAIRS` spin-orbitals/spinors by energy (above `JK_FROZEN_PAIRS`) are the fractional-occupation SQP window, `sum(n) = NELEC - 2*JK_FROZEN_PAIRS`; everything above that is deep virtual, pinned at exactly 0. `NELEC - 2*JK_FROZEN_PAIRS` must be strictly between 0 and `2*JK_ACTIVE_PAIRS`, else a clear error. |
| `TEMPERATURE` | double (> 0) | `1000` (Kelvin) | Smearing temperature, only used with `OCCUPATION_INIT FERMI_DIRAC`. |
| `PNOF_SUBSPACES` | int (>= 1) | `1` | Only with a PNOF `FUNCTIONAL`. Number of independent coupling subspaces built outward from HOMO (`Occ_opt/Orb_subspaces.h`); throws if it exceeds the occupied pairs available. |
| `PNOF_COUPLING` | int (>= 2) | `2` | Only with a PNOF `FUNCTIONAL`. Size of each subspace in pairs: 1 occupied + (`PNOF_COUPLING`-1) unoccupied. `2` is plain HOMO-LUMO pairing. |
| `SQP_PNOF_OCC` | bool | `FALSE` | Only with a PNOF `FUNCTIONAL`. `FALSE`: optimize via `Utils/LBFGS.h` over unconstrained gamma angles (DoNOF's own approach). `TRUE`: optimize via `Utils/SQP.h` over occupations directly, with explicit box+equality constraints. Both agree to full precision when both converge. |
| `FULL_OPTIMIZATION` | bool | `FALSE` | After occupation optimization (needs `FUNCTIONAL`), macro-iterate to convergence: an orbital-rotation step (`ORBITAL_OPTIMIZER`) at fixed occupations, then occupation re-optimization at the new orbitals, until `|E-E_old| < MACRO_ENERGY_TOLERANCE`. Works for `NON_REL` (real spin-orbitals), `X2C` (complex, Kramers-restricted rotations), and `C4_SPINOR` (complex, Kramers-restricted rotations *restricted to the positive-energy spinors only* -- see below). Validation checks gate the loop, and a final test verifies the optimized orbitals keep the expected symmetry (Kramers pairing for X2C/C4_SPINOR, spin symmetry for NON_REL, pair-symmetric energy for PNOF). |
| `MAX_MACRO_ITERATIONS` | int | `1000` | Maximum number of macro-iterations of `FULL_OPTIMIZATION`. |
| `MACRO_ENERGY_TOLERANCE` | float | `1e-9` | Energy convergence threshold of the macro-iteration loop. |
| `ORBITAL_GRADIENT_TOLERANCE` | float | `1e-5` | ADAM's/NEO's orbital-gradient convergence threshold (max gradient entry). |
| `ORBITAL_OPTIMIZER` | string | `ADAM` | Which method drives `FULL_OPTIMIZATION`'s orbital-rotation step. `ADAM`: DoNOF's own first-order optimizer (`Utils/ADAM.h`). `NEO`: `Utils/NEO.h`'s matrix-free, second-order Newton method targeting the ground state, using a row-based Hessian-vector product (no dense Hessian formed); also works with `CHOLESKY TRUE` and `C4_SPINOR`. Converges in far fewer macro-iterations than ADAM and usually matches its energy to 1e-6-1e-9. On some PNOF/GNOF NON_REL systems NEO can land on a different stationary point; a post-loop Hessian check detects this and automatically escapes a detected saddle (perturb along the negative-curvature eigenvector, retry up to 3 times), but a residual gap to a genuine alternate minimum is reported rather than silently fixed -- compare against `ADAM` as a routine cross-check. Templates: `examples/*_neo_full_optimization.inp`. |
| `FULL_OPTIMIZATION_4C_NEG` | bool | `FALSE` | Only meaningful for `C4_SPINOR` + `FULL_OPTIMIZATION` (any `FUNCTIONAL`, `CHOLESKY` TRUE or FALSE). After the positive-energy-only optimization has converged, runs the genuine **min-max** stage: orbital rotations now include the positive <-> negative-energy pairs, driven by NEO to a saddle point (whatever `ORBITAL_OPTIMIZER` says), alternating with a full re-minimization of the occupation numbers -- see below. Skipped, with a message, if the first stage did not converge. |
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
   (`X2C_DHF/X2C_OrbitalGradient.h`), the general dense-2-RDM
   cross-check at `VERBOSE > 0`, a finite-difference gradient/Hessian
   check, and the mixed real/imaginary Hessian block at `VERBOSE > 1`;
   and, under `HESSIAN_X2C`, the full orbital-rotation Hessian
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
`HESSIAN_X2C TRUE`), or `examples/water_X2C_debug.inp` for the same run
with `DEBUG TRUE` (all the extra cross-checks described above, plus
`HESSIAN_4C`/`HESSIAN_X2C`); `examples/water_debug_verbose2.inp` also
exercises X2C-HF's own mixed-Hessian check (`VERBOSE 2`) alongside
`C4_DHF`'s.

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
`target_order = 0` is then the physically correct target, confirmed by its own post-loop Hessian
check (`[PASS] the point is a genuine minimum`). The positive-energy branch is itself
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
* *Checks.* The integrals rotated to the starting point must reproduce the first stage's energy; at the end
  the Hessian is verified block-wise (minimum over the positive-energy rotations, maximum over the occupied
  electron-positron ones); `DEBUG TRUE` additionally counts all negative eigenvalues (about as many Hessian
  products as the order of the saddle).

For light systems the second stage moves the energy by ~1e-10 Hartree (the electron-positron gradient at the
no-pair minimum is tiny); it is a check that the no-pair minimum really is the min-max point and it is exact
for any `FUNCTIONAL`. See `examples/lih_gnof_c4_neg_full_optimization.inp` (GNOF, dense) and
`examples/lih_muller_c4_neg_full_optimization_cholesky.inp` (MULLER, `CHOLESKY TRUE`).

## Restart file

At the end of a `NON_RELATIVISTIC`, `X2C` and/or `C4_SPINOR` run with a `FUNCTIONAL`,
the final RDMFT state is written in **binary** to `<RESTART_FILE>.NON_REL`,
`<RESTART_FILE>.X2C_HF` and, for the 4-component path, `<RESTART_FILE>.4C`
(the positive-energy-only minimization) and `<RESTART_FILE>.4C_NEG` (the
`FULL_OPTIMIZATION_4C_NEG` min-max stage, written only when that stage ran and its
checks passed); default base name `RESTART`, in the working
directory; `RESTART_FILE NONE` disables it). The format, writer and reader
are in `Utils/Restart.h`; the program only writes the files for now (the
reader is used to verify them), reading a restart to start a calculation
is not implemented yet.

Contents:

- the **occupation numbers** (full vector, one entry per spin-orbital /
  spinor of the MO basis) -- for the JK-only functionals this is the state;
- for the **PNOF** functionals also the **gamma angles** (subspace after
  subspace, `PNOF_COUPLING - 1` angles each, the trigonometric
  parameterization of `Occ_opt/PNOFs.h`), obtained from the final
  occupations, so the SQP and the L-BFGS branches write the same thing;
- the final **molecular-orbital coefficients** `C = C_scf * U_total` in the
  AO spin-orbital basis (`4C`/`4C_NEG`: the 4-component RKB spinor basis
  [Large-alpha; Large-beta; Small; Small], columns in ascending energy with the
  negative-energy branch first and occupation 0; `4C_NEG`'s rotation includes the
  positive <-> negative mixing): `U_total` is the accumulated `FULL_OPTIMIZATION`
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

## Contributors

- Dr. M. Rodriguez-Mayorga 

