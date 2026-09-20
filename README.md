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
| `HESSIAN_X2C` | bool | `FALSE` | Same as `HESSIAN_NON_REL`/`HESSIAN_4C`, but for the converged approximate X2C-HF solution -- spans the full X2C-HF spinor space (2 components, no negative-energy branch at all), where a genuine MINIMUM (no negative eigenvalues) is physically expected, same as `HESSIAN_NON_REL`. |
| `HESSIAN_FUNCTIONAL` | bool | `FALSE` | Only with `FUNCTIONAL` (JK-only or PNOF): after the occupation-number optimization, builds the FULL real-step orbital-rotation Hessian of that functional at the OPTIMIZED occupations (`jkOnlyHessianMatrix` / `pnofHessianMatrix`), symmetrizes and diagonalizes it, and reports the number of negative / near-zero / positive eigenvalues, the lowest eigenvalues, and the orbital gradient (orbitals stay the converged HF/DHF ones, only the occupations are optimized, so the point is not orbital-stationary). Runs for every SCF path that evaluates the functional (`NON_RELATIVISTIC`, `X2C`, `C4_SPINOR`); for `C4_SPINOR` the negative-energy branch contributes the very negative (order `-2mc^2`) eigenvalues. For complex spinors (X2C, C4_DHF) it also builds and diagonalizes the JOINT real+imaginary (`t`,`y`) symmetric Hessian together with the joint gradient vector `[dE/dt; dE/dy]` (`jkOnlyJointHessianMatrix` / `pnofJointHessianMatrix`, `jointOrbitalGradient`) -- the input of a Newton-Raphson orbital step. Examples: `examples/water_muller_hessian.inp`, `examples/water_muller_as_hessian.inp`, `examples/lih_gnof_hessian.inp`; the finite-difference validation of every block is in `examples/water_X2C_muller_joint_blocks.inp` (JK_only, `DEBUG`) and `examples/lih_X2C_gnof_joint_blocks.inp` (PNOF, `DEBUG` + `VERBOSE 1`). |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (atomic units). A very large value probes the nonrelativistic limit; smaller values exaggerate relativistic effects. |
| `MIXING` | double, in `(0, 1]` | `0.4` | Linear density-matrix mixing weight for the `C4_DHF` SCF loop: the density fed into the next iteration is `mixing*P_new + (1-mixing)*P_current`. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum number of `C4_DHF` SCF cycles before giving up (the last cycle's results are still returned, with `converged = false`). |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` Hartree | `C4_DHF` SCF energy-change convergence threshold. Combined with `DENSITY_TOLERANCE` by OR: converged as soon as either is satisfied. |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold (see `ENERGY_TOLERANCE`). |
| `CHOLESKY` | bool | `FALSE` | Use a pivoted Cholesky decomposition (`Utils/Cholesky_Decomposition.h`) to reduce the cost of transforming two-electron integrals from one basis to another, instead of the original direct 4-leg transform: decompose once in the old basis, transform only the (often far fewer) Cholesky vectors, then reconstruct in the new basis. The decomposition itself is the same base (Beebe-Linderberg) algorithm the eT program uses (confirmed against Folkestad, Kjonstad, Koch, *J. Chem. Phys.* 150, 194112 (2019)), including their own "efficient algorithm" batching -- a whole qualified BATCH of candidate pivots is corrected against every prior batch's vectors via one GEMM at a time, rather than one growing GEMV per pivot, cutting the aggregate cost from O(Nchol^2*n^2) to O(Nchol^2*n^2/64). Applies to the UKB-Small -> RKB-Small projection (`C4_DHF/RkbTwoElectron.cpp`) and to every AO/spinor -> MO transform (`NON_REL`, `X2C`, `C4_DHF`). With `FULL_OPTIMIZATION TRUE` and `CHOLESKY TRUE` the macro-iteration loop itself runs on Cholesky vectors (`Utils/CholeskyEri.h`): the two-electron integrals are decomposed once, in the Coulomb grouping (rank ~ a few times n_AO, far below the n^2 of the bra-ket grouping used by the basis-transform helpers), every ADAM orbital rotation acts on the vectors (O(Nchol n^3) instead of O(n^5)), integral elements are evaluated on demand (O(Nchol)) and no dense n^4 tensor is kept inside the loop (the start-up/final Kramers-structure tests assemble one transiently). Energies agree with the dense loop to the decomposition threshold. Both paths reproduce the same converged SCF energies/gradients to full displayed precision; a handful of near-degenerate eigenvalues in the optional cheap-Hessian diagnostics (`HESSIAN_*`) can differ at the 1e-2 level purely from floating-point-order sensitivity of that near-degeneracy, not from an error in either path. The UKB-Small -> RKB-Small piece compresses genuinely well and is where `CHOLESKY TRUE` helps most; the AO/spinor -> MO transforms often show little to no rank reduction (the complex spinor pair-space lacks the symmetry real AO integrals have) -- confirmed directly on H2/cc-pVTZ, where BOTH `X2C`'s and `C4_DHF`'s own MO transforms need a Cholesky rank close to their full dimension (n^2, essentially no compression). For `X2C` there this is still fast in absolute terms (~4.9s total, matching `CHOLESKY FALSE`'s energy exactly) simply because that basis is small (n=60); for `C4_DHF`'s much larger RKB spinor basis (n=120) the same lack of compression makes it impractically slow (100s+) despite the batching improvement described above -- so it defaults to `FALSE` -- try `CHOLESKY TRUE` mainly for the UKB-Small -> RKB-Small projection, or for an AO/spinor -> MO transform only after confirming (e.g. with a quick timed run) that it actually compresses for your system. |
| `CHOLESKY_THRESHOLD` | double (> 0) | `1e-10` | Residual-diagonal cutoff for the `CHOLESKY TRUE` decomposition: a pivot below this is treated as numerical noise and decomposition stops. Loosening it (e.g. `1e-6`) finds fewer Cholesky vectors (faster, less accurate); tightening it (e.g. `1e-13`) finds more (slower, closer to an exact reconstruction). Only meaningful when `CHOLESKY` is `TRUE`. |
| `CACHE_INTEGRALS` | bool | `FALSE` | Cache the two-electron integral tensors to disk and reuse them on a later run with matching geometry+basis (see below). |
| `CACHE_DIR` | string | `.rerdmft_cache` | Directory (created if missing) used by `CACHE_INTEGRALS`. |
| `FUNCTIONAL` | string | *(none)* | Selects a density matrix functional approximation to evaluate on the converged `NON_REL`/`X2C`/`C4_DHF` orbitals: either a JK-only functional (`Occ_opt/JK_only.h`, Table 1 of Rodriguez-Mayorga et al., *Phys. Chem. Chem. Phys.* 2017) -- one of `SD`, `MBB` (or `MULLER`), `BBC2`, `CA`, `CGA`, `ML`, `MLSIC`, `GU`, `POWER` -- or a Piris natural orbital functional (`Occ_opt/PNOFs.h`) -- one of `PNOF5`, `PNOF7`, `PNOF7S`, `GNOF`. Setting this triggers the whole RDMFT functional evaluation described below; with no `FUNCTIONAL` keyword at all, that step is skipped entirely (it is not gated by `DEBUG`). |
| `OCCUPATION_INIT` | string | `PROPORTIONAL` | How to generate the initial fractional occupation numbers for a JK-only `FUNCTIONAL` (see below). `PROPORTIONAL`: an aufbau (idempotent) reference redistributed proportionally into an interior box -- temperature-independent. `FERMI_DIRAC`: smeared at `TEMPERATURE` instead. Only meaningful when `FUNCTIONAL` is one of the JK-only names (PNOF functionals build their own initial guess, see below). |
| `TEMPERATURE` | double (> 0) | `1000` (Kelvin) | Electronic temperature used to smear orbital energies into fractional Fermi-Dirac occupations. Only consumed when `OCCUPATION_INIT FERMI_DIRAC` is selected; ignored (but still validated) otherwise. |
| `PNOF_SUBSPACES` | int (>= 1) | `1` | Only meaningful when `FUNCTIONAL` is a PNOF name. How many independent coupling subspaces to build outward from HOMO (`Occ_opt/Orb_subspaces.h`): `1` builds just the HOMO subspace, `2` additionally builds a separate HOMO-1 subspace, and so on. Throws if this exceeds the number of occupied Kramers/spin pairs available. |
| `PNOF_COUPLING` | int (>= 2) | `2` | Only meaningful when `FUNCTIONAL` is a PNOF name. The SIZE of each subspace, in Kramers/spin pairs: 1 occupied pair + (`PNOF_COUPLING` - 1) unoccupied pairs. `2` (the default) is plain HOMO-LUMO perfect pairing; `3` couples each occupied pair with its two closest unoccupied pairs (LUMO and LUMO+1); and so on. Every subspace's unoccupied pairs are a disjoint block (never shared between subspaces); throws if `PNOF_SUBSPACES * (PNOF_COUPLING - 1)` exceeds the number of unoccupied pairs available. |
| `SQP_PNOF_OCC` | bool | `FALSE` | Only meaningful when `FUNCTIONAL` is a PNOF name. `FALSE` (the default): optimize the occupations via `Utils/LBFGS.h` over the UNCONSTRAINED gamma angles (`standalone_donof`'s own approach). `TRUE`: optimize via `Utils/SQP.h` over the occupations directly instead, subject to explicit box+equality constraints. The two methods solve the same problem and agree to full displayed precision whenever both converge cleanly (see the PNOF section below); only one runs per calculation, never both. |
| `FULL_OPTIMIZATION` | bool | `FALSE` | After the occupation-number optimization at the HF/X2C orbitals (needs `FUNCTIONAL`), macro-iterate to full convergence, as in DoNOF's driver: ADAM (`Utils/ADAM.h`, DoNOF's parameters: learning rate 0.01, beta1 0.7, beta2 0.9, 10 (+20 per restart) steps per call, learning rate x0.2 after a call that made no progress) over the orbital rotations at fixed occupations, then re-optimization of the occupation numbers at the new orbitals, until `|E - E_old| < MACRO_ENERGY_TOLERANCE` and ADAM asked for no restart. Runs on the MO integrals (rotated with an exact O(n^5) leg transform). **NON_REL** (real spin-orbitals) and **X2C** (complex spinors, with the rotations restricted to the time-reversal-symmetric Kramers-restricted subspace, `Utils/KramersRestriction.h`); not available for `C4_SPINOR` (a message is printed). BEFORE the loop, validation checks are run on the actual data -- exact rotation vs the Cholesky rotation, orbital gradient vs finite differences of the energy, and for X2C occupation equality within Kramers pairs, time-reversal symmetry of the gradient and of a probe Kramers-restricted rotation -- and the loop is only entered if every check passes (for PNOF also that the energy the orbital stage differentiates equals the energy the occupation optimizer minimizes). NON_REL rotations are spin-restricted (alpha and beta rotate identically). After the loop a FINAL TEST verifies the optimized orbitals: for X2C the Kramers pairing (`Theta|2k> = |2k+1>`) of the total rotation, of h and of the two-electron integrals, occupation equality and time-reversal symmetry of the gradient; for NON_REL the alpha/beta symmetry of h and the integrals; for PNOF that the pair-symmetric energy still equals the full two-RDM energy. Works for the JK_only and the PNOF functionals. |
| `MAX_MACRO_ITERATIONS` | int | `1000` | Maximum number of macro-iterations (ADAM orbital step + occupation re-optimization) of `FULL_OPTIMIZATION`. |
| `MACRO_ENERGY_TOLERANCE` | float | `1e-9` | Energy convergence threshold of the macro-iteration loop (`tolE` of the Fortran code). |
| `ORBITAL_GRADIENT_TOLERANCE` | float | `1e-5` | ADAM's orbital-gradient convergence threshold (max gradient entry; `10**-itolLambda` of the Fortran code). |
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
   subject to `sum(n_p) = NELEC` and `0 < n_p < 1`, and report the
   optimized occupations, their sum, and the optimized energy.

For `C4_SPINOR`, the negative-energy (Dirac sea) branch is excluded
from both steps entirely (pinned at exactly zero occupation, never an
optimization variable), preserving the no-pair approximation; for `X2C`
(like `NON_RELATIVISTIC`), every spinor competes for occupation, since
X2C's own decoupling already eliminated that branch entirely. Results
are printed after the corresponding SCF's own energy, gradient, and
(if requested) Hessian diagnostics -- optimized occupation numbers are
listed at fixed 5-decimal precision, in two columns for `C4_SPINOR` and
`X2C` (adjacent Kramers pairs side by side, which should read as
identical values -- confirmed to hold through the SQP optimization too)
and one column otherwise.

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

