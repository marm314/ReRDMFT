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
| `VERBOSE` | int (>= 0) | `0` | Only meaningful with `DEBUG TRUE`. At `0`, skips the extra O(n^5) dense-2-RDM cross-check (`Hessian_opt/GeneralizedFock.h`'s fully general path); at `> 0`, also runs it. All other `DEBUG` cross-checks are cheap and always run regardless of this setting. |
| `HESSIAN_NON_REL` | bool | `FALSE` | Build the FULL cheap (Hartree/exchange-ansatz) orbital-rotation Hessian for the converged `NON_REL` solution and diagonalize it, reporting whether it is a genuine minimum (no negative eigenvalues). Independent of `DEBUG`/`VERBOSE`; O(n^5) to build plus O(n^6) to diagonalize, so opt-in. |
| `HESSIAN_4C` | bool | `FALSE` | Same as `HESSIAN_NON_REL`, but for the converged `C4_DHF` solution -- spans the FULL RKB spinor space (including the negative-energy branch), where negative eigenvalues (a saddle point) are physically expected rather than a bug. |
| `SPEED_OF_LIGHT` | double (> 0) | CODATA value | Override the speed of light (atomic units). A very large value probes the nonrelativistic limit; smaller values exaggerate relativistic effects. |
| `MIXING` | double, in `(0, 1]` | `0.4` | Linear density-matrix mixing weight for the `C4_DHF` SCF loop: the density fed into the next iteration is `mixing*P_new + (1-mixing)*P_current`. |
| `MAX_ITERATIONS` | int (> 0) | `100` | Maximum number of `C4_DHF` SCF cycles before giving up (the last cycle's results are still returned, with `converged = false`). |
| `ENERGY_TOLERANCE` | double (> 0) | `1e-8` Hartree | `C4_DHF` SCF energy-change convergence threshold. Combined with `DENSITY_TOLERANCE` by OR: converged as soon as either is satisfied. |
| `DENSITY_TOLERANCE` | double (> 0) | `1e-6` | `C4_DHF` SCF density-change convergence threshold (see `ENERGY_TOLERANCE`). |
| `CACHE_INTEGRALS` | bool | `FALSE` | Cache the two-electron integral tensors to disk and reuse them on a later run with matching geometry+basis (see below). |
| `CACHE_DIR` | string | `.rerdmft_cache` | Directory (created if missing) used by `CACHE_INTEGRALS`. |

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
