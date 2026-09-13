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
