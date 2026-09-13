#ifndef RERDMFT_RKBDENSITYMATRIX_H
#define RERDMFT_RKBDENSITYMATRIX_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// Builds the RKB-basis molecular-spinor coefficient matrix
//   C_DHF = X_full * U,
// mapping the orthonormal eigenbasis (U, the eigenvectors that diagonalize
// H_RKB_ortho = X_full^dagger H_RKB X_full -- LinearAlgebra.h's
// diagonalizeHermitian) back to the original, non-orthonormal RKB spinor
// AO basis [Large-alpha, Large-beta, RKB-Small-alpha-partner, RKB-Small-
// beta-partner] (RkbHamiltonian.h). This is the standard relation solving
// the generalized eigenvalue problem H_RKB C = S_full C E (S_full:
// RkbOrthogonalization.h's sFullMatrix) via X_full = S_full^-1/2.
// Column c of C_DHF is the RKB-AO expansion of the c-th eigenstate, with
// the same ascending-energy column order as U/H_RKB_ortho's eigenvalues.
Matrix<std::complex<double>> rkbCoefficientMatrix(const Matrix<std::complex<double>>& x_full,
                                                   const Matrix<std::complex<double>>& u);

// Builds the RKB spinor-AO-basis density matrix
//   P(i,j) = sum_{c occupied} C_DHF(i,c) * conj(C_DHF(j,c))
// occupying the lowest `n_electrons` POSITIVE-energy spinors (each holds
// at most one electron -- spin is already explicit in the 4-component
// spinor basis, unlike a nonrelativistic spatial orbital). With the full
// RKB basis of dimension N = 4*nLarge, eigenvalues come in exactly two
// halves: the first N/2 (ascending-energy indices [0, N/2)) are the
// negative-energy ("no-pair") solutions clustered near -2c^2, and the
// second N/2 ([N/2, N)) are the positive-energy, chemically relevant ones
// (see main.cpp's even/odd Kramers-pair printout, which spans the full
// spectrum, and RkbPositiveEnergyHamiltonian.h, which isolates this upper
// half via an exact Feshbach reduction instead). So the occupied columns
// are [N/2, N/2 + n_electrons) -- e.g. for a 40-dimensional RKB basis
// (N/2 = 20), indices 20, 21, 22, ... . Throws std::runtime_error if
// n_electrons is not a positive number of spinors that fits within the
// positive-energy half.
Matrix<std::complex<double>> rkbDensityMatrix(const Matrix<std::complex<double>>& c_dhf,
                                               int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_RKBDENSITYMATRIX_H
