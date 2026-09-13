#ifndef RERDMFT_RKBORTHOGONALIZATION_H
#define RERDMFT_RKBORTHOGONALIZATION_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// Builds the full Loewdin orthonormalization matrix for the RKB core
// Hamiltonian's basis [Large-alpha, Large-beta, RKB-small-alpha,
// RKB-small-beta] (matching rkbHamiltonianMatrix's row/column ordering),
// by combining:
//   - x_large = X_Large = S_Large^-1/2 (LinearAlgebra.h), the (nLarge x
//     nLarge) real Loewdin matrix for the scalar Large AOs, repeated
//     (spin-block-diagonally) for the alpha and beta Large spin-orbitals, and
//   - x_small = X_small = S_small^-1/2 (LinearAlgebra.h, applied to
//     RkbOverlap.h's S_small), the (2*nLarge x 2*nLarge) complex Loewdin
//     matrix already spanning both RKB-small spin blocks.
//
// The result is block-diagonal (no Large-Small coupling) and
// (4*nLarge x 4*nLarge): X_full = diag(X_Large, X_Large, X_small).
// Throws std::runtime_error if x_small's dimensions don't match
// 2*x_large's.
Matrix<std::complex<double>> xFullMatrix(const Matrix<double>& x_large,
                                          const Matrix<std::complex<double>>& x_small);

// Builds the orthonormalized restricted-kinetic-balance core Hamiltonian
//   H_RKB_ortho = X_full^dagger H_RKB X_full
// (see xFullMatrix for X_full, RkbHamiltonian.h for H_RKB). Throws
// std::runtime_error if h_rkb and x_full have inconsistent dimensions.
Matrix<std::complex<double>> hRkbOrthoMatrix(const Matrix<std::complex<double>>& h_rkb,
                                              const Matrix<std::complex<double>>& x_full);

}  // namespace rerdmft

#endif  // RERDMFT_RKBORTHOGONALIZATION_H
