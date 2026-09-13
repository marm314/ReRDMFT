#ifndef RERDMFT_LINEARALGEBRA_H
#define RERDMFT_LINEARALGEBRA_H

#include "Matrix.h"

namespace rerdmft {

// Inverts a square, nonsingular real matrix via LAPACK (LU factorization
// with LAPACKE_dgetrf, then LAPACKE_dgetri). Throws std::runtime_error if
// `a` is singular (or not square).
Matrix<double> invert(const Matrix<double>& a);

// Computes S^-1/2 (the symmetric/Loewdin inverse square root) of a
// symmetric positive-definite matrix, via eigendecomposition S = U diag(w) U^T
// (LAPACKE_dsyev) and S^-1/2 = U diag(1/sqrt(w)) U^T. Throws
// std::runtime_error if `s` is not square, or has an eigenvalue at or
// below 1e-10 (not positive definite, or too close to linearly dependent
// to safely invert its square root).
Matrix<double> inverseSqrt(const Matrix<double>& s);

}  // namespace rerdmft

#endif  // RERDMFT_LINEARALGEBRA_H
