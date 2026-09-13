#ifndef RERDMFT_LINEARALGEBRA_H
#define RERDMFT_LINEARALGEBRA_H

#include <complex>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// Eigenvalues (real, ascending order) and eigenvectors (columns, in the
// same order) of a complex Hermitian matrix.
struct HermitianEigenResult {
  std::vector<double> eigenvalues;
  Matrix<std::complex<double>> eigenvectors;
};

// Same, for a real symmetric matrix.
struct SymmetricEigenResult {
  std::vector<double> eigenvalues;
  Matrix<double> eigenvectors;
};

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

// Diagonalizes a real symmetric matrix via LAPACK (LAPACKE_dsyev). Throws
// std::runtime_error if `a` is not square, or LAPACK fails to converge.
SymmetricEigenResult diagonalizeSymmetric(const Matrix<double>& a);

// Same as inverseSqrt, but for a complex Hermitian positive-definite
// matrix: S = U diag(w) U^dagger (LAPACKE_zheev, w real) and
// S^-1/2 = U diag(1/sqrt(w)) U^dagger. Throws std::runtime_error under the
// same conditions as inverseSqrt.
Matrix<std::complex<double>> inverseSqrtHermitian(const Matrix<std::complex<double>>& s);

// Diagonalizes a complex Hermitian matrix via LAPACK (LAPACKE_zheev).
// Throws std::runtime_error if `a` is not square, or LAPACK fails to
// converge.
HermitianEigenResult diagonalizeHermitian(const Matrix<std::complex<double>>& a);

// Inverts a complex Hermitian matrix via eigendecomposition,
// S^-1 = U diag(1/w) U^dagger (LAPACKE_zheev). Unlike inverseSqrtHermitian,
// this only requires each eigenvalue to be safely nonzero (not positive),
// since a Hermitian matrix being inverted here need not be positive
// definite. Throws std::runtime_error if `s` is not square, or has an
// eigenvalue with |w| at or below 1e-10 (too close to singular).
Matrix<std::complex<double>> invertHermitian(const Matrix<std::complex<double>>& s);

}  // namespace rerdmft

#endif  // RERDMFT_LINEARALGEBRA_H
