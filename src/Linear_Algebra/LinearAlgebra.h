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
// symmetric positive-definite matrix. Internally diagonally preconditions
// S (S' = D^-1 S D^-1, D = diag(sqrt(S_ii))) before eigendecomposing
// (LAPACKE_dsyev) and un-scales the result -- an exact identity, not an
// approximation, that removes any ill-conditioning coming from the
// matrix's own basis functions having wildly different self-overlaps
// (e.g. a restricted-kinetic-balance Small-component overlap for a heavy
// element) without changing the result for an already well-scaled S.
// Throws std::runtime_error if `s` is not square, or the (diagonally
// rescaled) matrix has an eigenvalue at or below 1e-10 (not positive
// definite, or too close to genuinely linearly dependent to safely
// invert its square root).
Matrix<double> inverseSqrt(const Matrix<double>& s);

// Diagonalizes a real symmetric matrix via LAPACK (LAPACKE_dsyev). Throws
// std::runtime_error if `a` is not square, or LAPACK fails to converge.
SymmetricEigenResult diagonalizeSymmetric(const Matrix<double>& a);

// Same as inverseSqrt (including the diagonal preconditioning), but for a
// complex Hermitian positive-definite matrix (LAPACKE_zheev). Throws
// std::runtime_error under the same conditions as inverseSqrt.
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

// Inverts a square, nonsingular GENERAL complex matrix via LAPACK (LU
// factorization with LAPACKE_zgetrf, then LAPACKE_zgetri) -- unlike
// invertHermitian, `a` need NOT be Hermitian (e.g. X2C_DHF's own
// large-component coefficient block C_L, an ordinary matrix of
// eigenvector components, not a physical operator). Throws
// std::runtime_error if `a` is singular (or not square).
Matrix<std::complex<double>> invertGeneral(const Matrix<std::complex<double>>& a);

// Diagnostics for pseudoInverseSymmetric: how many eigendirections were dropped as too close to
// the null space to safely invert (rather than throwing, as invert/invertHermitian do), and the
// eigenvalues bracketing that cutoff.
struct PseudoInverseReport {
  std::size_t n_dropped = 0;
  std::size_t n_kept = 0;
  double smallest_kept = 0.0;    // 0 if n_kept == 0
  double largest_dropped = 0.0;  // 0 if n_dropped == 0
};

// Moore-Penrose-style pseudo-inverse of a real symmetric matrix via eigendecomposition,
// S^+ = U diag(f(w)) U^T with f(w) = 1/w for w > threshold, f(w) = 0 otherwise -- unlike invert()
// (plain LU, no conditioning check at all) or inverseSqrt/invertHermitian (a hard floor that
// THROWS), this is for a matrix expected to be genuinely, legitimately rank-deficient at that
// threshold (e.g. a small-component AO overlap with near-linearly-dependent combinations from an
// uncontracted heavy-element basis -- see RkbTransformation.cpp): the near-null eigendirections
// are information nothing downstream can represent anyway, so they are dropped (treated as exactly
// unrepresentable) rather than amplified into numerical noise by 1/w blowing up a direction that
// was never reliably nonzero in the first place. Dimension-preserving: the result is always the
// same n x n shape, never a lower-rank factorization. Pass `report` to see what was dropped.
Matrix<double> pseudoInverseSymmetric(const Matrix<double>& s, double threshold = 1e-10,
                                       PseudoInverseReport* report = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_LINEARALGEBRA_H
