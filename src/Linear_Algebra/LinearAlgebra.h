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

// Like inverseSqrt, but for a complex Hermitian positive-definite matrix, and with one addition:
// internally diagonally preconditions S (S' = D^-1 S D^-1, D = diag(sqrt(S_ii))) before eigendecomposing
// (LAPACKE_zheev) and un-scales the result -- an exact identity, not an approximation, that removes any
// ill-conditioning coming from the matrix's own basis functions having wildly different self-overlaps
// (e.g. a restricted-kinetic-balance Small-component overlap for a heavy element); a near-identity
// rescaling for an already well-scaled S. inverseSqrt (the real case) deliberately does NOT do this --
// see its own comment. Throws std::runtime_error under the same conditions as inverseSqrt (eigenvalue
// of the diagonally-rescaled matrix at or below 1e-10).
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

// Diagnostics for canonicalOrthogonalize(Hermitian): how many eigendirections were dropped as too
// close to the null space to safely use, and the eigenvalues bracketing that cutoff.
struct RankReductionReport {
  std::size_t n_dropped = 0;
  std::size_t n_kept = 0;
  double smallest_kept = 0.0;    // 0 if n_kept == 0
  double largest_dropped = 0.0;  // 0 if n_dropped == 0
};

// Loewdin's CANONICAL orthonormalization with genuine linear-dependency removal -- DIRAC's own
// approach (Utils/LOWGEN, dirac/dirone.F) to exactly this problem, used for both its large- and
// small-component overlaps. Diagonalizes S = U diag(w) U^T and returns X = U_kept diag(1/sqrt(w_kept)),
// an (n x n_eff) matrix (n_eff = number of eigenvalues > threshold), NOT a square S^-1/2: directions
// with eigenvalue at or below threshold are genuinely DROPPED (X has fewer columns than rows) rather
// than kept-and-inverted (inverseSqrt, which throws instead). X^T S X = I_(n_eff) -- the orthonormal
// basis genuinely shrinks when the input is rank-deficient, exactly as DIRAC's own RKB/large-component
// treatment does (and explicitly accepts, printing a warning rather than failing).
Matrix<double> canonicalOrthogonalize(const Matrix<double>& s, double threshold = 1e-10,
                                       RankReductionReport* report = nullptr);

// Same as canonicalOrthogonalize, but for a complex Hermitian positive-semidefinite matrix -- and,
// like inverseSqrtHermitian (see its own comment), diagonally preconditioned first (S' = D^-1 S D^-1,
// D = diag(sqrt(S_ii))) so `threshold` compares genuine linear dependence, not raw scale, for a
// matrix like a restricted-kinetic-balance Small-component overlap whose diagonal spans many orders
// of magnitude. `threshold` applies to S''s eigenvalues.
Matrix<std::complex<double>> canonicalOrthogonalizeHermitian(const Matrix<std::complex<double>>& s,
                                                               double threshold = 1e-10,
                                                               RankReductionReport* report = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_LINEARALGEBRA_H
