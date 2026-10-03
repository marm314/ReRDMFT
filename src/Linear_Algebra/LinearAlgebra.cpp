#include "LinearAlgebra.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

// Makes LAPACKE's complex type identical to std::complex<double> (an
// officially supported override documented in lapack.h), so complex
// buffers can be passed directly with no manual reinterpretation.
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

namespace rerdmft {

Matrix<double> invert(const Matrix<double>& a) {
  if (a.rows() != a.cols()) {
    throw std::runtime_error("invert: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(a.rows());

  std::vector<double> data(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (lapack_int i = 0; i < n; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      data[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] =
          a(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    }
  }

  std::vector<lapack_int> ipiv(static_cast<std::size_t>(n));
  lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invert: LAPACKE_dgetrf failed (matrix is singular or invalid)");
  }
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invert: LAPACKE_dgetri failed (matrix is singular)");
  }

  Matrix<double> result(a.rows(), a.cols());
  for (lapack_int i = 0; i < n; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      result(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
          data[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
    }
  }
  return result;
}

SymmetricEigenResult diagonalizeSymmetric(const Matrix<double>& a_in) {
  if (a_in.rows() != a_in.cols()) {
    throw std::runtime_error("diagonalizeSymmetric: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(a_in.rows());
  const std::size_t un = static_cast<std::size_t>(n);

  // LAPACKE_dsyev only reads/writes the requested triangle on input, but we
  // hand it a full symmetric matrix either way.
  std::vector<double> a(un * un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      a[i * un + j] = a_in(i, j);
    }
  }

  std::vector<double> w(un);
  const lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, a.data(), n, w.data());
  if (info != 0) {
    throw std::runtime_error("diagonalizeSymmetric: LAPACKE_dsyev failed to converge");
  }

  SymmetricEigenResult result;
  result.eigenvalues.assign(w.begin(), w.end());
  result.eigenvectors = Matrix<double>(un, un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t k = 0; k < un; ++k) {
      result.eigenvectors(i, k) = a[i * un + k];
    }
  }
  return result;
}

Matrix<double> inverseSqrt(const Matrix<double>& s) {
  // Deliberately NOT diagonally preconditioned like inverseSqrtHermitian's own (see its comment): this is
  // used for the Large-component overlap (main.cpp's x_large), whose diagonal is already ~1 (a normalized
  // AO basis) -- rescaling changes nothing mathematically there, but DOES perturb which representative
  // eigenbasis LAPACK picks within an exactly-degenerate eigenspace (atoms with full p/d/f shells have
  // several), at the ~1e-15 level. That's an equally valid S^-1/2, but it cascades through the nonlinear
  // SCF/FULL_OPTIMIZATION far enough to flip an already-borderline gradient-vs-finite-difference check
  // (Xe/NON_REL/PCCD: 2.0e-6 vs its own 1e-6 threshold) from pass to fail for no actual accuracy gain --
  // confirmed by bisection (fine on the LU/no-cholesky-fix commit, broke exactly when this was added).
  const SymmetricEigenResult eig = diagonalizeSymmetric(s);
  const std::size_t un = eig.eigenvalues.size();

  constexpr double kMinEigenvalue = 1e-10;
  std::vector<double> inv_sqrt_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    if (eig.eigenvalues[i] <= kMinEigenvalue) {
      throw std::runtime_error(
          "inverseSqrt: matrix is not safely positive definite (eigenvalue " +
          std::to_string(eig.eigenvalues[i]) + " <= " + std::to_string(kMinEigenvalue) + ")");
    }
    inv_sqrt_w[i] = 1.0 / std::sqrt(eig.eigenvalues[i]);
  }

  // S^-1/2 = U diag(1/sqrt(w)) U^T: X(i,j) = sum_k U(i,k) (1/sqrt(w_k)) U(j,k).
  const Matrix<double>& u = eig.eigenvectors;
  Matrix<double> x(un, un, 0.0);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < un; ++k) {
        sum += u(i, k) * inv_sqrt_w[k] * u(j, k);
      }
      x(i, j) = sum;
    }
  }
  return x;
}

HermitianEigenResult diagonalizeHermitian(const Matrix<std::complex<double>>& a_in) {
  if (a_in.rows() != a_in.cols()) {
    throw std::runtime_error("diagonalizeHermitian: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(a_in.rows());
  const std::size_t un = static_cast<std::size_t>(n);

  std::vector<std::complex<double>> a(un * un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      a[i * un + j] = a_in(i, j);
    }
  }

  std::vector<double> w(un);
  const lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', n, a.data(), n, w.data());
  if (info != 0) {
    throw std::runtime_error("diagonalizeHermitian: LAPACKE_zheev failed to converge");
  }

  HermitianEigenResult result;
  result.eigenvalues.assign(w.begin(), w.end());
  result.eigenvectors = Matrix<std::complex<double>>(un, un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t k = 0; k < un; ++k) {
      result.eigenvectors(i, k) = a[i * un + k];
    }
  }
  return result;
}

Matrix<std::complex<double>> inverseSqrtHermitian(const Matrix<std::complex<double>>& s, bool precondition) {
  // Diagonal (Jacobi) preconditioning: S'(i,j) = S(i,j) / sqrt(S(i,i) S(j,j)), i.e. S' = D^-1 S D^-1 with
  // D = diag(sqrt(S_ii)) (real: a Hermitian matrix's diagonal is always real). S^-1/2 then satisfies
  // X = D^-1 Y, Y = S'^-1/2, EXACTLY -- not an approximation. Unlike inverseSqrt (the real, non-Hermitian
  // case -- deliberately NOT given this treatment, see its own comment), this one earns its keep: a
  // restricted-kinetic-balance Small-component overlap (RkbOverlap.h) inherits a genuinely huge diagonal
  // spread (self-overlaps differing by many orders of magnitude) from each primitive's raising-term weight
  // -2*a_k*c_k scaling with its own exponent -- a property of RKB-by-differentiation itself (true analytic
  // RKB or the old uKB projection alike; untouched by the large-component spherical/LOWGEN treatment,
  // which acts on angular redundancy and genuine linear dependence, not this scale). `precondition=false`
  // skips this (plain, un-preconditioned S^-1/2) for direct comparison on a hard case.
  const std::size_t un = s.rows();
  std::vector<double> d(un, 1.0);
  Matrix<std::complex<double>> s_scaled = s;
  if (precondition) {
    for (std::size_t i = 0; i < un; ++i) d[i] = std::sqrt(s(i, i).real());
    for (std::size_t i = 0; i < un; ++i)
      for (std::size_t j = 0; j < un; ++j) s_scaled(i, j) = s(i, j) / (d[i] * d[j]);
  }

  const HermitianEigenResult eig = diagonalizeHermitian(s_scaled);

  constexpr double kMinEigenvalue = 1e-10;
  std::vector<double> inv_sqrt_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    if (eig.eigenvalues[i] <= kMinEigenvalue) {
      throw std::runtime_error(
          "inverseSqrtHermitian: matrix is not safely positive definite (" +
          std::string(precondition ? "diagonally-rescaled " : "") + "eigenvalue " +
          std::to_string(eig.eigenvalues[i]) + " <= " + std::to_string(kMinEigenvalue) + ")");
    }
    inv_sqrt_w[i] = 1.0 / std::sqrt(eig.eigenvalues[i]);
  }

  // Y = S'^-1/2 = U diag(1/sqrt(w)) U^dagger, then X = D^-1 Y (row i scaled by 1/d[i] -- the identity
  // when precondition=false, d[i]=1 throughout).
  const Matrix<std::complex<double>>& u = eig.eigenvectors;
  Matrix<std::complex<double>> x(un, un, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < un; ++k) {
        sum += u(i, k) * inv_sqrt_w[k] * std::conj(u(j, k));
      }
      x(i, j) = sum / d[i];
    }
  }
  return x;
}

Matrix<std::complex<double>> invertHermitian(const Matrix<std::complex<double>>& s) {
  const HermitianEigenResult eig = diagonalizeHermitian(s);
  const std::size_t un = eig.eigenvalues.size();

  constexpr double kMinAbsEigenvalue = 1e-10;
  std::vector<double> inv_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    if (std::abs(eig.eigenvalues[i]) <= kMinAbsEigenvalue) {
      throw std::runtime_error(
          "invertHermitian: matrix is too close to singular (|eigenvalue| " +
          std::to_string(std::abs(eig.eigenvalues[i])) +
          " <= " + std::to_string(kMinAbsEigenvalue) + ")");
    }
    inv_w[i] = 1.0 / eig.eigenvalues[i];
  }

  // S^-1 = U diag(1/w) U^dagger: X(i,j) = sum_k U(i,k) (1/w_k) conj(U(j,k)).
  const Matrix<std::complex<double>>& u = eig.eigenvectors;
  Matrix<std::complex<double>> x(un, un, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < un; ++k) {
        sum += u(i, k) * inv_w[k] * std::conj(u(j, k));
      }
      x(i, j) = sum;
    }
  }
  return x;
}

Matrix<std::complex<double>> invertGeneral(const Matrix<std::complex<double>>& a_in) {
  if (a_in.rows() != a_in.cols()) {
    throw std::runtime_error("invertGeneral: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(a_in.rows());
  const std::size_t un = static_cast<std::size_t>(n);

  std::vector<std::complex<double>> data(un * un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      data[i * un + j] = a_in(i, j);
    }
  }

  std::vector<lapack_int> ipiv(un);
  lapack_int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invertGeneral: LAPACKE_zgetrf failed (matrix is singular or invalid)");
  }
  info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invertGeneral: LAPACKE_zgetri failed (matrix is singular)");
  }

  Matrix<std::complex<double>> result(un, un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      result(i, j) = data[i * un + j];
    }
  }
  return result;
}

Matrix<double> canonicalOrthogonalize(const Matrix<double>& s, double threshold, RankReductionReport* report) {
  const SymmetricEigenResult eig = diagonalizeSymmetric(s);
  const std::size_t un = eig.eigenvalues.size();

  std::vector<std::size_t> keep;
  double smallest_kept = 0.0, largest_dropped = 0.0;
  for (std::size_t i = 0; i < un; ++i) {
    if (eig.eigenvalues[i] > threshold) {
      keep.push_back(i);
      if (smallest_kept == 0.0 || eig.eigenvalues[i] < smallest_kept) smallest_kept = eig.eigenvalues[i];
    } else {
      largest_dropped = std::max(largest_dropped, eig.eigenvalues[i]);
    }
  }

  // X(:,j) = U(:,keep[j]) / sqrt(w_keep[j]): an (un x n_eff) matrix, n_eff = keep.size() <= un.
  const std::size_t n_eff = keep.size();
  const Matrix<double>& u = eig.eigenvectors;
  Matrix<double> x(un, n_eff, 0.0);
  for (std::size_t j = 0; j < n_eff; ++j) {
    const double fac = 1.0 / std::sqrt(eig.eigenvalues[keep[j]]);
    for (std::size_t i = 0; i < un; ++i) x(i, j) = u(i, keep[j]) * fac;
  }
  if (report) {
    report->n_kept = n_eff;
    report->n_dropped = un - n_eff;
    report->smallest_kept = smallest_kept;
    report->largest_dropped = largest_dropped;
  }
  return x;
}

Matrix<std::complex<double>> canonicalOrthogonalizeHermitian(const Matrix<std::complex<double>>& s, double threshold,
                                                              RankReductionReport* report) {
  // Same diagonal (Jacobi) preconditioning as inverseSqrtHermitian, and for the same reason: a
  // restricted-kinetic-balance Small-component overlap's diagonal spans many orders of magnitude
  // (differentiating Large AOs across a heavy element's full exponent range), which would otherwise
  // make `threshold` meaningless (an eigenvalue can be tiny purely from scale, not genuine linear
  // dependence). S' = D^-1 S D^-1 (D = diag(sqrt(S_ii))); the kept/dropped decision and `threshold`
  // itself apply to S''s eigenvalues, then X = D^-1 Y (Y = S'^-1/2-style kept/scaled eigenvectors)
  // un-scales back -- exact, not approximate, like inverseSqrtHermitian's own version.
  const std::size_t un = s.rows();
  std::vector<double> d(un);
  for (std::size_t i = 0; i < un; ++i) d[i] = std::sqrt(s(i, i).real());
  Matrix<std::complex<double>> s_scaled(un, un);
  for (std::size_t i = 0; i < un; ++i)
    for (std::size_t j = 0; j < un; ++j) s_scaled(i, j) = s(i, j) / (d[i] * d[j]);

  const HermitianEigenResult eig = diagonalizeHermitian(s_scaled);

  std::vector<std::size_t> keep;
  double smallest_kept = 0.0, largest_dropped = 0.0;
  for (std::size_t i = 0; i < un; ++i) {
    if (eig.eigenvalues[i] > threshold) {
      keep.push_back(i);
      if (smallest_kept == 0.0 || eig.eigenvalues[i] < smallest_kept) smallest_kept = eig.eigenvalues[i];
    } else {
      largest_dropped = std::max(largest_dropped, eig.eigenvalues[i]);
    }
  }

  const std::size_t n_eff = keep.size();
  const Matrix<std::complex<double>>& u = eig.eigenvectors;
  Matrix<std::complex<double>> x(un, n_eff, std::complex<double>(0.0, 0.0));
  for (std::size_t j = 0; j < n_eff; ++j) {
    const double fac = 1.0 / std::sqrt(eig.eigenvalues[keep[j]]);
    for (std::size_t i = 0; i < un; ++i) x(i, j) = u(i, keep[j]) * fac / d[i];
  }
  if (report) {
    report->n_kept = n_eff;
    report->n_dropped = un - n_eff;
    report->smallest_kept = smallest_kept;
    report->largest_dropped = largest_dropped;
  }
  return x;
}

}  // namespace rerdmft
