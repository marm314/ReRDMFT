#include "LinearAlgebra.h"

#include <cmath>
#include <stdexcept>
#include <vector>

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

Matrix<double> inverseSqrt(const Matrix<double>& s) {
  if (s.rows() != s.cols()) {
    throw std::runtime_error("inverseSqrt: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(s.rows());
  const std::size_t un = static_cast<std::size_t>(n);

  // LAPACKE_dsyev only reads/writes the requested triangle on input, but we
  // hand it a full symmetric matrix either way.
  std::vector<double> a(un * un);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      a[i * un + j] = s(i, j);
    }
  }

  std::vector<double> w(un);
  const lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, a.data(), n, w.data());
  if (info != 0) {
    throw std::runtime_error("inverseSqrt: LAPACKE_dsyev failed to converge");
  }

  constexpr double kMinEigenvalue = 1e-10;
  std::vector<double> inv_sqrt_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    if (w[i] <= kMinEigenvalue) {
      throw std::runtime_error(
          "inverseSqrt: matrix is not safely positive definite (eigenvalue " +
          std::to_string(w[i]) + " <= " + std::to_string(kMinEigenvalue) + ")");
    }
    inv_sqrt_w[i] = 1.0 / std::sqrt(w[i]);
  }

  // a(i,k) is now the i-th component of the k-th eigenvector (LAPACK's 'V'
  // job overwrites the input with eigenvectors as columns). Build
  // S^-1/2 = U diag(1/sqrt(w)) U^T directly: X(i,j) = sum_k U(i,k) (1/sqrt(w_k)) U(j,k).
  Matrix<double> x(un, un, 0.0);
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < un; ++k) {
        sum += a[i * un + k] * inv_sqrt_w[k] * a[j * un + k];
      }
      x(i, j) = sum;
    }
  }
  return x;
}

}  // namespace rerdmft
