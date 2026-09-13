#include "LinearAlgebra.h"

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

Matrix<std::complex<double>> inverseSqrtHermitian(const Matrix<std::complex<double>>& s) {
  const HermitianEigenResult eig = diagonalizeHermitian(s);
  const std::size_t un = eig.eigenvalues.size();

  constexpr double kMinEigenvalue = 1e-10;
  std::vector<double> inv_sqrt_w(un);
  for (std::size_t i = 0; i < un; ++i) {
    if (eig.eigenvalues[i] <= kMinEigenvalue) {
      throw std::runtime_error(
          "inverseSqrtHermitian: matrix is not safely positive definite (eigenvalue " +
          std::to_string(eig.eigenvalues[i]) + " <= " + std::to_string(kMinEigenvalue) + ")");
    }
    inv_sqrt_w[i] = 1.0 / std::sqrt(eig.eigenvalues[i]);
  }

  // S^-1/2 = U diag(1/sqrt(w)) U^dagger: X(i,j) = sum_k U(i,k) (1/sqrt(w_k)) conj(U(j,k)).
  const Matrix<std::complex<double>>& u = eig.eigenvectors;
  Matrix<std::complex<double>> x(un, un, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < un; ++i) {
    for (std::size_t j = 0; j < un; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < un; ++k) {
        sum += u(i, k) * inv_sqrt_w[k] * std::conj(u(j, k));
      }
      x(i, j) = sum;
    }
  }
  return x;
}

}  // namespace rerdmft
