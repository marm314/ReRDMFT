#include "X2C_hamiltonian.h"

#include <cmath>
#include <stdexcept>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

Matrix<std::complex<double>> subBlock(const Matrix<std::complex<double>>& m, std::size_t row0,
                                       std::size_t rows, std::size_t col0, std::size_t cols) {
  Matrix<std::complex<double>> result(rows, cols);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      result(i, j) = m(row0 + i, col0 + j);
    }
  }
  return result;
}

}  // namespace

X2CHamiltonianResult buildX2CHamiltonian(const Matrix<std::complex<double>>& h_rkb,
                                          const Matrix<std::complex<double>>& s_full,
                                          const Matrix<std::complex<double>>& c_tmp) {
  const std::size_t n = h_rkb.rows();
  if (h_rkb.cols() != n) {
    throw std::runtime_error("buildX2CHamiltonian: h_rkb is not square");
  }
  if (s_full.rows() != n || s_full.cols() != n) {
    throw std::runtime_error("buildX2CHamiltonian: s_full dimensions inconsistent with h_rkb");
  }
  if (c_tmp.rows() != n || c_tmp.cols() != n) {
    throw std::runtime_error("buildX2CHamiltonian: c_tmp dimensions inconsistent with h_rkb");
  }
  if (n % 2 != 0) {
    throw std::runtime_error(
        "buildX2CHamiltonian: h_rkb dimension must be even (Large/Small halves)");
  }
  const std::size_t n2 = n / 2;

  const auto h_ll = subBlock(h_rkb, 0, n2, 0, n2);
  const auto h_ls = subBlock(h_rkb, 0, n2, n2, n2);
  const auto h_ss = subBlock(h_rkb, n2, n2, n2, n2);
  const auto s_ll = subBlock(s_full, 0, n2, 0, n2);
  const auto s_ss = subBlock(s_full, n2, n2, n2, n2);

  // Positive-energy columns: [n2, n) (c_tmp's columns are in ascending
  // energy order, same as X2C_decoupling.h's own eigenvalues).
  const auto c_l = subBlock(c_tmp, 0, n2, n2, n2);
  const auto c_s = subBlock(c_tmp, n2, n2, n2, n2);

  X2CHamiltonianResult result;
  result.r_matrix = c_s * invertGeneral(c_l);
  const auto& r = result.r_matrix;
  const auto r_dagger = dagger(r);

  result.lambda = s_ll + r_dagger * (s_ss * r);

  const auto h_ls_r = h_ls * r;
  result.h_x2c = h_ll + h_ls_r + dagger(h_ls_r) + r_dagger * (h_ss * r);

  // X = inverseSqrtHermitian(Lambda) is diagonally preconditioned by default (X = D^-1 Y), so it
  // satisfies X^dagger Lambda X = I but is NOT Hermitian: the orthonormalized Hamiltonian must be
  // X^dagger h X, not X h X (which was off by ~1e-3 Ha for Ne/cc-pVDZ).
  const auto lambda_inv_sqrt = inverseSqrtHermitian(result.lambda);
  result.h_x2c_ortho = dagger(lambda_inv_sqrt) * (result.h_x2c * lambda_inv_sqrt);

  return result;
}

namespace {

// a^p for a Hermitian positive matrix, eigenvalues <= cutoff dropped.
Matrix<std::complex<double>> hermitianPower(const Matrix<std::complex<double>>& a, double p, double cutoff) {
  const auto eig = diagonalizeHermitian(a);
  const std::size_t n = a.rows();
  Matrix<std::complex<double>> scaled(n, n);
  for (std::size_t j = 0; j < n; ++j) {
    const double w = eig.eigenvalues[j];
    const double f = w > cutoff ? std::pow(w, p) : 0.0;
    for (std::size_t i = 0; i < n; ++i) scaled(i, j) = eig.eigenvectors(i, j) * f;
  }
  return scaled * dagger(eig.eigenvectors);
}

}  // namespace

Matrix<std::complex<double>> renormalizedX2CHamiltonian(const Matrix<std::complex<double>>& h_x2c,
                                                         const Matrix<std::complex<double>>& lambda,
                                                         const Matrix<std::complex<double>>& s_ll) {
  const std::size_t n = h_x2c.rows();
  if (h_x2c.cols() != n || lambda.rows() != n || lambda.cols() != n || s_ll.rows() != n || s_ll.cols() != n) {
    throw std::runtime_error("renormalizedX2CHamiltonian: inconsistent dimensions");
  }
  const double cutoff = 1e-14;
  const auto s_half = hermitianPower(s_ll, 0.5, cutoff);
  const auto s_inv_half = hermitianPower(s_ll, -0.5, cutoff);
  const auto mid = hermitianPower(s_inv_half * (lambda * s_inv_half), -0.5, cutoff);
  const auto r = s_inv_half * (mid * s_half);
  return dagger(r) * (h_x2c * r);
}

Matrix<std::complex<double>> approximateX2COrtho(const Matrix<std::complex<double>>& h_x2c,
                                                  const Matrix<std::complex<double>>& x_full) {
  const std::size_t n = h_x2c.rows();
  if (h_x2c.cols() != n) {
    throw std::runtime_error("approximateX2COrtho: h_x2c is not square");
  }
  // Only the LARGE-component block of X_full -- NOT the exact
  // Lambda^-1/2 renormalization h_x2c_ortho itself uses.
  const auto x_large_block = extractLargeComponentBlock(x_full, n);
  return dagger(x_large_block) * (h_x2c * x_large_block);
}

Matrix<std::complex<double>> extractLargeComponentBlock(const Matrix<std::complex<double>>& x_full,
                                                         std::size_t n) {
  if (x_full.rows() < n || x_full.cols() < n) {
    throw std::runtime_error("extractLargeComponentBlock: x_full is too small for n");
  }
  return subBlock(x_full, 0, n, 0, n);
}

}  // namespace rerdmft
