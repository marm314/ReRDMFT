#include "X2C_hamiltonian.h"

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

  const auto lambda_inv_sqrt = inverseSqrtHermitian(result.lambda);
  result.h_x2c_ortho = lambda_inv_sqrt * (result.h_x2c * lambda_inv_sqrt);

  return result;
}

Matrix<std::complex<double>> approximateX2COrtho(const Matrix<std::complex<double>>& h_x2c,
                                                  const Matrix<std::complex<double>>& x_full) {
  const std::size_t n = h_x2c.rows();
  if (h_x2c.cols() != n) {
    throw std::runtime_error("approximateX2COrtho: h_x2c is not square");
  }
  if (x_full.rows() < n || x_full.cols() < n) {
    throw std::runtime_error("approximateX2COrtho: x_full is too small for h_x2c");
  }
  // Only the LARGE-component block of X_full (its own upper-left n x n
  // block, diag(X_Large, X_Large)) -- NOT the exact Lambda^-1/2
  // renormalization h_x2c_ortho itself uses.
  const auto x_large_block = subBlock(x_full, 0, n, 0, n);
  return dagger(x_large_block) * (h_x2c * x_large_block);
}

}  // namespace rerdmft
