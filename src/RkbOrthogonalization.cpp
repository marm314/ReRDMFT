#include "RkbOrthogonalization.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

Matrix<std::complex<double>> xFullMatrix(const Matrix<double>& x_large,
                                          const Matrix<std::complex<double>>& x_small) {
  const std::size_t n_large = x_large.rows();
  if (x_large.cols() != n_large) {
    throw std::runtime_error("xFullMatrix: x_large is not square");
  }
  const std::size_t n_large2 = 2 * n_large;
  if (x_small.rows() != n_large2 || x_small.cols() != n_large2) {
    throw std::runtime_error(
        "xFullMatrix: x_small's dimensions are inconsistent with 2*x_large's");
  }

  const std::size_t n = n_large2 + n_large2;  // = 4*n_large
  Matrix<std::complex<double>> x_full(n, n, std::complex<double>(0.0, 0.0));

  // Top-left block: diag(X_Large, X_Large), spanning the Large-alpha and
  // Large-beta spin-orbitals.
  for (std::size_t i = 0; i < n_large; ++i) {
    for (std::size_t j = 0; j < n_large; ++j) {
      const std::complex<double> value(x_large(i, j), 0.0);
      x_full(i, j) = value;
      x_full(n_large + i, n_large + j) = value;
    }
  }
  // Bottom-right block: X_small, already spanning both RKB-small spin blocks.
  for (std::size_t i = 0; i < n_large2; ++i) {
    for (std::size_t j = 0; j < n_large2; ++j) {
      x_full(n_large2 + i, n_large2 + j) = x_small(i, j);
    }
  }

  return x_full;
}

Matrix<std::complex<double>> sFullMatrix(const Matrix<double>& s_large,
                                          const Matrix<std::complex<double>>& s_small) {
  const std::size_t n_large = s_large.rows();
  if (s_large.cols() != n_large) {
    throw std::runtime_error("sFullMatrix: s_large is not square");
  }
  const std::size_t n_large2 = 2 * n_large;
  if (s_small.rows() != n_large2 || s_small.cols() != n_large2) {
    throw std::runtime_error(
        "sFullMatrix: s_small's dimensions are inconsistent with 2*s_large's");
  }

  const std::size_t n = n_large2 + n_large2;  // = 4*n_large
  Matrix<std::complex<double>> s_full(n, n, std::complex<double>(0.0, 0.0));

  for (std::size_t i = 0; i < n_large; ++i) {
    for (std::size_t j = 0; j < n_large; ++j) {
      const std::complex<double> value(s_large(i, j), 0.0);
      s_full(i, j) = value;
      s_full(n_large + i, n_large + j) = value;
    }
  }
  for (std::size_t i = 0; i < n_large2; ++i) {
    for (std::size_t j = 0; j < n_large2; ++j) {
      s_full(n_large2 + i, n_large2 + j) = s_small(i, j);
    }
  }

  return s_full;
}

Matrix<std::complex<double>> hRkbOrthoMatrix(const Matrix<std::complex<double>>& h_rkb,
                                              const Matrix<std::complex<double>>& x_full) {
  if (h_rkb.rows() != x_full.rows() || h_rkb.cols() != x_full.cols()) {
    throw std::runtime_error("hRkbOrthoMatrix: H_RKB and X_full dimensions are inconsistent");
  }
  return dagger(x_full) * (h_rkb * x_full);
}

}  // namespace rerdmft
