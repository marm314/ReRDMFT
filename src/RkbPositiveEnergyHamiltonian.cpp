#include "RkbPositiveEnergyHamiltonian.h"

#include <cstddef>
#include <stdexcept>

#include "LinearAlgebra.h"

namespace rerdmft {

Matrix<std::complex<double>> rkbPositiveEnergyHamiltonian(
    const Matrix<std::complex<double>>& h_rkb, const Matrix<std::complex<double>>& s_small,
    const Matrix<std::complex<double>>& f_small, double speed_of_light) {
  const std::size_t n2 = s_small.rows();  // 2*nLarge
  if (s_small.cols() != n2 || f_small.rows() != n2 || f_small.cols() != n2) {
    throw std::runtime_error(
        "rkbPositiveEnergyHamiltonian: s_small and f_small must both be square and the same size");
  }
  if (h_rkb.rows() != 2 * n2 || h_rkb.cols() != 2 * n2) {
    throw std::runtime_error(
        "rkbPositiveEnergyHamiltonian: h_rkb dimensions are inconsistent with s_small/f_small");
  }

  // A = H_RKB's Large-Large block, B = H_RKB's Large-Small block.
  Matrix<std::complex<double>> a(n2, n2);
  Matrix<std::complex<double>> b(n2, n2);
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      a(i, j) = h_rkb(i, j);
      b(i, j) = h_rkb(i, n2 + j);
    }
  }

  const double two_c2 = 2.0 * speed_of_light * speed_of_light;

  // inner = S_small - F_small/(2c^2): well-scaled (O(1)), regardless of c.
  Matrix<std::complex<double>> inner(n2, n2);
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      inner(i, j) = s_small(i, j) - f_small(i, j) / two_c2;
    }
  }
  const Matrix<std::complex<double>> inner_inv = invertHermitian(inner);

  Matrix<std::complex<double>> b_dagger(n2, n2);
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      b_dagger(j, i) = std::conj(b(i, j));
    }
  }

  const Matrix<std::complex<double>> correction = b * (inner_inv * b_dagger);

  Matrix<std::complex<double>> h_eff(n2, n2);
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      h_eff(i, j) = a(i, j) + correction(i, j) / two_c2;
    }
  }
  return h_eff;
}

Matrix<std::complex<double>> positiveEnergyOrthoHamiltonian(
    const Matrix<std::complex<double>>& h_eff, const Matrix<double>& x_large) {
  const std::size_t n_large = x_large.rows();
  if (x_large.cols() != n_large || h_eff.rows() != 2 * n_large || h_eff.cols() != 2 * n_large) {
    throw std::runtime_error(
        "positiveEnergyOrthoHamiltonian: h_eff dimensions are inconsistent with 2*x_large's");
  }

  Matrix<std::complex<double>> x(2 * n_large, 2 * n_large, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_large; ++i) {
    for (std::size_t j = 0; j < n_large; ++j) {
      const std::complex<double> value(x_large(i, j), 0.0);
      x(i, j) = value;
      x(n_large + i, n_large + j) = value;
    }
  }

  return dagger(x) * (h_eff * x);
}

}  // namespace rerdmft
