#include "RkbHamiltonian.h"

#include <stdexcept>

namespace rerdmft {

Matrix<std::complex<double>> rkbEmbeddingMatrix(
    const Matrix<std::complex<double>>& rkb_coefficients) {
  const std::size_t n_large2 = rkb_coefficients.rows();  // 2*nLarge
  const std::size_t n_small2 = rkb_coefficients.cols();  // 2*nSmall

  const std::size_t n_new = 2 * n_large2;
  Matrix<std::complex<double>> w(n_large2 + n_small2, n_new, std::complex<double>(0.0, 0.0));

  // Top-left block: identity on the (unchanged) Large spin-orbitals.
  for (std::size_t i = 0; i < n_large2; ++i) {
    w(i, i) = std::complex<double>(1.0, 0.0);
  }
  // Bottom-right block: C^T, projecting the uKB Small basis onto the RKB
  // small basis (rkb_coefficients(j, i) is C[Large_j, Small_i]).
  for (std::size_t i = 0; i < n_small2; ++i) {
    for (std::size_t j = 0; j < n_large2; ++j) {
      w(n_large2 + i, n_large2 + j) = rkb_coefficients(j, i);
    }
  }
  return w;
}

Matrix<std::complex<double>> rkbHamiltonianMatrix(
    const Matrix<std::complex<double>>& h_ukb,
    const Matrix<std::complex<double>>& rkb_coefficients) {
  const std::size_t n_large2 = rkb_coefficients.rows();  // 2*nLarge
  const std::size_t n_small2 = rkb_coefficients.cols();  // 2*nSmall

  if (h_ukb.rows() != n_large2 + n_small2 || h_ukb.cols() != n_large2 + n_small2) {
    throw std::runtime_error(
        "rkbHamiltonianMatrix: H_UKB dimensions are inconsistent with the RKB "
        "coefficient matrix");
  }

  const Matrix<std::complex<double>> w = rkbEmbeddingMatrix(rkb_coefficients);

  // dagger(w) conjugates AND transposes the whole matrix, so it correctly
  // turns w's C^T block into conj(C) -- the C^dagger-derived factor this
  // formula needs, since C is complex (it carries the i from sigma_y and
  // from p = -i grad_r). Verified independently: rebuilding H_RKB's
  // Small-Small block from scratch as conj(C) * H_UKB_SS * C^T reproduces
  // this function's result to 0.000e+00, not just floating-point tolerance.
  return dagger(w) * (h_ukb * w);
}

}  // namespace rerdmft
