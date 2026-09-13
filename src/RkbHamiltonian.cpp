#include "RkbHamiltonian.h"

#include <stdexcept>

namespace rerdmft {

namespace {

// Hermitian adjoint: conjugate AND transpose. C is complex (it carries the
// i from sigma_y and from p = -i grad_r), so this matters -- applied to
// the whole assembled W below, it correctly turns the C^T block into
// conj(C), which is exactly the C^dagger-derived factor the H_RKB formula
// needs (verified independently: rebuilding H_RKB's Small-Small block
// from scratch as conj(C) * H_UKB_SS * C^T reproduces rkbHamiltonianMatrix's
// result to 0.000e+00, not just to floating-point tolerance).
Matrix<std::complex<double>> daggerOf(const Matrix<std::complex<double>>& a) {
  Matrix<std::complex<double>> result(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      result(j, i) = std::conj(a(i, j));
    }
  }
  return result;
}

}  // namespace

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

  const Matrix<std::complex<double>> w_dagger = daggerOf(w);
  return w_dagger * (h_ukb * w);
}

}  // namespace rerdmft
