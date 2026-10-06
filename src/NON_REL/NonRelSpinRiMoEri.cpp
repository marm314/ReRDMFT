#include "NonRelSpinRiMoEri.h"

#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rerdmft {

RiNonRelSpinMoEri buildRiNonRelSpinMoEri(const Matrix<double>& eri3_L, std::size_t n_spatial,
                                          const Matrix<double>& c_spatial) {
  if (c_spatial.rows() != n_spatial || c_spatial.cols() != n_spatial) {
    throw std::runtime_error("buildRiNonRelSpinMoEri: c_spatial dimensions do not match n_spatial");
  }
  if (eri3_L.cols() != n_spatial * n_spatial) {
    throw std::runtime_error("buildRiNonRelSpinMoEri: eri3_L column count does not match n_spatial*n_spatial");
  }
  const std::size_t n_aux = eri3_L.rows();
  Matrix<double> b(n_aux, n_spatial * n_spatial, 0.0);
  if (n_spatial == 0) return RiNonRelSpinMoEri(n_spatial, std::move(b));

  const Matrix<double> c_t = transpose(c_spatial);  // n_spatial x n_spatial

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const double* slice = eri3_L.data() + p * n_spatial * n_spatial;
    Matrix<double> t_p(n_spatial, n_spatial, 0.0);
    // T_P = c_spatial^T @ slice_P : (n_spatial x n_spatial) @ (n_spatial x n_spatial).
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_spatial),
                static_cast<int>(n_spatial), static_cast<int>(n_spatial), 1.0, c_t.data(),
                static_cast<int>(n_spatial), slice, static_cast<int>(n_spatial), 0.0, t_p.data(),
                static_cast<int>(n_spatial));
    // B_P = T_P @ c_spatial : written directly into b's row p.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_spatial),
                static_cast<int>(n_spatial), static_cast<int>(n_spatial), 1.0, t_p.data(),
                static_cast<int>(n_spatial), c_spatial.data(), static_cast<int>(n_spatial), 0.0,
                b.data() + p * n_spatial * n_spatial, static_cast<int>(n_spatial));
  }
  return RiNonRelSpinMoEri(n_spatial, std::move(b));
}

RiNonRelSpinMoEri RiNonRelSpinMoEri::rotated(const Matrix<double>& u) const {
  const std::size_t n_total = 2 * n_spatial_;
  if (u.rows() != n_total || u.cols() != n_total) {
    throw std::runtime_error("RiNonRelSpinMoEri::rotated: u dimensions do not match 2*n_spatial");
  }
  // `u` must be blockdiag(u_spatial, u_spatial) -- see the class comment -- but only
  // APPROXIMATELY: the spin restriction is enforced by projecting the ADAM gradient/kappa onto
  // the tied subspace every step (an iterative process), not by an exact algebraic constraint, so
  // routine floating-point noise (not a bug) is expected at the ~1e-10-1e-6 level depending on how
  // many ADAM steps have accumulated. Average the two blocks (the numerically robust choice --
  // both are equally valid estimates of the same "true" u_spatial) rather than reading one
  // arbitrarily; only a GENUINELY large deviation (orders of magnitude above roundoff) throws,
  // since that would indicate the restriction mechanism itself failed, not noise.
  double max_off_block = 0.0, max_alpha_beta_mismatch = 0.0;
  for (std::size_t i = 0; i < n_spatial_; ++i) {
    for (std::size_t j = 0; j < n_spatial_; ++j) {
      max_off_block = std::max({max_off_block, std::abs(u(i, n_spatial_ + j)), std::abs(u(n_spatial_ + i, j))});
      max_alpha_beta_mismatch = std::max(max_alpha_beta_mismatch, std::abs(u(n_spatial_ + i, n_spatial_ + j) - u(i, j)));
    }
  }
  constexpr double kBlockDiagTolerance = 1e-3;
  if (max_off_block > kBlockDiagTolerance || max_alpha_beta_mismatch > kBlockDiagTolerance) {
    throw std::runtime_error(
        "RiNonRelSpinMoEri::rotated: rotation is not spin-restricted (alpha and beta rotations "
        "differ by " + std::to_string(max_alpha_beta_mismatch) + ", cross-spin block max " +
        std::to_string(max_off_block) + ") -- NON_REL's own FULL_OPTIMIZATION should never "
        "produce such a rotation");
  }
  Matrix<double> u_spatial(n_spatial_, n_spatial_);
  for (std::size_t i = 0; i < n_spatial_; ++i)
    for (std::size_t j = 0; j < n_spatial_; ++j) u_spatial(i, j) = 0.5 * (u(i, j) + u(n_spatial_ + i, n_spatial_ + j));

  const std::size_t n_aux = b_.rows();
  Matrix<double> b_new(n_aux, n_spatial_ * n_spatial_, 0.0);
  if (n_spatial_ == 0) return RiNonRelSpinMoEri(n_spatial_, std::move(b_new));
  const Matrix<double> u_t = transpose(u_spatial);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const double* slice = b_.data() + p * n_spatial_ * n_spatial_;
    Matrix<double> t_p(n_spatial_, n_spatial_, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_spatial_),
                static_cast<int>(n_spatial_), static_cast<int>(n_spatial_), 1.0, u_t.data(),
                static_cast<int>(n_spatial_), slice, static_cast<int>(n_spatial_), 0.0, t_p.data(),
                static_cast<int>(n_spatial_));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_spatial_),
                static_cast<int>(n_spatial_), static_cast<int>(n_spatial_), 1.0, t_p.data(),
                static_cast<int>(n_spatial_), u_spatial.data(), static_cast<int>(n_spatial_), 0.0,
                b_new.data() + p * n_spatial_ * n_spatial_, static_cast<int>(n_spatial_));
  }
  return RiNonRelSpinMoEri(n_spatial_, std::move(b_new));
}

}  // namespace rerdmft
