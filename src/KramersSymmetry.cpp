#include "KramersSymmetry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "RkbHamiltonian.h"

namespace rerdmft {

namespace {

Matrix<std::complex<double>> toComplex(const Matrix<double>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) {
      result(i, j) = std::complex<double>(m(i, j), 0.0);
    }
  }
  return result;
}

Matrix<std::complex<double>> elementwiseConjugate(const Matrix<std::complex<double>>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) {
      result(i, j) = std::conj(m(i, j));
    }
  }
  return result;
}

// Shared by both maxKramersPartnerDeviation and
// maxKramersPartnerDeviationLarge once `psi` (each column a spinor,
// already in its own original AO representation), `theta`, and `s_full`
// (both sized to that same AO representation) are built.
double maxPartnerDeviationFromPsi(const Matrix<std::complex<double>>& psi,
                                   const Matrix<double>& theta, const Matrix<double>& s_full) {
  const std::size_t n_orig = psi.rows();
  const std::size_t n_final = psi.cols();
  const Matrix<std::complex<double>> theta_psi = toComplex(theta) * elementwiseConjugate(psi);
  const Matrix<std::complex<double>> s_complex = toComplex(s_full);
  const Matrix<std::complex<double>> s_theta_psi = s_complex * theta_psi;
  const Matrix<std::complex<double>> s_psi = s_complex * psi;

  double max_deviation = 0.0;
  for (std::size_t k = 0; k + 1 < n_final; k += 2) {
    std::complex<double> overlap(0.0, 0.0);
    std::complex<double> norm_odd(0.0, 0.0);
    std::complex<double> norm_theta_even(0.0, 0.0);
    for (std::size_t r = 0; r < n_orig; ++r) {
      overlap += std::conj(psi(r, k + 1)) * s_theta_psi(r, k);
      norm_odd += std::conj(psi(r, k + 1)) * s_psi(r, k + 1);
      norm_theta_even += std::conj(theta_psi(r, k)) * s_theta_psi(r, k);
    }
    const double denom = std::sqrt(norm_odd.real() * norm_theta_even.real());
    const double deviation = 1.0 - std::abs(overlap) / denom;
    max_deviation = std::max(max_deviation, std::abs(deviation));
  }
  return max_deviation;
}

}  // namespace

double maxKramersPartnerDeviation(const Matrix<std::complex<double>>& eigenvectors,
                                   const Matrix<std::complex<double>>& rkb_coefficients,
                                   const Matrix<std::complex<double>>& x_full,
                                   const Matrix<double>& s_large,
                                   const Matrix<double>& s_small_ukb) {
  const std::size_t n_large = s_large.rows();
  const std::size_t n_small = s_small_ukb.rows();
  const std::size_t n_orig = 2 * n_large + 2 * n_small;
  const std::size_t n_final = eigenvectors.rows();

  if (s_large.cols() != n_large || s_small_ukb.cols() != n_small ||
      rkb_coefficients.rows() != 2 * n_large || rkb_coefficients.cols() != 2 * n_small ||
      x_full.rows() != n_final || x_full.cols() != n_final || eigenvectors.cols() != n_final) {
    throw std::runtime_error("maxKramersPartnerDeviation: inconsistent input dimensions");
  }

  // V maps an H_RKB_ortho-basis vector back to the original
  // [Large-alpha, Large-beta, uKB-Small-alpha, uKB-Small-beta] spinor-AO
  // representation.
  const Matrix<std::complex<double>> v = rkbEmbeddingMatrix(rkb_coefficients) * x_full;
  const Matrix<std::complex<double>> psi = v * eigenvectors;  // columns are psi_k

  // Theta_matrix: Theta(AO*alpha) = +AO*beta, Theta(AO*beta) = -AO*alpha,
  // applied identically within the Large and (uKB) Small sectors.
  Matrix<double> theta(n_orig, n_orig, 0.0);
  for (std::size_t i = 0; i < n_large; ++i) {
    theta(n_large + i, i) = 1.0;
    theta(i, n_large + i) = -1.0;
  }
  for (std::size_t i = 0; i < n_small; ++i) {
    const std::size_t small_alpha = 2 * n_large + i;
    const std::size_t small_beta = 2 * n_large + n_small + i;
    theta(small_beta, small_alpha) = 1.0;
    theta(small_alpha, small_beta) = -1.0;
  }

  Matrix<double> s_full(n_orig, n_orig, 0.0);
  for (std::size_t i = 0; i < n_large; ++i) {
    for (std::size_t j = 0; j < n_large; ++j) {
      s_full(i, j) = s_large(i, j);
      s_full(n_large + i, n_large + j) = s_large(i, j);
    }
  }
  for (std::size_t i = 0; i < n_small; ++i) {
    for (std::size_t j = 0; j < n_small; ++j) {
      s_full(2 * n_large + i, 2 * n_large + j) = s_small_ukb(i, j);
      s_full(2 * n_large + n_small + i, 2 * n_large + n_small + j) = s_small_ukb(i, j);
    }
  }

  return maxPartnerDeviationFromPsi(psi, theta, s_full);
}

double maxKramersPartnerDeviationLarge(const Matrix<std::complex<double>>& c_matrix,
                                        const Matrix<double>& s_large) {
  const std::size_t n_large = s_large.rows();
  const std::size_t n_orig = 2 * n_large;

  if (s_large.cols() != n_large || c_matrix.rows() != n_orig) {
    throw std::runtime_error("maxKramersPartnerDeviationLarge: inconsistent input dimensions");
  }

  // Theta_matrix: Theta(AO*alpha) = +AO*beta, Theta(AO*beta) = -AO*alpha,
  // applied within the Large sector -- no small component at all here
  // (X2C's own decoupling already eliminated it).
  Matrix<double> theta(n_orig, n_orig, 0.0);
  for (std::size_t i = 0; i < n_large; ++i) {
    theta(n_large + i, i) = 1.0;
    theta(i, n_large + i) = -1.0;
  }

  Matrix<double> s_full(n_orig, n_orig, 0.0);
  for (std::size_t i = 0; i < n_large; ++i) {
    for (std::size_t j = 0; j < n_large; ++j) {
      s_full(i, j) = s_large(i, j);
      s_full(n_large + i, n_large + j) = s_large(i, j);
    }
  }

  // c_matrix's columns are already expressed in the original
  // [Large-alpha, Large-beta] AO representation (c_matrix = X_Large * U),
  // so no embedding step is needed -- unlike maxKramersPartnerDeviation,
  // which must first map H_RKB_ortho's own eigenvectors back through the
  // Large+Small RKB basis.
  return maxPartnerDeviationFromPsi(c_matrix, theta, s_full);
}

}  // namespace rerdmft
