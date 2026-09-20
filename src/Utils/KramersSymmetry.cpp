#include "KramersSymmetry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

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

// Per-pair phase of <psi_odd|Theta psi_even>_S -- shared by fixKramersPhase
// and fixKramersPhaseLarge. Theta(psi_even) is guaranteed (Kramers'
// theorem + the diagonalization's own orthogonality within the
// degenerate pair) to be a PURE PHASE multiple of psi_odd already, so
// this overlap's phase alone (its magnitude is ~1, given normalized
// input) is exactly the correction each pair's odd column needs.
std::vector<std::complex<double>> perPairOverlapPhase(const Matrix<std::complex<double>>& psi,
                                                        const Matrix<double>& theta,
                                                        const Matrix<double>& s_full) {
  const std::size_t n_orig = psi.rows();
  const std::size_t n_final = psi.cols();
  const Matrix<std::complex<double>> theta_psi = toComplex(theta) * elementwiseConjugate(psi);
  const Matrix<std::complex<double>> s_theta_psi = toComplex(s_full) * theta_psi;

  std::vector<std::complex<double>> phases(n_final / 2, std::complex<double>(1.0, 0.0));
  for (std::size_t k = 0; k + 1 < n_final; k += 2) {
    std::complex<double> overlap(0.0, 0.0);
    for (std::size_t r = 0; r < n_orig; ++r) {
      overlap += std::conj(psi(r, k + 1)) * s_theta_psi(r, k);
    }
    const double mag = std::abs(overlap);
    if (mag > 1e-12) phases[k / 2] = overlap / mag;
  }
  return phases;
}

// Multiplies each ODD column (2k+1) of `m` by `phases[k]`, leaving EVEN
// columns untouched.
Matrix<std::complex<double>> applyOddColumnPhases(const Matrix<std::complex<double>>& m,
                                                    const std::vector<std::complex<double>>& phases) {
  Matrix<std::complex<double>> result = m;
  const std::size_t n_final = m.cols();
  for (std::size_t k = 0; k + 1 < n_final; k += 2) {
    const std::complex<double> phase = phases[k / 2];
    for (std::size_t r = 0; r < m.rows(); ++r) {
      result(r, k + 1) = m(r, k + 1) * phase;
    }
  }
  return result;
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

Matrix<std::complex<double>> fixKramersPhase(const Matrix<std::complex<double>>& eigenvectors,
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
    throw std::runtime_error("fixKramersPhase: inconsistent input dimensions");
  }

  const Matrix<std::complex<double>> v = rkbEmbeddingMatrix(rkb_coefficients) * x_full;
  const Matrix<std::complex<double>> psi = v * eigenvectors;

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

  // Phases are computed from `psi` (the original AO representation,
  // where Theta acts simply) but applied to `eigenvectors` itself: since
  // psi = v * eigenvectors and v is linear, multiplying an eigenvector's
  // own column by a phase multiplies its psi column by the same phase --
  // so this reproduces the canonical psi_odd = Theta(psi_even) relation
  // exactly, while also correctly phase-fixing c_dhf = rkbCoefficientMatrix(
  // x_full, eigenvectors) downstream, since that construction is linear too.
  const auto phases = perPairOverlapPhase(psi, theta, s_full);
  return applyOddColumnPhases(eigenvectors, phases);
}

Matrix<std::complex<double>> fixKramersPairing(const Matrix<std::complex<double>>& eigenvectors,
                                                const std::vector<double>& energies,
                                                const Matrix<std::complex<double>>& rkb_coefficients,
                                                const Matrix<std::complex<double>>& x_full,
                                                const Matrix<double>& s_large,
                                                const Matrix<double>& s_small_ukb,
                                                double cluster_tolerance,
                                                KramersPairingReport* report) {
  using C = std::complex<double>;
  const std::size_t n_large = s_large.rows();
  const std::size_t n_small = s_small_ukb.rows();
  const std::size_t n_orig = 2 * n_large + 2 * n_small;
  const std::size_t n_final = eigenvectors.rows();
  if (s_large.cols() != n_large || s_small_ukb.cols() != n_small ||
      rkb_coefficients.rows() != 2 * n_large || rkb_coefficients.cols() != 2 * n_small ||
      x_full.rows() != n_final || x_full.cols() != n_final || eigenvectors.cols() != n_final ||
      energies.size() != n_final) {
    throw std::runtime_error("fixKramersPairing: inconsistent input dimensions");
  }

  // Original-representation Theta and metric, exactly as fixKramersPhase builds them.
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
  for (std::size_t i = 0; i < n_large; ++i)
    for (std::size_t j = 0; j < n_large; ++j) {
      s_full(i, j) = s_large(i, j);
      s_full(n_large + i, n_large + j) = s_large(i, j);
    }
  for (std::size_t i = 0; i < n_small; ++i)
    for (std::size_t j = 0; j < n_small; ++j) {
      s_full(2 * n_large + i, 2 * n_large + j) = s_small_ukb(i, j);
      s_full(2 * n_large + n_small + i, 2 * n_large + n_small + j) = s_small_ukb(i, j);
    }

  // V maps orthonormal-basis coefficients to the original representation. In that
  // coefficient space Theta acts as c -> T conj(c), T = V^dagger S Theta conj(V).
  const Matrix<C> v = rkbEmbeddingMatrix(rkb_coefficients) * x_full;
  const Matrix<C> s_v = toComplex(s_full) * v;
  const Matrix<C> theta_v = toComplex(s_full) * (toComplex(theta) * elementwiseConjugate(v));
  Matrix<C> gram(n_final, n_final, C{});
  Matrix<C> t(n_final, n_final, C{});
  for (std::size_t a = 0; a < n_final; ++a) {
    for (std::size_t b = 0; b < n_final; ++b) {
      C g{}, tt{};
      for (std::size_t r = 0; r < n_orig; ++r) {
        g += std::conj(v(r, a)) * s_v(r, b);
        tt += std::conj(v(r, a)) * theta_v(r, b);
      }
      gram(a, b) = g;
      t(a, b) = tt;
    }
  }
  double gram_error = 0.0;
  for (std::size_t a = 0; a < n_final; ++a)
    for (std::size_t b = 0; b < n_final; ++b)
      gram_error = std::max(gram_error, std::abs(gram(a, b) - (a == b ? 1.0 : 0.0)));
  if (gram_error > 1e-6) {
    throw std::runtime_error(
        "fixKramersPairing: the orthonormal RKB basis is not orthonormal in the original "
        "spinor-AO metric (max |V^dagger S V - 1| = " + std::to_string(gram_error) + ")");
  }
  return fixKramersPairingOrthonormal(eigenvectors, energies, t, cluster_tolerance, report);
}

Matrix<std::complex<double>> fixKramersPhaseLarge(const Matrix<std::complex<double>>& c_matrix,
                                                   const Matrix<double>& s_large) {
  const std::size_t n_large = s_large.rows();
  const std::size_t n_orig = 2 * n_large;

  if (s_large.cols() != n_large || c_matrix.rows() != n_orig) {
    throw std::runtime_error("fixKramersPhaseLarge: inconsistent input dimensions");
  }

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

  // c_matrix's columns are already in the original [Large-alpha,
  // Large-beta] AO representation, so no embedding is needed -- the
  // phase can be applied directly to c_matrix itself.
  const auto phases = perPairOverlapPhase(c_matrix, theta, s_full);
  return applyOddColumnPhases(c_matrix, phases);
}

}  // namespace rerdmft
