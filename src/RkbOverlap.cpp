#include "RkbOverlap.h"

#include <cstddef>
#include <stdexcept>

#include "Integrals.h"

namespace rerdmft {

Matrix<std::complex<double>> rkbSmallOverlapMatrix(
    const std::vector<BasisFunction>& small_basis,
    const Matrix<std::complex<double>>& rkb_coefficients) {
  const std::size_t n_small = small_basis.size();
  const std::size_t n_large2 = rkb_coefficients.rows();  // 2*nLarge
  const std::size_t n_small2 = rkb_coefficients.cols();  // 2*nSmall

  if (n_small2 != 2 * n_small) {
    throw std::runtime_error(
        "rkbSmallOverlapMatrix: rkb_coefficients column count is inconsistent with small_basis");
  }

  const Matrix<double> s_small = overlapMatrix(small_basis);

  // S_uKB, block-diagonal over the alpha/beta spin blocks.
  Matrix<std::complex<double>> s_ukb(n_small2, n_small2, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_small; ++i) {
    for (std::size_t j = 0; j < n_small; ++j) {
      const std::complex<double> value(s_small(i, j), 0.0);
      s_ukb(i, j) = value;
      s_ukb(n_small + i, n_small + j) = value;
    }
  }

  // conj(C): same shape as C (2*nLarge x 2*nSmall).
  Matrix<std::complex<double>> conj_c(n_large2, n_small2);
  for (std::size_t p = 0; p < n_large2; ++p) {
    for (std::size_t t = 0; t < n_small2; ++t) {
      conj_c(p, t) = std::conj(rkb_coefficients(p, t));
    }
  }

  // C^T: (2*nSmall x 2*nLarge).
  Matrix<std::complex<double>> c_transpose(n_small2, n_large2);
  for (std::size_t p = 0; p < n_large2; ++p) {
    for (std::size_t t = 0; t < n_small2; ++t) {
      c_transpose(t, p) = rkb_coefficients(p, t);
    }
  }

  return conj_c * (s_ukb * c_transpose);
}

}  // namespace rerdmft
