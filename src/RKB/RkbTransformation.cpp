#include "RkbTransformation.h"

#include <algorithm>
#include <cstddef>

namespace rerdmft {

namespace {

void addIfPresent(Matrix<std::complex<double>>& c, std::size_t row, std::size_t col_offset,
                   std::size_t term_index, std::complex<double> weight) {
  if (term_index == RkbDerivativeTerms::kNone) return;
  c(row, col_offset + term_index) += weight;
}

}  // namespace

Matrix<std::complex<double>> rkbCoefficients(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<RkbDerivativeTerms>& term_index) {
  const std::size_t n_large = large_basis.size();
  std::size_t n_small = 0;
  for (const auto& t : term_index) {
    for (std::size_t idx : {t.lower_x, t.raise_x, t.lower_y, t.raise_y, t.lower_z, t.raise_z}) {
      if (idx != RkbDerivativeTerms::kNone) n_small = std::max(n_small, idx + 1);
    }
  }

  const std::complex<double> i_unit(0.0, 1.0);
  Matrix<std::complex<double>> c(2 * n_large, 2 * n_small, std::complex<double>(0.0, 0.0));
  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 0;
  const std::size_t off_small_beta = n_small;

  for (std::size_t p = 0; p < n_large; ++p) {
    const RkbDerivativeTerms& t = term_index[p];
    const std::size_t row_alpha = off_large_alpha + p;
    const std::size_t row_beta = off_large_beta + p;

    // (Large-alpha, Small-alpha): -i * D_z(p)
    addIfPresent(c, row_alpha, off_small_alpha, t.lower_z, -i_unit);
    addIfPresent(c, row_alpha, off_small_alpha, t.raise_z, -i_unit);
    // (Large-alpha, Small-beta): -i * D_x(p) + D_y(p)
    addIfPresent(c, row_alpha, off_small_beta, t.lower_x, -i_unit);
    addIfPresent(c, row_alpha, off_small_beta, t.raise_x, -i_unit);
    addIfPresent(c, row_alpha, off_small_beta, t.lower_y, std::complex<double>(1.0, 0.0));
    addIfPresent(c, row_alpha, off_small_beta, t.raise_y, std::complex<double>(1.0, 0.0));
    // (Large-beta, Small-alpha): -i * D_x(p) - D_y(p)
    addIfPresent(c, row_beta, off_small_alpha, t.lower_x, -i_unit);
    addIfPresent(c, row_beta, off_small_alpha, t.raise_x, -i_unit);
    addIfPresent(c, row_beta, off_small_alpha, t.lower_y, std::complex<double>(-1.0, 0.0));
    addIfPresent(c, row_beta, off_small_alpha, t.raise_y, std::complex<double>(-1.0, 0.0));
    // (Large-beta, Small-beta): +i * D_z(p)
    addIfPresent(c, row_beta, off_small_beta, t.lower_z, i_unit);
    addIfPresent(c, row_beta, off_small_beta, t.raise_z, i_unit);
  }

  return c;
}

}  // namespace rerdmft
