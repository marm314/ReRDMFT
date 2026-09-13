#include "RkbDensityMatrix.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

Matrix<std::complex<double>> rkbCoefficientMatrix(const Matrix<std::complex<double>>& x_full,
                                                   const Matrix<std::complex<double>>& u) {
  if (x_full.cols() != u.rows()) {
    throw std::runtime_error("rkbCoefficientMatrix: X_full and U dimensions are inconsistent");
  }
  return x_full * u;
}

Matrix<std::complex<double>> rkbDensityMatrix(const Matrix<std::complex<double>>& c_dhf,
                                               int n_electrons) {
  const std::size_t n = c_dhf.rows();
  if (c_dhf.cols() != n) {
    throw std::runtime_error("rkbDensityMatrix: C_DHF is not square");
  }
  if (n % 2 != 0) {
    throw std::runtime_error("rkbDensityMatrix: RKB basis dimension must be even (4*nLarge)");
  }
  const std::size_t occ_start = n / 2;
  if (n_electrons <= 0 || occ_start + static_cast<std::size_t>(n_electrons) > n) {
    throw std::runtime_error(
        "rkbDensityMatrix: n_electrons must be a positive number of spinors that fits within "
        "the positive-energy half of the RKB basis");
  }
  const std::size_t occ_end = occ_start + static_cast<std::size_t>(n_electrons);

  Matrix<std::complex<double>> p(n, n, std::complex<double>(0.0, 0.0));
  for (std::size_t c = occ_start; c < occ_end; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::complex<double> c_ic = c_dhf(i, c);
      if (c_ic == std::complex<double>(0.0, 0.0)) continue;
      for (std::size_t j = 0; j < n; ++j) {
        p(i, j) += c_ic * std::conj(c_dhf(j, c));
      }
    }
  }
  return p;
}

}  // namespace rerdmft
