#include "X2C_DensityMatrix.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

Matrix<std::complex<double>> x2cDensityMatrix(const Matrix<std::complex<double>>& c_matrix,
                                               int n_electrons) {
  const std::size_t n = c_matrix.rows();
  if (c_matrix.cols() != n) {
    throw std::runtime_error("x2cDensityMatrix: c_matrix is not square");
  }
  if (n_electrons <= 0 || static_cast<std::size_t>(n_electrons) > n) {
    throw std::runtime_error(
        "x2cDensityMatrix: n_electrons must be a positive number of spinors that fits within "
        "the basis");
  }
  const std::size_t occ_end = static_cast<std::size_t>(n_electrons);

  Matrix<std::complex<double>> p(n, n, std::complex<double>(0.0, 0.0));
  for (std::size_t c = 0; c < occ_end; ++c) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::complex<double> c_ic = c_matrix(i, c);
      if (c_ic == std::complex<double>(0.0, 0.0)) continue;
      for (std::size_t j = 0; j < n; ++j) {
        p(i, j) += c_ic * std::conj(c_matrix(j, c));
      }
    }
  }
  return p;
}

}  // namespace rerdmft
