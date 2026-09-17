#include "X2C_OrbitalGradient.h"

#include <cstddef>

#include "X2C_DensityMatrix.h"
#include "X2C_FockMatrix.h"
#include "X2C_MoTransform.h"

namespace rerdmft {

Matrix<std::complex<double>> x2cOrbitalGradientEfficient(const Matrix<std::complex<double>>& h_x2c,
                                                          const Tensor4<double>& eri,
                                                          const Matrix<std::complex<double>>& c_matrix,
                                                          int n_electrons) {
  const auto p_new_ao = x2cDensityMatrix(c_matrix, n_electrons);
  const auto fock_like_ao = x2cFockMatrix(h_x2c, eri, p_new_ao);
  const auto fock_like_mo = x2cMoOneElectronTransform(fock_like_ao, c_matrix);

  const std::size_t n = c_matrix.cols();
  auto isOccupied = [n_electrons](std::size_t p) {
    return p < static_cast<std::size_t>(n_electrons);
  };

  Matrix<std::complex<double>> g(n, n, std::complex<double>(0.0, 0.0));
#pragma omp parallel for
  for (std::size_t p = 0; p < n; ++p) {
    const bool occ_p = isOccupied(p);
    for (std::size_t q = 0; q <= p; ++q) {
      const bool occ_q = isOccupied(q);
      if (occ_p == occ_q) continue;
      const std::complex<double> sign(occ_p ? 2.0 : -2.0, 0.0);
      g(p, q) = sign * fock_like_mo(p, q);
    }
  }
  return g;
}

}  // namespace rerdmft
