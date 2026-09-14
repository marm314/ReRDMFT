#include "RkbOrbitalGradient.h"

#include <cstddef>

#include "RkbDensityMatrix.h"
#include "RkbFockMatrix.h"
#include "RkbMoTransform.h"

namespace rerdmft {

Matrix<std::complex<double>> dhfOrbitalGradientEfficient(
    const Matrix<std::complex<double>>& h_rkb, const RkbTwoElectronTensor& eri_rkb,
    const Matrix<std::complex<double>>& c_dhf, int n_electrons) {
  const auto p_new_ao = rkbDensityMatrix(c_dhf, n_electrons);
  const auto fock_like_ao = rkbFockMatrix(h_rkb, eri_rkb, p_new_ao);
  const auto fock_like_mo = rkbMoOneElectronTransform(fock_like_ao, c_dhf);

  const std::size_t rkb_dim = c_dhf.cols();
  const std::size_t n_negative = rkb_dim / 2;
  auto isOccupied = [n_negative, n_electrons](std::size_t p) {
    return p >= n_negative && p < n_negative + static_cast<std::size_t>(n_electrons);
  };

  Matrix<std::complex<double>> g(rkb_dim, rkb_dim, std::complex<double>(0.0, 0.0));
#pragma omp parallel for
  for (std::size_t p = 0; p < rkb_dim; ++p) {
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
