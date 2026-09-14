#include "NonRelOrbitalGradient.h"

#include <cstddef>

#include "ClosedShellSpinOrbitals.h"
#include "MoIntegralTransform.h"
#include "NonRelHartreeFock.h"

namespace rerdmft {

Matrix<double> nonRelOrbitalGradientEfficient(const Matrix<double>& h_core_ao,
                                               const PackedTwoElectronTensor& eri_ao,
                                               const Matrix<double>& c_matrix, int n_electrons) {
  const auto p_new_ao = nonRelDensityMatrix(c_matrix, n_electrons);
  const auto fock_like_ao = nonRelFockMatrix(h_core_ao, eri_ao, p_new_ao);
  const auto fock_like_mo = moOneElectronTransform(fock_like_ao, c_matrix);

  const std::size_t n_spatial = c_matrix.cols();
  const auto fock_like_spin = closedShellSpinOrbitalOneElectron(fock_like_mo, n_spatial);

  const int n_occ_spatial = n_electrons / 2;
  const std::size_t n_spin = 2 * n_spatial;
  auto isOccupied = [n_spatial, n_occ_spatial](std::size_t p) {
    return (p % n_spatial) < static_cast<std::size_t>(n_occ_spatial);
  };

  Matrix<double> g(n_spin, n_spin, 0.0);
#pragma omp parallel for
  for (std::size_t p = 0; p < n_spin; ++p) {
    const bool occ_p = isOccupied(p);
    for (std::size_t q = 0; q <= p; ++q) {
      const bool occ_q = isOccupied(q);
      if (occ_p == occ_q) continue;
      g(p, q) = (occ_p ? 2.0 : -2.0) * fock_like_spin(p, q);
    }
  }
  return g;
}

}  // namespace rerdmft
