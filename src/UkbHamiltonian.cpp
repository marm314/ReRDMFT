#include "UkbHamiltonian.h"

#include "Vext.h"

namespace rerdmft {

Matrix<std::complex<double>> ukbHamiltonianMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, const std::vector<Atom>& geometry,
    double speed_of_light) {
  const auto kinetic = diracKineticMatrix(large_basis, small_basis, speed_of_light);
  const auto rest_energy = diracRestEnergyMatrix(large_basis, small_basis, speed_of_light);
  const auto vext = vextMatrix(large_basis, small_basis, geometry);

  return kinetic + rest_energy + vext;
}

}  // namespace rerdmft
