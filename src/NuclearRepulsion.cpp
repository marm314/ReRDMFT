#include "NuclearRepulsion.h"

#include <cmath>
#include <cstddef>

#include "Element.h"

namespace rerdmft {

double nuclearRepulsionEnergy(const std::vector<Atom>& geometry) {
  double energy = 0.0;
  for (std::size_t a = 0; a < geometry.size(); ++a) {
    const double z_a = static_cast<double>(atomicNumber(geometry[a].symbol));
    for (std::size_t b = a + 1; b < geometry.size(); ++b) {
      const double z_b = static_cast<double>(atomicNumber(geometry[b].symbol));
      const double dx = geometry[a].x - geometry[b].x;
      const double dy = geometry[a].y - geometry[b].y;
      const double dz = geometry[a].z - geometry[b].z;
      const double r_ab = std::sqrt(dx * dx + dy * dy + dz * dz);
      energy += z_a * z_b / r_ab;
    }
  }
  return energy;
}

}  // namespace rerdmft
