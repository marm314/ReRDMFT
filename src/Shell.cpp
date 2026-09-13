#include "Shell.h"

namespace rerdmft {

std::vector<CartesianExponents> cartesianComponents(int l) {
  std::vector<CartesianExponents> components;
  for (int lx = l; lx >= 0; --lx) {
    for (int ly = l - lx; ly >= 0; --ly) {
      const int lz = l - lx - ly;
      components.push_back({lx, ly, lz});
    }
  }
  return components;
}

}  // namespace rerdmft
