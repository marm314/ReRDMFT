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

char angularMomentumLabel(int l) {
  static const char labels[] = {'S', 'P', 'D', 'F', 'G', 'H', 'I'};
  if (l < 0 || l >= static_cast<int>(sizeof(labels))) return '?';
  return labels[l];
}

}  // namespace rerdmft
