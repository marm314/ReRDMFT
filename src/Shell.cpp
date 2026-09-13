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

int cartesianComponentIndex(int l, const CartesianExponents& c) {
  const std::vector<CartesianExponents> comps = cartesianComponents(l);
  for (std::size_t i = 0; i < comps.size(); ++i) {
    if (comps[i].lx == c.lx && comps[i].ly == c.ly && comps[i].lz == c.lz) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

char angularMomentumLabel(int l) {
  static const char labels[] = {'S', 'P', 'D', 'F', 'G', 'H', 'I'};
  if (l < 0 || l >= static_cast<int>(sizeof(labels))) return '?';
  return labels[l];
}

}  // namespace rerdmft
