#include "SmallComponentBasis.h"

#include <utility>

namespace rerdmft {

void SmallComponentBasis::build(const std::vector<Atom>& geometry,
                                 const BasisSet& basis_set) {
  functions_.clear();

  for (const auto& atom : geometry) {
    const std::vector<Shell>& large_shells =
        basis_set.shellsForElement(atom.symbol);
    for (const auto& large_shell : large_shells) {
      std::vector<int> small_ls;
      if (large_shell.l > 0) small_ls.push_back(large_shell.l - 1);
      small_ls.push_back(large_shell.l + 1);

      for (int small_l : small_ls) {
        for (const auto& cart : cartesianComponents(small_l)) {
          BasisFunction fn;
          fn.element = atom.symbol;
          fn.x = atom.x;
          fn.y = atom.y;
          fn.z = atom.z;
          fn.l = small_l;
          fn.cartesian = cart;
          fn.exponents = large_shell.exponents;
          fn.coefficients = large_shell.coefficients;
          functions_.push_back(std::move(fn));
        }
      }
    }
  }
}

}  // namespace rerdmft
