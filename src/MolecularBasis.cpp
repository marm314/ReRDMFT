#include "MolecularBasis.h"

#include <utility>

namespace rerdmft {

void MolecularBasis::build(const std::vector<Atom>& geometry,
                            const BasisSet& basis_set) {
  functions_.clear();

  for (const auto& atom : geometry) {
    const std::vector<Shell>& shells = basis_set.shellsForElement(atom.symbol);
    for (const auto& shell : shells) {
      for (const auto& cart : cartesianComponents(shell.l)) {
        BasisFunction fn;
        fn.element = atom.symbol;
        fn.x = atom.x;
        fn.y = atom.y;
        fn.z = atom.z;
        fn.l = shell.l;
        fn.cartesian = cart;
        fn.exponents = shell.exponents;
        fn.coefficients = shell.coefficients;
        functions_.push_back(std::move(fn));
      }
    }
  }
}

}  // namespace rerdmft
