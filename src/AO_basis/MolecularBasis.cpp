#include "MolecularBasis.h"

#include <stdexcept>
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

std::vector<ShellInfo> groupIntoShells(const std::vector<BasisFunction>& basis) {
  std::vector<ShellInfo> shells;
  std::size_t i = 0;
  while (i < basis.size()) {
    const BasisFunction& first = basis[i];
    const std::size_t count = cartesianComponents(first.l).size();
    // Defensive check (not just trust the construction invariant): every member of this shell
    // must actually share the same center/l/exponents, or something upstream built the flat list
    // in a way this grouping's callers cannot safely assume. Deliberately NOT checking
    // `coefficients` here: normalizeCartesianBasis rescales each Cartesian component by its own
    // scalar factor (see twoElectronShellQuartet's own comment), so coefficients legitimately
    // differ between a shell's components on a normalized basis -- exponents do not (normalization
    // only ever rescales coefficients, see Integrals.cpp's normalizeCartesianBasis), so that stays
    // the right invariant to check either before or after normalization has run.
    for (std::size_t k = 1; k < count; ++k) {
      if (i + k >= basis.size()) {
        throw std::runtime_error("groupIntoShells: basis list ends mid-shell");
      }
      const BasisFunction& member = basis[i + k];
      if (member.l != first.l || member.x != first.x || member.y != first.y || member.z != first.z ||
          member.exponents != first.exponents) {
        throw std::runtime_error("groupIntoShells: basis list is not laid out as consecutive shells");
      }
    }
    shells.push_back({i, count});
    i += count;
  }
  return shells;
}

}  // namespace rerdmft
