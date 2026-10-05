#ifndef RERDMFT_MOLECULARBASIS_H
#define RERDMFT_MOLECULARBASIS_H

#include <string>
#include <vector>

#include "BasisSet.h"
#include "Input.h"
#include "Shell.h"

namespace rerdmft {

// A single contracted cartesian atomic orbital, placed at an atomic center.
struct BasisFunction {
  std::string element;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int l = 0;
  CartesianExponents cartesian;
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

// Builds the full list of cartesian atomic orbitals for a molecule by
// placing a copy of each atom type's basis shells at every atomic position
// of that type (e.g. the Hydrogen basis is copied once per Hydrogen atom).
class MolecularBasis {
 public:
  void build(const std::vector<Atom>& geometry, const BasisSet& basis_set);

  const std::vector<BasisFunction>& functions() const { return functions_; }
  std::vector<BasisFunction>& functions() { return functions_; }

 private:
  std::vector<BasisFunction> functions_;
};

// One contracted shell's location within a flat BasisFunction list: `first` is the index of its
// first Cartesian component, `count` (== cartesianComponents(l).size()) how many consecutive
// entries belong to it. MolecularBasis::build's own construction loop guarantees shells appear as
// consecutive runs of exactly this many entries sharing the same center/l/exponents/coefficients
// (differing only in `.cartesian`) -- the same layout invariant SphericalTransform.h's own
// per-shell block-diagonal transform already relies on. Exposed so integral code can evaluate a
// whole shell quartet in ONE libcint call (any one member of the group works identically as the
// "representative" BasisFunction, since atm/bas/env never reference `.cartesian` -- see
// ElectronRepulsion.h's twoElectronShellQuartet) instead of once per individual AO quadruplet.
struct ShellInfo {
  std::size_t first = 0;
  std::size_t count = 0;
};
std::vector<ShellInfo> groupIntoShells(const std::vector<BasisFunction>& basis);

}  // namespace rerdmft

#endif  // RERDMFT_MOLECULARBASIS_H
