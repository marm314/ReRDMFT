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

}  // namespace rerdmft

#endif  // RERDMFT_MOLECULARBASIS_H
