#ifndef RERDMFT_SMALLCOMPONENTBASIS_H
#define RERDMFT_SMALLCOMPONENTBASIS_H

#include <vector>

#include "BasisSet.h"
#include "Input.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds the unrestricted-kinetic-balance (uKB) small-component cartesian
// AO basis from the large-component basis set: applying sigma.p to a large
// cartesian GTO of angular momentum l produces terms of angular momentum
// l-1 and l+1 (differentiating x^lx y^ly z^lz exp(-a r^2) lowers one
// cartesian exponent or raises one, never changing the exponent a). uKB
// simply adds shells of both resulting angular momenta as independent
// small-component basis functions, reusing the parent shell's exponents
// and contraction coefficients unchanged, at the same atomic center:
//   L shell l=0 (S)          -> S shells l=1 (P)
//   L shell l=1 (P)          -> S shells l=0 (S) and l=2 (D)
//   L shell l=2 (D)          -> S shells l=1 (P) and l=3 (F)
//   ... and so on.
//
// The result still needs libcint-based normalization (see
// normalizeCartesianBasis in Integrals.h), exactly like the large
// component.
class SmallComponentBasis {
 public:
  void build(const std::vector<Atom>& geometry, const BasisSet& basis_set);

  const std::vector<BasisFunction>& functions() const { return functions_; }
  std::vector<BasisFunction>& functions() { return functions_; }

 private:
  std::vector<BasisFunction> functions_;
};

}  // namespace rerdmft

#endif  // RERDMFT_SMALLCOMPONENTBASIS_H
