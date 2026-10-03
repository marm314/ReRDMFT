#ifndef RERDMFT_SMALLCOMPONENTBASIS_H
#define RERDMFT_SMALLCOMPONENTBASIS_H

#include <vector>

#include "MolecularBasis.h"
#include "RkbDerivativeTerms.h"

namespace rerdmft {

// The restricted-kinetic-balance (RKB) small-component basis, built directly and analytically from
// an already-normalized large (Cartesian) basis -- see buildRkbSmallBasis (RKB/RkbDerivativeTerms.h)
// for the construction itself. A thin, stateful wrapper around that free function, matching the
// project's other *Basis classes' build()/functions() interface.
class SmallComponentBasis {
 public:
  void build(const std::vector<BasisFunction>& large_basis);

  const std::vector<BasisFunction>& functions() const { return functions_; }
  std::vector<BasisFunction>& functions() { return functions_; }

  // Each large function's own term indices into functions() -- rkbCoefficients
  // (RKB/RkbTransformation.h) needs this to assign its closed-form weights.
  const std::vector<RkbDerivativeTerms>& termIndex() const { return term_index_; }

 private:
  std::vector<BasisFunction> functions_;
  std::vector<RkbDerivativeTerms> term_index_;
};

}  // namespace rerdmft

#endif  // RERDMFT_SMALLCOMPONENTBASIS_H
