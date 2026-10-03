#include "SmallComponentBasis.h"

namespace rerdmft {

void SmallComponentBasis::build(const std::vector<BasisFunction>& large_basis) {
  functions_ = buildRkbSmallBasis(large_basis, &term_index_);
}

}  // namespace rerdmft
