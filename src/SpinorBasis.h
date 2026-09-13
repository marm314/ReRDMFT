#ifndef RERDMFT_SPINORBASIS_H
#define RERDMFT_SPINORBASIS_H

#include <cstddef>
#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// The spin blocks of the 4-component spinor basis, in storage order: every
// Large-alpha function first, then every Large-beta function. (The Small
// component is not built yet.)
enum class SpinBlock { LargeAlpha, LargeBeta };

const char* spinBlockLabel(SpinBlock block);

// A 4-component spinor basis built from a scalar cartesian AO basis.
// Currently covers only the Large component: the same (already normalized)
// scalar AOs are reused, unchanged, for both the alpha and beta spin
// blocks, stored contiguously as [Large-alpha AOs][Large-beta AOs].
class SpinorBasis {
 public:
  void build(const std::vector<BasisFunction>& scalar_basis);

  // Number of scalar AOs per spin block.
  std::size_t nao() const { return scalar_basis_.size(); }

  // Total number of spinor basis functions (2 * nao(), since only the
  // Large component is currently present).
  std::size_t size() const { return 2 * nao(); }

  SpinBlock block(std::size_t index) const;

  // Index of the underlying scalar AO for a given spinor basis index,
  // within its spin block (0 <= aoIndex(index) < nao()).
  std::size_t aoIndex(std::size_t index) const;

  // The scalar cartesian AO underlying a given spinor basis index.
  const BasisFunction& ao(std::size_t index) const;

 private:
  std::vector<BasisFunction> scalar_basis_;
};

}  // namespace rerdmft

#endif  // RERDMFT_SPINORBASIS_H
