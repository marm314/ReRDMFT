#ifndef RERDMFT_SPINORBASIS_H
#define RERDMFT_SPINORBASIS_H

#include <cstddef>
#include <utility>
#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// The spin/component blocks of the 4-component spinor basis, in storage
// order: all Large-alpha, then all Large-beta, then all Small-alpha, then
// all Small-beta.
enum class SpinBlock { LargeAlpha, LargeBeta, SmallAlpha, SmallBeta };

const char* spinBlockLabel(SpinBlock block);

// The full 4-component spinor basis, built from a large-component scalar
// cartesian AO basis and a small-component one (see SmallComponentBasis).
// Both are already-normalized alpha-spin AO lists; the corresponding beta
// blocks are the same spatial AOs, reused unchanged, per unrestricted
// kinetic balance. Storage order is contiguous:
//   [Large-alpha][Large-beta][Small-alpha][Small-beta]
class SpinorBasis {
 public:
  void build(const std::vector<BasisFunction>& large_basis,
             const std::vector<BasisFunction>& small_basis);

  // Number of scalar AOs per large spin block (Large-alpha or Large-beta).
  std::size_t nLarge() const { return large_basis_.size(); }

  // Number of scalar AOs per small spin block (Small-alpha or Small-beta).
  std::size_t nSmall() const { return small_basis_.size(); }

  // Total number of spinor basis functions: 2 * nLarge() + 2 * nSmall().
  std::size_t size() const { return 2 * nLarge() + 2 * nSmall(); }

  SpinBlock block(std::size_t index) const { return locate(index).first; }

  // Index of the underlying scalar AO for a given spinor basis index,
  // within its spin/component block.
  std::size_t aoIndex(std::size_t index) const { return locate(index).second; }

  // The scalar cartesian AO underlying a given spinor basis index.
  const BasisFunction& ao(std::size_t index) const;

 private:
  std::pair<SpinBlock, std::size_t> locate(std::size_t index) const;

  std::vector<BasisFunction> large_basis_;
  std::vector<BasisFunction> small_basis_;
};

}  // namespace rerdmft

#endif  // RERDMFT_SPINORBASIS_H
