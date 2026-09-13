#include "SpinorBasis.h"

#include <stdexcept>

namespace rerdmft {

const char* spinBlockLabel(SpinBlock block) {
  switch (block) {
    case SpinBlock::LargeAlpha: return "Large-alpha";
    case SpinBlock::LargeBeta: return "Large-beta";
    case SpinBlock::SmallAlpha: return "Small-alpha";
    case SpinBlock::SmallBeta: return "Small-beta";
  }
  return "?";
}

void SpinorBasis::build(const std::vector<BasisFunction>& large_basis,
                         const std::vector<BasisFunction>& small_basis) {
  large_basis_ = large_basis;
  small_basis_ = small_basis;
}

std::pair<SpinBlock, std::size_t> SpinorBasis::locate(std::size_t index) const {
  const std::size_t n_large = nLarge();
  const std::size_t n_small = nSmall();
  if (index >= 2 * n_large + 2 * n_small) {
    throw std::out_of_range("spinor basis index out of range");
  }

  if (index < n_large) return {SpinBlock::LargeAlpha, index};
  index -= n_large;
  if (index < n_large) return {SpinBlock::LargeBeta, index};
  index -= n_large;
  if (index < n_small) return {SpinBlock::SmallAlpha, index};
  index -= n_small;
  return {SpinBlock::SmallBeta, index};
}

const BasisFunction& SpinorBasis::ao(std::size_t index) const {
  const auto [blk, ao_index] = locate(index);
  switch (blk) {
    case SpinBlock::LargeAlpha:
    case SpinBlock::LargeBeta:
      return large_basis_[ao_index];
    case SpinBlock::SmallAlpha:
    case SpinBlock::SmallBeta:
      return small_basis_[ao_index];
  }
  throw std::logic_error("unreachable spin block");
}

}  // namespace rerdmft
