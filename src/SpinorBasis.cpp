#include "SpinorBasis.h"

#include <stdexcept>

namespace rerdmft {

const char* spinBlockLabel(SpinBlock block) {
  switch (block) {
    case SpinBlock::LargeAlpha: return "Large-alpha";
    case SpinBlock::LargeBeta: return "Large-beta";
  }
  return "?";
}

void SpinorBasis::build(const std::vector<BasisFunction>& scalar_basis) {
  scalar_basis_ = scalar_basis;
}

SpinBlock SpinorBasis::block(std::size_t index) const {
  if (index >= size()) {
    throw std::out_of_range("spinor basis index out of range");
  }
  return index < nao() ? SpinBlock::LargeAlpha : SpinBlock::LargeBeta;
}

std::size_t SpinorBasis::aoIndex(std::size_t index) const {
  if (index >= size()) {
    throw std::out_of_range("spinor basis index out of range");
  }
  return index < nao() ? index : index - nao();
}

const BasisFunction& SpinorBasis::ao(std::size_t index) const {
  return scalar_basis_[aoIndex(index)];
}

}  // namespace rerdmft
