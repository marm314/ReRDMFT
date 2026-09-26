#ifndef RERDMFT_UTILS_BASISFINGERPRINT_H
#define RERDMFT_UTILS_BASISFINGERPRINT_H

#include <cstdint>
#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// Order-sensitive 64-bit hash over the concrete, already-built basis functions (element, center, angular momentum,
// exponents, contraction coefficients) -- not over the basis-file text. Two bases that hash equal are, short of an
// astronomically unlikely collision, the same basis. Stored in the RESTART file (Utils/Restart.h) so a restart can
// refuse a file written for a different basis.
std::uint64_t basisFingerprint(const std::vector<BasisFunction>& basis);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_BASISFINGERPRINT_H
