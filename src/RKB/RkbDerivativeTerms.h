#ifndef RERDMFT_RKBDERIVATIVETERMS_H
#define RERDMFT_RKBDERIVATIVETERMS_H

#include <cstddef>
#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// For one large (already-normalized) Cartesian basis function, where (if at all) its own
// elementary sigma.p-derivative pieces live in the RKB small-component basis buildRkbSmallBasis
// returns -- see that function's own comment for the exact construction. d/dx_k of a contracted
// Gaussian splits into a "lowering" piece (same contraction coefficients, Cartesian exponent along
// k reduced by 1 -- absent if that exponent is already 0) and a "raising" piece (coefficients
// rescaled by each primitive's own exponent, Cartesian exponent along k increased by 1 -- always
// present). kNone marks an absent lowering piece.
struct RkbDerivativeTerms {
  static constexpr std::size_t kNone = static_cast<std::size_t>(-1);
  std::size_t lower_x = kNone, raise_x = kNone;
  std::size_t lower_y = kNone, raise_y = kNone;
  std::size_t lower_z = kNone, raise_z = kNone;
};

// Builds the RKB small-component basis directly and analytically from an already-normalized large
// (Cartesian) basis: for each large function, its own d/dx, d/dy, d/dz elementary pieces (see
// RkbDerivativeTerms) are appended as ordinary BasisFunctions -- NOT deduplicated or shared across
// different large functions (unlike the project's earlier unrestricted-kinetic-balance basis),
// since true restricted kinetic balance ties each large function to its OWN small partner alone.
// Each large function contributes up to 6 elementary terms (fewer when a Cartesian exponent is
// already 0), so the result has at most 6 * large_basis.size() entries. These are NOT re-normalized
// (normalizeCartesianBasis must NOT be called on the result): their coefficients are the exact,
// closed-form derivative weights relative to the parent large function's own (already normalized)
// coefficients, and re-normalizing would destroy that relationship. `terms_out`, if given, is
// resized to large_basis.size() and filled with each large function's own term indices into the
// returned basis -- rkbCoefficients (RkbTransformation.h) needs this to assign its weights.
std::vector<BasisFunction> buildRkbSmallBasis(const std::vector<BasisFunction>& large_basis,
                                               std::vector<RkbDerivativeTerms>* terms_out = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_RKBDERIVATIVETERMS_H
