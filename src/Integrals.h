#ifndef RERDMFT_INTEGRALS_H
#define RERDMFT_INTEGRALS_H

#include <string>
#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// Diagnostic record for one cartesian atomic orbital's self-overlap check.
struct NormalizationCheck {
  std::string element;
  int l = 0;
  CartesianExponents cartesian;
  double self_overlap_before = 0.0;  // after primitive normalization, before the fix
  bool was_renormalized = false;
};

// Applies primitive Gaussian normalization (via libcint's CINTgto_norm) to
// every basis function, then uses libcint's cartesian overlap integral to
// verify each cartesian AO has unit self-overlap, rescaling its contraction
// coefficients whenever it does not. This is what happens for cartesian
// components that mix axes (e.g. d_xy, f_xyz): primitive normalization
// alone is not enough for them, since it does not depend on how the l
// quanta are split between lx, ly and lz.
//
// Returns a diagnostic report, one entry per basis function, in the same
// order as `functions`.
std::vector<NormalizationCheck> normalizeCartesianBasis(
    std::vector<BasisFunction>& functions, double tolerance = 1e-8);

}  // namespace rerdmft

#endif  // RERDMFT_INTEGRALS_H
