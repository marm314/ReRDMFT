#ifndef RERDMFT_NABLAINTEGRALS_H
#define RERDMFT_NABLAINTEGRALS_H

#include <array>

#include "MolecularBasis.h"

namespace rerdmft {

// <bra|d/dx_k|ket> for k=0(x),1(y),2(z), for one specific pair of
// individually-normalized cartesian AOs, placed at their real atomic
// centers. Built as a minimal, independent 2-shell/2-atom libcint system
// per pair (rather than a shared multi-shell system) because each cartesian
// AO carries its own individually rescaled contraction coefficients (see
// normalizeCartesianBasis) -- distinct cartesian components of the same
// physical shell no longer share one coefficient set once normalized, so
// each is its own libcint "shell" of its own angular momentum, and only the
// entry matching its own (lx,ly,lz) is read out of the full shell block
// libcint returns.
std::array<double, 3> nablaIntegral(const BasisFunction& bra, const BasisFunction& ket);

}  // namespace rerdmft

#endif  // RERDMFT_NABLAINTEGRALS_H
