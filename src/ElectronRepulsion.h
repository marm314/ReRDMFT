#ifndef RERDMFT_ELECTRONREPULSION_H
#define RERDMFT_ELECTRONREPULSION_H

#include <vector>

#include "MolecularBasis.h"
#include "Tensor4.h"

namespace rerdmft {

// Computes the full (real) electron-repulsion tensor in chemist's notation,
//   (pq|rs) = integral integral p(1) q(1) (1/r12) r(2) s(2) dr1 dr2,
// for a single already-normalized cartesian AO basis used on all four
// indices, via libcint's cint2e_cart. Exploits the standard 8-fold
// permutational symmetry of real-orbital ERIs (p<->q, r<->s, (pq)<->(rs))
// to only evaluate one representative of each symmetry-equivalent set.
Tensor4<double> twoElectronIntegrals(const std::vector<BasisFunction>& basis);

// Same integral, but with p,q drawn from `basis_pq` and r,s from
// `basis_rs` (generally a different basis, e.g. Large vs unrestricted-
// kinetic-balance Small). Exploits p<->q and r<->s symmetry (not the
// (pq)<->(rs) swap, since the two sides are generally different bases).
Tensor4<double> twoElectronIntegralsCross(const std::vector<BasisFunction>& basis_pq,
                                           const std::vector<BasisFunction>& basis_rs);

}  // namespace rerdmft

#endif  // RERDMFT_ELECTRONREPULSION_H
