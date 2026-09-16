#ifndef RERDMFT_NUCLEARATTRACTION_H
#define RERDMFT_NUCLEARATTRACTION_H

#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Computes the full (real, symmetric) nuclear-electron attraction matrix
// <AO_i| Vext(r) |AO_j> for an already-normalized cartesian AO basis, via
// libcint, where Vext(r) = -sum_A Z_A / |r - R_A| sums the Coulomb
// attraction to every nucleus in `geometry` (not just the ones the basis
// functions happen to be centered on).
Matrix<double> nuclearAttractionMatrix(const std::vector<BasisFunction>& basis,
                                        const std::vector<Atom>& geometry);

}  // namespace rerdmft

#endif  // RERDMFT_NUCLEARATTRACTION_H
