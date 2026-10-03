#ifndef RERDMFT_RKBTRANSFORMATION_H
#define RERDMFT_RKBTRANSFORMATION_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"
#include "RkbDerivativeTerms.h"

namespace rerdmft {

// Builds the restricted-kinetic-balance (RKB) expansion coefficients C that express sigma.p acting
// on each Large spin-orbital as a linear combination of the RKB small-component basis
// (buildRkbSmallBasis, RkbDerivativeTerms.h):
//
//   sigma.p |Large_p> = sum_q C_pq |Small_q>
//
// Unlike an unrestricted-kinetic-balance scheme (project history: SmallComponentBasis's earlier,
// now-removed uKB basis), this is an EXACT, closed-form identity, not a least-squares/numerical
// projection -- differentiating a Cartesian Gaussian is itself exact and closed-form (see
// RkbDerivativeTerms.h), and the small basis here is built to contain exactly the pieces each large
// function's own derivative needs, with no inversion of any overlap matrix required to recover C.
// sigma.p = sigma_x p_x + sigma_y p_y + sigma_z p_z (p_k = -i d/dx_k) couples the two spin flavors;
// writing D_k(p) for the elementary-term pair (lower_k(p), raise_k(p)) that RkbDerivativeTerms
// indexes (each already carrying its own closed-form coefficient, so each contributes with
// unit weight), direct application of the Pauli matrices gives:
//   (Large-alpha, Small-alpha): -i * D_z(p)
//   (Large-alpha, Small-beta ): -i * D_x(p) + D_y(p)
//   (Large-beta,  Small-alpha): -i * D_x(p) - D_y(p)
//   (Large-beta,  Small-beta ): +i * D_z(p)
// cross-checked two independent ways: against this project's own earlier (now-removed) numerical
// M*S^-1 construction's sign convention (itself validated against M. Rodriguez-Mayorga's
// m_relativistic.f90/MOLGW) via the integration-by-parts relation between the two, and directly
// against sigma.p applied to an explicit two-component spinor -- both agree exactly.
//
// `term_index` must be buildRkbSmallBasis's own output for this exact `large_basis` (same order).
// Returned as a (2*nLarge x 2*nSmall) matrix following the same spin block ordering as SpinorBasis:
// rows [Large-alpha, Large-beta], columns [Small-alpha, Small-beta] (nSmall = the RKB small basis
// size, i.e. term_index-implied basis, NOT a separately-counted spatial basis).
Matrix<std::complex<double>> rkbCoefficients(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<RkbDerivativeTerms>& term_index);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTRANSFORMATION_H
