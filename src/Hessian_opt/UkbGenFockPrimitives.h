#ifndef RERDMFT_UKBGENFOCKPRIMITIVES_H
#define RERDMFT_UKBGENFOCKPRIMITIVES_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// The two UKB-AO-direct primitives GenFock's H/X and two_rdm_l1/l2 terms both reduce to (see
// project memory project-ukb-direct-genfock-scheme for the full derivation):
//   J[D](mu,nu)     := sum_{lambda,sigma} (mu,nu|lambda,sigma) D(sigma,lambda)
//   K[D](mu,lambda) := sum_{nu,sigma}     (mu,nu|lambda,sigma) D(nu,sigma)
// computed, for an ARBITRARY LIST of UKB-AO-basis densities (same combined [Large-alpha,
// Large-beta, Small-alpha, Small-beta] layout UKB/UkbFockMatrixDirect.h's density_ukb already
// uses), via ONE shared pass over UKB AO integrals -- shell-batched Large-Large, Schwarz-screened
// Large-Small/Small-Small, full permutational symmetry already fixed this session -- so the
// expensive integral evaluation is paid exactly once no matter how many densities are supplied.
//
// Deliberately NOT a wrapper around UkbFockMatrixDirect.h's ukbFockTwoElectronDirect (left
// completely untouched -- still the production SCF_DIRECT_4C kernel, still single-density,
// combined-J-minus-K output): this file needs J and K SEPARATELY (GenFock's H and X terms
// generally weight them with DIFFERENT densities, not a shared reduction of one input), for
// potentially MANY densities at once, so it is its own implementation even though it mirrors that
// file's own shell-batching/screening/symmetry-scatter logic exactly (same already-validated
// patterns, same already-fixed shell-level-not-per-entry dedup for the Large-Large sector).
//
// Memory: O(R * n_ukb^2) for R input densities (each with its own 4x4 flavor-block
// decomposition) plus O(R * n_ukb^2) for the matching J/K accumulators -- cheap for PNOF (a
// handful of densities total, reused across every geminal) and 8 of JK_only's 10 functionals (1-3
// densities), still cheap for pCCD with a sane active space (O(n_occ)+O(n_vir) densities), but can
// reach tens of GB for ML/MLSIC's worst case (O(n_mo) densities) on a large basis -- splitting a
// large R into smaller batches (several integral passes instead of one) is a follow-up, not yet
// implemented here; this function always does R in one pass.
struct UkbGenFockBuild {
  std::vector<Matrix<std::complex<double>>> j;  // j[k](mu,nu) = J[densities[k]](mu,nu), n_ukb x n_ukb
  std::vector<Matrix<std::complex<double>>> k;  // k[k](mu,lambda) = K[densities[k]](mu,lambda), n_ukb x n_ukb
};

// `densities[k]` must each be n_ukb x n_ukb (n_ukb = 2*large_basis.size() + 2*small_basis.size()),
// same layout as UkbFockMatrixDirect.h's density_ukb -- general (not necessarily Hermitian; the
// two_rdm_l1/l2 densities built from paired NO columns, project memory's D_L^(q), are not in
// general). Returns `j`/`k`, each of size `densities.size()`, same n_ukb x n_ukb shape, UNASSEMBLED
// (the caller combines them with whatever row/column scaling and C^dagger(...)C or C^T(...)C
// transform its own functional's coupling decomposition calls for -- this function knows nothing
// about GenFock, PNOF, JK_only, or pCCD, only the two bare AO primitives above).
UkbGenFockBuild ukbGenFockBuild(const std::vector<BasisFunction>& large_basis,
                                 const std::vector<BasisFunction>& small_basis,
                                 const std::vector<Matrix<std::complex<double>>>& densities);

}  // namespace rerdmft

#endif  // RERDMFT_UKBGENFOCKPRIMITIVES_H
