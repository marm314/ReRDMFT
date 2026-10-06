#ifndef RERDMFT_UKBPNOFFOCK_H
#define RERDMFT_UKBPNOFFOCK_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "PNOFs.h"
#include "UkbFockMatrixDirect.h"

namespace rerdmft {

// Builds a PNOF functional's generalized Fock matrix (GenFock, Dyall Eq. 8.30 convention) DIRECTLY
// from on-demand UKB AO integrals (UkbGenFockPrimitives.h's ukbGenFockBuild) -- never a stored
// two-electron tensor. See project memory project-ukb-direct-genfock-scheme for the derivation,
// including the two_rdm_l1/l2 term's Kramers-pair-aware reduction (plain C^T, not C^dagger, on
// both legs of its own small transform -- a real asymmetry versus the H/X terms, not a
// simplification).
//
// GENERIC (always-correct) form, same choice as UkbJkOnlyFock.h: one UKB-AO density per output row
// `q` for each of the H, X, and L1/L2 terms, weighted directly from `buildPnofFullTwoRdm`'s own
// already-validated two_rdm_h/x/l1/l2 -- no per-geminal separable-sum decomposition is hand-coded
// here. Costs O(n_total) densities per call (not exploiting PNOF's own cheap, geminal-block-sized
// structure this file's memory establishes) -- a deliberate first-correctness-pass tradeoff, same
// as UkbJkOnlyFock.h's.
//
// `eri` supplies large_basis/small_basis/v_total (UkbFockMatrixDirect.h's UkbDirectEriSource,
// unchanged). `c_rkb` is the RKB-basis (n_rkb x n_total) natural-spinor coefficient matrix;
// `h_rkb` is the RKB-basis one-electron Hamiltonian; `geminals`/`occupations`/`relativistic` are
// exactly `buildPnofFullTwoRdm`'s own arguments (`n_total` is `occupations.size()`). Returns the
// n_total x n_total GenFock matrix, same convention `pnofFockMatrix` already uses.
Matrix<std::complex<double>> ukbPnofFockMatrix(PnofFunctional functional,
                                                const Matrix<std::complex<double>>& h_rkb,
                                                const UkbDirectEriSource& eri,
                                                const Matrix<std::complex<double>>& c_rkb,
                                                const std::vector<PnofGeminal>& geminals,
                                                const std::vector<double>& occupations,
                                                bool relativistic);

}  // namespace rerdmft

#endif  // RERDMFT_UKBPNOFFOCK_H
