#ifndef RERDMFT_UKBPCCDFOCK_H
#define RERDMFT_UKBPCCDFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "UkbFockMatrixDirect.h"
#include "pCCD.h"

namespace rerdmft {

// Builds pCCD's generalized Fock matrix (GenFock, Dyall Eq. 8.30 convention) DIRECTLY from
// on-demand UKB AO integrals (UkbGenFockPrimitives.h's ukbGenFockBuild) -- never a stored
// two-electron tensor. See project memory project-ukb-direct-genfock-scheme for the derivation.
//
// Same generic (always-correct) pattern as UkbJkOnlyFock.h/UkbPnofFock.h: one UKB-AO density per
// output row `q` for each of the H, X, and L1/L2 terms, weighted directly from
// `buildPccdFullTwoRdm`'s own already-validated unfold -- no hand-derived decomposition of pCCD's
// own T/Z-amplitude-built two_rdm_h/x/l1/l2 is done here. `PccdFullTwoRdm` feeds the EXACT same
// `two_rdm_h/x/l1/l2` shape `PnofFullTwoRdm` does (`PccdFock.h`'s own documented mapping), so the
// SAME H/X and two_rdm_l1/l2 reductions this scheme already derived and validated for PNOF apply
// here unchanged -- the only difference from `ukbPnofFockMatrix` is which `build*FullTwoRdm`/
// `build*PairOf` pair is called. Costs O(n_total) densities per call (not exploiting pCCD's own
// `O(n_occ)+O(n_vir)` bound this file's memory establishes) -- a deliberate first-correctness-pass
// tradeoff, same as the other two files.
//
// `eri` supplies large_basis/small_basis/v_total (UkbFockMatrixDirect.h's UkbDirectEriSource,
// unchanged). `c_rkb` is the RKB-basis (n_rkb x n_total) natural-spinor coefficient matrix;
// `h_rkb` is the RKB-basis one-electron Hamiltonian; `reps`/`bar`/`n_core`/`n_occ`/`n_vir`/`rdm`/
// `occupations` are exactly `buildPccdFullTwoRdm`'s own arguments (`n_total` is
// `occupations.size()`). Returns the n_total x n_total GenFock matrix, same convention
// `pccdFockMatrix` already uses.
Matrix<std::complex<double>> ukbPccdFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                                const UkbDirectEriSource& eri,
                                                const Matrix<std::complex<double>>& c_rkb,
                                                const std::vector<std::size_t>& reps,
                                                const std::vector<std::size_t>& bar, std::size_t n_core,
                                                std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                                const std::vector<double>& occupations);

}  // namespace rerdmft

#endif  // RERDMFT_UKBPCCDFOCK_H
