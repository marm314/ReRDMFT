#ifndef RERDMFT_PCCDFOCK_H
#define RERDMFT_PCCDFOCK_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"
#include "pCCD.h"

namespace rerdmft {

// Unfolds pCCD's pair-indexed 1-/2-RDM (Occ_opt/pCCD.h's `PccdRdm`, over the active
// occupied+virtual window) into the FULL actual-orbital-indexed (n_total x n_total)
// two_rdm_h/x/l1/l2 matrices Hessian_opt/HartreeExchangeGradient.h's
// hartreeExchangeFockMatrix/hartreeExchangeEnergy expect -- mirroring EXACTLY the role
// Hessian_opt/PnofFock.h's buildPnofFullTwoRdm plays for PNOF, and feeding the SAME
// generic contraction code (no new Fock machinery needed): a frozen-core pair is simply
// one with x == 0 identically (no t/z amplitude at all), which is already what
// `Q_ij = 1 - x_i - x_j`/`D_ij = x^j_i` reduce to at x = 0 -- a frozen core pair's
// Q with anything else is just the plain, uncorrelated n_p*n_q product, and its D with
// anything is exactly 0, with NO separate formula needed.
//
// Mapping (tex doc, Sec. "Density matrices"), for two DISTINCT pair representatives p, q
// (p, q ranging over core ++ active-occupied ++ active-virtual; `rep(P)` = P's own pair's
// representative, matching PnofFock.h's own terminology):
//   two_rdm_h_full(P,Q) = two_rdm_x_full(P,Q) =
//     0                            if P == Q
//     n_{rep(P)}                   if P != Q, rep(P) == rep(Q)            (block (a), r=p)
//     Q_{rep(P),rep(Q)}            if rep(P) != rep(Q), EVERY bar-parity  (block (b))
//   two_rdm_l1_full(P,Q) =
//     +D_{rep(Q),rep(P)}/4         if rep(P) != rep(Q), MATCHING bar-parity   (block (a), r!=p)
//     -D_{rep(Q),rep(P)}/4         if rep(P) != rep(Q), MISMATCHED bar-parity
//     0                            if rep(P) == rep(Q) (the r=p case is already in
//                                  two_rdm_h/x above, not here -- see PnofFock.h's own
//                                  identical convention)
//   two_rdm_l2_full(P,Q) = -two_rdm_l1_full(P,Q)   (always; empty/unused for NON_REL,
//                                                    exactly like PnofFock.h)
// The /4 (not the naive /2 a direct Gamma_{p,pbar,r,rbar} = D_rp/2 lookup would give) is
// EXACTLY PnofFock.cpp's own Pi/4-vs-Pi/2 story: hartreeExchangeEnergy/Fock's own L1+L2
// double loop visits every pair-transfer tuple TWICE (once as (P,Q)'s L1, once as its
// bar-parity-flipped L2 partner), doubling the naive value -- confirmed empirically
// (tests/test_pccd_fock.cpp), not re-derived a priori.
// "MATCHING bar-parity" means P, Q are both their own pair's representative, or both the
// partner -- same meaning as PnofFock.h's own `matching_parity`. Unlike PNOF's Pi_pq
// (symmetric), pCCD's D_pq is NOT symmetric (D_pq != D_qp in general -- tex doc, Sec.
// "Density matrices"), so two_rdm_l1_full(P,Q) and two_rdm_l1_full(Q,P) are built from
// D's TWO different entries, not one shared value.
//
// `reps`/`bar` are the SAME combined (core ++ occupied ++ virtual) actual-orbital index
// lists `buildPccdCoefficients` takes; `occupations` is the FULL (n_total) occupation
// vector (frozen core at 1, deep-virtual outside the combined list at 0, active
// occupied/virtual at `rdm.n_occ`/`rdm.n_vir` -- the caller's job to assemble, exactly
// like main.cpp already does for JK_only/PNOF).
struct PccdFullTwoRdm {
  Matrix<double> two_rdm_h;
  Matrix<double> two_rdm_x;
  Matrix<double> two_rdm_l1;
  Matrix<double> two_rdm_l2;
};

PccdFullTwoRdm buildPccdFullTwoRdm(const std::vector<std::size_t>& reps,
                                    const std::vector<std::size_t>& bar, std::size_t n_core,
                                    std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                    const std::vector<double>& occupations, std::size_t n_total);

// pair_of over the full n_total space: `bar[a]` for every combined-list representative
// `reps[a]`, identity (p <-> p) everywhere else (PnofFock.h's buildPnofPairOf convention)
// -- "everywhere else" never actually arises here since every active-window actual
// orbital index is either a `reps` entry or a `bar` entry by construction, but the
// identity fallback keeps this safe for a caller that passes a window not covering all
// of n_total (e.g. C4_DHF's excluded negative-energy branch, later).
std::vector<std::size_t> buildPccdPairOf(const std::vector<std::size_t>& reps,
                                          const std::vector<std::size_t>& bar,
                                          std::size_t n_total);

// buildPccdFullTwoRdm + buildPccdPairOf, then hartreeExchangeFockMatrix -- the pCCD
// counterpart of PnofFock.h's pnofFockMatrix.
template <typename T, typename Eri>
Matrix<T> pccdFockMatrix(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                          const std::vector<std::size_t>& bar, std::size_t n_core,
                          std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                          const std::vector<double>& occupations);

}  // namespace rerdmft

#endif  // RERDMFT_PCCDFOCK_H
