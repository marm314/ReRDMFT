#ifndef RERDMFT_PNOFFOCK_H
#define RERDMFT_PNOFFOCK_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Orb_subspaces.h"
#include "PNOFs.h"
#include "Tensor4.h"

namespace rerdmft {

// Unfolds a PNOF functional's GEMINAL-representative-indexed occupation
// data into the FULL actual-orbital-indexed (n x n) two_rdm_h/x/l1/l2
// matrices Hessian_opt/HartreeExchangeGradient.h's hartreeExchangeFockMatrix
// (and, as of this piece, Hessian_opt/HartreeExchangeHessian.h's
// hartreeExchangeHessianElement) already expect -- i.e. the "cheap"
// (O(n^2) to build, O(n^3)/O(n) once fed through those existing cheap
// contractions) alternative to ever constructing an O(n^4) two-electron-
// reduced-density-matrix tensor for a PNOF functional.
//
// Derivation: doc/rel_pnofs.tex's own Eqs.~98a-f (the general,
// functional-independent reconstruction of Appendix D), read LITERALLY
// (not folded into an equivalent-for-the-energy-only form -- an earlier
// version of this file did exactly that, which happened to reproduce
// the correct energy AND generalized Fock matrix exactly [confirmed via
// an independent doc-literal dense-tensor cross-check, both to machine
// precision] but was proven WRONG for the generalized Hessian
// specifically -- see this file's own git history/project memory for
// the full story). Reading Eqs.~98a-f off directly for every actual-
// orbital quadruple:
//
//   two_rdm_h_full(P,Q) =
//     0                                                if P == Q
//     n_{rep(P)}                                       if P != Q, rep(P) == rep(Q)
//     n_{rep(P)}*n_{rep(Q)} - Delta_{rep(P),rep(Q)}     if rep(P) != rep(Q)
//   two_rdm_x_full(P,Q) =
//     0                                                if P == Q
//     n_{rep(P)}                                       if P != Q, rep(P) == rep(Q)
//     n_{rep(P)}*n_{rep(Q)} - Delta_{rep(P),rep(Q)}     if rep(P) != rep(Q), MATCHING bar-parity
//     [above], if `relativistic`, else 0                if rep(P) != rep(Q), MISMATCHED bar-parity
//   two_rdm_l1_full(P,Q) =
//     +Pi_{rep(P),rep(Q)}/4   if rep(P) != rep(Q), MATCHING bar-parity
//     -Pi_{rep(P),rep(Q)}/4   if rep(P) != rep(Q), MISMATCHED bar-parity
//   (HALF the Gamma element +-Pi/2 of Eqs. 98: each pair-transfer tuple is
//   enumerated twice by the L1+L2 pattern and the pair term carries weight 1 --
//   see buildPnofFullTwoRdm's .cpp comment. An earlier version stored +-Pi/2,
//   which doubled the pair energy relative to Occ_opt/PNOFs.h's
//   pnofElectronicEnergy; corrected 2026-09-20, energies now agree to machine
//   precision, also for the exact 2-electron pair-CI energy.)
//     0                                                otherwise
//   two_rdm_l2_full(P,Q) = -two_rdm_l1_full(P,Q)   [always, at every (P,Q)]
//
// where rep(P) is P's own geminal's representative index, "matching
// bar-parity" means P and Q are both a representative or both a bar-
// partner of their respective geminals, Delta_{a,b} = n_a*n_b when a,b
// are in the SAME PNOF subspace (Om_p) and 0 otherwise
// (Occ_opt/Orb_subspaces.h's own intra/inter split), and Pi_{a,b} is
// pnofPiIntra (same subspace) or pnofPiInter/pnofPiInterGnof (different
// subspaces) -- EXACTLY the same dispatch Occ_opt/PNOFs.h's own
// buildPnofTwoRdm already uses for the geminal-indexed energy formula,
// so the two are guaranteed consistent by construction.
//
// two_rdm_l1/l2 are built UNCONDITIONALLY (no `relativistic` gate),
// unlike two_rdm_x's own mismatched-bar-parity entry: Eqs.~98a-f
// themselves carry no such gate (they are "common in form to all four
// functionals" and to both the relativistic and non-relativistic case
// -- doc/rel_pnofs.tex never singles out an L1/L2-only relativistic
// restriction the way it does for the mismatched-parity EXCHANGE term).
// For NON_REL, the actual-orbital integrals these two_rdm_l1/l2
// coefficients multiply (Hessian_opt/HartreeExchangeGradient.h's own
// `eri(p,pbar,q,qbar)`/`eri(p,pbar,qbar,q)` pattern, `pbar`/`qbar`
// being NON_REL's own opposite-spin partner) are exactly zero by an
// ordinary spin-selection rule whenever the pattern is spin-forbidden,
// so this does not change any already-validated NON_REL result --
// confirmed numerically, not just argued, before trusting this here.
//
// `n_total` is `h.rows()`/`eri.dim0()` in the CALLER's own full
// (possibly larger than the geminal-covered "active" window, e.g.
// C4_DHF's excluded negative-energy branch) orbital space --
// `geminals[*].i`/`.ibar` must already be expressed in THIS SAME full
// index space (main.cpp's own convention: buildPnofGeminals's raw,
// active-window-relative indices shifted by `n_inactive_below` -- see
// main.cpp's own buildPnofFunctionalReport). Actual orbitals not
// covered by any geminal (deep, permanently unoccupied virtuals, or an
// entirely inactive/excluded block) are left at their default-
// constructed zero in every returned matrix -- correct, since their
// occupation is always exactly 0 there.
//
// Verified against Occ_opt/PNOFs.h's independently-implemented
// pnofElectronicEnergy (energy), and against a doc-Eqs.98a-f-literal
// dense 2-RDM fed through both Hessian_opt/GeneralizedFock.h's
// generalizedFockMatrix (Fock) and Hessian_opt/GeneralizedHessian.h's
// generalizedOrbitalHessianElement (Hessian), on real, converged SCF
// integrals, for GNOF, both relativistic and non-relativistic, before
// being trusted here.
struct PnofFullTwoRdm {
  Matrix<double> two_rdm_h;
  Matrix<double> two_rdm_x;
  Matrix<double> two_rdm_l1;
  Matrix<double> two_rdm_l2;
};

// `relativistic` matches Occ_opt/PNOFs.h's buildPnofTwoRdm's own flag of
// the same name: the mismatched-bar-parity EXCHANGE contribution (row 5/
// L_ij, needing a genuinely relativistic/spinor L_ij = <ibar j|j ibar>-
// type integral) is left at zero when false -- two_rdm_l1/l2 are NOT
// gated by it (see the struct's own header comment above for why).
PnofFullTwoRdm buildPnofFullTwoRdm(PnofFunctional functional,
                                    const std::vector<PnofGeminal>& geminals,
                                    const std::vector<double>& occupations, std::size_t n_total,
                                    bool relativistic);

// The `pair_of` array Hessian_opt/HartreeExchangeGradient.h's own L1/L2
// ansatz needs (`pair_of[p]` = p's bar/Kramers partner), built directly
// from `geminals` over the SAME full `n_total`-sized index space as
// `buildPnofFullTwoRdm` -- indices not covered by any geminal map to
// themselves (never read, since two_rdm_l1/l2 are zero there).
std::vector<std::size_t> buildPnofPairOf(const std::vector<PnofGeminal>& geminals,
                                          std::size_t n_total);

// The "cheap" (O(n^3), no two-electron-reduced-density-matrix tensor
// ever built) generalized Fock matrix for a PNOF functional: unfolds via
// buildPnofFullTwoRdm/buildPnofPairOf above, then feeds the result
// straight into Hessian_opt/HartreeExchangeGradient.h's own
// hartreeExchangeFockMatrix (already validated against the fully
// general, O(n^5) GeneralizedFock.h::generalizedFockMatrix). Feed the
// result into Hessian_opt/OrbitalGradient.h's orbitalGradient for the
// orbital-rotation gradient, and Hessian_opt/PnofHessian.h's
// pnofHessianElement/pnofHessianMatrix for the Hessian.
template <typename T, typename Eri>
Matrix<T> pnofFockMatrix(PnofFunctional functional, const Matrix<T>& h, const Eri& eri,
                          const std::vector<PnofGeminal>& geminals,
                          const std::vector<double>& occupations, bool relativistic);

}  // namespace rerdmft

#endif  // RERDMFT_PNOFFOCK_H
