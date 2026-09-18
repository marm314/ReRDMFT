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
// data into the FULL actual-orbital-indexed (n x n) two_rdm_h/x matrices
// Hessian_opt/HartreeExchangeGradient.h's hartreeExchangeFockMatrix
// already expects -- i.e. the "cheap" (O(n^2) to build, O(n^3) once fed
// through that existing O(n^3) Fock-matrix contraction) alternative to
// ever constructing an O(n^4) two-electron-reduced-density-matrix
// tensor for a PNOF functional.
//
// Derivation: Occ_opt/PNOFs.h's own geminal-representative-indexed
// PnofTwoRdm/pnofElectronicEnergy (the already-validated reference this
// file's own energy is cross-checked against) contracts, for every
// ordered pair of DISTINCT geminals (a,b):
//   + two_rdm_h(a,b) * J_ij           [J_ij = eri(i,j,i,j)]
//   - two_rdm_x(a,b) * K_ij           [K_ij = eri(i,j,j,i), MATCHING bar-parity]
//   + two_rdm_l1(a,b) * K_ij          [SAME K_ij, PNOFs.h's row 6]
//   - two_rdm_x(a,b) * L_ij           [L_ij = eri(ibar,j,j,ibar), MISMATCHED
//                                       bar-parity -- relativistic only]
//   + two_rdm_l2(a,b) * L_ij          [SAME L_ij, PNOFs.h's row 7 -- rel. only]
// with two_rdm_l1(a,b) = two_rdm_l2(a,b) = Pi(n_a,n_b) (PNOFs.h's own
// naming -- NOT the same thing as this project's OTHER, unrelated
// Hessian_opt/HartreeExchangeGradient.h "L1/L2" naming for its distinct
// two_rdm(p,pbar,q,qbar)/two_rdm(p,pbar,qbar,q) pairing-integral pattern;
// an earlier version of this file conflated the two identically-named
// but numerically INEQUIVALENT quantities, which happened to still
// reproduce the right NON_REL energy -- real orbitals' extra integral
// permutation symmetry papers over the difference -- but broke for
// X2C/C4_DHF's genuinely complex spinors, where the two patterns
// diverge. The correct reading: Pi is just an ADDITIONAL coefficient on
// the SAME K_ij/L_ij exchange-type integral already used by the
// ordinary exchange term, not a separate pairing integral at all -- so
// it folds directly into two_rdm_x_full, and
// Hessian_opt/HartreeExchangeGradient.h's own L1/L2 slots (`pair_of`,
// `two_rdm_l1`, `two_rdm_l2`) are simply UNUSED for PNOF functionals.
//
// Reading this off in the FULL actual-orbital-quadruple language that
// Hessian_opt/HartreeExchangeGradient.h's own two ansatz patterns need
// -- (P,Q,P,Q) [H] and (P,Q,Q,P) [X] -- and using the SAME "H is
// uniform across every bar-combination" fact this project already
// establishes for plain Hartree/exchange (doc/rel_pnofs.tex's own
// `eq:coulomb-elems` first relation), plus Hermiticity to cover the
// bar-combinations PNOFs.h's own formula does not enumerate explicitly:
//
//   two_rdm_h_full(P,Q) =
//     0                                                if P == Q
//     n_{rep(P)}                                       if P != Q, rep(P) == rep(Q)
//     n_{rep(P)}*n_{rep(Q)} - Delta_{rep(P),rep(Q)}     if rep(P) != rep(Q)
//   two_rdm_x_full(P,Q) =
//     0                                                if P == Q
//     n_{rep(P)}                                       if P != Q, rep(P) == rep(Q)
//     [n_{rep(P)}*n_{rep(Q)} - Delta_{rep(P),rep(Q)}] - Pi_{rep(P),rep(Q)}
//                                                       if rep(P) != rep(Q), MATCHING bar-parity
//     [above], if `relativistic`, else 0                if rep(P) != rep(Q), MISMATCHED bar-parity
//
// where rep(P) is P's own geminal's representative index, "matching
// bar-parity" means P and Q are both a representative or both a bar-
// partner of their respective geminals (pp == qq below), Delta_{a,b} =
// n_a*n_b when a,b are in the SAME PNOF subspace and 0 otherwise
// (Occ_opt/Orb_subspaces.h's intra/inter split), and Pi_{a,b} is
// pnofPiIntra/pnofPiInter/pnofPiInterGnof -- the SAME dispatch
// Occ_opt/PNOFs.h's own buildPnofTwoRdm already uses, so the two are
// guaranteed consistent by construction.
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
// pnofElectronicEnergy on real, converged SCF integrals, for all four
// functionals, both relativistic (X2C/C4_DHF) and non-relativistic
// (NON_REL), before being trusted for the generalized Fock matrix below.
struct PnofFullTwoRdm {
  Matrix<double> two_rdm_h;
  Matrix<double> two_rdm_x;
};

// `relativistic` matches Occ_opt/PNOFs.h's buildPnofTwoRdm's own flag of
// the same name: the mismatched-bar-parity exchange contribution (row 5/
// L_ij and its Pi-pairing counterpart row 7, both needing a genuinely
// relativistic/spinor L_ij = <ibar j|j ibar>-type integral) is left at
// zero when false.
PnofFullTwoRdm buildPnofFullTwoRdm(PnofFunctional functional,
                                    const std::vector<PnofGeminal>& geminals,
                                    const std::vector<double>& occupations, std::size_t n_total,
                                    bool relativistic);

// The "cheap" (O(n^3), no two-electron-reduced-density-matrix tensor
// ever built) generalized Fock matrix for a PNOF functional: unfolds via
// buildPnofFullTwoRdm above, then feeds the result straight into
// Hessian_opt/HartreeExchangeGradient.h's own hartreeExchangeFockMatrix
// (already validated against the fully general, O(n^5)
// GeneralizedFock.h::generalizedFockMatrix) -- H/X only, no `pair_of`/
// L1/L2 needed (see buildPnofFullTwoRdm's own header comment for why).
// Feed the result into Hessian_opt/OrbitalGradient.h's orbitalGradient
// for the orbital-rotation gradient, exactly like hartreeExchangeFockMatrix's
// own existing JK_only callers already do.
template <typename T>
Matrix<T> pnofFockMatrix(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                          const std::vector<PnofGeminal>& geminals,
                          const std::vector<double>& occupations, bool relativistic);

}  // namespace rerdmft

#endif  // RERDMFT_PNOFFOCK_H
