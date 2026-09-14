#ifndef RERDMFT_HARTREEEXCHANGEGRADIENT_H
#define RERDMFT_HARTREEEXCHANGEGRADIENT_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the generalized Fock matrix directly from Dyall & Faegri
// Eq. (8.30)'s rewritten form (see GeneralizedFock.h),
//   F_pq = sum_r h_rp D_rq + 2 sum_rst <sr|tp> two_rdm_srtq,
// assuming the 1-RDM D is diagonal (the natural-orbital/natural-spinor
// basis, D_rq = n_r delta_rq -- `occupations`, general and possibly
// fractional, NOT necessarily idempotent) and the 2-RDM contains ONLY
// Hartree, exchange, and (optionally, for Piris-style PNOF functionals)
// two further "L1/L2" pair-coupling terms, each with its OWN,
// independently supplied, real coupling matrix:
//   two_rdm_pqrs = (1/2) [ two_rdm_h(p,q) delta_pr delta_qs
//                          - two_rdm_x(p,q) delta_ps delta_qr ]
//                  + two_rdm_l1(p,q) delta_{r,pbar} delta_{s,qbar}   (*)
//                  + two_rdm_l2(p,q) delta_{r,pbar} delta_{s,qbar}'  (**)
// where pbar/qbar denote each orbital's PAIR PARTNER (`pair_of`) --
// PNOF's degenerate, same-occupation-number orbital pairs, which in the
// nonrelativistic limit become the ordinary spin-up/spin-down partners
// of one spatial orbital; term (*) is two_rdm(p,pbar,q,qbar) = L1(p,q)
// (bra pair (p,pbar), ket pair (q,qbar) in the SAME order) and term
// (**) is two_rdm(p,pbar,qbar,q) = L2(p,q) (ket pair order SWAPPED).
// None of two_rdm_h/two_rdm_x/two_rdm_l1/two_rdm_l2 need be equal,
// symmetric, or tied to any particular formula -- that is entirely the
// caller's choice of RDMFT functional; ordinary HF/DHF is simply the
// case two_rdm_h(p,q) = two_rdm_x(p,q) = occupations[p]*occupations[q]
// with idempotent occupations and no L1/L2 terms at all (see main.cpp
// for that as a numerical test) -- `pair_of`/`two_rdm_l1`/`two_rdm_l2`
// are optional and, left at their defaults (all empty), contribute
// nothing, so existing Hartree+exchange-only callers are unaffected.
//
// Substituting all of the above directly into the sums (see the .cpp
// for the full derivation, verified against the unsimplified formula
// on random data with two_rdm_h/two_rdm_x/two_rdm_l1/two_rdm_l2 all
// deliberately UNEQUAL, asymmetric, and unrelated to any occupation-
// number formula) gives, with NO 2-RDM tensor ever built:
//   F_pq = occupations[q] * h(q,p)
//          + sum_s [ two_rdm_h(s,q) * <q s|p s> - two_rdm_x(q,s) * <q s|s p> ]
//          + 2 * sum_s <s sbar|qbar p> * [ two_rdm_l1(s,qbar) + two_rdm_l2(s,q) ]
// (the last line only evaluated when `pair_of` is non-empty).
//
// `h` and `eri` (physics notation: eri(A,B,C,D) == <A B|C D>), and
// `occupations`/`two_rdm_h`/`two_rdm_x`/`two_rdm_l1`/`two_rdm_l2`, must
// all be expressed in the SAME diagonalizing (natural-orbital or
// natural-spinor) basis -- building that basis (an AO -> natural-
// orbital transform) is the caller's job, e.g.
// NON_REL/MoIntegralTransform.h or C4_DHF/RkbMoTransform.h. `pair_of`
// (length n, pair_of[p] = index of p's pair partner, an involution:
// pair_of[pair_of[p]] == p) is also the caller's job to build, since
// the pairing convention differs by basis (e.g. NON_REL/
// ClosedShellSpinOrbitals.h's block offset p <-> p+n_spatial, vs
// C4_DHF's adjacent Kramers pairs p <-> p xor 1).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `two_rdm_h`/`two_rdm_x`/`two_rdm_l1`/
// `two_rdm_l2` are always real (Matrix<double>) regardless of T, since
// occupation-number-derived coupling coefficients are always real.
// Feed the resulting F into OrbitalGradient.h's orbitalGradient to get
// g_pq = 2*(F_qp - conj(F_pq)) (see that header for the factor of 2).
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x,
                                     const std::vector<std::size_t>& pair_of = {},
                                     const Matrix<double>& two_rdm_l1 = Matrix<double>(),
                                     const Matrix<double>& two_rdm_l2 = Matrix<double>());

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEGRADIENT_H
