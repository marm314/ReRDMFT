#ifndef RERDMFT_HARTREEEXCHANGEGRADIENT_H
#define RERDMFT_HARTREEEXCHANGEGRADIENT_H

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
// Hartree and exchange terms, each with its OWN, independently
// supplied, real coupling matrix:
//   two_rdm_pqrs = (1/2) [ two_rdm_h(p,q) delta_pr delta_qs
//                          - two_rdm_x(p,q) delta_ps delta_qr ]
// (two_rdm_h and two_rdm_x need not be equal, symmetric, or tied to any
// particular formula -- that is entirely the caller's choice of RDMFT
// functional; ordinary HF/DHF is simply the case two_rdm_h(p,q) =
// two_rdm_x(p,q) = occupations[p]*occupations[q] with idempotent
// occupations, see main.cpp for that as a numerical test).
//
// Substituting both directly into the sums above (see the .cpp for the
// full derivation, verified against the unsimplified formula on random
// data with two_rdm_h and two_rdm_x deliberately UNEQUAL and
// asymmetric) gives, with NO 2-RDM tensor ever built:
//   F_pq = occupations[q] * h(q,p)
//          + sum_s [ two_rdm_h(s,q) * <q s|p s> - two_rdm_x(q,s) * <q s|s p> ]
//
// `h` and `eri` (physics notation: eri(A,B,C,D) == <A B|C D>), and
// `occupations`/`two_rdm_h`/`two_rdm_x`, must all be expressed in the
// SAME diagonalizing (natural-orbital or natural-spinor) basis --
// building that basis (an AO -> natural-orbital transform) is the
// caller's job, e.g. NON_REL/MoIntegralTransform.h or
// C4_DHF/RkbMoTransform.h.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `two_rdm_h`/`two_rdm_x` are always real
// (Matrix<double>) regardless of T, since occupation-number-derived
// coupling coefficients are always real. Feed the resulting F into
// OrbitalGradient.h's orbitalGradient to get g_pq = F_qp - conj(F_pq).
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEGRADIENT_H
