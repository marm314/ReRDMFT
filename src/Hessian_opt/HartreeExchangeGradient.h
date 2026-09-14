#ifndef RERDMFT_HARTREEEXCHANGEGRADIENT_H
#define RERDMFT_HARTREEEXCHANGEGRADIENT_H

#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the generalized Fock matrix directly from Dyall & Faegri
// Eq. (8.30)'s rewritten form (see GeneralizedFock.h),
//   F_pq = sum_r h_rp D_rq + 2 sum_rst <sr|tp> two_rdm_srtq,
// SIMPLIFYING THE SUMS under two assumptions: the 1-RDM D is diagonal
// (the natural-orbital/natural-spinor basis, D_rq = n_r delta_rq, n_r
// the occupation numbers -- general and possibly fractional, NOT
// necessarily idempotent), and the 2-RDM contains ONLY Hartree,
// exchange, and opposite-spin-exchange terms, i.e. the same closed
// form already used to validate GeneralizedFock.h:
//   two_rdm_pqrs = (1/2) n_p n_q (delta_pr delta_qs - delta_ps delta_qr).
// Substituting both directly into the sums above collapses the O(n^5)
// triple sum to a single sum over s per (p,q) pair -- O(n^3) overall
// given h/eri already in the natural-orbital/spinor basis, and NO
// 2-RDM tensor is ever built:
//   FockLike(r,p) = h(r,p) + sum_s n_s * (<r s|p s> - <r s|s p>)
//   F_pq = n_q * FockLike(q,p)
// (FockLike is exactly "Hartree - exchange", built directly from the
// occupation numbers and the two-electron integrals -- this does NOT
// call, or rely on, any ordinary HF/DHF Fock-matrix routine; see the
// .cpp for the full derivation, verified against the unsimplified
// generalizedFockMatrix formula on random data, both with fractional
// and with idempotent occupations). The opposite-spin-exchange
// contribution (whatever it evaluates to for a given basis/spin
// structure) is already included in the unrestricted sum over s -- it
// is not a separate term requiring separate bookkeeping here.
//
// `h` and `eri` (physics notation: eri(A,B,C,D) == <A B|C D>) must
// already be expressed in the SAME diagonalizing (natural-orbital or
// natural-spinor) basis as `occupations` -- building that basis (an
// AO -> natural-orbital transform) is the caller's job, e.g.
// NON_REL/MoIntegralTransform.h or C4_DHF/RkbMoTransform.h.
//
// Ordinary HF/DHF is the special case occupations[p] in {0,1} (spin-
// orbital occupation numbers); feed a converged HF/DHF density's own
// occupations in as a numerical test against the already-validated
// HF/DHF gradient (see main.cpp).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. Feed the resulting F into OrbitalGradient.h's
// orbitalGradient to get g_pq = F_qp - conj(F_pq).
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEGRADIENT_H
