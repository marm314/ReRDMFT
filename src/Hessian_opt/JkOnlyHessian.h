#ifndef RERDMFT_JKONLYHESSIAN_H
#define RERDMFT_JKONLYHESSIAN_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Cheap orbital-rotation Hessian element for JK_only functionals
// (Occ_opt/JK_only.h), Hess_pq,rs = d^2E/dkappa_pq dkappa_rs, for the SAME
// diagonal-D, Hartree/exchange-only 2-RDM ansatz HartreeExchangeHessian.h's
// hartreeExchangeHessianElement uses with an empty pair_of:
//   two_rdm_pqrs = (1/2) [ two_rdm_h(p,q) delta_pr delta_qs
//                          - two_rdm_x(p,q) delta_ps delta_qr ]
// with no L1/L2 pair-term extension at all -- JK_only's own two_rdm_h/x
// (jkHartreeCoupling/jkExchangeCoupling) carry no pairing structure,
// unlike PNOF's.
//
// Deliberately a SEPARATE, self-contained implementation from
// HartreeExchangeHessian.h/PnofHessian.h, not a thin wrapper around
// them, even though the underlying formula is currently identical to
// hartreeExchangeHessianElement's own H/X-only path. The relativistic
// (T=complex<double>) version of this formula is confirmed WRONG for
// JK_only's non-idempotent, non-Kramers-bar-uniform 2-RDM (project
// memory: project_jk_only_relativistic_hessian_gap.md) and is expected
// to need its own fix/derivation, independent of PNOF's already-
// validated, already-correct Hessian. Forking this out FIRST, before
// any such fix is attempted, means future changes made while chasing
// that bug cannot accidentally regress PnofFock.h/PnofHessian.h's own
// Hessian (per explicit user instruction: have two different functions
// so solving the JK_only problem does not destroy the correct PNOF
// Hessian).
//
// `h`/`eri` (physics notation), `occupations`, `two_rdm_h`/`two_rdm_x`
// follow HartreeExchangeGradient.h's own conventions; `fock` MUST be
// `hartreeExchangeFockMatrix(h, eri, occupations, two_rdm_h,
// two_rdm_x)`'s own output (same Dyall free-index-swap convention
// documented in HartreeExchangeHessian.h/GeneralizedHessian.h --
// fock(A,B) == F_tex(B,A)). Works for either a real (T=double, the
// only case currently validated/used in production) or complex (T=
// std::complex<double>, NOT YET validated for a genuinely non-
// idempotent, non-bar-uniform 2-RDM -- see the caveat above) T.
template <typename T>
T jkOnlyHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, const Matrix<T>& fock, std::size_t p,
                        std::size_t q, std::size_t r, std::size_t s);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYHESSIAN_H
