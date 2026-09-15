#ifndef RERDMFT_HARTREEEXCHANGEHESSIAN_H
#define RERDMFT_HARTREEEXCHANGEHESSIAN_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds one element of the orbital-rotation Hessian
//   Hess_pq,rs = d^2E/dt_pq dt_rs
// directly from GeneralizedHessian.h's own "boxed" formula, assuming --
// exactly as HartreeExchangeGradient.h's hartreeExchangeFockMatrix does
// for the gradient -- the 1-RDM D is diagonal (`occupations`, general,
// not necessarily idempotent) and the 2-RDM contains ONLY Hartree and
// exchange terms, each with its own, independently supplied, real
// coupling matrix:
//   two_rdm_pqrs = (1/2) [ two_rdm_h(p,q) delta_pr delta_qs
//                          - two_rdm_x(p,q) delta_ps delta_qr ]
// (no PNOF-style L1/L2 pair terms here -- unlike
// hartreeExchangeFockMatrix, this Hessian has not been extended to
// cover them). Substituting this directly into GeneralizedHessian.h's
// boxed formula (see the .cpp for the full derivation, verified against
// the unsimplified, general-2-RDM formula on random data with
// two_rdm_h/two_rdm_x deliberately UNEQUAL, asymmetric, and unrelated
// to any occupation-number formula -- same validation style as
// hartreeExchangeFockMatrix's own) collapses every O(n^2)-cost double
// sum in the general formula to an O(n)-cost single sum, with NO dense
// 2-RDM tensor ever built -- the cheap counterpart to
// GeneralizedHessian.h's generalizedOrbitalHessianElement, exactly as
// hartreeExchangeFockMatrix is to generalizedFockMatrix.
//
// `fock` MUST be `hartreeExchangeFockMatrix(h, eri, occupations,
// two_rdm_h, two_rdm_x)`'s own output, passed in by the caller (not
// recomputed here) since it is needed, unchanged, for every
// (p,q,r,s) element -- this is what "generalized Fock elements are
// used in the definition" means for this cheap path too (Dyall's own
// free-index convention means `fock(A,B) == F_tex(B,A)`, the SAME
// index-swap already documented in GeneralizedHessian.h -- confirmed
// again here since hartreeExchangeFockMatrix computes the identical F
// as generalizedFockMatrix for this ansatz, just cheaper).
//
// `h` and `eri` (physics notation) and `occupations`/`two_rdm_h`/
// `two_rdm_x` follow EXACTLY HartreeExchangeGradient.h's own
// conventions -- see that header.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `two_rdm_h`/`two_rdm_x` are always real
// (Matrix<double>) regardless of T.
template <typename T>
T hartreeExchangeHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t p, std::size_t q,
                                 std::size_t r, std::size_t s);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEHESSIAN_H
