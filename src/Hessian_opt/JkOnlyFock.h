#ifndef RERDMFT_JKONLYFOCK_H
#define RERDMFT_JKONLYFOCK_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Generalized Fock matrix and electronic energy for JK_only functionals
// (Occ_opt/JK_only.h), for the SAME diagonal-D, Hartree/exchange-only
// 2-RDM ansatz HartreeExchangeGradient.h's hartreeExchangeFockMatrix/
// hartreeExchangeEnergy use with an empty pair_of:
//   two_rdm_pqrs = (1/2) [ two_rdm_h(p,q) delta_pr delta_qs
//                          - two_rdm_x(p,q) delta_ps delta_qr ]
// with no L1/L2 pair-coupling extension at all -- JK_only's own
// two_rdm_h/x (jkHartreeCoupling/jkExchangeCoupling) carry no pairing
// structure, unlike PNOF's.
//
// Deliberately a SEPARATE, self-contained implementation from
// HartreeExchangeGradient.h/PnofFock.h, not a thin wrapper around them,
// mirroring the same fork already done for the Hessian
// (Hessian_opt/JkOnlyHessian.h). JK_only's relativistic Hessian is
// already confirmed broken for its non-idempotent, non-Kramers-bar-
// uniform 2-RDM (project memory: project_jk_only_relativistic_hessian_
// gap.md); building/testing a corrected, genuinely GENERALIZED Fock for
// JK_only is expected to need experimentation of the same kind. Forking
// this out FIRST, before any such work starts, means future changes
// here cannot accidentally regress PnofFock.h/PnofHessian.h's own,
// already-validated Fock/gradient/energy (per explicit user
// instruction: dedicated functions per functional family so solving
// the JK_only problem never destroys what already works for PNOFs).
//
// `h`/`eri` (physics notation) and `occupations`/`two_rdm_h`/
// `two_rdm_x` follow HartreeExchangeGradient.h's own conventions. Works
// for either a real (T=double) or complex (T=std::complex<double>)
// orbital basis.
template <typename T>
Matrix<T> jkOnlyFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                            const std::vector<double>& occupations,
                            const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);

// The electronic energy (one- + two-electron; caller adds nuclear
// repulsion separately) for the same ansatz as jkOnlyFockMatrix.
// Returned as `double` even for T = std::complex<double> (via
// std::real, valid for real T too).
template <typename T>
double jkOnlyEnergy(const Matrix<T>& h, const Tensor4<T>& eri,
                     const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                     const Matrix<double>& two_rdm_x);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYFOCK_H
