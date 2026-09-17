#ifndef RERDMFT_HARTREEEXCHANGEHESSIAN_H
#define RERDMFT_HARTREEEXCHANGEHESSIAN_H

#include <cstddef>
#include <utility>
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
// GeneralizedHessian.h's generalizedOrbitalHessianElement (O(n^2) per
// element there), exactly as hartreeExchangeFockMatrix is to
// generalizedFockMatrix. This single ELEMENT costs O(n); a full dense
// Hessian tensor (looping p,q,r,s, i.e. O(n^4) elements) would cost
// O(n^5).
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
// `hartreeExchangeHessianElementImag` is the analogous cheap path for
// GeneralizedHessian.h's generalizedOrbitalHessianElementImag (the
// IMAGINARY-step direction, only meaningful for T =
// std::complex<double> -- see that header for the derivation and the
// same "compiles for T = double but not meaningful there" caveat).
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

template <typename T>
T hartreeExchangeHessianElementImag(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                     std::size_t p, std::size_t q, std::size_t r, std::size_t s);

// The MIXED real/imaginary second derivative,
//   Hess^{ty}_pq,rs = d^2E/dt_pq dy_rs
// (t_pq = Re(kappa_pq)-direction on the FIRST pair, y_rs = Im(kappa_rs)-
// direction on the SECOND pair) -- the block GeneralizedHessian.h's own
// derivation history flagged as "not implemented" (only the pure
// real-real and imaginary-imaginary diagonal blocks were covered).
// Derived the same way those two were, from the SAME four raw terms
// (GeneralizedHessian.h's/this file's own internal G_pq,rs): writing
// kappa_pq = t+iy, kappa_qp = -t+iy (t on pair (p,q)) and
// kappa_rs = t'+iy', kappa_sr = -t'+iy' (y' on pair (r,s)) and expanding
// the tex's own E(kappa) ~ (1/2) sum kappa_pq kappa_rs G_pq,rs quadratic
// term, the coefficient of t*y' (as a REAL number, after pulling out an
// overall factor of i -- see the .cpp) is:
//   Hess^{ty}_pq,rs = i * [ G_rs,pq + G_rs,qp - G_sr,pq - G_sr,qp ]
// -- note the "y" pair (r,s) sits in the flipping (first) slot of each
// raw term and the "t" pair (p,q) in the fixed (second) slot, the
// OPPOSITE of the naive reading of the symbol names above. This was
// caught by, not just avoided by, the numerical check below (an
// earlier version of this derivation had the two pairs' roles swapped
// and reproduced d^2E/dy_pq dt_rs instead -- exactly right in
// magnitude, wrong in which pair got which direction -- until the
// finite-difference comparison caught it): a real-vs-imaginary-step
// mixed-direction 2D finite difference (t on (p,q), y on (r,s)) against
// this exact function, and the complementary check with the roles
// swapped against Hess^{ty}_rs,pq, at BOTH a converged (water_debug.inp)
// and a deliberately non-converged, 1-SCF-iteration
// (water_debug_1iter.inp) DHF density, matching to O(step^2) (down to
// ~1e-8 at step=1e-3) in both cases. By the same double-commutator-
// Hermiticity argument as the other two blocks (K_pq = E_pq-E_qp and
// K'_rs = i(E_rs+E_sr) are BOTH anti-Hermitian, so [[H,K_pq],K'_rs] is
// Hermitian too), this is exactly real for physical (Hermitian
// h/eri/D/2-RDM) input -- confirmed numerically alongside the
// magnitude check (imaginary part ~1e-16-1e-17, floating-point
// roundoff, at both test points).
//
// The OTHER mixed direction, Hess^{yt}_pq,rs = d^2E/dy_pq dt_rs, is NOT
// a new function: by Schwarz's theorem (mixed partials commute for any
// smooth function, regardless of which physical pair each one belongs
// to), Hess^{yt}_pq,rs = Hess^{ty}_rs,pq -- i.e. just call this same
// function with the two pairs swapped. Confirmed numerically, not just
// asserted, at both test points above.
template <typename T>
T hartreeExchangeHessianElementMixed(const Matrix<T>& h, const Tensor4<T>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                      std::size_t p, std::size_t q, std::size_t r, std::size_t s);

// The independent real-step orbital-rotation parameters are exactly
// the (p,q) pairs with p > q (see OrbitalGradient.h) -- this lists them
// all, in a FIXED order, for a basis of dimension `n`: pair I is
// (pair[I].first, pair[I].second), first > second. Used to give
// `hartreeExchangeHessianMatrix`'s rows/columns a definite meaning.
std::vector<std::pair<std::size_t, std::size_t>> hessianPairIndices(std::size_t n);

// Builds the FULL, dense orbital-rotation Hessian, indexed by the
// INDEPENDENT rotation pairs `pair_indices` (typically
// `hessianPairIndices(h.rows())`) rather than by individual orbital
// indices: element (I,J) is `hartreeExchangeHessianElement` evaluated
// at (p,q) = pair_indices[I], (r,s) = pair_indices[J]. This is exactly
// the real-symmetric (T = double) or complex-Hermitian (T =
// std::complex<double>) matrix to diagonalize (LinearAlgebra.h's
// diagonalizeSymmetric/diagonalizeHermitian) to check whether a
// converged SCF solution is a genuine minimum (all eigenvalues >= 0)
// or a saddle point (some strictly negative) with respect to real
// orbital rotations -- e.g. DHF's admission of rotations into the
// negative-energy branch is expected to show up as negative
// eigenvalues here, unlike NON_REL's genuine minimum.
//
// EXPENSIVE: `pair_indices.size()` is O(n^2), and each element costs
// O(n) (hartreeExchangeHessianElement's own cost), so this costs
// O(n^5) overall, plus whatever the caller's subsequent diagonalization
// costs (O(n^6) for a dense eigensolver on the O(n^2)-dimensional
// result) -- usable for a small-to-moderate basis (e.g. water/STO-3G),
// not a production Newton-Raphson step on a realistic basis.
template <typename T>
Matrix<T> hartreeExchangeHessianMatrix(
    const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEHESSIAN_H
