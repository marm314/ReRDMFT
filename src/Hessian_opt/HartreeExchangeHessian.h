#ifndef RERDMFT_HARTREEEXCHANGEHESSIAN_H
#define RERDMFT_HARTREEEXCHANGEHESSIAN_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds one element of the orbital-rotation Hessian
//   Hess_pq,rs = d^2E/dkappa_pq dkappa_rs
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
// Optional `pair_of`/`two_rdm_l1`/`two_rdm_l2` (default empty ==
// unused, exactly like `hartreeExchangeFockMatrix`'s own parameters of
// the same name/meaning) extend the 2-RDM ansatz with the SAME PNOF-
// style pair terms, `two_rdm(p,pbar,q,qbar) += two_rdm_l1(p,q)`,
// `two_rdm(p,pbar,qbar,q) += two_rdm_l2(p,q)`. **This extension is
// REQUIRED, not optional in practice, for any 2-RDM that genuinely has
// nonzero L1/L2 pairing entries** (e.g. Hessian_opt/PnofFock.h's own
// PNOF 2-RDM): substituting ONLY the H/X part of the ansatz into
// GeneralizedHessian.h's boxed Eq. 9 and leaving out a physically
// nonzero L1/L2 part gives an INCORRECT Hessian element, even though
// the SAME omission has NO effect on the Fock matrix/gradient/energy
// (an exact integral identity for Kramers-paired spinors,
// `eri(p,pbar,q,qbar) = eri(p,q,q,p)`, makes the Fock/energy blind to
// whether Pi-pairing weight sits at the L1/L2 tensor position or is
// folded into `two_rdm_x` instead -- Eq. 9's Hessian, fixing all four
// indices explicitly, is not similarly forgiving). Discovered and
// derived by the SAME direct-substitution method as the base H/X
// formula: substitute the FULL ansatz's `two_rdm(p,pbar,q,qbar)=
// two_rdm_l1(p,q)`/`two_rdm(p,pbar,qbar,q)=two_rdm_l2(p,q)` into each
// of GeneralizedHessian.h's own six raw exchange-type sums, collapsing
// the double sums the same way the base derivation did (see the .cpp);
// verified against `generalizedOrbitalHessianElement` fed an
// EXPLICIT, densely-built Tensor4 using the identical full ansatz, on
// random data (both `T=double` and `T=complex<double>`, with
// `two_rdm_h`/`two_rdm_x`/`two_rdm_l1`/`two_rdm_l2` deliberately
// unequal, asymmetric, and unrelated to any occupation-number formula
// -- the same validation style as the base H/X-only formula's own),
// to machine precision.
//
// `fock` must still be `hartreeExchangeFockMatrix`'s own output, called
// WITH the same `pair_of`/`two_rdm_l1`/`two_rdm_l2` this time (its
// existing L1/L2 extension, already validated -- see that header).
//
// `hartreeExchangeHessianElementImag` is the analogous cheap path for
// GeneralizedHessian.h's generalizedOrbitalHessianElementImag (the
// IMAGINARY-step direction, only meaningful for T =
// std::complex<double> -- see that header for the derivation and the
// same "compiles for T = double but not meaningful there" caveat). It
// and the Mixed element below now ALSO accept the optional
// `pair_of`/`two_rdm_l1`/`two_rdm_l2` (all three blocks are fixed sign
// combinations of the same bare G_pq,rs, whose L1/L2 part is
// rawL1L2HessianTerm); validated for PNOF against finite differences of
// the gradient (see main.cpp's pnofJointBlocksReport).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `two_rdm_h`/`two_rdm_x`/`two_rdm_l1`/
// `two_rdm_l2` are always real (Matrix<double>) regardless of T.
template <typename T, typename Eri>
T hartreeExchangeHessianElement(const Matrix<T>& h, const Eri& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t p, std::size_t q,
                                 std::size_t r, std::size_t s,
                                 const std::vector<std::size_t>& pair_of = {},
                                 const Matrix<double>& two_rdm_l1 = Matrix<double>(),
                                 const Matrix<double>& two_rdm_l2 = Matrix<double>());

// Optional `pair_of`/`two_rdm_l1`/`two_rdm_l2` (default empty == unused),
// same meaning as for hartreeExchangeHessianElement: they add the PNOF
// pair-term contribution to the BARE G_pq,rs that both this function and
// the Mixed one below combine (see rawL1L2HessianTerm in the .cpp).
// Required whenever the 2-RDM has nonzero L1/L2 entries.
template <typename T, typename Eri>
T hartreeExchangeHessianElementImag(const Matrix<T>& h, const Eri& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                     std::size_t p, std::size_t q, std::size_t r, std::size_t s,
                                     const std::vector<std::size_t>& pair_of = {},
                                     const Matrix<double>& two_rdm_l1 = Matrix<double>(),
                                     const Matrix<double>& two_rdm_l2 = Matrix<double>());

// The MIXED real/imaginary second derivative,
//   Hess^{ty}_pq,rs = d^2E/dkappa_pq dy_rs
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
// and reproduced d^2E/dy_pq dkappa_rs instead -- exactly right in
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
// The OTHER mixed direction, Hess^{yt}_pq,rs = d^2E/dy_pq dkappa_rs, is NOT
// a new function: by Schwarz's theorem (mixed partials commute for any
// smooth function, regardless of which physical pair each one belongs
// to), Hess^{yt}_pq,rs = Hess^{ty}_rs,pq -- i.e. just call this same
// function with the two pairs swapped. Confirmed numerically, not just
// asserted, at both test points above.
template <typename T, typename Eri>
T hartreeExchangeHessianElementMixed(const Matrix<T>& h, const Eri& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                      std::size_t p, std::size_t q, std::size_t r, std::size_t s,
                                      const std::vector<std::size_t>& pair_of = {},
                                      const Matrix<double>& two_rdm_l1 = Matrix<double>(),
                                      const Matrix<double>& two_rdm_l2 = Matrix<double>());

// The TRUE (symmetric) second-derivative matrix of the energy with
// respect to the real orbital-rotation parameters [t_0..t_{n-1},
// y_0..y_{n-1}] (t_I = Re kappa_pq: kappa_pq=+t, kappa_qp=-t; y_I =
// Im kappa_pq: kappa_pq=kappa_qp=iy; pairs p>q in `pair_indices`), i.e.
// the Hessian of E(exp(-kappa)) at kappa=0 in these coordinates -- what
// a Newton-Raphson step needs. Off orbital stationarity the bare
// G_pq,rs combinations above are NOT symmetric (they are SEQUENTIAL
// derivatives, d/dkappa_rs of the gradient measured after rotating), so
// this symmetrizes each block from the same bare G (with its L1/L2 part
// when `pair_of`/`two_rdm_l1`/`two_rdm_l2` are given):
//   tt(I,J) = (1/2)[SS_tt(I;J)+SS_tt(J;I)],  SS_tt(pq;rs) = G_pq,rs-G_pq,sr-G_qp,rs+G_qp,sr
//   yy(I,J) = (1/2)[SS_yy(I;J)+SS_yy(J;I)],  SS_yy(pq;rs) = -(G_pq,rs+G_pq,sr+G_qp,rs+G_qp,sr)
//   ty(I,J) = (1/2)[SS_ty(I;J)+SS_yt(J;I)],  SS_ty(pq;rs) = i(G_pq,rs+G_pq,sr-G_qp,rs-G_qp,sr),
//                                            SS_yt(pq;rs) = i(G_pq,rs-G_pq,sr+G_qp,rs-G_qp,sr)
// (SS_ab(I;J) = d/da_J of the b-component gradient of pair I). Returned
// as a real symmetric 2*n_pairs matrix (real parts; the imaginary parts
// are roundoff for physical input). Complex orbitals only. O(n^5).
// Independent of hartreeExchangeJointHessianMatrix below, which keeps its
// original (unsymmetrized, converged-SCF-oriented) convention.
Matrix<double> hartreeExchangeSymmetricJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(),
    const Matrix<double>& two_rdm_l2 = Matrix<double>());

// The independent real-step orbital-rotation parameters are exactly
// the (p,q) pairs with p > q (see OrbitalGradient.h) -- this lists them
// all, in a FIXED order, for a basis of dimension `n`: pair I is
// (pair[I].first, pair[I].second), first > second. Used to give
// `hartreeExchangeHessianMatrix`'s rows/columns a definite meaning.
std::vector<std::pair<std::size_t, std::size_t>> hessianPairIndices(std::size_t n);

// Hessian-VECTOR product w = H v of hartreeExchangeSymmetricJointHessianMatrix's own matrix H,
// over the joint real parameters [t_0..t_{n_pairs-1}; y_0..y_{n_pairs-1}], WITHOUT ever forming
// that matrix: for every unordered pair (I,J) (I<=J) it computes the same four block values
// (tt, yy, ty(I,J), ty(J,I)) the matrix builder does and immediately accumulates their
// contribution into `w`, discarding them. Same O(n_pairs^2) element evaluations (O(n) each, so
// O(n^5) total, exactly hartreeExchangeSymmetricJointHessianMatrix's own cost) but O(n_pairs)
// memory instead of O(n_pairs^2) -- the callback NEO.h's NeoProblem::hessianVector needs without
// ever materializing the (potentially huge) dense Hessian. `v`/the returned vector have size
// 2*pair_indices.size(); optional `pair_of`/`two_rdm_l1`/`two_rdm_l2` as
// hartreeExchangeSymmetricJointHessianMatrix's own. Complex orbitals only (T = std::complex
// <double>; the joint [t;y] parametrization only exists for those -- NON_REL orbitals have no y
// direction at all, so a plain vector of hartreeExchangeHessianElement/pnofHessianElement
// evaluations already IS memory-light: see FullOptimization.cpp's realHessianVector).
template <typename Eri>
std::vector<double> hartreeExchangeJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v, const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(),
    const Matrix<double>& two_rdm_l2 = Matrix<double>());

// The DIAGONAL of the matrix hartreeExchangeJointHessianVector applies, in the joint ordering
// [t_0..t_{n-1}, y_0..y_{n-1}] (size 2*pair_indices.size()): tt(I,I) and yy(I,I) from the same
// bare G_pq,rs combinations as the matrix builder with (r,s) = (p,q). O(n_pairs) elements
// (O(n) each) -- negligible next to one Hessian-vector product -- so NEO can use it as its Davidson
// preconditioner (NeoProblem::hessianDiagonal). Complex spinors only.
template <typename Eri>
std::vector<double> hartreeExchangeJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const Eri& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(),
    const Matrix<double>& two_rdm_l2 = Matrix<double>());



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
// Optional `pair_of`/`two_rdm_l1`/`two_rdm_l2`, same meaning and
// default as `hartreeExchangeHessianElement`'s own -- passed through to
// every element.
template <typename T>
Matrix<T> hartreeExchangeHessianMatrix(
    const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(),
    const Matrix<double>& two_rdm_l2 = Matrix<double>());

// The FULL, JOINT, real-parameter orbital-rotation Hessian for a
// complex (relativistic, spin-orbit-coupled) spinor basis -- the
// matrix a genuine Newton-Raphson/NEO step needs, unlike
// `hartreeExchangeHessianMatrix` above, which only ever assembles the
// real-real (t-t) block. Each independent pair `pair_indices[I] =
// (p,q)` carries TWO real rotation parameters, `t_I = Re(kappa_pq)`
// and `y_I = Im(kappa_pq)` (kappa_qp = -conj(kappa_pq) is not
// independent) -- so the full real Hessian over ALL such parameters is
// `2*n_pairs x 2*n_pairs`, block-ordered [t_0..t_{n_pairs-1},
// y_0..y_{n_pairs-1}] (all t's first, then all y's, each in
// `pair_indices` order):
//
//   [ Hess_tt   Hess_ty ]
//   [ Hess_yt   Hess_yy ]
//
// where block (I,J), 0-indexed within each n_pairs x n_pairs quadrant,
// is:
//   Hess_tt(I,J) = hartreeExchangeHessianElement(p_I,q_I,p_J,q_J)
//   Hess_yy(I,J) = hartreeExchangeHessianElementImag(p_I,q_I,p_J,q_J)
//   Hess_ty(I,J) = hartreeExchangeHessianElementMixed(p_I,q_I,p_J,q_J)
//   Hess_yt(I,J) = hartreeExchangeHessianElementMixed(p_J,q_J,p_I,q_I)
//                = Hess_ty(J,I)  [Schwarz's theorem, see
//                  hartreeExchangeHessianElementMixed's own header
//                  comment -- NOT a separate formula]
// Note Hess_yt(I,J) is defined to equal Hess_ty(J,I), which is exactly
// what makes the assembled matrix symmetric: entry (I, n_pairs+J)
// [row t_I, col y_J] and entry (n_pairs+J, I) [row y_J, col t_I] are
// both set from the SAME single `hartreeExchangeHessianElementMixed`
// call, by construction, rather than from two independently-derived
// formulas that would need to be checked against each other.
//
// Every element is real for physical (Hermitian h/eri/D/2-RDM) input
// (the same double-commutator-Hermiticity argument as each individual
// element function -- see their own headers), so this returns a plain
// `Matrix<double>`, ready for `LinearAlgebra.h`'s
// `diagonalizeSymmetric` exactly like `hartreeExchangeHessianMatrix`'s
// own real-orbital (T=double) instantiation -- `.real()` is taken of
// each underlying complex element value (confirmed, not just assumed,
// to leave only floating-point roundoff behind: see
// hartreeExchangeHessianElementMixed's own validation history).
//
// IMPORTANT CAVEAT, discovered while validating this exact function:
// the returned matrix is symmetric to FLOATING-POINT PRECISION ONLY AT
// (or very near) a converged, stationary SCF solution -- off
// stationarity, `Hess_tt(I,J)` and `Hess_tt(J,I)` (and, separately,
// `Hess_yy(I,J)`/`Hess_yy(J,I)`) can differ by an amount that tracks
// the SCF's own residual orbital gradient, whenever pairs I and J
// share a common orbital index (Hess_ty/Hess_yt are exactly symmetric
// by construction regardless, per the note above -- only the diagonal
// blocks are affected). Confirmed on water/STO-3G: max asymmetry
// 2.65e-6 at the default DENSITY_TOLERANCE=1e-6 (matching the SCF's
// own known residual-gradient scale at that tolerance), dropping to
// 1.82e-11 (floating-point roundoff) at DENSITY_TOLERANCE=1e-13/
// ENERGY_TOLERANCE=1e-14. This is expected, not a bug: `Hess_pq,rs` is
// built from the SAME double-commutator formula validated (against
// finite differences) to equal the true `d^2E/dkappa_pq dkappa_rs` for one
// FIXED (p,q,r,s) ordering at a time; Schwarz's theorem guarantees the
// two orderings agree only where the reference state is an actual
// stationary point (converged SCF; the standard setting a
// Newton-Raphson/NEO step is taken from anyway). A caller working with
// a not-fully-converged density should symmetrize the result
// (`0.5*(H+H^T)`) before diagonalizing/inverting it.
//
// Only meaningful for a genuinely complex spinor basis (T =
// std::complex<double> inputs) -- a real orbital basis has no
// y-direction at all, so there is nothing for this function to add
// over `hartreeExchangeHessianMatrix<double>` there; this function
// does not attempt to support that case (no `T` template parameter,
// unlike every OTHER function in this file -- matching
// hartreeExchangeHessianElementMixed's own deliberate departure from
// the template-over-T pattern for the same reason).
//
// EXPENSIVE: three O(n) element evaluations per (I,J) pair instead of
// `hartreeExchangeHessianMatrix`'s one, so this costs 3x as much for
// the same `pair_indices` -- still O(n^5) overall, plus O(n^6) for the
// caller's subsequent diagonalization of the (now twice as large)
// `2*n_pairs`-dimensional result.
Matrix<double> hartreeExchangeJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGEHESSIAN_H
