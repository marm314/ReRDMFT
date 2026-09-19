#ifndef RERDMFT_ORBITALROTATIONFINITEDIFFERENCE_H
#define RERDMFT_ORBITALROTATIONFINITEDIFFERENCE_H

#include <cstddef>
#include <functional>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// An RDMFT functional's electronic energy at FIXED occupations, as a
// function of the (h, eri) it is evaluated on -- e.g.
// [&](const Matrix<T>& h, const Tensor4<T>& eri) { return
// hartreeExchangeEnergy(h, eri, occupations, two_rdm_h, two_rdm_x); } for
// JK_only, or the analogous pnofFockMatrix/pnofElectronicEnergy-based
// closure for a PNOF functional -- occupations and every other
// functional-specific quantity are captured by the caller's closure, so
// this file needs no knowledge of which functional is being checked.
template <typename T>
using RdmftEnergyFn = std::function<double(const Matrix<T>&, const Tensor4<T>&)>;

template <typename T>
struct OrbitalRotationGradientCheck {
  double analytic = 0.0;
  double finite_difference = 0.0;
  double abs_diff = 0.0;
};

// Genuinely independent validation of an analytic orbital-rotation
// gradient (`gradient`, e.g. OrbitalGradient.h's orbitalGradient applied
// to Hessian_opt/PnofFock.h's pnofFockMatrix or
// HartreeExchangeGradient.h's hartreeExchangeFockMatrix) at a FIXED set
// of occupations, generalizing this project's existing HF/DHF finite-
// difference checks (main.cpp's finiteDifferenceCheckReport, which
// rotates the DENSITY via idempotent occupations -- invalid for a
// fractional-occupation RDMFT functional's own converged, non-
// stationary-for-orbitals occupations) by rotating the ORBITALS/
// INTEGRALS via IntegralRotation.h's rotateIntegrals instead, at a
// single real rotation angle t along kappa_pq = t, kappa_qp = -t (the
// SAME (p,q) real parametrization OrbitalGradient.h's own g_pq =
// 2*(F_qp - conj(F_pq)) is built to satisfy g_pq = dE/dt for exactly),
// central-differenced:
//   finite_difference = [E(h(+t),eri(+t)) - E(h(-t),eri(-t))] / (2t)
// compared against g_pq's own real part (g_pq is guaranteed real for a
// Hermitian h/eri and real, diagonal occupations, exactly like every
// *_result.electronic_energy in this project). `p`/`q` may be given in
// EITHER order: OrbitalGradient.h's own `gradient` only stores its
// p >= q "lower triangle", so this reconstructs g_pq = -conj(g_qp) for
// an upper-triangle request rather than reading that matrix's zero
// placeholder there.
template <typename T>
OrbitalRotationGradientCheck<T> orbitalRotationGradientCheck(const Matrix<T>& h,
                                                               const Tensor4<T>& eri,
                                                               const Matrix<T>& gradient,
                                                               const RdmftEnergyFn<T>& energy_fn,
                                                               std::size_t p, std::size_t q,
                                                               double step = 1e-4);

// The FULL analytic orbital-rotation gradient matrix (OrbitalGradient.h's
// own p>=q "lower triangle" storage convention) at FIXED occupations, as
// a function of the (h, eri) it is evaluated on -- e.g.
// [&](const Matrix<T>& h, const Tensor4<T>& eri) { return orbitalGradient(
// pnofFockMatrix(functional, h, eri, geminals, occupations, relativistic));
// } for a PNOF functional. Used by orbitalRotationHessianCheck below to
// probe the SECOND derivative by finite-differencing the FIRST.
template <typename T>
using RdmftGradientFn = std::function<Matrix<T>(const Matrix<T>&, const Tensor4<T>&)>;

template <typename T>
struct OrbitalRotationHessianCheck {
  double analytic = 0.0;
  double finite_difference = 0.0;
  double abs_diff = 0.0;
};

// Genuinely independent validation of an analytic orbital-rotation
// Hessian ELEMENT (`hessian_element`, e.g.
// Hessian_opt/PnofHessian.h's pnofHessianElement or
// Hessian_opt/HartreeExchangeHessian.h's hartreeExchangeHessianElement)
// at a FIXED set of occupations -- the same e^kappa integral-rotation
// idea as orbitalRotationGradientCheck above, one derivative order
// higher: `Hess_pq,rs = d^2E/dt_pq dt_rs = d(g_pq)/dt_rs`, so this
// rotates by kappa_rs = +-t (the SECOND pair (r,s), the direction the
// derivative is taken along), evaluates the FULL analytic gradient
// matrix at each rotated point via the caller-supplied `gradient_fn`,
// reads its (p,q) element (reconstructing `g_pq = -conj(g_qp)` for an
// upper-triangle (p,q) request, exactly like orbitalRotationGradientCheck
// does for its own gradient argument), and central-differences:
//   finite_difference = [g_pq(kappa_rs=+t) - g_pq(kappa_rs=-t)] / (2t)
// compared against `hessian_element`'s own real part (guaranteed real
// for physical Hermitian input, exactly like every other Hessian
// element in this project -- see HartreeExchangeHessian.h's own
// header). `(p,q)` and `(r,s)` may each be given in either order.
template <typename T>
OrbitalRotationHessianCheck<T> orbitalRotationHessianCheck(const Matrix<T>& h,
                                                             const Tensor4<T>& eri,
                                                             T hessian_element,
                                                             const RdmftGradientFn<T>& gradient_fn,
                                                             std::size_t p, std::size_t q,
                                                             std::size_t r, std::size_t s,
                                                             double step = 1e-4);

}  // namespace rerdmft

#endif  // RERDMFT_ORBITALROTATIONFINITEDIFFERENCE_H
