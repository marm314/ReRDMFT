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

}  // namespace rerdmft

#endif  // RERDMFT_ORBITALROTATIONFINITEDIFFERENCE_H
