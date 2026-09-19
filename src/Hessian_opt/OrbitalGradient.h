#ifndef RERDMFT_ORBITALGRADIENT_H
#define RERDMFT_ORBITALGRADIENT_H

#include "Matrix.h"

namespace rerdmft {

// Builds the orbital-rotation gradient
//   g_pq = 2 * (F_qp - conj(F_pq))
// from the generalized Fock matrix F (GeneralizedFock.h's
// generalizedFockMatrix). The base combination F_qp - conj(F_pq)
// matches Dyall & Faegri, "Introduction to Relativistic Quantum
// Chemistry", Eq. (8.32) (p. 124): g_pq = f_qp - f*_pq, with F here
// playing the role of Dyall's f -- but note the factor of 2 in front is
// a DELIBERATE departure from Dyall's own (unscaled) equation. It is
// there so that g_pq equals dE/dkappa EXACTLY for the single real
// orbital-rotation degree of freedom t used throughout this project's
// exponential parametrization (Hessian_opt/SpinorRotation.h's U_rot =
// exp(-kappa), one independent real t per antisymmetric-Hermitian pair
// kappa_pq = +t, kappa_qp = -conj(t)): the unscaled Dyall combination is
// only the coefficient of ONE of that pair's two entries, and the
// finite-difference path that moves both at once (main.cpp's DEBUG
// validation) picks up the other's equal contribution too -- see that
// derivation for the full algebra. At a stationary point of the energy
// (e.g. a converged RDMFT/CASSCF orbital optimization), g_pq = 0 for
// all p,q regardless of this scalar factor -- the generalized Brillouin
// condition is unaffected.
//
// g is antisymmetric-Hermitian: g_qp = -conj(g_pq) (immediate from the
// definition), so only the p >= q "lower triangle" (including the
// diagonal, where g_pp = 2*(F_pp - conj(F_pp)) = 4i*Im(F_pp) is
// generally nonzero unless F_pp is real) is computed and stored here;
// g(p,q) for p < q is left at its default-constructed zero --
// reconstruct it as -conj(g(q,p)) if ever needed, rather than reading
// it directly.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `fock` must be square.
template <typename T>
Matrix<T> orbitalGradient(const Matrix<T>& fock);

}  // namespace rerdmft

#endif  // RERDMFT_ORBITALGRADIENT_H
