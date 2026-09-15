#ifndef RERDMFT_GENERALIZEDHESSIAN_H
#define RERDMFT_GENERALIZEDHESSIAN_H

#include <cstddef>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds one element of the orbital-rotation Hessian
//   Hess_pq,rs = d^2E/dt_pq dt_rs
// for the SAME single-real-parameter exponential-rotation convention
// already used by OrbitalGradient.h's orbitalGradient (Hessian_opt/
// SpinorRotation.h's U_rot = exp(-kappa), kappa_pq = +t / kappa_qp = -t
// for a fixed pair (p,q), t real): the full analogue of that gradient,
// for a GENERAL (not necessarily idempotent) 1-RDM `d` and a GENERAL
// (not necessarily Wick-derivable/single-determinant) dense 2-RDM
// `two_rdm` -- exactly the RDMFT case, no shortcuts.
//
// Derived from doc/orbital_hessian.tex (energy expansion
// E(kappa) ~ E(0) + sum kappa_pq g_pq + (1/2) sum kappa_pq kappa_rs
// G_pq,rs, G_pq,rs = <0|[[H,a+_p a_q],a+_r a_s]|0>), whose "boxed"
// Fock-matrix form (the tex's Eq. for G_pq,rs, expressed via
// GeneralizedFock.h's own generalizedFockMatrix output F -- see the
// .cpp for the exact F index correspondence) is evaluated here for the
// four sign/index combinations needed to antisymmetrize BOTH pairs
// (matching orbitalGradient's own kappa_pq/kappa_qp antisymmetrization,
// done twice, once per pair):
//   Hess_pq,rs = G_pq,rs - G_pq,sr - G_qp,rs + G_qp,sr
// This combination is what was actually validated (see below), not the
// tex's own bare G_pq,rs alone.
//
// Only the REAL-step direction (t = Re(kappa_pq), t' = Re(kappa_rs)) is
// implemented/validated here -- the analogous imaginary-direction and
// mixed Re/Im second derivatives (needed for a fully complex/DHF
// Newton-Raphson step) are NOT yet covered; extending this the same
// way OrbitalGradient.h's imaginary-step validation was done is future
// work.
//
// `h`, `eri`, `d`, `two_rdm` follow EXACTLY GeneralizedFock.h's own
// conventions (physics-notation eri, standard physicist 2-RDM ordering
// with N(N-1)/2 normalization, general/non-idempotent d) -- see that
// header. `fock` MUST be `generalizedFockMatrix(h, eri, d, two_rdm)`'s
// own output, passed in by the caller (not recomputed here) since it
// is needed, unchanged, for every (p,q,r,s) element -- "generalized
// Fock elements are used in the definition" is exactly this F.
//
// EXPENSIVE and deliberately unoptimized (this is the first, general
// version): each of the four G_pq,rs-style terms costs O(n^4)
// (six independent double sums over the remaining orbital indices), so
// this single ELEMENT already costs O(n^4); a full dense Hessian
// tensor (looping p,q,r,s) would cost O(n^8) -- expect this to be
// usable only for small test systems (e.g. water/STO-3G) or for
// spot-checking a handful of elements, not for a production Newton-
// Raphson step on a realistic basis. A cheaper, HF/DHF-specific
// shortcut (mirroring NonRelOrbitalGradient.h/RkbOrbitalGradient.h's
// relationship to the general gradient) is not implemented here.
//
// Validated (see feedback/project memory for the scratch scripts, not
// committed) via: (1) a small explicit-Fock-space numerical check
// against the true many-body double commutator
// <0|[K1_pq,[K1_rs,H]]|0> (K1_ab = a+_a a_b - a+_b a_a), for a GENERAL
// (non-idempotent, non-Wick-derivable) random 2-RDM built from an
// actual few-particle quantum state -- confirming this formula needs
// only the 1-RDM and 2-RDM (no 3-RDM), matching several independent
// (p,q,r,s) index quadruples to floating-point precision; (2) a
// finite-difference check against water/STO-3G's own converged
// NON_REL HF data (see main.cpp, DEBUG TRUE).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
T generalizedOrbitalHessianElement(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                    const Tensor4<T>& two_rdm, const Matrix<T>& fock,
                                    std::size_t p, std::size_t q, std::size_t r, std::size_t s);

}  // namespace rerdmft

#endif  // RERDMFT_GENERALIZEDHESSIAN_H
