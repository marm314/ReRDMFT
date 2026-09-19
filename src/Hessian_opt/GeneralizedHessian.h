#ifndef RERDMFT_GENERALIZEDHESSIAN_H
#define RERDMFT_GENERALIZEDHESSIAN_H

#include <cstddef>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds one element of the orbital-rotation Hessian
//   Hess_pq,rs = d^2E/dkappa_pq dkappa_rs
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
// `generalizedOrbitalHessianElementImag` (below) is the analogous
// IMAGINARY-step direction (y = Im(kappa_pq), y' = Im(kappa_rs); only
// meaningful for T = std::complex<double> -- C4_DHF's genuinely complex
// spinors, mirroring exactly why OrbitalGradient.h's own imaginary-step
// check only applies there, not to NON_REL's real orbitals):
//   Hess^{yy}_pq,rs = d^2E/dy_pq dy_rs
//                   = -(G_pq,rs + G_pq,sr + G_qp,rs + G_qp,sr)
// (note ALL FOUR terms add here, unlike the real-real case's
// alternating signs, plus an overall minus sign -- both come from
// kappa_pq = kappa_qp = iy for a pure-imaginary perturbation, i.e. this
// direction moves the (p,q) AND (q,p) entries to the SAME value with a
// leading factor of i, rather than to opposite values, exactly as
// already established for the gradient's own imaginary direction; see
// main.cpp's finiteDifferenceCheckReport for that derivation). The
// MIXED real/imaginary second derivative (d^2E/dkappa_pq dy_rs) is a
// separate, distinctly-signed combination not implemented here --
// only the "pure" real-real and imaginary-imaginary diagonal blocks of
// the full complex Hessian are covered so far.
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
// version): each of the four G_pq,rs-style terms costs O(n^2) (six
// independent DOUBLE sums, each over just two of the remaining orbital
// indices -- NOT a dense O(n^4) contraction; the other two indices of
// each eri/two_rdm element are already pinned to p,q,r,s), so this
// single ELEMENT costs O(n^2); a full dense Hessian tensor (looping
// p,q,r,s, i.e. O(n^4) elements) would cost O(n^6) -- expect this to
// be usable for a small-to-moderate basis (e.g. water/STO-3G, or
// spot-checking a handful of elements on something larger), but not
// for a production Newton-Raphson step on a realistic basis, where
// O(n^6) for the full tensor is still prohibitive. A cheaper,
// HF/DHF-specific shortcut (mirroring NonRelOrbitalGradient.h/
// RkbOrbitalGradient.h's relationship to the general gradient) is
// Hessian_opt/HartreeExchangeHessian.h's hartreeExchangeHessianElement
// (O(n) per element, O(n^5) for a full tensor).
//
// Both are validated (see feedback/project memory for the scratch
// scripts, not committed) via: (1) a small explicit-Fock-space
// numerical check -- the real-real combination against the true
// many-body double commutator <0|[K1_pq,[K1_rs,H]]|0>
// (K1_ab = a+_a a_b - a+_b a_a), the imaginary-imaginary combination
// against a 2D finite difference on the exact rotated-determinant
// energy -- for a GENERAL (non-idempotent, non-Wick-derivable) random
// 2-RDM built from an actual few-particle quantum state, confirming
// both formulas need only the 1-RDM and 2-RDM (no 3-RDM), matching
// several independent (p,q,r,s) index quadruples to floating-point/
// finite-difference-limited precision; (2) a finite-difference check
// against water/STO-3G's own converged NON_REL HF data (real-real) and
// C4_DHF's own converged DHF data (imaginary-imaginary) (see main.cpp,
// DEBUG TRUE).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `generalizedOrbitalHessianElementImag` compiles
// for T = double too (so callers need not branch on T at the call
// site) but is not meaningful there (a real orbital basis has no
// imaginary kappa direction at all) -- callers should gate its use on
// T = std::complex<double> the same way main.cpp's gradient check does
// (`if constexpr`).
template <typename T>
T generalizedOrbitalHessianElement(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                    const Tensor4<T>& two_rdm, const Matrix<T>& fock,
                                    std::size_t p, std::size_t q, std::size_t r, std::size_t s);

template <typename T>
T generalizedOrbitalHessianElementImag(const Matrix<T>& h, const Tensor4<T>& eri,
                                        const Matrix<T>& d, const Tensor4<T>& two_rdm,
                                        const Matrix<T>& fock, std::size_t p, std::size_t q,
                                        std::size_t r, std::size_t s);

}  // namespace rerdmft

#endif  // RERDMFT_GENERALIZEDHESSIAN_H
