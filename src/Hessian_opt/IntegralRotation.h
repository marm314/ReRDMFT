#ifndef RERDMFT_INTEGRALROTATION_H
#define RERDMFT_INTEGRALROTATION_H

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// One- and two-electron integrals rotated into a new orbital basis
// C_new = C_old * U (SpinorRotation.h's own convention for what "U"
// means), i.e. h_rot = U^dagger h U and eri_rot(p,q,r,s) =
// sum_{a,b,c,d} conj(U(a,p)) U(b,q) conj(U(c,r)) U(d,s) eri(a,b,c,d)
// (physics notation, bra legs conjugated -- the same convention every
// other MO transform in this project uses, e.g. C4_DHF/RkbMoTransform.h).
template <typename T>
struct RotatedIntegrals {
  Matrix<T> h;
  Tensor4<T> eri;
};

// `h`/`eri` must already be expressed in SOME orthonormal orbital basis
// (e.g. a converged HF/DHF MO/natural-spinor basis); `u` (typically
// SpinorRotation.h's own spinorRotationMatrix(kappa)) must be unitary in
// that same basis. Used by OrbitalRotationFiniteDifference.h to probe an
// RDMFT functional's energy at a rotated-orbital, FIXED-occupation point
// without ever needing a picture-change/AO-basis re-transform -- unlike
// this project's existing HF/DHF finite-difference checks (main.cpp's
// finiteDifferenceCheckReport), which instead rotate the DENSITY at
// fixed integrals (only valid for idempotent occupations), this rotates
// the INTEGRALS at fixed (possibly fractional) occupations, the
// generalization fractional-occupation RDMFT functionals need.
//
// Costs O(n^5) for the two-electron leg (reuses Utils/
// Cholesky_Decomposition.h's own GEMM-accelerated choleskyTransformEri,
// itself O(Nchol*(n^2*n + n*n^2)) to decompose/transform plus
// O(Nchol*n^4) to reconstruct -- cheaper than a naive direct 4-leg
// transform, though still the dominant cost here) -- affordable for the
// small-to-moderate active spaces this project's own finite-difference
// checks already run on (see main.cpp's existing VERBOSE-gated O(n^5)
// dense-2-RDM gradient path for the established precedent of what this
// project already considers an acceptable, occasionally-run cost).
template <typename T>
RotatedIntegrals<T> rotateIntegrals(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& u);

}  // namespace rerdmft

#endif  // RERDMFT_INTEGRALROTATION_H
