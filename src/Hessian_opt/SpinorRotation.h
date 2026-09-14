#ifndef RERDMFT_SPINORROTATION_H
#define RERDMFT_SPINORROTATION_H

#include "Matrix.h"

namespace rerdmft {

// Builds the orbital-rotation matrix U_rot = exp(-kappa) from an anti-
// Hermitian rotation generator kappa (kappa^dagger = -kappa; for T =
// double this is real antisymmetric, kappa^T = -kappa) -- the standard
// exponential parametrization of an orbital rotation in SCF/MCSCF/RDMFT
// orbital optimization (e.g. built from OrbitalGradient.h's g_pq and an
// approximate or exact orbital Hessian in a Newton-Raphson-like step),
// used to rotate MOLECULAR ORBITAL (or natural-spinor) coefficients:
//   C_new = C_old * U_rot
// -- NEVER applied directly to an AO or RKB-spinor basis.
//
// Computed via the standard trick for a skew-Hermitian matrix
// exponential: i*kappa is Hermitian (since kappa is anti-Hermitian), so
// diagonalizing i*kappa = V diag(w) V^dagger (w real,
// LinearAlgebra.h's diagonalizeHermitian) gives
//   U_rot = exp(-kappa) = exp(i * V diag(w) V^dagger)
//         = V diag(exp(i*w)) V^dagger,
// which is unitary since |exp(i*w_k)| = 1 for real w_k. For T = double
// (kappa real antisymmetric), this reconstruction is carried out in
// complex arithmetic internally (i*kappa is complex Hermitian even
// though kappa itself is real) and only the real part is returned --
// mathematically exp(-kappa) is exactly real in that case (exp of a
// real matrix is real), so the imaginary part is only nonzero here due
// to floating-point rounding; it is checked and this throws if it is
// unexpectedly large (which would mean kappa was not actually
// antisymmetric).
//
// Throws std::runtime_error if kappa is not square, or is not
// (anti-)Hermitian to within a small numerical tolerance.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
Matrix<T> spinorRotationMatrix(const Matrix<T>& kappa);

}  // namespace rerdmft

#endif  // RERDMFT_SPINORROTATION_H
