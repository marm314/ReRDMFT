#ifndef RERDMFT_SINGLEDETERMINANTRDM_H
#define RERDMFT_SINGLEDETERMINANTRDM_H

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the two-electron reduced density matrix (2-RDM) of a single
// Slater determinant, given its (idempotent) one-electron reduced
// density matrix `d`, via Wick's theorem:
//   two_rdm_pqrs = (1/2) (D_pr D_qs - D_ps D_qr)
// -- the single-determinant special case of
// Hessian_opt/GeneralizedFock.h's `two_rdm` convention (standard
// physicist operator ordering, N(N-1)/2-normalized; verified against
// an explicit second-quantized calculation, see that file's history).
// `d` need not actually be idempotent for this function to run (the
// formula is evaluated as written regardless), but the RESULT is only
// the true 2-RDM of a physical single determinant when it is -- e.g.
// D built by C4_DHF/RkbDensityMatrix.h or NON_REL/
// ClosedShellSpinOrbitals.h's closedShellSpinOrbitalDensity. Feeding it
// generalizedFockMatrix's `two_rdm` argument together with the same
// (idempotent) `d` reproduces the ordinary HF/DHF Fock matrix (up to
// the orbital-rotation-gradient sign convention there), so a converged
// SCF's orbital gradient (OrbitalGradient.h) built this way should
// vanish -- the generalized Brillouin condition at a stationary point.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
Tensor4<T> singleDeterminantTwoRdm(const Matrix<T>& d);

}  // namespace rerdmft

#endif  // RERDMFT_SINGLEDETERMINANTRDM_H
