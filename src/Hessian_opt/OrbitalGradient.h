#ifndef RERDMFT_ORBITALGRADIENT_H
#define RERDMFT_ORBITALGRADIENT_H

#include "Matrix.h"

namespace rerdmft {

// Builds the orbital-rotation gradient
//   g_pq = F_qp - conj(F_pq)
// from the generalized Fock matrix F (GeneralizedFock.h's
// generalizedFockMatrix), matching Dyall & Faegri, "Introduction to
// Relativistic Quantum Chemistry", Eq. (8.32) (p. 124): g_pq = f_qp -
// f*_pq, with F here playing the role of Dyall's f. At a stationary
// point of the energy (e.g. a converged RDMFT/CASSCF orbital
// optimization), g_pq = 0 for all p,q -- the generalized Brillouin
// condition.
//
// g is antisymmetric-Hermitian: g_qp = -conj(g_pq) (immediate from the
// definition), so only the p >= q "lower triangle" (including the
// diagonal, where g_pp = F_pp - conj(F_pp) = 2i*Im(F_pp) is generally
// nonzero unless F_pp is real) is computed and stored here; g(p,q) for
// p < q is left at its default-constructed zero -- reconstruct it as
// -conj(g(q,p)) if ever needed, rather than reading it directly.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp. `fock` must be square.
template <typename T>
Matrix<T> orbitalGradient(const Matrix<T>& fock);

}  // namespace rerdmft

#endif  // RERDMFT_ORBITALGRADIENT_H
