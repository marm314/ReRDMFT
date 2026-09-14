#ifndef RERDMFT_GENERALIZEDFOCK_H
#define RERDMFT_GENERALIZEDFOCK_H

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the generalized Fock matrix
//   F_pq = sum_r h_rp D_rq + sum_rst (st|rp) P_strq
// entirely in an orthonormal molecular-orbital (or natural-spin-orbital)
// basis -- never the AO or RKB-spinor basis. This is the first sum on
// the right-hand side of Dyall & Faegri, "Introduction to Relativistic
// Quantum Chemistry", Eq. (8.30) (p. 124): the gradient there is
// g_pq = [this sum] - [the mirrored sum with p,q swapped and complex
// conjugated], and its antisymmetric combination is the orbital
// gradient/generalized Brillouin condition; F_pq itself is the basic
// building block for the orbital Hessian built elsewhere in this
// directory. Note the free indices as Dyall writes them: h_rp (not
// h_pr) and D_rq -- implemented exactly as printed, not silently
// transposed/conjugated, even though h and D are both Hermitian.
//
// `d` is the one-electron reduced density matrix (1-RDM) over the same
// orbital basis, D_rq = <Psi| a^dagger_r a_q |Psi> -- general: NOT
// assumed idempotent (fractional occupations, and off-diagonal elements
// in any basis other than the natural-orbital one, are both fine),
// unlike C4_DHF/RkbFockMatrix.h and NON_REL/NonRelHartreeFock.h's Fock
// matrices, which both assume D comes from an idempotent HF/DHF
// occupied-orbital projector.
//
// `two_rdm` is the two-electron reduced density matrix, in Dyall's own
// index convention (Eq. 8.28):
//   P_pqrs = <Psi| a^dagger_p a^dagger_r a_s a_q |Psi>
// -- NOT the more common physicist convention Gamma_pqrs = <a^dagger_p
// a^dagger_q a_s a_r> (Dyall's 2nd and 3rd indices are swapped relative
// to that). This convention is deliberate: it is exactly what pairs,
// with no reindexing, against a CHEMIST-notation integral (see `eri`
// below) in the Hamiltonian's two-electron term and in F_pq itself.
// Zero unless all four indices are occupied for a single determinant
// (Eq. 8.28's own note); for a general idempotent D (not necessarily
// diagonal), the single-determinant value generalizes via Wick's
// theorem to
//   P_pqrs = D_pq D_rs - D_ps D_rq
// (which reduces to Dyall's Eq. (8.29), P_pqrs = delta_pq delta_rs -
// delta_ps delta_rq, when D is diagonal 1/0 -- used to cross-check this
// routine against C4_DHF/RkbFockMatrix.h's already-validated HF/DHF
// Fock matrix). Passed in as a plain, general (not symmetry-packed)
// spin-orbital tensor (no spin-summation/factor-of-2 bookkeeping here)
// so the SAME routine works for a CASSCF two_rdm (for testing/
// validation) or an RDMFT-functional one built from natural orbitals
// and occupation numbers -- two_rdm's construction is entirely the
// caller's concern.
//
// `eri` is the two-electron integral tensor over the SAME orbital
// basis, in CHEMIST notation directly: eri(s,t,r,p) == (st|rp) -- the
// SAME convention as ElectronRepulsion.h's twoElectronIntegrals/
// twoElectronIntegralsPacked (unlike RkbTwoElectronTensor, which
// returns physics notation directly): a caller transforming AO
// integrals into this MO basis needs no notation conversion, only the
// basis transform itself.
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
Matrix<T> generalizedFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                 const Tensor4<T>& two_rdm);

}  // namespace rerdmft

#endif  // RERDMFT_GENERALIZEDFOCK_H
