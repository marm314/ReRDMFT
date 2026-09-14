#ifndef RERDMFT_GENERALIZEDFOCK_H
#define RERDMFT_GENERALIZEDFOCK_H

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the generalized Fock matrix
//   F_pq = sum_r h_rp D_rq + 2 sum_rst <sr|tp> two_rdm_strq
// entirely in an orthonormal molecular-orbital (or natural-spin-orbital)
// basis -- never the AO or RKB-spinor basis. This is the first sum on
// the right-hand side of Dyall & Faegri, "Introduction to Relativistic
// Quantum Chemistry", Eq. (8.30) (p. 124),
//   F_pq = sum_r h_rp D_rq + sum_rst (st|rp) P_strq,
// rewritten for THIS project's two conventions (both deliberately
// different from Dyall's own, see below): `eri` in physics notation
// instead of chemist, and `two_rdm` normalized to N(N-1)/2 (the number
// of electron PAIRS) instead of Dyall's N(N-1). Its antisymmetric
// combination F_pq - F*_qp is the orbital gradient/generalized
// Brillouin condition (Dyall's Eq. 8.32, up to the same two rewrites),
// and F_pq itself is the basic building block for the orbital Hessian
// built elsewhere in this directory. Note the free indices as Dyall
// writes them: h_rp (not h_pr) and D_rq -- implemented exactly as
// printed, not silently transposed/conjugated, even though h and D are
// both Hermitian.
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
// index convention (Eq. 8.28: P_pqrs = <a^dagger_p a^dagger_r a_s a_q>
// -- NOT the more common physicist ordering Gamma_pqrs = <a^dagger_p
// a^dagger_q a_s a_r>, which has the 2nd/3rd indices swapped relative
// to Dyall's), but normalized to N(N-1)/2 rather than Dyall's own
// N(N-1): two_rdm = P_Dyall / 2, i.e. sum_pq two_rdm_pqpq = N(N-1)/2 =
// the number of electron pairs. That halving is exactly why a
// compensating factor of 2 appears in front of the sum above (absent
// from Dyall's own Eq. 8.30, which uses P_Dyall directly). For a
// single determinant with idempotent D, Wick's theorem gives
//   two_rdm_pqrs = (1/2) (D_pq D_rs - D_ps D_rq)
// (i.e. half of Dyall's Eq. 8.29 special case D_pq D_rs - D_ps D_rq for
// a diagonal 0/1 D) -- used to cross-check this routine against
// C4_DHF/RkbFockMatrix.h's already-validated HF/DHF Fock matrix (which
// reduces to Dyall's Eq. 8.31 through the SAME substitution, since the
// factor of 2 here exactly cancels the factor of 1/2 in two_rdm).
// Passed in as a plain, general (not symmetry-packed) spin-orbital
// tensor (no spin-summation/factor-of-2-for-double-occupancy
// bookkeeping here) so the SAME routine works for a CASSCF two_rdm
// (for testing/validation) or an RDMFT-functional one built from
// natural orbitals and occupation numbers -- two_rdm's construction is
// entirely the caller's concern.
//
// `eri` is the two-electron integral tensor over the SAME orbital
// basis, in PHYSICS notation directly: eri(A,B,C,D) == <AB|CD> (A,C on
// electron 1; B,D on electron 2) -- the SAME convention as
// RkbTwoElectronTensor (unlike ElectronRepulsion.h's
// twoElectronIntegrals/twoElectronIntegralsPacked, which return chemist
// notation directly). Dyall's chemist-notation (st|rp) is therefore
// read here as <sr|tp> = eri(s,r,t,p) (using physics<A B|C D> =
// chemist(A,C,B,D), i.e. chemist(s,t,r,p) = physics<s r|t p>): a caller
// transforming AO integrals into this MO basis must convert notation
// accordingly if starting from a chemist-notation AO tensor (see
// NON_REL/NonRelHartreeFock.cpp's nonRelFockMatrix for that relation).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
Matrix<T> generalizedFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                 const Tensor4<T>& two_rdm);

}  // namespace rerdmft

#endif  // RERDMFT_GENERALIZEDFOCK_H
