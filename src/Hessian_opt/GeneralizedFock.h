#ifndef RERDMFT_GENERALIZEDFOCK_H
#define RERDMFT_GENERALIZEDFOCK_H

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the generalized Fock matrix
//   F_pq = sum_r h_pr D_rq + sum_rst <pr|st> Gamma_qrst
// entirely in an orthonormal molecular-orbital (or natural-spin-orbital)
// basis -- never the AO or RKB-spinor basis. This is the standard
// MCSCF/RDMFT generalized Fock matrix (see e.g. Helgaker, Jorgensen &
// Olsen, "Molecular Electronic-Structure Theory", or M. Piris's PNOF
// papers): its antisymmetric part F_pq - F_qp is the orbital gradient
// (the generalized Brillouin condition F_pq = F_qp holds at a
// stationary point), and it is the basic building block for the orbital
// Hessian built elsewhere in this directory.
//
// `d` is the one-electron reduced density matrix (1-RDM) over the same
// orbital basis, D_rq = <Psi| a^dagger_r a_q |Psi> -- general: NOT
// assumed idempotent (fractional occupations, and off-diagonal elements
// in any basis other than the natural-orbital one, are both fine),
// unlike C4_DHF/RkbFockMatrix.h and NON_REL/NonRelHartreeFock.h's Fock
// matrices, which both assume D comes from an idempotent HF/DHF
// occupied-orbital projector.
//
// `gamma` is the two-electron reduced density matrix (2-RDM),
//   Gamma_pqrs = <Psi| a^dagger_p a^dagger_q a_s a_r |Psi>
// (physicist operator ordering: antisymmetric under p<->q and under
// r<->s, Hermitian via Gamma_pqrs* = Gamma_rspq), over spin-orbitals
// (no separate spin summation/factor-of-2 bookkeeping here -- a spin-
// summed spatial-orbital Gamma, e.g. the usual closed-shell CASSCF
// convention, must be expanded to spin-orbitals by the caller first).
// Passed in as a plain, general (not symmetry-packed) tensor so the
// SAME routine works for a CASSCF Gamma (for testing/validation) or an
// RDMFT-functional Gamma built from natural orbitals and occupation
// numbers -- Gamma's construction is entirely the caller's concern.
//
// `eri` is the two-electron integral tensor over the SAME orbital
// basis, in PHYSICS notation directly: eri(p,r,s,t) == <pr|st> (p,r on
// electron 1; s,t on electron 2) -- NOT chemist notation (pq|rs). This
// matches RkbTwoElectronTensor's convention but NOT
// ElectronRepulsion.h's twoElectronIntegrals/twoElectronIntegralsPacked,
// which return chemist notation directly; a caller transforming AO
// integrals into this MO basis must convert notation accordingly (see
// NON_REL/NonRelHartreeFock.cpp's nonRelFockMatrix for the chemist/
// physics relation physics<A B|C D> = chemist(A,C,B,D)).
//
// Works for either a real (T = double) or complex (T = std::complex
// <double>) orbital basis -- explicit instantiations for both are
// provided in the .cpp.
template <typename T>
Matrix<T> generalizedFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                 const Tensor4<T>& gamma);

}  // namespace rerdmft

#endif  // RERDMFT_GENERALIZEDFOCK_H
