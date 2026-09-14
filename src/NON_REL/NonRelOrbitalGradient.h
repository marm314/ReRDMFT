#ifndef RERDMFT_NONRELORBITALGRADIENT_H
#define RERDMFT_NONRELORBITALGRADIENT_H

#include "ElectronRepulsion.h"
#include "Matrix.h"

namespace rerdmft {

// Builds the same spin-orbital orbital gradient as
// Hessian_opt/GeneralizedFock.h + OrbitalGradient.h would, applied to a
// closed-shell single-determinant 1-/2-RDM (Hessian_opt/
// SingleDeterminantRdm.h) -- but WITHOUT ever materializing a 2-RDM or
// paying its O(n^5) contraction. For a single determinant, the 2-RDM's
// only nonzero pieces are Hartree, (same-spin) exchange, and (would-be,
// here identically zero for a real closed-shell spin-orbital basis)
// opposite-spin exchange -- exactly the terms nonRelFockMatrix already
// computes directly from the density and the two-electron integrals,
// at O(n^4).
//
// Derivation: substituting two_rdm_pqrs = (1/2)(D_pr D_qs - D_ps D_qr)
// into generalizedFockMatrix's formula and using D's idempotency
// collapses it to F_pq = FockLike(q,p) if q is occupied, else 0 (Dyall
// & Faegri's own remark, Eq. 8.30's surrounding text: "only nonzero if
// the second index is an occupied spinor index"), where FockLike is
// the ORDINARY closed-shell Fock matrix -- built from the density of
// the CONVERGED `c_matrix` (via nonRelDensityMatrix), NOT the SCF
// loop's stored, one-iteration-stale density -- transformed into the
// spin-orbital MO basis. Then g_pq = F_qp - conj(F_pq) simplifies
// (FockLike is Hermitian) to
//   g_pq = FockLike(p,q) * ([p occupied] - [q occupied])
// i.e. g is nonzero only in the occupied-virtual and virtual-occupied
// blocks (occ-occ and virt-virt are identically zero -- the well-known
// redundancy of those orbital rotations). Verified to reproduce
// Hessian_opt's general (slower) path numerically; see main.cpp.
//
// Only p >= q is computed/stored, matching OrbitalGradient.h's
// convention (p < q is -conj(g(q,p)), reconstructible if ever needed).
Matrix<double> nonRelOrbitalGradientEfficient(const Matrix<double>& h_core_ao,
                                               const PackedTwoElectronTensor& eri_ao,
                                               const Matrix<double>& c_matrix, int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_NONRELORBITALGRADIENT_H
