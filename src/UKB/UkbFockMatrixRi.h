#ifndef RERDMFT_UKBFOCKMATRIXRI_H
#define RERDMFT_UKBFOCKMATRIXRI_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// RI/density-fitting counterpart of UkbFockMatrixDirect.h's ukbFockTwoElectronDirect: builds the
// two-electron (J-K) contribution to the Fock matrix in the UKB spinor basis [Large-alpha,
// Large-beta, Small-alpha, Small-beta], from a general (not necessarily spin-block-diagonal)
// Hermitian UKB density matrix PLUS its own occupied-orbital decomposition -- but from a
// PRE-BUILT, metric-orthogonalized RI 3-center tensor (AO_ints/ThreeCenterIntegrals.h's
// riThreeCenterTensor) instead of recomputing AO integrals fresh every build. `eri3_LL`/`eri3_SS`
// are real, shape (n_aux, nl*nl)/(n_aux, ns*ns), row P holding the (contiguous) nl x nl / ns x ns
// block (mu nu|P) -- see ThreeCenterIntegrals.h's own convention.
//
// No (LS|P) tensor exists or is needed: see project memory project-ri-pauto-kr-validation. The
// derivation this function implements (verified algebraically before trusting it, same discipline
// as UkbFockMatrixDirect.h's own header comment, which this directly extends to RI):
//
//   J_f(a,b) = sum_P (ab|P)_type(f) * X_P,   X_P = sum_g sum_{q,r in g} (qr|P)_type(g) * P_gg(r,q)
//
// (type(f) in {LL,SS} depending on whether f is spatially Large or Small; X_P is REAL and shared
// by both spatial types, since the AO-level integrals are real and only Re(P) survives a Hermitian
// density's trace against them -- same simplification MOLGW's own setup_hartree_ri uses for a
// complex p_matrix).
//
//   K_fu(a,b) = -sum_i occ_i * conj(tmp_f(i,a)) * tmp_u(i,b),
//   tmp_f(i,a) = sum_{c in f} conj(c_occ(offset[f]+c, i)) * (ac|P)_type(f)
//
// where `c_occ` (n_ukb x n_occ, caller-supplied) and `occupations` are the UKB-AO-basis occupied
// orbitals and their occupation numbers.
//
// Why these are CALLER-supplied rather than recovered here from `density_ukb` itself (an earlier
// version tried exactly that, twice, and got it wrong both times -- see below): the UKB AO basis
// is not orthonormal AND, for the Small component specifically, genuinely linearly DEPENDENT
// (duplicate (l,exponent) shells from buildRkbSmallBasis's raising/lowering construction), so
// `density_ukb` is idempotent only via P S P = P, not P P = P. Recovering orbitals from
// `density_ukb` therefore needs SOME orthogonalization of the (rank-deficient) UKB overlap, and
// doing that FRESH, independently from whatever orthogonalization built `density_ukb` in the
// first place, is fragile: a direction sitting just above a from-scratch threshold -- and thus
// KEPT, with a correspondingly large 1/sqrt(eigenvalue) -- can amplify ordinary floating-point
// noise in `density_ukb` by many orders of magnitude, and that amplified noise feeds back into the
// next SCF iteration's density, growing geometrically (confirmed: SPEED_OF_LIGHT far from its
// physical value, e.g. an artificial 100000, made a fresh UKB-overlap orthogonalization this
// unstable within 2-3 iterations -- energies reaching 1e+204 -- even though it worked cleanly on
// LiH/6-31G and CO/6-31G at the physical speed of light; a yet earlier version skipped the
// non-orthogonality altogether, as if the UKB basis were orthonormal, caught by a ~0.5 Hartree
// discrepancy on LiH; both found and fixed 2026-10-06). The robust fix: never build a new
// orthogonalization of the UKB overlap at all -- recover the occupied RKB-basis orbitals using
// `x_full`/`s_full`, the SAME canonical-orthogonalization objects the SCF loop itself already uses
// every iteration to diagonalize the Fock matrix (so by construction x_full^dagger s_full x_full =
// I to the SAME precision the SCF loop already trusts, and the occupied RKB density was built FROM
// x_full in the first place, so it has zero support outside x_full's span -- no leakage, no
// amplification), then embed the RECOVERED coefficients -- not the density -- into the UKB AO
// basis via the ordinary, un-inverted `v_total` (see `rkbFockMatrix` below). Generalizes MOLGW's
// own setup_exchange_ri_x2c_1/_2 (m_hamiltonian_twobodies.f90) -- which builds exactly this
// half-transform/cross-GEMM pattern across the two SPIN channels of its (real-AO, no small
// component) X2C basis -- to all 4 UKB blocks (spin AND spatial flavor) at once: f=u gives the 4
// diagonal blocks (same pattern as x2c_1's ZHERK), f!=u gives the 6 independent cross blocks (same
// pattern as x2c_2's cross ZGEMM) -- K_uf = K_fu^dagger, not independently built.
//
// Cost: O(n_aux * n_occ * (nl^2+ns^2+nl*ns)) for K (n_aux small GEMMs) + O(n_aux*(nl^2+ns^2)) for J
// -- categorically cheaper than ukbFockTwoElectronDirect's O(n^4)-shaped AO loops, at the cost of
// the (small, see project memory) one-time RI tensor storage this function is handed.
Matrix<std::complex<double>> riFockTwoElectronDirect(std::size_t nl, std::size_t ns,
                                                       const Matrix<double>& eri3_LL,
                                                       const Matrix<double>& eri3_SS,
                                                       const Matrix<std::complex<double>>& density_ukb,
                                                       const Matrix<std::complex<double>>& c_occ_ukb,
                                                       const std::vector<double>& occupations);

// SCF_DIRECT_4C's RI counterpart -- the "ERI source" for C4_DHF.h's generic dhfScfImpl, exactly
// mirroring UkbDirectEriSource (UkbFockMatrixDirect.h) except it holds the (small, see project
// memory) pre-built RI 3-center tensors instead of the raw AO bases + recomputing everything every
// build. `v_total` is the SAME RKB<->UKB embedding matrix UkbDirectEriSource uses; `s_full`/
// `x_full` are the SAME RKB-basis metric/canonical-orthogonalization the SCF loop itself already
// builds and uses every iteration (C4_DHF.cpp's dhfScfImpl) -- reused here, not rebuilt, for the
// occupied-orbital recovery riFockTwoElectronDirect's own header comment explains.
struct RiDirectEriSource {
  std::size_t nl = 0, ns = 0;
  Matrix<double> eri3_LL, eri3_SS;
  Matrix<std::complex<double>> v_total;
  Matrix<std::complex<double>> s_full, x_full;
};

// F = h_rkb + V_total^dagger * riFockTwoElectronDirect(..., V_total*P*V_total^dagger, V_total*C_occ_rkb,
// occ) * V_total, where C_occ_rkb/occ come from eigendecomposing P_eff_rkb = X_full^dagger (S_full
// density_matrix S_full) X_full (a genuine plain-matrix projector -- see riFockTwoElectronDirect's
// header comment) and embedding its eigenvalue~1 eigenvectors via C_occ_rkb = X_full * Y_occ. This
// drops into C4_DHF.cpp's dhfScfImpl<Eri> with no other change to the SCF loop.
Matrix<std::complex<double>> rkbFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                            const RiDirectEriSource& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_UKBFOCKMATRIXRI_H
