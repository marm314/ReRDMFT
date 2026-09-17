#ifndef RERDMFT_RKBMOTRANSFORM_H
#define RERDMFT_RKBMOTRANSFORM_H

#include <complex>
#include <cstddef>

#include "Matrix.h"
#include "RkbTwoElectron.h"
#include "Tensor4.h"

namespace rerdmft {

// Transforms the RKB spinor AO-basis one-electron Hamiltonian into the
// converged DHF spinor ("molecular orbital") basis spanned by the
// columns of `c_dhf` (C4_DHF.h's DiracHartreeFockResult::c_dhf = X_full
// * U -- linear combinations of the RKB basis, S_full-orthonormal by
// construction): H_MO = C_dhf^dagger H_RKB C_dhf.
Matrix<std::complex<double>> rkbMoOneElectronTransform(
    const Matrix<std::complex<double>>& h_rkb, const Matrix<std::complex<double>>& c_dhf);

// Transforms the packed, physics-notation RKB spinor AO-basis two-
// electron tensor (RkbTwoElectron.h) into a DENSE tensor over the SAME
// `c_dhf` spinor basis, still in physics notation directly (no chemist/
// physics conversion needed here -- unlike NON_REL/MoIntegralTransform.h,
// RkbTwoElectronTensor already stores <A B|C D> in physics notation, so
// the MO tensor comes out in the SAME convention Hessian_opt/
// GeneralizedFock.h expects with no reindexing):
//   <p q|r s>_MO = sum_ABCD conj(c_Ap) conj(c_Bq) c_Cr c_Ds <A B|C D>_AO
// (bra legs p,q get CONJUGATED coefficients; ket legs r,s do not --
// c_dhf is complex, unlike NON_REL's real C), performed as four
// sequential O(n_ao^4 * n_mo) leg transforms rather than one O(n^8)
// direct sum (same technique as RkbTwoElectron.cpp's transformLeg and
// NON_REL/MoIntegralTransform.cpp's transformLeg1..4, reimplemented
// independently here rather than reusing either).
//
// Intended for small systems: this densifies an already-packed
// (electron-exchange-halved) O(n_ao^4) tensor into a full O(n_mo^4)
// dense one and costs O(n^5) time -- the RKB spinor dimension n_ao =
// 4*n_large, so this is proportionally heavier than the analogous
// NON_REL transform for the same molecule (dimension 4x larger, cost
// ~4^5 = 1024x more for the two-electron step).
Tensor4<std::complex<double>> rkbMoTwoElectronTransformPhysics(
    const RkbTwoElectronTensor& eri_ao_physics, const Matrix<std::complex<double>>& c_dhf);

// Same result as rkbMoTwoElectronTransformPhysics above (verified to
// agree to floating-point precision at the default threshold before
// being trusted -- see main.cpp), computed via Utils/
// Cholesky_Decomposition.h's pivoted Cholesky decomposition instead of
// a direct 4-leg transform. RkbTwoElectronTensor's own Hermiticity
// relation <A B|C D> = conj(<C D|A B>) (documented in RkbTwoElectron.h)
// is exactly the bra=(A,B)/ket=(C,D) grouping choleskyDecomposeEri
// requires, so this is a thin, direct pass-through -- densify then
// choleskyTransformEri, no chemist/physics reindexing needed at all
// (unlike NON_REL's own Cholesky counterpart).
Tensor4<std::complex<double>> rkbMoTwoElectronTransformPhysicsCholesky(
    const RkbTwoElectronTensor& eri_ao_physics, const Matrix<std::complex<double>>& c_dhf,
    double threshold = 1e-10);

// The idempotent 1-RDM, in the `c_dhf` MO basis, of the single Slater
// determinant occupying the lowest `n_electrons` POSITIVE-energy
// states -- NOT the lowest `n_electrons` states overall. The RKB
// working basis of dimension `rkb_dim` (= 4*n_large) is ordered by
// ascending Fock_ortho/H_RKB_ortho eigenvalue, with the first rkb_dim/2
// states being the negative-energy ("Dirac sea") branch, which is left
// entirely unoccupied in the standard no-pair DHF approximation used
// throughout this project (see C4_DHF/RkbDensityMatrix.h, which this
// mirrors for the AO basis; here it is diagonal in the MO/eigenvector
// basis instead): D_MO(p,p) = 1 for p in [rkb_dim/2, rkb_dim/2 +
// n_electrons), 0 otherwise (off-diagonal 0, since this basis IS the
// one that diagonalizes the converged Fock operator).
Matrix<std::complex<double>> occupiedPositiveEnergyDensity(std::size_t rkb_dim, int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_RKBMOTRANSFORM_H
