#ifndef RERDMFT_RKBTWOELECTRON_H
#define RERDMFT_RKBTWOELECTRON_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the full two-electron Coulomb repulsion tensor in the restricted-
// kinetic-balance (RKB) 4-component spinor basis, in PHYSICS notation:
//   <Spinor_A Spinor_B | Spinor_C Spinor_D>
//     = integral integral Spinor_A^*(r1) Spinor_B^*(r2) (1/r12)
//                          Spinor_C(r1)   Spinor_D(r2)   dr1 dr2,
// with Spinor_A, Spinor_C acting on electron 1 and Spinor_B, Spinor_D on
// electron 2 (i.e. <AB|CD> = (AC|BD) in chemist's notation). Each spinor
// index ranges over the same (4*nLarge)-dimensional RKB basis as H_RKB
// (RkbHamiltonian.h): [Large-alpha, Large-beta, RKB-Small-alpha-partner,
// RKB-Small-beta-partner], each block of size nLarge.
//
// 1/r12 is a scalar operator (spin- and component-independent), so a
// nonzero <A|Op(r1)|C>-type density requires A and C to be the same
// "flavor" (both Large-alpha, both Large-beta, or both the same RKB-Small
// partner type); cross-flavor combinations are exactly zero and are not
// separately stored. Building the RKB-Small blocks requires the RKB
// coefficients (RkbTransformation.h): a genuine 2-spinor RKB-Small basis
// function has contributions from BOTH the unrestricted-kinetic-balance
// Small-alpha and Small-beta functions, so its two-electron integrals need
// the underlying unrestricted-kinetic-balance Small AO integrals
// transformed through both spin blocks of `rkb_coefficients` and summed.
//
// Computed via three real spatial-AO electron-repulsion tensors
// (ElectronRepulsion.h) -- (Large,Large|Large,Large), (Large,Large|Small,
// Small), and (Small,Small|Small,Small), Small meaning the unrestricted-
// kinetic-balance basis -- each transformed through `rkb_coefficients` on
// whichever legs are Small-type. Both time and memory scale steeply with
// basis size (the result alone is (4*nLarge)^4 complex numbers), so this
// is intended for small test systems.
Tensor4<std::complex<double>> rkbTwoElectronIntegrals(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis,
    const Matrix<std::complex<double>>& rkb_coefficients);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTWOELECTRON_H
