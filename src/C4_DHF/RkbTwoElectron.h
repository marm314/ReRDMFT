#ifndef RERDMFT_RKBTWOELECTRON_H
#define RERDMFT_RKBTWOELECTRON_H

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

// Storage for <Spinor_A Spinor_B|Spinor_C Spinor_D>: only the integrals not related by the electron-exchange
// symmetry <AB|CD> = <BA|DC> and the Hermiticity <AB|CD> = conj <CD|AB> are kept (about n^4/4 complex numbers),
// every other element is rebuilt on access -- see Utils/SymmetricEri.h.
using RkbTwoElectronTensor = SymmetricEri<std::complex<double>>;

// Builds the full two-electron Coulomb repulsion tensor in the restricted-
// kinetic-balance (RKB) 4-component spinor basis, in PHYSICS notation:
//   <Spinor_A Spinor_B | Spinor_C Spinor_D>
//     = integral integral Spinor_A^*(r1) Spinor_B^*(r2) (1/r12)
//                          Spinor_C(r1)   Spinor_D(r2)   dr1 dr2,
// with Spinor_A, Spinor_C acting on electron 1 and Spinor_B, Spinor_D on
// electron 2 (i.e. <AB|CD> = (AC|BD) in chemist's notation). Each spinor
// index ranges over the same (4*nLarge)-dimensional RKB basis as H_RKB
// (RkbHamiltonian.h): [Large-alpha, Large-beta, RKB-Small-alpha-partner,
// RKB-Small-beta-partner], each block of size nLarge. The result is
// stored in an RkbTwoElectronTensor (electron-exchange symmetry only
// -- see its own comment); Hermiticity is verified, not exploited for
// storage, as a debug-only check (main.cpp).
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
// whichever legs are Small-type. One of the four RKB-Small(y1),RKB-
// Small(y2) electron-1/electron-2 partner-flavor combinations is skipped
// and recovered by an index permutation of another (see the .cpp), via
// the same electron-exchange symmetry used for storage. Both time and
// memory still scale steeply with basis size, so this remains intended
// for small test systems.
// `use_cholesky` (default false, matching Input.h's CHOLESKY keyword
// default) decomposes the dominant-cost (Small,Small|Small,Small) piece
// via Utils/Cholesky_Decomposition.h's pivoted Cholesky decomposition
// before projecting it into the four RKB-Small(y1),RKB-Small(y2) blocks,
// instead of directly quarter-transforming the full (n_small)^4 tensor on
// each of its two electron-pair legs (see the .cpp for the derivation of
// why this is a legitimate, verified-equivalent reformulation) -- this is
// the one Cholesky application that compresses well, so worth enabling
// explicitly (CHOLESKY TRUE) for larger small-component bases.
RkbTwoElectronTensor rkbTwoElectronIntegrals(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis,
                                              const Matrix<std::complex<double>>& rkb_coefficients,
                                              bool use_cholesky = false, double cholesky_threshold = 1e-10);

// Projects one real symmetric (n_small x n_small) pair vector of the Small-basis Coulomb matrix into the
// RKB-Small(y) partner flavor (y = 0: alpha-partner, 1: beta-partner): W(p,q) = sum over the Small basis's
// two spin blocks of conj(c(p,a)) v(a,b) c(q,b) -- the projection rkbTwoElectronIntegrals applies to each
// Cholesky vector of (SS|SS), exposed for C4_DHF/RkbCholesky.h. Returns an (n_large x n_large) Hermitian matrix.
Matrix<std::complex<double>> rkbProjectSmallVector(const Matrix<double>& v,
                                                    const Matrix<std::complex<double>>& rkb_coefficients,
                                                    std::size_t y, std::size_t n_large, std::size_t n_small);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTWOELECTRON_H
