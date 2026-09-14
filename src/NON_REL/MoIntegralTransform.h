#ifndef RERDMFT_MOINTEGRALTRANSFORM_H
#define RERDMFT_MOINTEGRALTRANSFORM_H

#include "ElectronRepulsion.h"
#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Transforms a real AO-basis one-electron matrix into the MO basis
// given real AO->MO coefficients `c` (n_ao x n_mo): H_MO = C^T H_AO C.
Matrix<double> moOneElectronTransform(const Matrix<double>& h_ao, const Matrix<double>& c);

// Transforms a real, packed (chemist-notation, 8-fold-symmetry-unique)
// AO-basis two-electron tensor into a DENSE MO-basis tensor, in PHYSICS
// notation directly: eri_mo(p,q,r,s) == <p q|r s>_MO (p,r on electron
// 1; q,s on electron 2) -- the convention Hessian_opt/GeneralizedFock.h
// expects, NOT chemist notation. Computed via
//   <p r|q s>_MO = sum_ABCD c_Ap c_Br c_Cq c_Ds (A B|C D)_AO
// (using physics<P Q|R S> = chemist(P,R,Q,S), so eri_mo(p,q,r,s) reads
// the chemist-notation MO tensor at (p,r,q,s)), performed as four
// sequential O(n_ao^4 * n_mo) leg transforms (same technique as
// C4_DHF/RkbTwoElectron.cpp's transformLeg, reimplemented here
// independently for real doubles rather than reusing that complex-only,
// file-local helper) instead of one O(n^8) direct sum. `c` is real, so
// no conjugation is needed anywhere in the transform.
//
// Intended for small systems: this densifies an O(n_ao^4)-packed tensor
// into an O(n_mo^4) dense one and costs O(n^5) time, the same caveat
// already carried by this project's other MO-basis/Hessian_opt code.
Tensor4<double> moTwoElectronTransformPhysics(const PackedTwoElectronTensor& eri_ao_chemist,
                                               const Matrix<double>& c);

}  // namespace rerdmft

#endif  // RERDMFT_MOINTEGRALTRANSFORM_H
