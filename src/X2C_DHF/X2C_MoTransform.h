#ifndef RERDMFT_X2C_DHF_X2C_MOTRANSFORM_H
#define RERDMFT_X2C_DHF_X2C_MOTRANSFORM_H

#include <complex>

#include "Matrix.h"
#include "ElectronRepulsion.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

// Transforms the (fixed, one-electron-only) X2C Hamiltonian into the
// converged X2C-HF spinor ("molecular orbital") basis spanned by the
// columns of `c_matrix` (X2C_HF.h's X2CHartreeFockResult::c_matrix =
// X_Large * U, S_Large-orthonormal by construction): H_MO = C^dagger
// H_x2c C. Purely generic (no X2C-specific structure at all -- same
// operation as C4_DHF/RkbMoTransform.h's rkbMoOneElectronTransform,
// reimplemented here rather than reused, matching this project's
// established "each physics context gets its own small wrapper" style
// even for a mathematically-shared one-liner).
Matrix<std::complex<double>> x2cMoOneElectronTransform(const Matrix<std::complex<double>>& h_x2c,
                                                        const Matrix<std::complex<double>>& c_matrix);

// Transforms the dense, REAL, physics-notation Large-component
// spin-orbital two-electron Coulomb tensor (NON_REL/
// ClosedShellSpinOrbitals.h's closedShellSpinOrbitalTwoElectron output
// -- the SAME ordinary, non-relativistic integrals X2C_FockMatrix.h
// uses, no picture-change correction) into a DENSE COMPLEX tensor over
// the `c_matrix` spinor basis, still in physics notation:
//   <p q|r s>_MO = sum_ABCD conj(c_Ap) conj(c_Bq) c_Cr c_Ds <A B|C D>_AO
// (bra legs p,q get CONJUGATED coefficients; ket legs r,s do not --
// `c_matrix` is complex even though the AO-basis `eri` is real, since
// X2C's spin-orbit coupling only enters via `h_x2c`/`c_matrix`, never
// the two-electron integrals themselves). Performed as four sequential
// O(n^4 * n_mo) leg transforms (the AO tensor is converted to complex
// once, up front, then reuses EXACTLY C4_DHF/RkbMoTransform.cpp's own
// leg-transform technique, reimplemented independently here per this
// project's established convention -- see that file's own comment for
// why NON_REL/MoIntegralTransform.cpp and it don't share code either)
// rather than one O(n^8) direct sum.
//
// Intended for small systems: this costs O(n^5) time, same caveat as
// every other MO-basis transform in this project.
Tensor4<std::complex<double>> x2cMoTwoElectronTransformPhysics(
    const Tensor4<double>& eri_ao_physics, const Matrix<std::complex<double>>& c_matrix);

// Same result as x2cMoTwoElectronTransformPhysics above (verified to
// agree to floating-point precision at the default threshold before
// being trusted -- see main.cpp), computed via Utils/
// Cholesky_Decomposition.h's pivoted Cholesky decomposition instead of
// a direct 4-leg transform. `eri_ao_physics` is already dense and
// already in physics notation, so this is a thin, direct pass-through
// to choleskyTransformEriMixed (no notation conversion needed at all,
// unlike NON_REL's own Cholesky counterpart) -- kept as its own named
// function purely for API consistency with every other MO-transform
// pair in this project.
Tensor4<std::complex<double>> x2cMoTwoElectronTransformPhysicsCholesky(
    const Tensor4<double>& eri_ao_physics, const Matrix<std::complex<double>>& c_matrix,
    double threshold = 1e-10);

// The MO-basis spinor integrals as a unique-element store (Utils/SymmetricEri.h), computed in slabs straight
// from the packed SPATIAL AO integrals (closed-shell spin structure <AB|CD> = (AC|BD) delta delta) -- neither the
// dense spin-orbital AO tensor nor a dense MO tensor is formed.
SymmetricEri<std::complex<double>> x2cMoTwoElectronSymmetric(const PackedTwoElectronTensor& eri_spatial_chemist,
                                                              const Matrix<std::complex<double>>& c_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_MOTRANSFORM_H
