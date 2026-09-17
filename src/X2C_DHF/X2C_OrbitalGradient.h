#ifndef RERDMFT_X2C_DHF_X2C_ORBITALGRADIENT_H
#define RERDMFT_X2C_DHF_X2C_ORBITALGRADIENT_H

#include <complex>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Same idea as C4_DHF/RkbOrbitalGradient.h's dhfOrbitalGradientEfficient
// (itself mirroring NON_REL/NonRelOrbitalGradient.h), for the X2C-HF
// case: avoids Hessian_opt/GeneralizedFock.h's O(n^2)-per-element dense-
// 2-RDM contraction by reusing X2C_DHF/X2C_FockMatrix.h's own
// x2cFockMatrix directly (already exactly Hartree + same-flavor
// exchange + opposite-flavor exchange, at O(n) per element) on the
// density built fresh from the converged `c_matrix` (via
// X2C_DHF/X2C_DensityMatrix.h's x2cDensityMatrix; NOT the SCF loop's
// stored, one-iteration-stale density), then transforming into the MO
// basis via X2C_MoTransform.h's own x2cMoOneElectronTransform:
//   g_pq = 2 * FockLike(p,q) * ([p occupied] - [q occupied])
// (same derivation as C4_DHF/RkbOrbitalGradient.h's own -- Dyall Eq.
// 8.30's "nonzero only if the second index is occupied" remark, plus
// FockLike's Hermiticity, and Hessian_opt/OrbitalGradient.h's
// deliberate factor of 2 -- see that header), with "occupied" meaning
// the lowest `n_electrons` spinors directly (no negative-energy branch
// to offset past at all, unlike the RKB case -- X2C's own decoupling
// already eliminated it). No spin-orbital expansion is needed here
// (the Large-component spin-orbital basis already IS spin-orbital-
// like). Verified to reproduce Hessian_opt's general (slower) path
// numerically; see main.cpp.
//
// Only p >= q is computed/stored, matching OrbitalGradient.h.
Matrix<std::complex<double>> x2cOrbitalGradientEfficient(const Matrix<std::complex<double>>& h_x2c,
                                                          const Tensor4<double>& eri,
                                                          const Matrix<std::complex<double>>& c_matrix,
                                                          int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_ORBITALGRADIENT_H
