#ifndef RERDMFT_RKBORBITALGRADIENT_H
#define RERDMFT_RKBORBITALGRADIENT_H

#include <complex>

#include "Matrix.h"
#include "RkbTwoElectron.h"

namespace rerdmft {

// Same idea as NON_REL/NonRelOrbitalGradient.h's
// nonRelOrbitalGradientEfficient, for the RKB spinor DHF case: avoids
// Hessian_opt/GeneralizedFock.h's O(n^5) dense-2-RDM contraction (n =
// 4*n_large here, so that contraction is ~4^5 = 1024x more expensive
// than the analogous NON_REL one for the same molecule -- impractical
// past small STO-3G-scale bases) by reusing rkbFockMatrix directly
// (already exactly Hartree + same-flavor exchange + opposite-flavor
// exchange, at O(n^4)) on the density built fresh from the converged
// `c_dhf` (via rkbDensityMatrix; NOT the SCF loop's stored, one-
// iteration-stale density), then transforming into the MO basis:
//   g_pq = 2 * FockLike(p,q) * ([p occupied] - [q occupied])
// (same derivation as the NON_REL header -- Dyall Eq. 8.30's own
// "nonzero only if the second index is occupied" remark, plus
// FockLike's Hermiticity, and Hessian_opt/OrbitalGradient.h's
// deliberate factor of 2 -- see that header), with "occupied" meaning
// the lowest `n_electrons` POSITIVE-energy states (RkbDensityMatrix.h/
// RkbMoTransform.h's convention -- the negative-energy branch is never
// occupied). No spin-orbital expansion is needed here (the RKB spinor
// basis already IS spin-orbital-like). Verified to reproduce
// Hessian_opt's general (slower) path numerically; see main.cpp.
//
// Only p >= q is computed/stored, matching OrbitalGradient.h.
Matrix<std::complex<double>> dhfOrbitalGradientEfficient(
    const Matrix<std::complex<double>>& h_rkb, const RkbTwoElectronTensor& eri_rkb,
    const Matrix<std::complex<double>>& c_dhf, int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_RKBORBITALGRADIENT_H
