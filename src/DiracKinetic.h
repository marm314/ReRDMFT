#ifndef RERDMFT_DIRACKINETIC_H
#define RERDMFT_DIRACKINETIC_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Speed of light in atomic units, as used throughout this project.
inline constexpr double kSpeedOfLight = 137.036;

// Builds the Dirac kinetic-energy matrix T = -i c (alpha . grad_r) in the
// 4-component spinor basis [Large-alpha, Large-beta, Small-alpha,
// Small-beta] (see SpinorBasis), from the already-normalized large- and
// small-component scalar cartesian AO bases.
//
// alpha_x, alpha_y, alpha_z are the standard 4x4 Dirac matrices built from
// the 2x2 Pauli matrices,
//   alpha_k = [[0, sigma_k], [sigma_k, 0]],
//   sigma_x = [[0,1],[1,0]], sigma_y = [[0,-i],[i,0]], sigma_z = [[1,0],[0,-1]],
// so the operator is zero within the Large-Large and Small-Small blocks and
// couples only Large <-> Small. The resulting matrix is complex (from the
// sigma_y term) and Hermitian.
Matrix<std::complex<double>> diracKineticMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis,
    double speed_of_light = kSpeedOfLight);

}  // namespace rerdmft

#endif  // RERDMFT_DIRACKINETIC_H
