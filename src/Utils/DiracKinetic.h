#ifndef RERDMFT_DIRACKINETIC_H
#define RERDMFT_DIRACKINETIC_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"
#include "PhysicalConstants.h"

namespace rerdmft {

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

// Builds the rest-energy-alignment matrix
//   spinor_a^dagger [[0_2, 0_2], [0_2, -2 c^2 I_2]] spinor_b
// in the same 4-component spinor basis: zero within the Large-Large block,
// -2c^2 * <AO_i|AO_j> within the Small-Small block (both spin blocks
// alike), and zero Large-Small coupling (block-diagonal, like the
// standard Dirac beta matrix). This is the standard beta*m*c^2 rest-mass
// term with a global -c^2 shift applied (i.e. diag(+c^2,-c^2) shifted by
// -c^2), chosen to align the resulting positive-energy branch with the
// nonrelativistic energy scale. Verified against an independent reference
// implementation (M. Rodriguez-Mayorga's m_relativistic.f90, MOLGW): its
// H4c_me only adds Vext (no additional +1*S) to the Large-Large block.
Matrix<std::complex<double>> diracRestEnergyMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis,
    double speed_of_light = kSpeedOfLight);

}  // namespace rerdmft

#endif  // RERDMFT_DIRACKINETIC_H
