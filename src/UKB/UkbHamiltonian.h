#ifndef RERDMFT_UKBHAMILTONIAN_H
#define RERDMFT_UKBHAMILTONIAN_H

#include <complex>
#include <vector>

#include "DiracKinetic.h"
#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds the unrestricted-kinetic-balance core Hamiltonian
//   H_UKB = T (Dirac kinetic) + rest-energy alignment + Vext
// by summing diracKineticMatrix, diracRestEnergyMatrix (DiracKinetic.h) and
// vextMatrix (Vext.h) in the 4-component spinor basis [Large-alpha,
// Large-beta, Small-alpha, Small-beta] (see SpinorBasis).
Matrix<std::complex<double>> ukbHamiltonianMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, const std::vector<Atom>& geometry,
    double speed_of_light = kSpeedOfLight);

}  // namespace rerdmft

#endif  // RERDMFT_UKBHAMILTONIAN_H
