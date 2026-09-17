#ifndef RERDMFT_RKBPOSITIVEENERGYHAMILTONIAN_H
#define RERDMFT_RKBPOSITIVEENERGYHAMILTONIAN_H

#include <complex>

#include "Matrix.h"
#include "PhysicalConstants.h"

namespace rerdmft {

// Builds the positive-energy (Large-component-dominant, nonrelativistic-
// comparable) branch of the RKB core Hamiltonian by EXACT Feshbach/Loewdin
// partitioning of H_RKB, in a way that never forms the catastrophically
// cancelling -2c^2*S_small + F_small sum that H_RKB's Small-Small block
// otherwise contains (see UkbHamiltonian.cpp/DiracKinetic.cpp: that block
// is -2c^2*S_uKB + V_uKB, summed in double precision inside
// ukbHamiltonianMatrix well before the RKB transformation -- at large c
// this destroys any part of V_uKB smaller than roughly c^2 * 1e-16, which
// is what makes H_RKB_ortho's diagonalization (main.cpp) lose the
// nonrelativistic limit and Kramers-pair symmetry at extreme
// SPEED_OF_LIGHT values).
//
// Writing H_RKB in 2x2 block form (each block 2*nLarge x 2*nLarge),
//   H_RKB = [[A, B], [B^dagger, D]],   D = -2c^2*S_small + F_small,
// with A, B exactly H_RKB's Large-Large and Large-Small blocks (both
// entirely free of the cancellation, since neither ever contains a -2c^2
// term), and S_small, F_small the RKB-transformed Small-component overlap
// (RkbOverlap.h's rkbSmallOverlapMatrix) and nuclear-attraction
// (rkbSmallVextMatrix) matrices -- both exactly independent of c -- the
// Feshbach-eliminated effective Hamiltonian for the E~0 (positive-energy)
// solutions is
//   H_eff = A - B D^-1 B^dagger.
// Since D = -2c^2*(S_small - F_small/(2c^2)) exactly for any nonzero c,
//   D^-1 = -1/(2c^2) * (S_small - F_small/(2c^2))^-1
// exactly too, so
//   H_eff = A + (1/(2c^2)) * B * (S_small - F_small/(2c^2))^-1 * B^dagger,
// computed here by inverting the well-scaled, O(1) matrix
// (S_small - F_small/(2c^2)) directly (via LinearAlgebra.h's
// invertHermitian) -- never the ill-conditioned D itself. The result is
// (2*nLarge x 2*nLarge), matching A's dimensions.
Matrix<std::complex<double>> rkbPositiveEnergyHamiltonian(
    const Matrix<std::complex<double>>& h_rkb, const Matrix<std::complex<double>>& s_small,
    const Matrix<std::complex<double>>& f_small, double speed_of_light = kSpeedOfLight);

// Loewdin-orthonormalizes H_eff (rkbPositiveEnergyHamiltonian's result)
// with X_Large repeated block-diagonally across the Large-alpha and
// Large-beta spin blocks:
//   H_eff_ortho = diag(X_Large,X_Large)^dagger H_eff diag(X_Large,X_Large).
// Throws std::runtime_error if h_eff's dimensions are not 2*x_large.rows().
Matrix<std::complex<double>> positiveEnergyOrthoHamiltonian(
    const Matrix<std::complex<double>>& h_eff, const Matrix<double>& x_large);

}  // namespace rerdmft

#endif  // RERDMFT_RKBPOSITIVEENERGYHAMILTONIAN_H
