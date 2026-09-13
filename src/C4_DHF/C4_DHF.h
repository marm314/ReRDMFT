#ifndef RERDMFT_C4_DHF_H
#define RERDMFT_C4_DHF_H

#include <complex>
#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "RkbTwoElectron.h"

namespace rerdmft {

// One SCF cycle's diagnostics: the one-body (Fock_ortho) eigenvalues
// obtained by diagonalizing with X_full that iteration, plus the energy
// and convergence metrics evaluated for that same cycle.
struct ScfIteration {
  int iteration = 0;
  double energy = 0.0;
  double density_change = 0.0;  // max|P_new - P_current|; NaN on the first iteration
  double energy_change = 0.0;   // |energy - previous energy|; NaN on the first iteration
  std::vector<double> orbital_energies;  // Fock_ortho eigenvalues (ascending), this cycle
};

// Result of a converged (or not) 4-component Dirac-Hartree-Fock SCF run.
struct DiracHartreeFockResult {
  bool converged = false;
  int iterations = 0;
  double electronic_energy = 0.0;         // (1/2) Re Tr[P (H_RKB + F)] at convergence
  double nuclear_repulsion_energy = 0.0;  // sum_{A<B} Z_A Z_B / R_AB (constant, geometry only)
  double total_energy = 0.0;              // electronic_energy + nuclear_repulsion_energy
  std::vector<double> orbital_energies;   // final Fock_ortho eigenvalues, ascending
  Matrix<std::complex<double>> density_matrix;  // final (unmixed) density, RKB spinor AO basis
  Matrix<std::complex<double>> fock_matrix;     // final Fock matrix, RKB spinor AO basis
  Matrix<std::complex<double>> c_dhf;           // final C_DHF = X_full * U
  // Final Fock_ortho = X_full^dagger F X_full eigenvectors, in the SAME
  // orthonormalized RKB basis as H_RKB_ortho's -- so KramersSymmetry.h's
  // maxKramersPartnerDeviation applies to these directly, to check that
  // diagonalizing the (self-consistent) Fock matrix still yields genuine
  // Kramers pairs, not just degenerate-looking eigenvalues.
  Matrix<std::complex<double>> fock_ortho_eigenvectors;
  std::vector<ScfIteration> history;  // one entry per iteration, in order
};

// Sum_{A<B} Z_A*Z_B / |R_A - R_B| -- the classical nucleus-nucleus Coulomb
// repulsion energy, constant for a fixed geometry (Bohr, as stored in
// Input::geometry()). Z_A from Element.h's atomicNumber.
double nuclearRepulsionEnergy(const std::vector<Atom>& geometry);

// Runs the 4-component Dirac-Hartree-Fock (DHF) self-consistent field
// procedure in the restricted-kinetic-balance (RKB) spinor AO basis,
// starting from `initial_density` (typically the core-Hamiltonian-guess
// density already built by RkbDensityMatrix.h in main.cpp). Each
// iteration:
//   1. Builds F = rkbFockMatrix(h_rkb, eri, P_current) (RkbFockMatrix.h).
//   2. Orthonormalizes and diagonalizes it: F_ortho = X_full^dagger F
//      X_full, eigenvectors U (LinearAlgebra.h's diagonalizeHermitian).
//   3. Builds C_DHF = X_full * U and the corresponding density P_new
//      (RkbDensityMatrix.h), occupying the lowest n_electrons
//      positive-energy spinors.
//   4. Evaluates the SCF energy with the density that actually built F:
//      E = (1/2) Re Tr[P_current (H_RKB + F)] -- the standard
//      self-consistent-pair HF energy expression.
//   5. Checks convergence: both the density change |P_new - P_current|
//      and the energy change from the previous iteration must fall below
//      their tolerances (skipped on the first iteration, which has no
//      previous energy to compare against).
//   6. If not converged, linearly mixes the density fed into the next
//      iteration: P_current <- mixing*P_new + (1-mixing)*P_current
//      (`mixing`, e.g. Input::mixing(), is a damping aid for convergence
//      only -- it does not change what a converged result looks like,
//      only how the iterations get there).
//
// Stops after `max_iterations` regardless of convergence (converged is
// false in that case, but the last iteration's results are still
// returned). total_energy adds the constant nuclear_repulsion_energy.
DiracHartreeFockResult runDiracHartreeFockScf(
    const Matrix<std::complex<double>>& h_rkb, const RkbTwoElectronTensor& eri,
    const Matrix<std::complex<double>>& x_full,
    const Matrix<std::complex<double>>& initial_density, int n_electrons,
    const std::vector<Atom>& geometry, double mixing, int max_iterations = 100,
    double energy_tolerance = 1e-8, double density_tolerance = 1e-6);

}  // namespace rerdmft

#endif  // RERDMFT_C4_DHF_H
