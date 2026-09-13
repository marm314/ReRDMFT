#ifndef RERDMFT_NONRELHARTREEFOCK_H
#define RERDMFT_NONRELHARTREEFOCK_H

#include <vector>

#include "ElectronRepulsion.h"
#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// One SCF cycle's diagnostics -- see C4_DHF.h's ScfIteration, which this
// mirrors exactly, just with real (not complex) orbital energies.
struct NonRelScfIteration {
  int iteration = 0;
  double energy = 0.0;
  double density_change = 0.0;  // max|P_new - P_current|; NaN on the first iteration
  double energy_change = 0.0;   // |energy - previous energy|; NaN on the first iteration
  std::vector<double> orbital_energies;  // Fock_ortho eigenvalues (ascending), this cycle
};

// Result of a converged (or not) nonrelativistic (restricted, closed-
// shell) Hartree-Fock SCF run. Mirrors C4_DHF.h's DiracHartreeFockResult;
// everything here is real (Large-component-only, no spinor structure).
struct NonRelHartreeFockResult {
  bool converged = false;
  int iterations = 0;
  double electronic_energy = 0.0;         // (1/2) Tr[P (H_core + F)] at convergence
  double nuclear_repulsion_energy = 0.0;  // sum_{A<B} Z_A Z_B / R_AB (constant, geometry only)
  double total_energy = 0.0;              // electronic_energy + nuclear_repulsion_energy
  std::vector<double> orbital_energies;   // final Fock_ortho eigenvalues, ascending
  Matrix<double> density_matrix;          // final (unmixed) density, Large AO basis
  Matrix<double> fock_matrix;             // final Fock matrix, Large AO basis
  Matrix<double> c_matrix;                // final C = X_large * U
  std::vector<NonRelScfIteration> history;  // one entry per iteration, in order
};

// Builds the closed-shell (restricted) nonrelativistic density matrix
//   P(p,r) = 2 * sum_{i=1}^{n_electrons/2} C(p,i) * C(r,i),
// doubly occupying the lowest n_electrons/2 spatial orbitals (columns of
// C, ascending energy order -- each spatial orbital holds two electrons,
// unlike the 4-component spinor basis where each spinor holds at most
// one). Throws std::runtime_error if n_electrons is not a positive even
// number (closed-shell RHF requires it) that fits within C's column
// count.
Matrix<double> nonRelDensityMatrix(const Matrix<double>& c, int n_electrons);

// Builds the standard (restricted, closed-shell) nonrelativistic Fock
// matrix, in physics notation (p,r electron 1; q,s electron 2):
//   F(p,r) = H_core(p,r) + J(p,r) - (1/2) K(p,r)
//   J(p,r) = sum_{q,s} P(s,q) <p q|r s>   (Hartree/Coulomb)
//   K(p,r) = sum_{q,s} P(s,q) <p q|s r>   (exchange)
// `eri` (ElectronRepulsion.h's twoElectronIntegralsPacked -- storing only
// the unique values, unlike twoElectronIntegrals' dense form used
// internally by RkbTwoElectron.cpp) is CHEMIST notation (pq|rs) directly,
// not physics notation -- <A B|C D> = chemist(A,C,B,D) (see .cpp), so
// <p q|r s> is eri(p,r,q,s), not eri(p,q,r,s).
// The 1/2 on K (absent from C4_DHF/RkbFockMatrix.h's spinor-basis
// formula) is exactly the standard closed-shell RHF factor: with P the
// TOTAL density (already including the factor of 2 for double
// occupancy), each spatial orbital's exchange contribution only comes
// from its OWN spin channel, not both -- unlike the Hartree term, which
// sees the total density of both spins. There is no "opposite-spin
// exchange" term here (contrast RkbFockMatrix.h's K_opposite): P is
// spin-block-diagonal with IDENTICAL alpha and beta blocks by
// construction (real, nonrelativistic, spin is a good quantum number),
// so that contribution is identically zero and is not computed at all.
Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const PackedTwoElectronTensor& eri,
                                 const Matrix<double>& density_matrix);

// Runs the standard nonrelativistic (restricted, closed-shell) Hartree-
// Fock SCF procedure in the Large-component AO basis. Builds the
// (Large,Large|Large,Large) two-electron repulsion tensor internally
// (ElectronRepulsion.h's twoElectronIntegralsPacked -- real, no
// restricted-kinetic-balance small component involved at all, storing only
// the unique values), then starting from
// `initial_density` (typically the core-Hamiltonian-guess density built
// from H_core's own eigenvectors in main.cpp), mirrors C4_DHF.h's
// runDiracHartreeFockScf exactly (same linear-mixing and OR-combined
// convergence conventions -- see there for the full derivation), only
// using X_large in place of X_full (no RKB small-component metric here):
// each iteration builds F, orthonormalizes/diagonalizes with X_large,
// builds the new density, evaluates E=(1/2)Tr[P_current(H_core+F)], and
// checks convergence (density change OR energy change below its
// tolerance, skipped on the first iteration). Stops after
// `max_iterations` regardless of convergence.
NonRelHartreeFockResult runNonRelativisticHartreeFock(
    const std::vector<BasisFunction>& large_basis, const Matrix<double>& h_core,
    const Matrix<double>& x_large, const Matrix<double>& initial_density, int n_electrons,
    const std::vector<Atom>& geometry, double mixing, int max_iterations,
    double energy_tolerance, double density_tolerance);

}  // namespace rerdmft

#endif  // RERDMFT_NONRELHARTREEFOCK_H
