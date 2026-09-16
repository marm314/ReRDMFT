#ifndef RERDMFT_X2C_DHF_X2C_HF_H
#define RERDMFT_X2C_DHF_X2C_HF_H

#include <complex>
#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// One SCF cycle's diagnostics -- see C4_DHF.h's ScfIteration, its own
// template.
struct X2CScfIteration {
  int iteration = 0;
  double energy = 0.0;
  double density_change = 0.0;  // max|P_new - P_current|; NaN on the first iteration
  double energy_change = 0.0;   // |energy - previous energy|; NaN on the first iteration
  std::vector<double> orbital_energies;
};

// Result of a converged (or not) X2C-HF SCF run -- see C4_DHF.h's
// DiracHartreeFockResult, its own template.
struct X2CHartreeFockResult {
  bool converged = false;
  int iterations = 0;
  double electronic_energy = 0.0;         // (1/2) Re Tr[P (h_x2c + F)] at convergence
  double nuclear_repulsion_energy = 0.0;  // sum_{A<B} Z_A Z_B / R_AB (constant, geometry only)
  double total_energy = 0.0;              // electronic_energy + nuclear_repulsion_energy
  std::vector<double> orbital_energies;   // final Fock_ortho eigenvalues, ascending
  Matrix<std::complex<double>> density_matrix;  // final (unmixed) density
  Matrix<std::complex<double>> fock_matrix;     // final Fock matrix (AO/Large-component basis)
  Matrix<std::complex<double>> c_matrix;        // final C = x_large * U
  Matrix<std::complex<double>> fock_ortho_eigenvectors;
  std::vector<X2CScfIteration> history;
};

// Runs the (approximate) X2C Hartree-Fock SCF -- EXACTLY the same
// self-consistent-pair procedure as C4_DHF.h's runDiracHartreeFockScf
// (that header documents each step in full; this is its direct
// template), just in the much smaller 2*nLarge-dimensional Large-
// component-only spin-orbital space instead of the full 4*nLarge RKB
// one, and with two DELIBERATE simplifications relative to exact X2C:
//
//   1. `h_x2c` (X2C_hamiltonian.h's own exact one-electron X2C
//      Hamiltonian) is built ONCE, from the bare one-electron RKB
//      Hamiltonian, and stays FIXED throughout the SCF -- no picture-
//      change correction is applied to the one-electron part as the
//      density changes (a "one-electron X2C" treatment, in the
//      standard nomenclature, not "X2C with an updated mean-field
//      picture-change correction").
//   2. `eri` is the ORDINARY (real, non-relativistic) two-electron
//      Coulomb tensor over the Large-component spin-orbital basis
//      (X2C_FockMatrix.h) -- no two-electron picture-change correction
//      of any kind is applied either.
//
// Each iteration:
//   1. F = x2cFockMatrix(h_x2c, eri, P_current) (X2C_FockMatrix.h) --
//      a GENERALIZED (not spin-restricted) Fock build, since h_x2c's
//      own spin-orbit coupling can make P genuinely spin-off-diagonal.
//   2. F_ortho = x_large^dagger F x_large, where `x_large` is ONLY the
//      large-component Loewdin orthogonalization
//      (X2C_hamiltonian.h's extractLargeComponentBlock(x_full, n) --
//      passed in directly here, already extracted by the caller) --
//      NEVER the exact Lambda renormalization h_x2c_ortho itself uses.
//      This is why the overall procedure is an APPROXIMATE X2C-HF, not
//      an exact one, on top of the two simplifications above.
//   3. C = x_large * U; P_new = x2cDensityMatrix(C, n_electrons)
//      (X2C_DensityMatrix.h), occupying the LOWEST n_electrons columns
//      -- there is no negative-energy branch to skip in this space at
//      all (X2C's own decoupling already eliminated it).
//   4. Evaluates the SCF energy with the density that actually built
//      F: E = (1/2) Re Tr[P_current (h_x2c + F)].
//   5. Checks convergence: EITHER the density change or the energy
//      change falling below its own tolerance is enough (OR, not AND)
//      -- skipped on the first iteration.
//   6. If not converged, linearly mixes the density fed into the next
//      iteration: P_current <- mixing*P_new + (1-mixing)*P_current.
//
// Stops after `max_iterations` regardless of convergence. total_energy
// adds the constant nuclear_repulsion_energy.
X2CHartreeFockResult runX2CHartreeFockScf(const Matrix<std::complex<double>>& h_x2c,
                                           const Tensor4<double>& eri,
                                           const Matrix<std::complex<double>>& x_large,
                                           const Matrix<std::complex<double>>& initial_density,
                                           int n_electrons, const std::vector<Atom>& geometry,
                                           double mixing, int max_iterations = 100,
                                           double energy_tolerance = 1e-8,
                                           double density_tolerance = 1e-6);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_HF_H
