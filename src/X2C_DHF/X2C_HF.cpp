#include "X2C_HF.h"

#include <cmath>
#include <cstddef>

#include "DIIS.h"
#include "LinearAlgebra.h"
#include "NuclearRepulsion.h"
#include "X2C_DensityMatrix.h"
#include "X2C_FockMatrix.h"

namespace rerdmft {

namespace {

std::complex<double> traceOfProduct(const Matrix<std::complex<double>>& a,
                                     const Matrix<std::complex<double>>& b) {
  std::complex<double> trace(0.0, 0.0);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      trace += a(i, j) * b(j, i);
    }
  }
  return trace;
}

double maxAbsDifference(const Matrix<std::complex<double>>& a,
                         const Matrix<std::complex<double>>& b) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
    }
  }
  return max_diff;
}

}  // namespace

X2CHartreeFockResult runX2CHartreeFockScf(const Matrix<std::complex<double>>& h_x2c,
                                           const Tensor4<double>& eri,
                                           const Matrix<std::complex<double>>& x_large,
                                           const Matrix<std::complex<double>>& initial_density,
                                           int n_electrons, const std::vector<Atom>& geometry,
                                           double mixing, const Matrix<std::complex<double>>& overlap,
                                           int diis_size, int max_iterations,
                                           double energy_tolerance, double density_tolerance) {
  X2CHartreeFockResult result;
  result.nuclear_repulsion_energy = nuclearRepulsionEnergy(geometry);

  Matrix<std::complex<double>> p_current = initial_density;
  double previous_energy = 0.0;
  const bool use_diis = diis_size >= 2;
  Diis<std::complex<double>> diis(static_cast<std::size_t>(use_diis ? diis_size : 0));

  for (int iteration = 1; iteration <= max_iterations; ++iteration) {
    const Matrix<std::complex<double>> fock = x2cFockMatrix(h_x2c, eri, p_current);
    // DIIS: extrapolate F (the first iteration has a single pair and returns F unchanged).
    const Matrix<std::complex<double>> fock_scf =
        use_diis ? diis.extrapolate(scfCommutatorError(fock, p_current, overlap), fock) : fock;
    const Matrix<std::complex<double>> fock_ortho = dagger(x_large) * (fock_scf * x_large);
    const HermitianEigenResult eig = diagonalizeHermitian(fock_ortho);
    const Matrix<std::complex<double>> c_matrix = x_large * eig.eigenvectors;
    const Matrix<std::complex<double>> p_new = x2cDensityMatrix(c_matrix, n_electrons);

    // Self-consistent-pair energy: evaluated with the SAME density that
    // built this Fock matrix, not the (not yet computed) mixed one.
    const double energy = 0.5 * traceOfProduct(p_current, h_x2c + fock).real();
    const double density_change = maxAbsDifference(p_new, p_current);
    const double energy_change = std::abs(energy - previous_energy);

    result.iterations = iteration;
    result.electronic_energy = energy;
    result.orbital_energies = eig.eigenvalues;
    result.density_matrix = p_current;
    result.fock_matrix = fock_scf;
    result.c_matrix = c_matrix;
    result.fock_ortho_eigenvectors = eig.eigenvectors;

    X2CScfIteration record;
    record.iteration = iteration;
    record.energy = energy;
    record.density_change = iteration > 1 ? density_change : std::nan("");
    record.energy_change = iteration > 1 ? energy_change : std::nan("");
    record.orbital_energies = eig.eigenvalues;
    result.history.push_back(std::move(record));

    // Converged as soon as EITHER metric is satisfied (OR, not AND) --
    // see the header's note.
    const bool converged = iteration > 1 && (density_change < density_tolerance ||
                                              energy_change < energy_tolerance);
    previous_energy = energy;
    if (converged) {
      result.converged = true;
      break;
    }

    if (use_diis) {
      p_current = p_new;  // DIIS replaces the linear density mixing
      continue;
    }
    // Linear mixing: only affects the density fed into the NEXT
    // iteration's Fock build, not the energy/results just recorded above.
    Matrix<std::complex<double>> p_mixed(p_current.rows(), p_current.cols());
    for (std::size_t i = 0; i < p_mixed.rows(); ++i) {
      for (std::size_t j = 0; j < p_mixed.cols(); ++j) {
        p_mixed(i, j) = mixing * p_new(i, j) + (1.0 - mixing) * p_current(i, j);
      }
    }
    p_current = std::move(p_mixed);
  }

  result.total_energy = result.electronic_energy + result.nuclear_repulsion_energy;
  return result;
}

}  // namespace rerdmft
