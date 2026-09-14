#include "NonRelHartreeFock.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "ElectronRepulsion.h"
#include "LinearAlgebra.h"
#include "NuclearRepulsion.h"

namespace rerdmft {

namespace {

double traceOfProduct(const Matrix<double>& a, const Matrix<double>& b) {
  double trace = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      trace += a(i, j) * b(j, i);
    }
  }
  return trace;
}

double maxAbsDifference(const Matrix<double>& a, const Matrix<double>& b) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
    }
  }
  return max_diff;
}

}  // namespace

Matrix<double> nonRelDensityMatrix(const Matrix<double>& c, int n_electrons) {
  const std::size_t n = c.rows();
  if (c.cols() != n) {
    throw std::runtime_error("nonRelDensityMatrix: C is not square");
  }
  if (n_electrons <= 0 || n_electrons % 2 != 0) {
    throw std::runtime_error(
        "nonRelDensityMatrix: n_electrons must be a positive even number (closed-shell "
        "restricted Hartree-Fock)");
  }
  const std::size_t n_occ = static_cast<std::size_t>(n_electrons) / 2;
  if (n_occ > n) {
    throw std::runtime_error(
        "nonRelDensityMatrix: n_electrons/2 exceeds the number of available spatial orbitals");
  }

  Matrix<double> p(n, n, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t row = 0; row < n; ++row) {
      const double c_row_i = c(row, i);
      if (c_row_i == 0.0) continue;
      for (std::size_t col = 0; col < n; ++col) {
        p(row, col) += 2.0 * c_row_i * c(col, i);
      }
    }
  }
  return p;
}

Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const PackedTwoElectronTensor& eri,
                                 const Matrix<double>& density_matrix) {
  const std::size_t n = h_core.rows();
  if (h_core.cols() != n) {
    throw std::runtime_error("nonRelFockMatrix: H_core is not square");
  }
  if (eri.dim() != n) {
    throw std::runtime_error("nonRelFockMatrix: ERI tensor dimension is inconsistent with H_core");
  }
  if (density_matrix.rows() != n || density_matrix.cols() != n) {
    throw std::runtime_error(
        "nonRelFockMatrix: density matrix dimensions are inconsistent with H_core");
  }

  // `eri` is ElectronRepulsion.h's twoElectronIntegralsPacked result, which
  // is CHEMIST notation (pq|rs) directly (p,q electron-1 pair; r,s
  // electron-2 pair) -- NOT physics notation. Physics <A B|C D> =
  // chemist(A,C,B,D) (RkbTwoElectron.h uses the same relation), so
  // <p q|r s> = eri(p,r,q,s) and <p q|s r> = eri(p,s,q,r); using
  // eri(p,q,r,s)/eri(p,q,s,r) directly would silently compute a different
  // (and wrong) pair of integrals.
  Matrix<double> fock(n, n, 0.0);
  // Each (p,r) owns its own disjoint output position and only reads the
  // shared, const `eri`/`density_matrix` -- safe to parallelize.
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t r = 0; r < n; ++r) {
      double hartree = 0.0;
      double exchange = 0.0;
      for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t s = 0; s < n; ++s) {
          const double p_sq = density_matrix(s, q);
          if (p_sq == 0.0) continue;
          hartree += p_sq * eri(p, r, q, s);
          exchange += p_sq * eri(p, s, q, r);
        }
      }
      fock(p, r) = h_core(p, r) + hartree - 0.5 * exchange;
    }
  }
  return fock;
}

NonRelHartreeFockResult runNonRelativisticHartreeFock(
    const PackedTwoElectronTensor& eri, const Matrix<double>& h_core,
    const Matrix<double>& x_large, const Matrix<double>& initial_density, int n_electrons,
    const std::vector<Atom>& geometry, double mixing, int max_iterations,
    double energy_tolerance, double density_tolerance) {
  NonRelHartreeFockResult result;
  result.nuclear_repulsion_energy = nuclearRepulsionEnergy(geometry);

  Matrix<double> p_current = initial_density;
  double previous_energy = 0.0;

  for (int iteration = 1; iteration <= max_iterations; ++iteration) {
    const Matrix<double> fock = nonRelFockMatrix(h_core, eri, p_current);
    // X_large is symmetric (S_Large^-1/2), so this is X^dagger F X exactly.
    const Matrix<double> fock_ortho = x_large * (fock * x_large);
    const SymmetricEigenResult eig = diagonalizeSymmetric(fock_ortho);
    const Matrix<double> c_matrix = x_large * eig.eigenvectors;
    const Matrix<double> p_new = nonRelDensityMatrix(c_matrix, n_electrons);

    // Self-consistent-pair energy: evaluated with the SAME density that
    // built this Fock matrix, not the (not yet computed) mixed one.
    const double energy = 0.5 * traceOfProduct(p_current, h_core + fock);
    const double density_change = maxAbsDifference(p_new, p_current);
    const double energy_change = std::abs(energy - previous_energy);

    result.iterations = iteration;
    result.electronic_energy = energy;
    result.orbital_energies = eig.eigenvalues;
    result.density_matrix = p_current;
    result.fock_matrix = fock;
    result.c_matrix = c_matrix;

    NonRelScfIteration record;
    record.iteration = iteration;
    record.energy = energy;
    record.density_change = iteration > 1 ? density_change : std::nan("");
    record.energy_change = iteration > 1 ? energy_change : std::nan("");
    record.orbital_energies = eig.eigenvalues;
    result.history.push_back(std::move(record));

    // Converged as soon as EITHER metric is satisfied (OR, not AND),
    // matching C4_DHF.h's runDiracHartreeFockScf.
    const bool converged = iteration > 1 && (density_change < density_tolerance ||
                                              energy_change < energy_tolerance);
    previous_energy = energy;
    if (converged) {
      result.converged = true;
      break;
    }

    // Linear mixing, same convention as runDiracHartreeFockScf: only
    // affects the density fed into the NEXT iteration's Fock build.
    Matrix<double> p_mixed(p_current.rows(), p_current.cols());
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
