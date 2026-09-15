#include "FermiDirac.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "PhysicalConstants.h"

namespace rerdmft {

namespace {

double occupationSum(const std::vector<double>& orbital_energies, double mu, double kt) {
  double sum = 0.0;
  for (const double e : orbital_energies) {
    // exp() overflowing to +inf for a very large (e-mu)/kt is fine here
    // (IEEE 754 gives 1/(inf+1) = 0, not NaN) -- no special-casing needed.
    sum += 1.0 / (std::exp((e - mu) / kt) + 1.0);
  }
  return sum;
}

}  // namespace

std::vector<double> fermiDiracOccupations(const std::vector<double>& orbital_energies,
                                           double n_electrons, double temperature_kelvin,
                                           double tolerance, int max_bisection_iterations) {
  const std::size_t n = orbital_energies.size();
  if (n_electrons < 0.0 || n_electrons > static_cast<double>(n)) {
    throw std::runtime_error(
        "fermiDiracOccupations: n_electrons out of range [0, orbital_energies.size()]");
  }
  if (temperature_kelvin <= 0.0) {
    throw std::runtime_error("fermiDiracOccupations: temperature_kelvin must be positive");
  }
  const double kt = kBoltzmannHartreePerKelvin * temperature_kelvin;

  const double e_min = *std::min_element(orbital_energies.begin(), orbital_energies.end());
  const double e_max = *std::max_element(orbital_energies.begin(), orbital_energies.end());
  const double buffer = std::max(1.0, 50.0 * kt);
  double lo = e_min - buffer;
  double hi = e_max + buffer;

  double mu = 0.5 * (lo + hi);
  bool converged = false;
  for (int iter = 0; iter < max_bisection_iterations; ++iter) {
    mu = 0.5 * (lo + hi);
    const double sum = occupationSum(orbital_energies, mu, kt);
    if (std::abs(sum - n_electrons) <= tolerance) {
      converged = true;
      break;
    }
    if (sum < n_electrons) {
      lo = mu;
    } else {
      hi = mu;
    }
  }
  if (!converged && std::abs(occupationSum(orbital_energies, mu, kt) - n_electrons) > tolerance) {
    throw std::runtime_error(
        "fermiDiracOccupations: bisection for the chemical potential did not converge");
  }

  std::vector<double> occupations(n);
  for (std::size_t i = 0; i < n; ++i) {
    occupations[i] = 1.0 / (std::exp((orbital_energies[i] - mu) / kt) + 1.0);
  }
  return occupations;
}

}  // namespace rerdmft
