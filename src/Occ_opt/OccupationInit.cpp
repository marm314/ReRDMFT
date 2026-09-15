#include "OccupationInit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "FermiDirac.h"
#include "StringUtils.h"

namespace rerdmft {

OccupationInitMethod parseOccupationInitMethod(const std::string& name) {
  const std::string upper = toUpper(name);
  if (upper == "PROPORTIONAL") return OccupationInitMethod::kProportional;
  if (upper == "FERMI_DIRAC") return OccupationInitMethod::kFermiDirac;
  throw std::runtime_error("parseOccupationInitMethod: unrecognized method name '" + name + "'");
}

std::vector<double> aufbauOccupations(const std::vector<double>& orbital_energies_active,
                                       double n_electrons) {
  const std::size_t n = orbital_energies_active.size();
  const std::size_t n_occ = static_cast<std::size_t>(std::lround(n_electrons));
  if (n_occ > n) {
    throw std::runtime_error("aufbauOccupations: n_electrons exceeds orbital_energies_active.size()");
  }
  std::vector<double> occupations(n, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) occupations[i] = 1.0;
  return occupations;
}

std::vector<double> redistributeIntoInteriorBox(std::vector<double> occupations,
                                                 double n_electrons, double epsilon) {
  const std::size_t n = occupations.size();
  const std::vector<double> lb(n, epsilon);
  const std::vector<double> ub(n, 1.0 - epsilon);

  for (double& v : occupations) {
    v = std::min(1.0 - epsilon, std::max(epsilon, v));
  }
  double clamped_sum = 0.0;
  for (const double v : occupations) clamped_sum += v;

  const double correction = n_electrons - clamped_sum;
  if (n > 0 && correction != 0.0) {
    std::vector<double> headroom(n);
    double total_headroom = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      headroom[i] = (correction > 0.0) ? (ub[i] - occupations[i]) : (occupations[i] - lb[i]);
      total_headroom += headroom[i];
    }
    if (total_headroom > 0.0) {
      for (std::size_t i = 0; i < n; ++i) {
        occupations[i] += correction * (headroom[i] / total_headroom);
      }
    }
  }
  return occupations;
}

std::vector<double> generateInitialOccupations(OccupationInitMethod method,
                                                const std::vector<double>& orbital_energies_active,
                                                double n_electrons, double temperature_kelvin,
                                                double epsilon) {
  std::vector<double> occupations;
  switch (method) {
    case OccupationInitMethod::kProportional:
      occupations = aufbauOccupations(orbital_energies_active, n_electrons);
      break;
    case OccupationInitMethod::kFermiDirac:
      occupations =
          fermiDiracOccupations(orbital_energies_active, n_electrons, temperature_kelvin);
      break;
  }
  return redistributeIntoInteriorBox(std::move(occupations), n_electrons, epsilon);
}

}  // namespace rerdmft
