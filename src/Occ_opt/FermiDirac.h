#ifndef RERDMFT_OCC_OPT_FERMI_DIRAC_H
#define RERDMFT_OCC_OPT_FERMI_DIRAC_H

#include <vector>

namespace rerdmft {

// Fermi-Dirac fractional occupation numbers,
//   n_i(mu) = 1 / (exp((e_i - mu) / (kB*T)) + 1),
// for a set of SPIN-orbital/spinor energies `orbital_energies` (Hartree,
// atomic units -- each entry may be occupied by at most ONE electron,
// i.e. these are already spin-orbitals/spinors, not spatial orbitals
// with an implicit factor-of-2 double occupancy). The chemical
// potential `mu` is found by bisection so that
// `sum_i n_i(mu) == n_electrons` to within `tolerance` -- well-posed
// since each n_i(mu), and hence their sum, is monotonically increasing
// in mu. `temperature_kelvin` is the ELECTRONIC temperature in Kelvin
// (converted to the atomic-units energy scale kB*T via
// PhysicalConstants.h's kBoltzmannHartreePerKelvin); a HIGH temperature
// (the caller's TEMPERATURE input keyword defaults to 500 K, see
// Input.h) smears the T=0 step-function occupation (n_i in {0,1}) into
// a smooth fractional distribution, needed to evaluate a genuinely
// fractional-occupation RDMFT functional (Occ_opt/JK_only.h) on
// otherwise-idempotent converged HF/DHF orbitals.
//
// Throws std::runtime_error if `n_electrons` is not in [0,
// orbital_energies.size()] (no chemical potential can satisfy the
// electron-count constraint outside that range) or if bisection fails
// to converge within `max_bisection_iterations` (should not happen in
// practice -- sum_i n_i(mu) is continuous and strictly monotonic in mu
// away from the pathological T=0 limit).
std::vector<double> fermiDiracOccupations(const std::vector<double>& orbital_energies,
                                           double n_electrons, double temperature_kelvin,
                                           double tolerance = 1e-12,
                                           int max_bisection_iterations = 200);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_FERMI_DIRAC_H
