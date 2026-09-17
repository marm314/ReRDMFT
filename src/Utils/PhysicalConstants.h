#ifndef RERDMFT_PHYSICALCONSTANTS_H
#define RERDMFT_PHYSICALCONSTANTS_H

namespace rerdmft {

// Speed of light in atomic units (CODATA), used throughout this project as
// the default value of c. Overridable via the SPEED_OF_LIGHT input keyword
// (see Input.h) to probe the nonrelativistic limit (c -> infinity) or
// otherwise vary relativistic effects.
inline constexpr double kSpeedOfLight = 137.036;

// Boltzmann constant in Hartree/Kelvin (CODATA k_B = 1.380649e-23 J/K,
// 1 Hartree = 4.3597447222071e-18 J), used to convert an electronic
// temperature in Kelvin (see the TEMPERATURE input keyword, Input.h)
// into the atomic-units energy scale k_B*T needed by
// Occ_opt/FermiDirac.h's fermiDiracOccupations.
inline constexpr double kBoltzmannHartreePerKelvin = 3.166811563e-6;

}  // namespace rerdmft

#endif  // RERDMFT_PHYSICALCONSTANTS_H
