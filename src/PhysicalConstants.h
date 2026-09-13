#ifndef RERDMFT_PHYSICALCONSTANTS_H
#define RERDMFT_PHYSICALCONSTANTS_H

namespace rerdmft {

// Speed of light in atomic units (CODATA), used throughout this project as
// the default value of c. Overridable via the SPEED_OF_LIGHT input keyword
// (see Input.h) to probe the nonrelativistic limit (c -> infinity) or
// otherwise vary relativistic effects.
inline constexpr double kSpeedOfLight = 137.036;

}  // namespace rerdmft

#endif  // RERDMFT_PHYSICALCONSTANTS_H
