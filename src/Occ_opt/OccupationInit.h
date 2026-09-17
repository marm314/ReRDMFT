#ifndef RERDMFT_OCC_OPT_OCCUPATION_INIT_H
#define RERDMFT_OCC_OPT_OCCUPATION_INIT_H

#include <string>
#include <vector>

namespace rerdmft {

// How to generate the initial fractional occupation numbers a JK-only
// functional (Occ_opt/JK_only.h) is evaluated at / an SQP occupation-
// number optimization (Utils/SQP.h) starts from, for a FIXED set of
// (already-converged HF/DHF) orbitals.
enum class OccupationInitMethod {
  // Aufbau (T=0, idempotent: 1.0 for the n_electrons lowest-energy
  // active orbitals, 0.0 otherwise) redistributed into the interior
  // box (see redistributeIntoInteriorBox) -- temperature-independent,
  // and the DEFAULT (Input.h's OCCUPATION_INIT keyword): unlike Fermi-
  // Dirac, it never depends on picking a "good" electronic temperature,
  // and its own redistribution step already guards the same n->0/n->1
  // boundary-divergence issue that motivated introducing this method
  // (see [[project-occ-opt]] -- CO/STO-3G's Fermi-Dirac occupations at
  // TEMPERATURE 1000 K underflowed to exactly 0.0 for several virtuals,
  // producing an infinite Hessian entry for MBB-family functionals).
  kProportional,
  // Occ_opt/FermiDirac.h's fermiDiracOccupations at the input's
  // TEMPERATURE (Input.h), then ALSO redistributed into the interior
  // box for the same reason.
  kFermiDirac,
};

// Maps Input.h's OCCUPATION_INIT keyword string (already validated
// there against {"PROPORTIONAL","FERMI_DIRAC"}, case-insensitive but
// stored uppercase) to the corresponding OccupationInitMethod -- kept
// here, not in Input.h, so Input.h stays independent of Occ_opt (same
// reasoning as JK_only.h's own parseJkFunctional).
OccupationInitMethod parseOccupationInitMethod(const std::string& name);

// The aufbau (T=0) reference: 1.0 for the n_electrons lowest-energy
// orbitals, 0.0 for the rest. `orbital_energies_active` MUST already be
// sorted ascending (every caller in this project already keeps orbital
// energies in ascending order); `n_electrons` MUST be (to floating-
// point precision) a non-negative integer no larger than
// orbital_energies_active.size().
std::vector<double> aufbauOccupations(const std::vector<double>& orbital_energies_active,
                                       double n_electrons);

// Redistributes `occupations` into the OPEN interior box
// [epsilon, 1-epsilon] while preserving sum(occupations) == n_electrons
// EXACTLY: first clamps every entry into the box, then spreads the
// resulting (small) sum residual across ALL entries, weighted by each
// one's own remaining HEADROOM to the relevant bound (room to grow,
// (1-epsilon)-x_i, if the residual needs adding; room to shrink,
// x_i-epsilon, if it needs removing) -- this guarantees no entry can
// overshoot its own bound (each absorbs at most its own share of
// headroom), unlike nudging just a single "most central" entry, which
// can overshoot when EVERY entry sits near a boundary simultaneously
// (the failure mode this replaced -- see [[project-occ-opt]]).
std::vector<double> redistributeIntoInteriorBox(std::vector<double> occupations,
                                                 double n_electrons, double epsilon = 1e-6);

// Generates the initial occupations via `method` (see
// OccupationInitMethod above), already redistributed into the interior
// box and exactly summing to n_electrons -- i.e. ready to use directly
// both as the functional-evaluation occupations AND as Utils/SQP.h's
// feasible starting point, no further clamping needed by the caller.
std::vector<double> generateInitialOccupations(OccupationInitMethod method,
                                                const std::vector<double>& orbital_energies_active,
                                                double n_electrons, double temperature_kelvin,
                                                double epsilon = 1e-6);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_OCCUPATION_INIT_H
