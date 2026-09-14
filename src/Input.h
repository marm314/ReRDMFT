#ifndef RERDMFT_INPUT_H
#define RERDMFT_INPUT_H

#include <string>
#include <vector>

#include "PhysicalConstants.h"

namespace rerdmft {

// Coordinates are in Bohr (atomic units), regardless of the input file's
// units (see Input::read).
struct Atom {
  std::string symbol;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

// Parses the ReRDMFT input file and stores the run parameters: number of
// electrons, gaussian basis set file name, and molecular geometry. The
// input file's geometry is in Angstrom; it is converted to Bohr on read.
class Input {
 public:
  void read(const std::string& filename);

  int n_electrons() const { return n_electrons_; }
  const std::string& basis_file() const { return basis_file_; }
  const std::vector<Atom>& geometry() const { return geometry_; }
  // Optional; defaults to false when the DEBUG keyword is absent from the
  // input file. When true, the program prints detailed basis and matrix
  // diagnostics; otherwise it only prints a concise summary.
  bool debug() const { return debug_; }
  // Optional; defaults to the standard CODATA value (PhysicalConstants.h)
  // when the SPEED_OF_LIGHT keyword is absent. Overriding it (e.g. to a
  // very large number) lets you probe the nonrelativistic limit
  // (c -> infinity) or otherwise vary relativistic effects.
  double speed_of_light() const { return speed_of_light_; }
  // Optional; defaults to false when the NON_RELATIVISTIC keyword is
  // absent. When true, additionally builds and diagonalizes the
  // nonrelativistic Schrodinger core Hamiltonian in the Large-component AO
  // basis (SchrodingerKinetic.h), printing its eigenvalues for comparison
  // against the relativistic ones.
  bool non_relativistic() const { return non_relativistic_; }
  // Optional; defaults to false when the C4_SPINOR keyword is absent.
  // When true, additionally builds the full 4-component Dirac-Hartree-Fock
  // two-electron Coulomb repulsion tensor in the RKB spinor basis
  // (C4_DHF/RkbTwoElectron.h) -- opt-in since both its time and memory
  // cost scale steeply with basis size.
  bool c4_spinor() const { return c4_spinor_; }
  // Optional; defaults to 0.4 when the MIXING keyword is absent. Linear
  // density-matrix mixing weight for the C4_DHF SCF loop (C4_DHF/C4_DHF.h):
  // the density fed into the next iteration's Fock build is
  // mixing*P_new + (1-mixing)*P_current. Must be in (0, 1].
  double mixing() const { return mixing_; }
  // Optional; defaults to 100 when MAX_ITERATIONS is absent. Maximum
  // number of C4_DHF SCF cycles (C4_DHF/C4_DHF.h) before giving up
  // (converged is false in that case, but the last cycle's results are
  // still returned). Must be positive.
  int max_iterations() const { return max_iterations_; }
  // Optional; defaults to 1e-8 Hartree when ENERGY_TOLERANCE is absent.
  // The C4_DHF SCF loop's energy-change convergence threshold. Combined
  // with density_tolerance() by OR (see its note): converged as soon as
  // EITHER one is satisfied, not only when both are. Must be positive.
  double energy_tolerance() const { return energy_tolerance_; }
  // Optional; defaults to 1e-6 when DENSITY_TOLERANCE is absent. The
  // C4_DHF SCF loop's density-change convergence threshold. Combined with
  // energy_tolerance() by OR: the SCF loop calls itself converged (after
  // at least one iteration) as soon as EITHER the density change or the
  // energy change falls below its own tolerance. Must be positive.
  double density_tolerance() const { return density_tolerance_; }
  // Optional; defaults to false when CACHE_INTEGRALS is absent. When
  // true, the two-electron integral tensors (C4_DHF's RKB spinor tensor
  // and/or NON_REL's Large-basis tensor, whichever apply) are cached to
  // disk under cache_dir() and reused on a later run with the same
  // geometry+basis (regardless of SPEED_OF_LIGHT, which these tensors do
  // not depend on -- see IntegralCache.h) instead of being recomputed
  // via libcint from scratch. Off by default since it writes files to
  // disk and a cache built by a different code version is only detected
  // (and safely ignored) via an embedded format-version tag, not a
  // content check.
  bool cache_integrals() const { return cache_integrals_; }
  // Optional; defaults to ".rerdmft_cache" when CACHE_DIR is absent.
  // Directory (created if missing) that cache_integrals() cache files
  // are written to/read from.
  const std::string& cache_dir() const { return cache_dir_; }

 private:
  int n_electrons_ = 0;
  std::string basis_file_;
  std::vector<Atom> geometry_;
  bool debug_ = false;
  double speed_of_light_ = kSpeedOfLight;
  bool non_relativistic_ = false;
  bool c4_spinor_ = false;
  double mixing_ = 0.4;
  int max_iterations_ = 100;
  double energy_tolerance_ = 1e-8;
  double density_tolerance_ = 1e-6;
  bool cache_integrals_ = false;
  std::string cache_dir_ = ".rerdmft_cache";
};

}  // namespace rerdmft

#endif  // RERDMFT_INPUT_H
