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
  // Optional; defaults to false when the TWO_ELECTRON keyword is absent.
  // When true, additionally builds the full two-electron Coulomb
  // repulsion tensor in the RKB spinor basis (RkbTwoElectron.h) -- opt-in
  // since both its time and memory cost scale steeply with basis size.
  bool two_electron() const { return two_electron_; }

 private:
  int n_electrons_ = 0;
  std::string basis_file_;
  std::vector<Atom> geometry_;
  bool debug_ = false;
  double speed_of_light_ = kSpeedOfLight;
  bool non_relativistic_ = false;
  bool two_electron_ = false;
};

}  // namespace rerdmft

#endif  // RERDMFT_INPUT_H
