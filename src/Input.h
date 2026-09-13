#ifndef RERDMFT_INPUT_H
#define RERDMFT_INPUT_H

#include <string>
#include <vector>

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

 private:
  int n_electrons_ = 0;
  std::string basis_file_;
  std::vector<Atom> geometry_;
};

}  // namespace rerdmft

#endif  // RERDMFT_INPUT_H
