#include "Input.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "StringUtils.h"

namespace rerdmft {

namespace {

// CODATA Bohr radius: 1 Bohr = 0.52917721067 Angstrom. Geometries in the
// input file are given in Angstrom (the common convention for a readable
// input format) but integral evaluation requires atomic units, so atomic
// coordinates are converted to Bohr immediately after parsing and are
// stored (and used everywhere else in the program) in Bohr.
constexpr double kAngstromToBohr = 1.0 / 0.52917721067;

}  // namespace

void Input::read(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("could not open input file '" + filename + "'");
  }

  bool has_n_electrons = false;
  bool has_basis_file = false;

  std::string raw_line;
  int line_number = 0;
  while (std::getline(file, raw_line)) {
    ++line_number;
    const std::string line = stripComment(raw_line);
    if (line.empty()) continue;

    std::istringstream iss(line);
    std::string keyword;
    iss >> keyword;
    keyword = toUpper(keyword);

    if (keyword == "NELEC" || keyword == "NELECTRONS") {
      if (!(iss >> n_electrons_)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected an integer after " + keyword);
      }
      has_n_electrons = true;
    } else if (keyword == "BASIS") {
      if (!(iss >> basis_file_)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected a file name after BASIS");
      }
      has_basis_file = true;
    } else if (keyword == "GEOMETRY") {
      geometry_.clear();
      while (std::getline(file, raw_line)) {
        ++line_number;
        const std::string geom_line = stripComment(raw_line);
        if (geom_line.empty()) continue;
        if (toUpper(geom_line) == "END") break;

        std::istringstream geom_iss(geom_line);
        Atom atom;
        if (!(geom_iss >> atom.symbol >> atom.x >> atom.y >> atom.z)) {
          throw std::runtime_error(
              "line " + std::to_string(line_number) +
              ": expected '<symbol> <x> <y> <z>' in GEOMETRY block");
        }
        atom.x *= kAngstromToBohr;
        atom.y *= kAngstromToBohr;
        atom.z *= kAngstromToBohr;
        geometry_.push_back(atom);
      }
    } else {
      throw std::runtime_error("line " + std::to_string(line_number) +
                                ": unrecognized keyword '" + keyword + "'");
    }
  }

  if (!has_n_electrons) {
    throw std::runtime_error("missing required keyword NELEC");
  }
  if (!has_basis_file) {
    throw std::runtime_error("missing required keyword BASIS");
  }
  if (geometry_.empty()) {
    throw std::runtime_error("missing or empty GEOMETRY block");
  }
}

}  // namespace rerdmft
