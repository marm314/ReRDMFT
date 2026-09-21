#include "Input.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "StringUtils.h"

namespace rerdmft {

namespace {

// CODATA Bohr radius: 1 Bohr = 0.52917721067 Angstrom. Geometries in the
// input file are given in Angstrom (the common convention for a readable
// input format) but integral evaluation requires atomic units, so atomic
// coordinates are converted to Bohr immediately after parsing and are
// stored (and used everywhere else in the program) in Bohr.
constexpr double kAngstromToBohr = 1.0 / 0.52917721067;

// Parses a boolean value token, accepting an optional '=' before it (with
// or without surrounding whitespace, e.g. "DEBUG True", "DEBUG = True",
// "DEBUG=True") since that reads naturally for a flag like DEBUG.
bool parseBool(std::istringstream& stream, int line_number, const std::string& keyword) {
  std::string token;
  if (!(stream >> token)) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected a value after " +
                              keyword);
  }
  if (!token.empty() && token[0] == '=') {
    token.erase(0, 1);
    if (token.empty() && !(stream >> token)) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                                ": expected a value after " + keyword + " =");
    }
  }
  const std::string upper = toUpper(token);
  if (upper == "TRUE" || upper == "1" || upper == "YES") return true;
  if (upper == "FALSE" || upper == "0" || upper == "NO") return false;
  throw std::runtime_error("line " + std::to_string(line_number) + ": invalid boolean value '" +
                            token + "' for " + keyword);
}

// Parses a double value token, accepting an optional '=' before it (with or
// without surrounding whitespace), same as parseBool.
double parseDouble(std::istringstream& stream, int line_number, const std::string& keyword) {
  std::string token;
  if (!(stream >> token)) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected a value after " +
                              keyword);
  }
  if (!token.empty() && token[0] == '=') {
    token.erase(0, 1);
    if (token.empty() && !(stream >> token)) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                                ": expected a value after " + keyword + " =");
    }
  }
  try {
    return std::stod(token);
  } catch (const std::exception&) {
    throw std::runtime_error("line " + std::to_string(line_number) +
                              ": invalid floating point value '" + token + "' for " + keyword);
  }
}

// Parses an integer value token, accepting an optional '=' before it (with
// or without surrounding whitespace), same as parseBool/parseDouble.
int parseInt(std::istringstream& stream, int line_number, const std::string& keyword) {
  std::string token;
  if (!(stream >> token)) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": expected a value after " +
                              keyword);
  }
  if (!token.empty() && token[0] == '=') {
    token.erase(0, 1);
    if (token.empty() && !(stream >> token)) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                                ": expected a value after " + keyword + " =");
    }
  }
  try {
    return std::stoi(token);
  } catch (const std::exception&) {
    throw std::runtime_error("line " + std::to_string(line_number) + ": invalid integer value '" +
                              token + "' for " + keyword);
  }
}

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
    } else if (keyword == "DEBUG") {
      debug_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "VERBOSE") {
      verbose_ = parseInt(iss, line_number, keyword);
      if (!(verbose_ >= 0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": VERBOSE must be non-negative");
      }
    } else if (keyword == "NON_RELATIVISTIC") {
      non_relativistic_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "C4_SPINOR") {
      c4_spinor_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "X2C") {
      x2c_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "HESSIAN_NON_REL") {
      hessian_non_rel_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "HESSIAN_4C") {
      hessian_4c_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "HESSIAN_X2C") {
      hessian_x2c_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "HESSIAN_FUNCTIONAL") {
      hessian_functional_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "MIXING") {
      mixing_ = parseDouble(iss, line_number, keyword);
      if (!(mixing_ > 0.0 && mixing_ <= 1.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": MIXING must be in (0, 1]");
      }
    } else if (keyword == "DIIS") {
      diis_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "DIIS_SIZE") {
      diis_size_ = parseInt(iss, line_number, keyword);
      if (!(diis_size_ >= 2)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": DIIS_SIZE must be at least 2");
      }
    } else if (keyword == "MAX_ITERATIONS") {
      max_iterations_ = parseInt(iss, line_number, keyword);
      if (!(max_iterations_ > 0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": MAX_ITERATIONS must be positive");
      }
    } else if (keyword == "ENERGY_TOLERANCE") {
      energy_tolerance_ = parseDouble(iss, line_number, keyword);
      if (!(energy_tolerance_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": ENERGY_TOLERANCE must be positive");
      }
    } else if (keyword == "DENSITY_TOLERANCE") {
      density_tolerance_ = parseDouble(iss, line_number, keyword);
      if (!(density_tolerance_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": DENSITY_TOLERANCE must be positive");
      }
    } else if (keyword == "CHOLESKY") {
      cholesky_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "CHOLESKY_THRESHOLD") {
      cholesky_threshold_ = parseDouble(iss, line_number, keyword);
      if (!(cholesky_threshold_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": CHOLESKY_THRESHOLD must be positive");
      }
    } else if (keyword == "CACHE_INTEGRALS") {
      cache_integrals_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "CACHE_DIR") {
      if (!(iss >> cache_dir_)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected a directory path after CACHE_DIR");
      }
    } else if (keyword == "RESTART_FILE") {
      if (!(iss >> restart_file_)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected a file name (or NONE) after RESTART_FILE");
      }
    } else if (keyword == "FUNCTIONAL") {
      std::string token;
      if (!(iss >> token)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected a functional name after FUNCTIONAL");
      }
      std::string upper = toUpper(token);
      // MULLER is accepted as a synonym for MBB (Table 1's own name,
      // Muller/Buijse-Baerends) -- the literature commonly calls this
      // functional by either name.
      if (upper == "MULLER") upper = "MBB";
      static const std::vector<std::string> kKnownFunctionals = {
          "SD",    "MBB",   "BBC2",     "CA",    "CGA",   "ML",
          "MLSIC", "GU",    "POWER",    "MULLER_AS",
          "PNOF5", "PNOF7", "PNOF7S", "GNOF"};
      if (std::find(kKnownFunctionals.begin(), kKnownFunctionals.end(), upper) ==
          kKnownFunctionals.end()) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": unrecognized FUNCTIONAL '" + token +
                                  "' (expected one of SD, MBB (or MULLER), BBC2, CA, CGA, ML, "
                                  "MLSIC, GU, POWER, MULLER_AS, PNOF5, PNOF7, PNOF7S, GNOF)");
      }
      functional_ = upper;
      has_functional_ = true;
    } else if (keyword == "TEMPERATURE") {
      temperature_ = parseDouble(iss, line_number, keyword);
      if (!(temperature_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": TEMPERATURE must be positive");
      }
    } else if (keyword == "OCCUPATION_INIT") {
      std::string token;
      if (!(iss >> token)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": expected a method name after OCCUPATION_INIT");
      }
      const std::string upper = toUpper(token);
      static const std::vector<std::string> kKnownMethods = {"PROPORTIONAL", "FERMI_DIRAC"};
      if (std::find(kKnownMethods.begin(), kKnownMethods.end(), upper) == kKnownMethods.end()) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": unrecognized OCCUPATION_INIT '" + token +
                                  "' (expected one of PROPORTIONAL, FERMI_DIRAC)");
      }
      occupation_init_ = upper;
    } else if (keyword == "PNOF_SUBSPACES") {
      pnof_subspaces_ = parseInt(iss, line_number, keyword);
      if (pnof_subspaces_ < 1) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": PNOF_SUBSPACES must be at least 1");
      }
    } else if (keyword == "PNOF_COUPLING") {
      pnof_coupling_ = parseInt(iss, line_number, keyword);
      if (pnof_coupling_ < 2) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": PNOF_COUPLING must be at least 2");
      }
    } else if (keyword == "SQP_PNOF_OCC") {
      sqp_pnof_occ_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "FULL_OPTIMIZATION") {
      full_optimization_ = parseBool(iss, line_number, keyword);
    } else if (keyword == "MAX_MACRO_ITERATIONS") {
      max_macro_iterations_ = parseInt(iss, line_number, keyword);
      if (!(max_macro_iterations_ > 0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": MAX_MACRO_ITERATIONS must be positive");
      }
    } else if (keyword == "MACRO_ENERGY_TOLERANCE") {
      macro_energy_tolerance_ = parseDouble(iss, line_number, keyword);
      if (!(macro_energy_tolerance_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": MACRO_ENERGY_TOLERANCE must be positive");
      }
    } else if (keyword == "ORBITAL_GRADIENT_TOLERANCE") {
      orbital_gradient_tolerance_ = parseDouble(iss, line_number, keyword);
      if (!(orbital_gradient_tolerance_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": ORBITAL_GRADIENT_TOLERANCE must be positive");
      }
    } else if (keyword == "SPEED_OF_LIGHT") {
      speed_of_light_ = parseDouble(iss, line_number, keyword);
      if (!(speed_of_light_ > 0.0)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": SPEED_OF_LIGHT must be positive");
      }
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

void Input::print(std::ostream& out) const {
  const std::ios_base::fmtflags saved_flags = out.flags();
  const std::streamsize saved_precision = out.precision();
  out << std::defaultfloat << std::setprecision(10);
  const auto flag = [](bool value) { return value ? "TRUE" : "FALSE"; };
  const auto line = [&out](const char* name) -> std::ostream& {
    return out << "  " << std::left << std::setw(22) << name << std::right << " ";
  };

  out << "Input variables (current status):\n";
  line("NELEC") << n_electrons_ << "\n";
  line("BASIS") << basis_file_ << "\n";
  line("DEBUG") << flag(debug_) << "\n";
  line("VERBOSE") << verbose_ << "\n";
  line("NON_RELATIVISTIC") << flag(non_relativistic_) << "\n";
  line("C4_SPINOR") << flag(c4_spinor_) << "\n";
  line("X2C") << flag(x2c_) << "\n";
  line("HESSIAN_NON_REL") << flag(hessian_non_rel_) << "\n";
  line("HESSIAN_4C") << flag(hessian_4c_) << "\n";
  line("HESSIAN_X2C") << flag(hessian_x2c_) << "\n";
  line("HESSIAN_FUNCTIONAL") << flag(hessian_functional_) << "\n";
  line("MIXING") << mixing_ << "\n";
  line("DIIS") << flag(diis_) << "\n";
  line("DIIS_SIZE") << diis_size_ << "\n";
  line("MAX_ITERATIONS") << max_iterations_ << "\n";
  line("ENERGY_TOLERANCE") << energy_tolerance_ << "\n";
  line("DENSITY_TOLERANCE") << density_tolerance_ << "\n";
  line("CHOLESKY") << flag(cholesky_) << "\n";
  line("CHOLESKY_THRESHOLD") << cholesky_threshold_ << "\n";
  line("CACHE_INTEGRALS") << flag(cache_integrals_) << "\n";
  line("CACHE_DIR") << cache_dir_ << "\n";
  line("RESTART_FILE") << restart_file_ << "\n";
  line("FUNCTIONAL") << functional_ << (has_functional_ ? "" : " (default; not set in input)")
                      << "\n";
  line("TEMPERATURE") << temperature_ << "\n";
  line("OCCUPATION_INIT") << occupation_init_ << "\n";
  line("PNOF_SUBSPACES") << pnof_subspaces_ << "\n";
  line("PNOF_COUPLING") << pnof_coupling_ << "\n";
  line("SQP_PNOF_OCC") << flag(sqp_pnof_occ_) << "\n";
  line("FULL_OPTIMIZATION") << flag(full_optimization_) << "\n";
  line("MAX_MACRO_ITERATIONS") << max_macro_iterations_ << "\n";
  line("MACRO_ENERGY_TOLERANCE") << macro_energy_tolerance_ << "\n";
  line("ORBITAL_GRADIENT_TOLERANCE") << orbital_gradient_tolerance_ << "\n";
  line("SPEED_OF_LIGHT") << speed_of_light_ << "\n";

  out.flags(saved_flags);
  out.precision(saved_precision);
}

}  // namespace rerdmft
