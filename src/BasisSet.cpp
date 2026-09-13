#include "BasisSet.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "StringUtils.h"

namespace rerdmft {

namespace {

int angularMomentumFromLetter(char letter) {
  switch (letter) {
    case 'S': return 0;
    case 'P': return 1;
    case 'D': return 2;
    case 'F': return 3;
    case 'G': return 4;
    case 'H': return 5;
    case 'I': return 6;
    default:
      throw std::runtime_error(std::string("unrecognized shell type '") +
                                letter + "'");
  }
}

// A "****" line (Gaussian's atom block separator).
bool isSeparator(const std::string& line) {
  return !line.empty() && line.find_first_not_of('*') == std::string::npos;
}

}  // namespace

void BasisSet::read(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("could not open basis set file '" + filename +
                              "'");
  }

  shells_by_element_.clear();

  std::string raw_line;
  int line_number = 0;
  auto nextContentLine = [&](std::string& out) -> bool {
    while (std::getline(file, raw_line)) {
      ++line_number;
      out = stripComment(raw_line);
      if (!out.empty()) return true;
    }
    return false;
  };

  std::string line;
  while (nextContentLine(line)) {
    if (isSeparator(line)) continue;  // leading/blank "****" separators

    // Atom header: "<Symbol> <ignored integer>".
    std::istringstream header(line);
    std::string symbol;
    header >> symbol;
    symbol = toUpper(symbol);

    if (shells_by_element_.count(symbol)) {
      throw std::runtime_error("line " + std::to_string(line_number) +
                                ": duplicate basis definition for element '" +
                                symbol + "'");
    }

    std::vector<Shell> shells;
    while (nextContentLine(line) && !isSeparator(line)) {
      std::istringstream shell_header(line);
      std::string shell_type;
      int n_primitives = 0;
      double scale = 1.0;
      if (!(shell_header >> shell_type >> n_primitives >> scale)) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                  ": malformed shell header '" + line + "'");
      }
      shell_type = toUpper(shell_type);
      const bool is_sp = (shell_type == "SP" || shell_type == "L");

      Shell shell, s_shell, p_shell;
      if (is_sp) {
        s_shell.l = 0;
        p_shell.l = 1;
      } else {
        if (shell_type.size() != 1) {
          throw std::runtime_error("line " + std::to_string(line_number) +
                                    ": unsupported shell type '" +
                                    shell_type + "'");
        }
        shell.l = angularMomentumFromLetter(shell_type[0]);
      }

      for (int i = 0; i < n_primitives; ++i) {
        if (!nextContentLine(line)) {
          throw std::runtime_error(
              "unexpected end of file while reading primitives for element '" +
              symbol + "'");
        }
        std::istringstream prim(line);
        std::string exp_token, coef_token, coef2_token;
        if (is_sp) {
          if (!(prim >> exp_token >> coef_token >> coef2_token)) {
            throw std::runtime_error(
                "line " + std::to_string(line_number) +
                ": expected 'exponent coeff_S coeff_P' for SP shell");
          }
          const double exponent = parseFortranDouble(exp_token);
          s_shell.exponents.push_back(exponent);
          s_shell.coefficients.push_back(parseFortranDouble(coef_token));
          p_shell.exponents.push_back(exponent);
          p_shell.coefficients.push_back(parseFortranDouble(coef2_token));
        } else {
          if (!(prim >> exp_token >> coef_token)) {
            throw std::runtime_error("line " + std::to_string(line_number) +
                                      ": expected 'exponent coefficient'");
          }
          shell.exponents.push_back(parseFortranDouble(exp_token));
          shell.coefficients.push_back(parseFortranDouble(coef_token));
        }
      }

      if (scale != 1.0) {
        const double scale2 = scale * scale;
        for (double& e : shell.exponents) e *= scale2;
        for (double& e : s_shell.exponents) e *= scale2;
        for (double& e : p_shell.exponents) e *= scale2;
      }

      if (is_sp) {
        shells.push_back(std::move(s_shell));
        shells.push_back(std::move(p_shell));
      } else {
        shells.push_back(std::move(shell));
      }
    }

    if (shells.empty()) {
      throw std::runtime_error("no shells found for element '" + symbol +
                                "'");
    }
    shells_by_element_[symbol] = std::move(shells);
  }
}

bool BasisSet::hasElement(const std::string& symbol) const {
  return shells_by_element_.count(toUpper(symbol)) != 0;
}

const std::vector<Shell>& BasisSet::shellsForElement(
    const std::string& symbol) const {
  const auto it = shells_by_element_.find(toUpper(symbol));
  if (it == shells_by_element_.end()) {
    throw std::runtime_error("no basis defined for element '" + symbol + "'");
  }
  return it->second;
}

}  // namespace rerdmft
