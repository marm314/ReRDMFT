#include "JK_only.h"

#include <cmath>
#include <stdexcept>

namespace rerdmft {

namespace {

// ML/MLSIC (Marques-Lathiotakis) literature-fitted coefficients, Table 1
// footnote 13.
constexpr double kMlA0 = 126.3101;
constexpr double kMlA1 = 2213.33;
constexpr double kMlB1 = 2338.64;

constexpr double kMlsicA0 = 1298.78;
constexpr double kMlsicA1 = 35114.4;
constexpr double kMlsicB1 = 36412.2;

double marquesLathiotakisForm(double n_i, double n_j, double a0, double a1, double b1) {
  const double ninj = n_i * n_j;
  return ninj * (a0 + a1 * ninj) / (1.0 + b1 * ninj);
}

}  // namespace

double jkExchangeFunction(JkFunctional functional, double n_i, double n_j, std::size_t i,
                           std::size_t j, std::size_t f_l, double power_alpha) {
  switch (functional) {
    case JkFunctional::kSd:
      return n_i * n_j;

    case JkFunctional::kMbb:
      return std::sqrt(n_i * n_j);

    case JkFunctional::kBbc2: {
      if (i == j) return n_i;
      const bool i_weak = i >= f_l;
      const bool j_weak = j >= f_l;
      const bool i_strong = i < f_l;
      const bool j_strong = j < f_l;
      if (i_weak && j_weak) return -std::sqrt(n_i * n_j);
      if (i_strong && j_strong) return n_i * n_j;
      return std::sqrt(n_i * n_j);  // otherwise (one strong, one weak)
    }

    case JkFunctional::kCa:
      return std::sqrt(n_i * (1.0 - n_i) * n_j * (1.0 - n_j)) + n_i * n_j;

    case JkFunctional::kCga:
      return 0.5 * (n_i * n_j + std::sqrt(n_i * (2.0 - n_i) * n_j * (2.0 - n_j)));

    case JkFunctional::kMl:
      return marquesLathiotakisForm(n_i, n_j, kMlA0, kMlA1, kMlB1);

    case JkFunctional::kMlsic:
      if (i == j) return n_i * n_j;
      return marquesLathiotakisForm(n_i, n_j, kMlsicA0, kMlsicA1, kMlsicB1);

    case JkFunctional::kGu:
      if (i == j) return n_i * n_j;
      return std::sqrt(n_i * n_j);

    case JkFunctional::kPower:
      return std::pow(n_i * n_j, power_alpha);
  }
  throw std::runtime_error("jkExchangeFunction: unhandled JkFunctional");
}

Matrix<double> jkHartreeCoupling(const std::vector<double>& occupations) {
  const std::size_t n = occupations.size();
  Matrix<double> result(n, n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      result(p, q) = occupations[p] * occupations[q];
    }
  }
  return result;
}

Matrix<double> jkExchangeCoupling(JkFunctional functional, const std::vector<double>& occupations,
                                   std::size_t f_l, double power_alpha) {
  const std::size_t n = occupations.size();
  Matrix<double> result(n, n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      result(p, q) =
          jkExchangeFunction(functional, occupations[p], occupations[q], p, q, f_l, power_alpha);
    }
  }
  return result;
}

}  // namespace rerdmft
