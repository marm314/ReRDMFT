#include "JK_only.h"

#include <cmath>
#include <stdexcept>

#include "StringUtils.h"

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

// g(x) = x*(a0+a1*x)/(1+b1*x) is the Marques-Lathiotakis form written
// as a function of x = n_i*n_j alone (marquesLathiotakisForm(n_i,n_j) =
// g(n_i*n_j)) -- g'/g'' below let the chain rule (x = n_i*n_j) produce
// the D1/D11/D12 partials in one place for both ML and MLSIC (which
// only differ in a0/a1/b1).
double marquesLathiotakisGPrime(double x, double a0, double a1, double b1) {
  const double d = 1.0 + b1 * x;
  return (a0 + 2.0 * a1 * x + a1 * b1 * x * x) / (d * d);
}

double marquesLathiotakisGDoublePrime(double x, double a0, double a1, double b1) {
  const double d = 1.0 + b1 * x;
  const double m = a0 + 2.0 * a1 * x + a1 * b1 * x * x;
  return 2.0 * (a1 * d * d - b1 * m) / (d * d * d);
}

// D1/D11/D12 for f(a,b) = g(a*b) (ML/MLSIC's shared form, x=a*b):
//   D1  = g'(x)*b
//   D11 = g''(x)*b^2
//   D12 = g''(x)*x + g'(x)
void marquesLathiotakisDerivatives(double a, double b, double a0, double a1, double b1,
                                    double& d1, double& d11, double& d12) {
  const double x = a * b;
  const double gp = marquesLathiotakisGPrime(x, a0, a1, b1);
  const double gpp = marquesLathiotakisGDoublePrime(x, a0, a1, b1);
  d1 = gp * b;
  d11 = gpp * b * b;
  d12 = gpp * x + gp;
}

// CA: f(a,b) = sqrt(u(a)*v(b)) + a*b, u(t)=t*(1-t), v(t)=t*(1-t) (same
// function, different argument) -- D1/D11/D12 wrt `a` (b held fixed),
// derived via phi(a) = (1-2a)/(2*sqrt(u(a))) so that D1 = phi(a)*sqrt(v(b)) + b:
//   D11 = phi'(a)*sqrt(v(b)),
//     phi'(a) = -u(a)^{-1/2} - (1/4)*(1-2a)^2*u(a)^{-3/2}
//   D12 = phi(a)*(1-2b)/(2*sqrt(v(b))) + 1
void csanyiAriasDerivatives(double a, double b, double& d1, double& d11, double& d12) {
  const double u = a * (1.0 - a);
  const double v = b * (1.0 - b);
  const double sqrt_u = std::sqrt(u);
  const double sqrt_v = std::sqrt(v);
  const double phi = (1.0 - 2.0 * a) / (2.0 * sqrt_u);
  const double phi_prime =
      -1.0 / sqrt_u - 0.25 * (1.0 - 2.0 * a) * (1.0 - 2.0 * a) / (u * sqrt_u);
  d1 = phi * sqrt_v + b;
  d11 = phi_prime * sqrt_v;
  d12 = phi * (1.0 - 2.0 * b) / (2.0 * sqrt_v) + 1.0;
}

// CGA: f(a,b) = [a*b + sqrt(p(a)*q(b))]/2, p(t)=t*(2-t), q(t)=t*(2-t) --
// psi(a) = (1-a)/(2*sqrt(p(a))) so that D1 = 0.5*b + psi(a)*sqrt(q(b)):
//   D11 = psi'(a)*sqrt(q(b)),
//     psi'(a) = -0.5*p(a)^{-1/2} - 0.5*(1-a)^2*p(a)^{-3/2}
//   D12 = 0.5 + psi(a)*(1-b)/sqrt(q(b))
void csanyiGoedeckerAriasDerivatives(double a, double b, double& d1, double& d11, double& d12) {
  const double p = a * (2.0 - a);
  const double q = b * (2.0 - b);
  const double sqrt_p = std::sqrt(p);
  const double sqrt_q = std::sqrt(q);
  const double psi = 0.5 * (1.0 - a) / sqrt_p;
  const double psi_prime = -0.5 / sqrt_p - 0.5 * (1.0 - a) * (1.0 - a) / (p * sqrt_p);
  d1 = 0.5 * b + psi * sqrt_q;
  d11 = psi_prime * sqrt_q;
  d12 = 0.5 + psi * (1.0 - b) / sqrt_q;
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

namespace {

double mbbD1(double a, double b) { return 0.5 * std::sqrt(b / a); }
double mbbD11(double a, double b) { return -0.25 * std::sqrt(b) * std::pow(a, -1.5); }
double mbbD12(double a, double b) { return 0.25 / std::sqrt(a * b); }

}  // namespace

double jkExchangeFunctionD1(JkFunctional functional, double n_i, double n_j, std::size_t i,
                             std::size_t j, std::size_t f_l, double power_alpha) {
  switch (functional) {
    case JkFunctional::kSd:
      return n_j;

    case JkFunctional::kMbb:
      return mbbD1(n_i, n_j);

    case JkFunctional::kBbc2: {
      // i==j: f_X(n_i,n_i;i,i) = n_i is only ever EVALUATED at n_i==n_j
      // (whenever i==j, every caller passes occupations[i]==occupations[j]
      // for the same orbital) -- unlike GU/MLSIC's own diag branch
      // n_i*n_j, this "n_i" formula is NOT the restriction of any
      // smooth two-variable function used elsewhere in this table, so
      // there is no single "natural" off-diagonal extension to
      // differentiate. The gradient/Hessian formulas
      // (Occ_opt/OccupationEnergy.h) need a SYMMETRIC extension whose
      // two partials sum to the TRUE 1-variable derivative
      // d/dt[f_X(t,t;i,i)] = d/dt[t] = 1 at the diagonal -- the
      // symmetric affine extension f_X(a,b;i,i) := (a+b)/2 (reduces to
      // `a` exactly at a==b) does this with D1=D2=0.5 each (confirmed
      // by the full multi-orbital energy finite-difference check in
      // Occ_opt/OccupationEnergy.h's own validation -- the naive
      // "D1=1" choice silently passed jkExchangeFunction's own
      // per-element FD check, since that only ever probes this branch
      // at a==b too, but gave a gradient exactly 2x too large once
      // summed into the real energy).
      if (i == j) return 0.5;
      const bool i_weak = i >= f_l;
      const bool j_weak = j >= f_l;
      const bool i_strong = i < f_l;
      const bool j_strong = j < f_l;
      if (i_weak && j_weak) return -mbbD1(n_i, n_j);
      if (i_strong && j_strong) return n_j;
      return mbbD1(n_i, n_j);
    }

    case JkFunctional::kCa: {
      double d1, d11, d12;
      csanyiAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d1;
    }

    case JkFunctional::kCga: {
      double d1, d11, d12;
      csanyiGoedeckerAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d1;
    }

    case JkFunctional::kMl: {
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlA0, kMlA1, kMlB1, d1, d11, d12);
      return d1;
    }

    case JkFunctional::kMlsic: {
      if (i == j) return n_j;
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlsicA0, kMlsicA1, kMlsicB1, d1, d11, d12);
      return d1;
    }

    case JkFunctional::kGu:
      if (i == j) return n_j;
      return mbbD1(n_i, n_j);

    case JkFunctional::kPower: {
      const double x = n_i * n_j;
      return power_alpha * std::pow(x, power_alpha - 1.0) * n_j;
    }
  }
  throw std::runtime_error("jkExchangeFunctionD1: unhandled JkFunctional");
}

double jkExchangeFunctionD11(JkFunctional functional, double n_i, double n_j, std::size_t i,
                              std::size_t j, std::size_t f_l, double power_alpha) {
  switch (functional) {
    case JkFunctional::kSd:
      return 0.0;

    case JkFunctional::kMbb:
      return mbbD11(n_i, n_j);

    case JkFunctional::kBbc2: {
      if (i == j) return 0.0;
      const bool i_weak = i >= f_l;
      const bool j_weak = j >= f_l;
      const bool i_strong = i < f_l;
      const bool j_strong = j < f_l;
      if (i_weak && j_weak) return -mbbD11(n_i, n_j);
      if (i_strong && j_strong) return 0.0;
      return mbbD11(n_i, n_j);
    }

    case JkFunctional::kCa: {
      double d1, d11, d12;
      csanyiAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d11;
    }

    case JkFunctional::kCga: {
      double d1, d11, d12;
      csanyiGoedeckerAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d11;
    }

    case JkFunctional::kMl: {
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlA0, kMlA1, kMlB1, d1, d11, d12);
      return d11;
    }

    case JkFunctional::kMlsic: {
      if (i == j) return 0.0;
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlsicA0, kMlsicA1, kMlsicB1, d1, d11, d12);
      return d11;
    }

    case JkFunctional::kGu:
      if (i == j) return 0.0;
      return mbbD11(n_i, n_j);

    case JkFunctional::kPower: {
      const double x = n_i * n_j;
      return power_alpha * (power_alpha - 1.0) * std::pow(x, power_alpha - 2.0) * n_j * n_j;
    }
  }
  throw std::runtime_error("jkExchangeFunctionD11: unhandled JkFunctional");
}

double jkExchangeFunctionD12(JkFunctional functional, double n_i, double n_j, std::size_t i,
                              std::size_t j, std::size_t f_l, double power_alpha) {
  switch (functional) {
    case JkFunctional::kSd:
      return 1.0;

    case JkFunctional::kMbb:
      return mbbD12(n_i, n_j);

    case JkFunctional::kBbc2: {
      if (i == j) return 0.0;
      const bool i_weak = i >= f_l;
      const bool j_weak = j >= f_l;
      const bool i_strong = i < f_l;
      const bool j_strong = j < f_l;
      if (i_weak && j_weak) return -mbbD12(n_i, n_j);
      if (i_strong && j_strong) return 1.0;
      return mbbD12(n_i, n_j);
    }

    case JkFunctional::kCa: {
      double d1, d11, d12;
      csanyiAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d12;
    }

    case JkFunctional::kCga: {
      double d1, d11, d12;
      csanyiGoedeckerAriasDerivatives(n_i, n_j, d1, d11, d12);
      return d12;
    }

    case JkFunctional::kMl: {
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlA0, kMlA1, kMlB1, d1, d11, d12);
      return d12;
    }

    case JkFunctional::kMlsic: {
      if (i == j) return 1.0;
      double d1, d11, d12;
      marquesLathiotakisDerivatives(n_i, n_j, kMlsicA0, kMlsicA1, kMlsicB1, d1, d11, d12);
      return d12;
    }

    case JkFunctional::kGu:
      if (i == j) return 1.0;
      return mbbD12(n_i, n_j);

    case JkFunctional::kPower: {
      const double x = n_i * n_j;
      return power_alpha * power_alpha * std::pow(x, power_alpha - 1.0);
    }
  }
  throw std::runtime_error("jkExchangeFunctionD12: unhandled JkFunctional");
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

JkFunctional parseJkFunctional(const std::string& name) {
  const std::string upper = toUpper(name);
  if (upper == "SD") return JkFunctional::kSd;
  if (upper == "MBB" || upper == "MULLER") return JkFunctional::kMbb;
  if (upper == "BBC2") return JkFunctional::kBbc2;
  if (upper == "CA") return JkFunctional::kCa;
  if (upper == "CGA") return JkFunctional::kCga;
  if (upper == "ML") return JkFunctional::kMl;
  if (upper == "MLSIC") return JkFunctional::kMlsic;
  if (upper == "GU") return JkFunctional::kGu;
  if (upper == "POWER") return JkFunctional::kPower;
  throw std::runtime_error("parseJkFunctional: unrecognized functional name '" + name + "'");
}

}  // namespace rerdmft
