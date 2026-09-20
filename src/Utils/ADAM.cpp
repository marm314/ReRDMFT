#include "ADAM.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rerdmft {

namespace {

// |x|^2 for both scalar types (g * conj(g) of the Fortran, real).
inline double squaredMagnitude(double x) { return x * x; }
inline double squaredMagnitude(const std::complex<double>& x) { return std::norm(x); }

}  // namespace

// =====================================================================
// Adam
// =====================================================================

template <typename T>
Adam<T>::Adam(std::size_t n, const AdamOptions& options)
    : opt_(options), lr_(options.learning_rate), m1_(n, T{}), m2_(n, 0.0), m2max_(n, 0.0) {}

template <typename T>
void Adam<T>::clean() {
  t_ = 0;
  std::fill(m1_.begin(), m1_.end(), T{});
  std::fill(m2_.begin(), m2_.end(), 0.0);
  std::fill(m2max_.begin(), m2max_.end(), 0.0);
}

template <typename T>
std::vector<T> Adam<T>::step(const std::vector<T>& gradient) {
  if (gradient.size() != m1_.size()) {
    throw std::runtime_error("Adam::step: gradient size differs from the ADAM size");
  }
  ++t_;
  const double correction1 = 1.0 - std::pow(opt_.beta1, t_);
  const double correction2 = 1.0 - std::pow(opt_.beta2, t_);
  std::vector<T> out(m1_.size());
  for (std::size_t i = 0; i < m1_.size(); ++i) {
    m1_[i] = opt_.beta1 * m1_[i] + (1.0 - opt_.beta1) * gradient[i];
    m2_[i] = opt_.beta2 * m2_[i] + (1.0 - opt_.beta2) * squaredMagnitude(gradient[i]);
    const T m1_hat = m1_[i] / correction1;
    const double m2_hat = m2_[i] / correction2;
    m2max_[i] = std::max(m2max_[i], m2_hat);
    out[i] = -lr_ * m1_hat / (std::sqrt(m2max_[i]) + opt_.epsilon);
  }
  return out;
}

template class Adam<double>;
template class Adam<std::complex<double>>;

// =====================================================================
// pairs, gradient vector, kappa assembly
// =====================================================================

std::vector<AdamPair> adamPairIndices(std::size_t n, bool include_diagonal) {
  std::vector<AdamPair> pairs;
  pairs.reserve(n * (n + 1) / 2);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < p; ++q) pairs.emplace_back(p, q);
    if (include_diagonal) pairs.emplace_back(p, p);
  }
  return pairs;
}

template <typename T>
std::vector<T> adamGradientVector(const Matrix<T>& gradient, const std::vector<AdamPair>& pairs) {
  std::vector<T> out;
  out.reserve(pairs.size());
  for (const auto& [p, q] : pairs) {
    if (p < q) throw std::runtime_error("adamGradientVector: pairs must have p >= q");
    if (p >= gradient.rows() || q >= gradient.cols()) {
      throw std::runtime_error("adamGradientVector: pair out of range");
    }
    out.push_back(gradient(p, q));
  }
  return out;
}

template std::vector<double> adamGradientVector(const Matrix<double>&,
                                                const std::vector<AdamPair>&);
template std::vector<std::complex<double>> adamGradientVector(
    const Matrix<std::complex<double>>&, const std::vector<AdamPair>&);

Matrix<double> adamKappaMatrix(std::size_t n, const std::vector<AdamPair>& pairs,
                               const std::vector<double>& step) {
  if (step.size() != pairs.size()) {
    throw std::runtime_error("adamKappaMatrix: step size differs from the number of pairs");
  }
  Matrix<double> kappa(n, n, 0.0);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const auto [p, q] = pairs[i];
    if (p < q || p >= n) throw std::runtime_error("adamKappaMatrix: pairs must have n > p >= q");
    if (p == q) continue;  // no phase rotation for real orbitals
    kappa(p, q) = step[i];
    kappa(q, p) = -step[i];
  }
  return kappa;
}

Matrix<std::complex<double>> adamKappaMatrix(std::size_t n, const std::vector<AdamPair>& pairs,
                                             const std::vector<std::complex<double>>& step) {
  if (step.size() != pairs.size()) {
    throw std::runtime_error("adamKappaMatrix: step size differs from the number of pairs");
  }
  Matrix<std::complex<double>> kappa(n, n, std::complex<double>{});
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const auto [p, q] = pairs[i];
    if (p < q || p >= n) throw std::runtime_error("adamKappaMatrix: pairs must have n > p >= q");
    if (p == q) {
      kappa(p, p) = std::complex<double>(0.0, step[i].imag());
    } else {
      kappa(p, q) = step[i];
      kappa(q, p) = -std::conj(step[i]);
    }
  }
  return kappa;
}

// =====================================================================
// AdamOptimizer
// =====================================================================

template <typename T>
AdamOptimizer<T>::AdamOptimizer(const AdamOptions& options)
    : opt_(options), learning_rate_(options.learning_rate) {}

template <typename T>
void AdamOptimizer<T>::reset() {
  learning_rate_ = opt_.learning_rate;
  extra_iterations_ = 0;
  restart_ = false;
}

template <typename T>
AdamResult AdamOptimizer<T>::run(AdamProblem<T>& problem) {
  AdamResult result;
  restart_ = false;  // clean_adam: ADAM allows convergence for small energy differences again

  int max_calls = opt_.base_iterations + extra_iterations_;
  if (max_calls > opt_.max_iterations_limit) max_calls = opt_.base_iterations;

  AdamOptions run_options = opt_;
  run_options.learning_rate = learning_rate_;
  Adam<T> adam(problem.dimension(), run_options);
  result.learning_rate = learning_rate_;

  double energy_old = problem.energy();
  double energy_best = energy_old;
  double energy_difference = 0.0;
  double max_gradient_best = 0.0;
  double max_gradient = 0.0;
  result.energy_start = energy_old;

  for (;;) {
    const std::vector<T> gradient = problem.gradient();
    if (gradient.size() != problem.dimension()) {
      throw std::runtime_error("AdamOptimizer::run: gradient size differs from dimension()");
    }
    max_gradient = 0.0;
    for (const T& g : gradient) max_gradient = std::max(max_gradient, std::abs(g));

    if (max_gradient < opt_.gradient_tolerance) {
      // Gradient converged: back to the initial rate and iteration budget.
      result.gradient_converged = true;
      learning_rate_ = opt_.learning_rate;
      extra_iterations_ = 0;
      break;
    }

    const std::vector<T> step = adam.step(gradient);
    problem.rotate(step);
    const double energy = problem.energy();
    if (energy < energy_best) {
      result.improved = true;
      energy_best = energy;
      max_gradient_best = max_gradient;
      problem.saveBest();
    }

    energy_difference = energy - energy_old;
    if (adam.iterations() > 1 && std::abs(energy_difference) < opt_.energy_tolerance) {
      result.energy_converged = true;
      break;
    }
    energy_old = energy;
    if (adam.iterations() == max_calls) break;
  }

  result.iterations = adam.iterations();
  result.energy_best = energy_best;
  result.energy_difference = energy_difference;
  result.max_gradient = result.improved ? max_gradient_best : max_gradient;

  // Nothing better was found although the energy was still changing:
  // ask for a redo with a lower learning rate and more iterations.
  if (!result.improved && std::abs(energy_difference) > opt_.energy_tolerance) {
    restart_ = true;
    learning_rate_ *= opt_.rate_factor;
    extra_iterations_ += opt_.iterations_increment;
  }
  result.restart_requested = restart_;

  problem.restoreBest();
  return result;
}

template class AdamOptimizer<double>;
template class AdamOptimizer<std::complex<double>>;

}  // namespace rerdmft
