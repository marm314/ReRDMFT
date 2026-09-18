#include "LBFGS.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace rerdmft {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

double infNorm(const std::vector<double>& a) {
  double m = 0.0;
  for (double v : a) m = std::max(m, std::abs(v));
  return m;
}

std::vector<double> axpy(double alpha, const std::vector<double>& x,
                          const std::vector<double>& y) {
  std::vector<double> result(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) result[i] = alpha * x[i] + y[i];
  return result;
}

struct CurvaturePair {
  std::vector<double> s;
  std::vector<double> y;
  double rho;  // 1 / (y^T s)
};

// Two-loop recursion (Nocedal & Wright, Algorithm 7.4): returns
// d = -H_k * g, the L-BFGS search direction, from the stored history
// (oldest first, newest last) and the current gradient g.
std::vector<double> twoLoopRecursion(const std::deque<CurvaturePair>& history,
                                      const std::vector<double>& g) {
  const std::size_t n = g.size();
  std::vector<double> q = g;
  const std::size_t m = history.size();
  std::vector<double> alpha(m);

  for (std::size_t idx = m; idx-- > 0;) {
    const auto& pair = history[idx];
    alpha[idx] = pair.rho * dot(pair.s, q);
    q = axpy(-alpha[idx], pair.y, q);
  }

  // Initial Hessian approximation H0 = gamma * I, gamma the standard
  // Barzilai-Borwein-style scaling from the most recent pair (Nocedal &
  // Wright, Eq. 7.20) -- falls back to the identity (gamma=1) before
  // any curvature pair has been accepted.
  double gamma = 1.0;
  if (m > 0) {
    const auto& newest = history.back();
    const double y_dot_y = dot(newest.y, newest.y);
    if (y_dot_y > 0.0) gamma = 1.0 / (newest.rho * y_dot_y);
  }
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) r[i] = gamma * q[i];

  for (std::size_t idx = 0; idx < m; ++idx) {
    const auto& pair = history[idx];
    const double beta = pair.rho * dot(pair.y, r);
    r = axpy(alpha[idx] - beta, pair.s, r);
  }

  std::vector<double> d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = -r[i];
  return d;
}

}  // namespace

LbfgsResult solveLbfgs(const LbfgsValueFn& value, const LbfgsGradientFn& gradient,
                        std::vector<double> x0, const LbfgsOptions& options) {
  std::vector<double> x = std::move(x0);
  const std::size_t n = x.size();

  std::vector<double> g = gradient(x);
  if (g.size() != n) {
    throw std::runtime_error("solveLbfgs: gradient() returned a vector of the wrong size");
  }

  std::deque<CurvaturePair> history;

  LbfgsResult result;
  double f = value(x);

  for (int iter = 0; iter < options.max_iterations; ++iter) {
    result.iterations = iter + 1;

    if (infNorm(g) <= options.gradient_tolerance) {
      result.x = x;
      result.objective_value = f;
      result.gradient = g;
      result.converged = true;
      return result;
    }

    std::vector<double> d = twoLoopRecursion(history, g);
    double directional_derivative = dot(g, d);
    if (!(directional_derivative < 0.0)) {
      // Safeguard against a bad (non-descent) direction, e.g. from an
      // ill-conditioned/degenerate history right after a skipped
      // curvature update -- reset to plain steepest descent for this
      // one step (always a descent direction since g != 0 here, having
      // already failed the convergence check above) rather than
      // diverging, and drop the stale history that produced it.
      history.clear();
      d.assign(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) d[i] = -g[i];
      directional_derivative = dot(g, d);
    }

    double alpha = 1.0;
    std::vector<double> x_trial(n);
    double f_trial = f;
    bool accepted = false;
    for (int ls = 0; ls < options.max_line_search_steps; ++ls) {
      for (std::size_t i = 0; i < n; ++i) x_trial[i] = x[i] + alpha * d[i];
      f_trial = value(x_trial);
      if (f_trial <= f + options.armijo_c1 * alpha * directional_derivative) {
        accepted = true;
        break;
      }
      alpha *= options.backtrack_factor;
    }
    if (!accepted) {
      // Line search exhausted without satisfying Armijo -- report the
      // best point found so far rather than taking a step that may not
      // decrease the objective.
      result.x = x;
      result.objective_value = f;
      result.gradient = g;
      result.converged = false;
      return result;
    }

    const std::vector<double> g_trial = gradient(x_trial);
    std::vector<double> s(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
      s[i] = x_trial[i] - x[i];
      y[i] = g_trial[i] - g[i];
    }
    const double sy = dot(s, y);
    const double s_norm = std::sqrt(dot(s, s));
    const double y_norm = std::sqrt(dot(y, y));
    if (sy > options.curvature_skip_tolerance * std::max(s_norm * y_norm, 1.0)) {
      if (history.size() >= options.history_size) history.pop_front();
      history.push_back(CurvaturePair{s, y, 1.0 / sy});
    }

    x = x_trial;
    f = f_trial;
    g = g_trial;
  }

  result.x = x;
  result.objective_value = f;
  result.gradient = g;
  result.converged = false;
  return result;
}

}  // namespace rerdmft
