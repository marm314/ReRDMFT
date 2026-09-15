#include "SQP.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

constexpr double kFeasibilityTolerance = 1e-8;
constexpr double kZeroTolerance = 1e-10;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

// Result of fixing the components of d listed by `status` (-1 == pinned
// at lb, +1 == pinned at ub, 0 == free) and optimizing the local
// quadratic model over the remaining free variables subject to
// a_eq * d = r_eq -- the equality-constrained QP that
// solveBoxEqualityQp's active-set loop re-solves every iteration for
// its current working set.
struct EqualityQpSolution {
  std::vector<double> d;                 // full vector, size n
  std::vector<double> lambda;            // one per row of a_eq
  std::vector<double> reduced_gradient;  // (g + B d - A^T lambda)_i, size n
};

EqualityQpSolution solveEqualityQpGivenFixed(const Matrix<double>& b, const std::vector<double>& g,
                                              const Matrix<double>& a_eq,
                                              const std::vector<double>& r_eq,
                                              const std::vector<int>& status,
                                              const std::vector<double>& fixed_value) {
  const std::size_t n = g.size();
  const std::size_t m = a_eq.rows();

  std::vector<std::size_t> free_idx;
  free_idx.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (status[i] == 0) free_idx.push_back(i);
  }
  const std::size_t nf = free_idx.size();

  std::vector<double> d(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (status[i] != 0) d[i] = fixed_value[i];
  }
  std::vector<double> lambda(m, 0.0);

  if (nf > 0) {
    const std::size_t dim = nf + m;
    Matrix<double> kkt(dim, dim, 0.0);
    std::vector<double> rhs(dim, 0.0);

    for (std::size_t jj = 0; jj < nf; ++jj) {
      const std::size_t j = free_idx[jj];
      double ghat_j = g[j];
      for (std::size_t i = 0; i < n; ++i) {
        if (status[i] != 0) ghat_j += b(j, i) * d[i];
      }
      rhs[jj] = -ghat_j;
      for (std::size_t kk = 0; kk < nf; ++kk) {
        kkt(jj, kk) = b(j, free_idx[kk]);
      }
      for (std::size_t r = 0; r < m; ++r) {
        kkt(jj, nf + r) = a_eq(r, j);
        kkt(nf + r, jj) = a_eq(r, j);
      }
    }
    for (std::size_t r = 0; r < m; ++r) {
      double rhat_r = r_eq[r];
      for (std::size_t i = 0; i < n; ++i) {
        if (status[i] != 0) rhat_r -= a_eq(r, i) * d[i];
      }
      rhs[nf + r] = rhat_r;
    }

    // The KKT matrix can be singular or ill-conditioned in practice --
    // an occupation-number Hessian is NOT guaranteed positive definite
    // (RDMFT energy vs. occupations is generally non-convex, and some
    // functionals' second derivatives diverge/vanish near n=0 or n=1),
    // and a particular active-set working set can make the reduced
    // system exactly singular even when nearby working sets are fine.
    // Standard Levenberg-Marquardt-style fix: add a small, increasing
    // multiple of the identity to the FREE-FREE Hessian block only
    // (never to the Lagrange-multiplier rows/columns, which have no
    // diagonal to damp) and retry -- this perturbs the local quadratic
    // model slightly without changing what a converged (zero-gradient,
    // correct-sign-multiplier) solution looks like, since the damping
    // only pushes the SEARCH DIRECTION, not the stationarity conditions
    // solveBoxEqualityQp's own outer loop checks.
    Matrix<double> kkt_inv;
    bool solved = false;
    std::string last_error;
    double damping = 0.0;
    for (int attempt = 0; attempt < 20; ++attempt) {
      Matrix<double> kkt_damped = kkt;
      for (std::size_t jj = 0; jj < nf; ++jj) kkt_damped(jj, jj) += damping;
      try {
        kkt_inv = invert(kkt_damped);
        solved = true;
        break;
      } catch (const std::exception& e) {
        last_error = e.what();
        damping = (damping <= 0.0) ? 1e-10 : damping * 10.0;
      }
    }
    if (!solved) {
      throw std::runtime_error("solveBoxEqualityQp: KKT solve failed even after "
                                "Levenberg-Marquardt regularization (" +
                                last_error + ")");
    }
    std::vector<double> sol(dim, 0.0);
    for (std::size_t r = 0; r < dim; ++r) {
      double val = 0.0;
      for (std::size_t c = 0; c < dim; ++c) val += kkt_inv(r, c) * rhs[c];
      sol[r] = val;
    }
    for (std::size_t jj = 0; jj < nf; ++jj) d[free_idx[jj]] = sol[jj];
    for (std::size_t r = 0; r < m; ++r) lambda[r] = sol[nf + r];
  } else if (m > 0) {
    // Every variable pinned, yet an equality constraint remains: only
    // consistent if the pinned values already happen to satisfy it (no
    // freedom left to enforce it otherwise) -- a degenerate working set
    // that should not arise from solveBoxEqualityQp's own ratio test
    // starting from a feasible point, but checked here defensively.
    for (std::size_t r = 0; r < m; ++r) {
      double val = 0.0;
      for (std::size_t i = 0; i < n; ++i) val += a_eq(r, i) * d[i];
      if (std::abs(val - r_eq[r]) > kFeasibilityTolerance) {
        throw std::runtime_error(
            "solveBoxEqualityQp: degenerate working set pins every variable "
            "without satisfying a_eq * d = r_eq");
      }
    }
  }

  std::vector<double> reduced_gradient(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double val = g[i];
    for (std::size_t k = 0; k < n; ++k) val += b(i, k) * d[k];
    for (std::size_t r = 0; r < m; ++r) val -= a_eq(r, i) * lambda[r];
    reduced_gradient[i] = val;
  }

  return {d, lambda, reduced_gradient};
}

}  // namespace

BoxEqualityQpResult solveBoxEqualityQp(const Matrix<double>& b, const std::vector<double>& g,
                                        const Matrix<double>& a_eq,
                                        const std::vector<double>& r_eq,
                                        const std::vector<double>& lb,
                                        const std::vector<double>& ub, std::vector<double> d0,
                                        int max_iterations) {
  const std::size_t n = g.size();
  if (b.rows() != n || b.cols() != n) {
    throw std::runtime_error("solveBoxEqualityQp: b dimensions inconsistent with g");
  }
  const std::size_t m = a_eq.rows();
  if (m != 0 && a_eq.cols() != n) {
    throw std::runtime_error("solveBoxEqualityQp: a_eq dimensions inconsistent with g");
  }
  if (r_eq.size() != m) {
    throw std::runtime_error("solveBoxEqualityQp: r_eq size inconsistent with a_eq");
  }
  if (lb.size() != n || ub.size() != n) {
    throw std::runtime_error("solveBoxEqualityQp: lb/ub size inconsistent with g");
  }
  if (d0.size() != n) {
    throw std::runtime_error("solveBoxEqualityQp: d0 size inconsistent with g");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (d0[i] < lb[i] - kFeasibilityTolerance || d0[i] > ub[i] + kFeasibilityTolerance) {
      throw std::runtime_error("solveBoxEqualityQp: d0 violates lb <= d0 <= ub at index " +
                                std::to_string(i));
    }
  }
  for (std::size_t r = 0; r < m; ++r) {
    double val = 0.0;
    for (std::size_t i = 0; i < n; ++i) val += a_eq(r, i) * d0[i];
    if (std::abs(val - r_eq[r]) > kFeasibilityTolerance) {
      throw std::runtime_error("solveBoxEqualityQp: d0 violates a_eq * d0 = r_eq at row " +
                                std::to_string(r));
    }
  }

  std::vector<double> d = d0;
  std::vector<int> status(n, 0);  // 0 free, -1 pinned at lb, +1 pinned at ub
  std::vector<double> fixed_value(n, 0.0);

  BoxEqualityQpResult result;
  for (int iter = 0; iter < max_iterations; ++iter) {
    result.iterations = iter + 1;
    for (std::size_t i = 0; i < n; ++i) {
      if (status[i] == -1) fixed_value[i] = lb[i];
      if (status[i] == 1) fixed_value[i] = ub[i];
    }

    const EqualityQpSolution eq_sol =
        solveEqualityQpGivenFixed(b, g, a_eq, r_eq, status, fixed_value);

    std::vector<double> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = eq_sol.d[i] - d[i];

    if (norm(p) <= kZeroTolerance) {
      // Current iterate already solves the equality-QP for this working
      // set -- optimal for the FULL box+equality QP iff every active
      // bound's multiplier has the sign that makes relaxing it
      // unprofitable (>= 0 at an active lower bound, <= 0 at an active
      // upper bound, the standard KKT sign test for a minimization).
      int worst_idx = -1;
      double worst_violation = kZeroTolerance;
      for (std::size_t i = 0; i < n; ++i) {
        if (status[i] == -1 && -eq_sol.reduced_gradient[i] > worst_violation) {
          worst_violation = -eq_sol.reduced_gradient[i];
          worst_idx = static_cast<int>(i);
        } else if (status[i] == 1 && eq_sol.reduced_gradient[i] > worst_violation) {
          worst_violation = eq_sol.reduced_gradient[i];
          worst_idx = static_cast<int>(i);
        }
      }
      if (worst_idx < 0) {
        result.d = d;
        result.eq_multipliers = eq_sol.lambda;
        result.bound_multipliers.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
          if (status[i] != 0) result.bound_multipliers[i] = eq_sol.reduced_gradient[i];
        }
        result.converged = true;
        return result;
      }
      status[static_cast<std::size_t>(worst_idx)] = 0;
      continue;
    }

    // Ratio test: how far can we move from `d` towards `eq_sol.d` along
    // p before a currently-free variable would cross a bound? (Pinned
    // variables have p[i] == 0 identically, so they never block here.)
    double alpha = 1.0;
    int blocking_idx = -1;
    int blocking_bound = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (status[i] != 0) continue;
      if (p[i] > kZeroTolerance) {
        const double candidate = (ub[i] - d[i]) / p[i];
        if (candidate < alpha) {
          alpha = candidate;
          blocking_idx = static_cast<int>(i);
          blocking_bound = 1;
        }
      } else if (p[i] < -kZeroTolerance) {
        const double candidate = (lb[i] - d[i]) / p[i];
        if (candidate < alpha) {
          alpha = candidate;
          blocking_idx = static_cast<int>(i);
          blocking_bound = -1;
        }
      }
    }
    if (alpha < 0.0) alpha = 0.0;

    for (std::size_t i = 0; i < n; ++i) d[i] += alpha * p[i];

    if (blocking_idx >= 0 && alpha < 1.0) {
      const auto bi = static_cast<std::size_t>(blocking_idx);
      status[bi] = blocking_bound;
      d[bi] = (blocking_bound == 1) ? ub[bi] : lb[bi];  // snap, avoid roundoff drift
    }
    // Otherwise alpha == 1 (no blocking variable): d now equals
    // eq_sol.d, so next iteration's equality-QP solve for the SAME
    // working set will find p ~ 0 and fall into the optimality check.
  }

  result.d = d;
  result.converged = false;
  return result;
}

SqpResult solveSqp(const SqpValueFn& value, const SqpGradientFn& gradient,
                    const SqpHessianFn& hessian, const Matrix<double>& a_eq,
                    const std::vector<double>& b_eq, const std::vector<double>& lb,
                    const std::vector<double>& ub, std::vector<double> x0,
                    const SqpOptions& options) {
  const std::size_t n = x0.size();
  if (lb.size() != n || ub.size() != n) {
    throw std::runtime_error("solveSqp: lb/ub size inconsistent with x0");
  }
  const std::size_t m = a_eq.rows();
  if (m != 0 && a_eq.cols() != n) {
    throw std::runtime_error("solveSqp: a_eq dimensions inconsistent with x0");
  }
  if (b_eq.size() != m) {
    throw std::runtime_error("solveSqp: b_eq size inconsistent with a_eq");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (x0[i] < lb[i] - kFeasibilityTolerance || x0[i] > ub[i] + kFeasibilityTolerance) {
      throw std::runtime_error("solveSqp: x0 violates lb <= x0 <= ub at index " +
                                std::to_string(i));
    }
  }
  for (std::size_t r = 0; r < m; ++r) {
    double val = 0.0;
    for (std::size_t i = 0; i < n; ++i) val += a_eq(r, i) * x0[i];
    if (std::abs(val - b_eq[r]) > kFeasibilityTolerance) {
      throw std::runtime_error("solveSqp: x0 violates a_eq * x0 = b_eq at row " +
                                std::to_string(r));
    }
  }

  std::vector<double> x = std::move(x0);
  const std::vector<double> zero_r_eq(m, 0.0);
  const std::vector<double> d0(n, 0.0);

  SqpResult result;
  for (int iter = 0; iter < options.max_iterations; ++iter) {
    result.iterations = iter + 1;
    const std::vector<double> g = gradient(x);
    const Matrix<double> h = hessian(x);

    std::vector<double> lb_step(n), ub_step(n);
    for (std::size_t i = 0; i < n; ++i) {
      lb_step[i] = lb[i] - x[i];
      ub_step[i] = ub[i] - x[i];
    }

    const BoxEqualityQpResult qp = solveBoxEqualityQp(h, g, a_eq, zero_r_eq, lb_step, ub_step, d0);

    if (norm(qp.d) <= options.step_tolerance) {
      result.x = x;
      result.objective_value = value(x);
      result.converged = true;
      return result;
    }

    // Backtracking Armijo line search along d. The box+equality
    // feasible set is convex, and both x and x+d lie in it (d solves
    // the QP subproblem's OWN shifted bounds/equality exactly), so
    // every x + alpha*d for alpha in [0,1] is feasible too -- no
    // separate feasibility check needed while backtracking.
    const double directional_derivative = dot(g, qp.d);
    const double f0 = value(x);
    double alpha = 1.0;
    std::vector<double> x_trial(n);
    for (int ls = 0; ls < options.max_line_search_steps; ++ls) {
      for (std::size_t i = 0; i < n; ++i) x_trial[i] = x[i] + alpha * qp.d[i];
      if (value(x_trial) <= f0 + options.armijo_c1 * alpha * directional_derivative) {
        break;
      }
      alpha *= options.backtrack_factor;
    }
    x = x_trial;
  }

  result.x = x;
  result.objective_value = value(x);
  result.converged = false;
  return result;
}

}  // namespace rerdmft
