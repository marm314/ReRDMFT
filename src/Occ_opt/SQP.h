#ifndef RERDMFT_OCC_OPT_SQP_H
#define RERDMFT_OCC_OPT_SQP_H

#include <cstddef>
#include <functional>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// Generic, problem-agnostic sequential quadratic programming (SQP)
// solver for
//
//   minimize    value(x)
//   subject to  a_eq * x = b_eq
//               lb <= x <= ub
//
// i.e. a smooth objective with LINEAR equality constraints and simple
// box bounds -- exactly the shape of an occupation-number optimization
// (sum n_p = N_electrons, 0 <= n_p <= 1), which is the motivating use
// case, but nothing here is specific to occupation numbers: `value`/
// `gradient`/`hessian` are opaque callbacks, so this file only depends
// on Matrix.h/std::vector, never on eri/RKB/basis-specific types.
//
// At each iterate x_k, builds the local quadratic model
//   g_k^T d + (1/2) d^T B_k d,   g_k = gradient(x_k), B_k = hessian(x_k)
// and solves it (via solveBoxEqualityQp, below) for the step d subject
// to the SAME equality constraint restricted to the step (a_eq * d = 0,
// since x_k is kept feasible exactly throughout -- see solveSqp's own
// comment) and the box constraint shifted to the step,
// lb - x_k <= d <= ub - x_k. A backtracking Armijo line search along d
// then picks the step length. This is Newton's method specialized to
// linear equality + box constraints (an exact-Hessian SQP, not a
// quasi-Newton/BFGS one) -- `hessian` is assumed cheap/available, as it
// is for the RDMFT occupation-number objectives this is built for.
struct SqpOptions {
  int max_iterations = 100;
  // Converged once the step norm ||d|| falls below this (the QP
  // subproblem's own solution barely moves the iterate any further).
  double step_tolerance = 1e-10;
  // Backtracking line search: Armijo sufficient-decrease constant.
  double armijo_c1 = 1e-4;
  // Backtracking line search: step-length shrink factor per rejection.
  double backtrack_factor = 0.5;
  int max_line_search_steps = 50;
};

struct SqpResult {
  std::vector<double> x;
  double objective_value = 0.0;
  bool converged = false;
  int iterations = 0;
};

using SqpValueFn = std::function<double(const std::vector<double>&)>;
using SqpGradientFn = std::function<std::vector<double>(const std::vector<double>&)>;
using SqpHessianFn = std::function<Matrix<double>(const std::vector<double>&)>;

// `x0` MUST already be feasible (a_eq * x0 == b_eq, lb <= x0 <= ub, both
// to within a loose numerical tolerance) -- solveSqp throws
// std::runtime_error otherwise, rather than silently starting from an
// infeasible point (finding an initial feasible point, "Phase 1" in the
// standard SQP literature, is the CALLER's job here; for occupation
// numbers, e.g. n_p = N_electrons / n_orbitals for every p is always
// feasible when 0 < N_electrons < n_orbitals).
SqpResult solveSqp(const SqpValueFn& value, const SqpGradientFn& gradient,
                    const SqpHessianFn& hessian, const Matrix<double>& a_eq,
                    const std::vector<double>& b_eq, const std::vector<double>& lb,
                    const std::vector<double>& ub, std::vector<double> x0,
                    const SqpOptions& options = {});

// The box- and linear-equality-constrained QP subproblem itself, solved
// via a primal active-set method specialized to simple bounds (Nocedal &
// Wright, "Numerical Optimization", the equality-constrained-QP KKT
// solve of Ch. 16 plus an active-set loop over which bounds are
// currently pinned, restricted to bounds -- as opposed to the fully
// general linear-inequality case -- since each bound touches only one
// variable, which is what keeps the ratio test and multiplier check
// below simple):
//
//   minimize    g^T d + (1/2) d^T b d
//   subject to  a_eq * d = r_eq
//               lb <= d <= ub
//
// Exposed separately (rather than folded into solveSqp) so it can be
// validated on its own, against known analytic QP solutions, before
// trusting the outer Newton-SQP loop that repeatedly calls it.
//
// `d0` MUST already be feasible for THIS subproblem (a_eq * d0 == r_eq,
// lb <= d0 <= ub) -- throws std::runtime_error otherwise. solveSqp
// always calls this with d0 = 0 (the zero step is always feasible for
// the shifted subproblem, since x_k itself is feasible).
struct BoxEqualityQpResult {
  std::vector<double> d;
  // One multiplier per row of a_eq.
  std::vector<double> eq_multipliers;
  // One multiplier per component of d; 0 where the bound is inactive.
  // Sign convention: positive at an active LOWER bound, negative at an
  // active UPPER bound, exactly like a standard KKT multiplier for a
  // constraint written as (lb - d <= 0) / (d - ub <= 0) in a
  // minimization problem.
  std::vector<double> bound_multipliers;
  bool converged = false;
  int iterations = 0;
};

BoxEqualityQpResult solveBoxEqualityQp(const Matrix<double>& b, const std::vector<double>& g,
                                        const Matrix<double>& a_eq,
                                        const std::vector<double>& r_eq,
                                        const std::vector<double>& lb,
                                        const std::vector<double>& ub, std::vector<double> d0,
                                        int max_iterations = 200);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_SQP_H
