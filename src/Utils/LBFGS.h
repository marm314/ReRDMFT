#ifndef RERDMFT_UTILS_LBFGS_H
#define RERDMFT_UTILS_LBFGS_H

#include <cstddef>
#include <functional>
#include <vector>

namespace rerdmft {

// Generic, problem-agnostic limited-memory BFGS (L-BFGS) solver for
//
//   minimize  value(x)
//
// an unconstrained smooth minimization given only function values and
// gradients (no explicit Hessian, unlike Utils/SQP.h's exact-Hessian
// Newton-SQP) -- the standard choice when the Hessian is unavailable or
// too expensive to form, and `x` has no box/equality constraints.
//
// Maintains the last `history_size` (s_i, y_i) curvature pairs
// (s_i = x_{i+1}-x_i, y_i = g_{i+1}-g_i) and applies the classic
// two-loop recursion (Nocedal & Wright, "Numerical Optimization",
// Algorithm 7.4) to compute an approximate Newton direction
// d_k = -H_k g_k without ever forming the n-by-n H_k explicitly, then
// takes a backtracking Armijo line search step along d_k.
//
// A curvature pair (s_i, y_i) is only pushed onto the history when
// y_i^T s_i is sufficiently positive (the standard L-BFGS safeguard,
// "damped"/skipped update): this keeps the implicit Hessian
// approximation positive definite (hence d_k always a descent
// direction) even when the line search's Armijo-only acceptance would
// otherwise admit a step that violates the curvature (secant)
// condition -- the price of not doing a full strong-Wolfe line search.
struct LbfgsOptions {
  int max_iterations = 500;
  // Number of (s_i, y_i) pairs kept; the "L" (limited) in L-BFGS. 5-20
  // is the usual practical range -- more improves the Hessian
  // approximation's quality per iteration at the cost of O(history_size
  // * n) work per step.
  std::size_t history_size = 10;
  // Converged once ||gradient(x)||_inf falls below this.
  double gradient_tolerance = 1e-8;
  // Backtracking line search: Armijo sufficient-decrease constant.
  double armijo_c1 = 1e-4;
  // Backtracking line search: step-length shrink factor per rejection.
  double backtrack_factor = 0.5;
  int max_line_search_steps = 60;
  // A curvature pair is only kept if y^T s exceeds this (relative to
  // ||s||*||y||) -- guards the positive-definiteness of the implicit
  // Hessian approximation (see this file's own header comment above).
  double curvature_skip_tolerance = 1e-10;
};

struct LbfgsResult {
  std::vector<double> x;
  double objective_value = 0.0;
  std::vector<double> gradient;
  bool converged = false;
  int iterations = 0;
};

using LbfgsValueFn = std::function<double(const std::vector<double>&)>;
using LbfgsGradientFn = std::function<std::vector<double>(const std::vector<double>&)>;

// `x0` is the (unconstrained, so always feasible) starting point.
// Throws std::runtime_error if `gradient` returns a vector whose size
// does not match `x0`.
LbfgsResult solveLbfgs(const LbfgsValueFn& value, const LbfgsGradientFn& gradient,
                        std::vector<double> x0, const LbfgsOptions& options = {});

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_LBFGS_H
