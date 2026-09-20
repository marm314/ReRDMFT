#ifndef RERDMFT_UTILS_ADAM_H
#define RERDMFT_UTILS_ADAM_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// ADAM (adaptive moment estimation) for orbital rotations, ported from
// standalone_donof's m_adam.F90 (the ADAM step), m_optorb.F90 (the
// orbital-optimization loop that drives it, its restart rule and the
// convergence tests) and m_noft_driver.F90 (allocation, macro-level
// use of `restart`). The one difference: the GRADIENT is provided by the
// caller (grad_pq / grad_pq_cmplx are NOT computed here -- in the Fortran
// they were built from the Lagrange-multiplier matrix Lambda_pq).
//
// CONVENTIONS ARE THIS PROJECT'S, NOT THE FORTRAN'S (Hessian_opt/OrbitalGradient.h,
// SpinorRotation.h): U_rot = exp(-kappa), C_new = C_old U_rot; one
// independent parameter per orbital pair (p,q), p > q, kappa_pq = t + i y,
// kappa_qp = -conj(kappa_pq) (real t: kappa_pq = +t, kappa_qp = -t;
// imaginary y: kappa_pq = kappa_qp = iy); and the gradient entry of a pair
// is g_pq = dE/dt + i dE/dy (orbitalGradient's g_pq, whose Re/Im parts are
// exactly what jointOrbitalGradient returns), stored in the LOWER triangle
// p >= q. So ADAM's step for the pair is directly the update of kappa_pq
// (Re = dt, Im = dy), with no sign flip. The set of pairs is NOT fixed
// here: every entry point takes an explicit pair list (`adamPairIndices`
// gives the same list/order as hessianPairIndices, optionally with the
// diagonal p = q), so nothing is restricted to a triangle by this file.
//
// Everything is generic over T = double (real orbitals, e.g. NON_REL) and
// T = std::complex<double> (complex orbitals, e.g. X2C / 4-component).
// The two cases are NOT the same algorithm applied twice -- see the notes
// on Adam::step below.

// Parameters exactly as used in the Fortran (adam_t defaults and
// opt_orb's hard-coded numbers).
struct AdamOptions {
  double learning_rate = 0.01;   // l_rate (also the value it is reset to on gradient convergence)
  double rate_factor = 0.2;      // fact_rate: l_rate *= this on a failed (restart) run
  double beta1 = 0.7;            // first-moment decay
  double beta2 = 0.9;            // second-moment decay
  double epsilon = 1e-15;        // added to sqrt(second moment) in the denominator
  // Iterations per run: icall_max = base_iterations + extra_iterations,
  // where extra_iterations (icall_max_add) grows by iterations_increment
  // after every failed run and is reset to 0 on gradient convergence; if
  // icall_max would exceed max_iterations_limit it falls back to
  // base_iterations (opt_orb: "if(icall_max>200) icall_max=10").
  int base_iterations = 10;
  int iterations_increment = 20;
  int max_iterations_limit = 200;
  // Convergence thresholds. In the Fortran both are INPUT values:
  // 10**-itolLambda for the gradient (itolLambda = 5 in the shipped
  // driver) and tolE (1e-9) for the energy.
  double gradient_tolerance = 1e-5;
  double energy_tolerance = 1e-9;
};

// ---------------------------------------------------------------------
// The ADAM step itself (build_adam / clean_adam)
// ---------------------------------------------------------------------

// Holds the moments for `n` gradient entries and turns a gradient into a
// step. `step()` returns the update Delta (the "kappa" entry of the
// Fortran: it already contains the minus sign and the learning rate), so
// x_new = x + step.
//
// Real (T = double), per entry:
//   m1 = b1 m1 + (1-b1) g
//   m2 = b2 m2 + (1-b2) g^2
//   m1hat = m1/(1-b1^t),  m2hat = m2/(1-b2^t)
//   m2max = max(m2max, m2hat)                        (AMSGrad-style running maximum)
//   step  = -lr m1hat / (sqrt(m2max) + eps)
// Complex (T = std::complex<double>), per entry, same formulas but with a
// COMPLEX first moment and the REAL second moment of |g|^2 = g conj(g),
// i.e. the real and imaginary parts of one complex entry are normalized
// by a SHARED denominator (so the step keeps the direction of the complex
// gradient); this is deliberately not two independent real ADAMs on
// (Re g, Im g). The running maximum uses the real parts, as in the
// Fortran. t counts calls to step() since the last clean().
template <typename T>
class Adam {
 public:
  explicit Adam(std::size_t n, const AdamOptions& options = {});

  // Zeroes the moments and the iteration counter (clean_adam).
  void clean();

  // One ADAM step for `gradient` (size n). Throws std::runtime_error on a
  // size mismatch.
  std::vector<T> step(const std::vector<T>& gradient);

  std::size_t size() const { return m1_.size(); }
  int iterations() const { return t_; }
  double learningRate() const { return lr_; }
  void setLearningRate(double lr) { lr_ = lr; }

 private:
  AdamOptions opt_;
  double lr_;
  int t_ = 0;
  std::vector<T> m1_;
  std::vector<double> m2_;      // |g|^2 moment: real for both T (the Fortran's complex
  std::vector<double> m2max_;   // second moment has identically zero imaginary part)
};

// ---------------------------------------------------------------------
// Pairs, gradient vector and rotation generator kappa (our conventions)
// ---------------------------------------------------------------------
using AdamPair = std::pair<std::size_t, std::size_t>;

// The parameter pairs for `n` orbitals, lower triangle in row order:
// (1,0), (2,0), (2,1), (3,0), ... -- identical to hessianPairIndices(n).
// With include_diagonal the diagonal pair (p,p) is listed after the
// off-diagonal pairs of its row, i.e. (0,0), (1,0), (1,1), (2,0), ...
// A diagonal entry is a single-entry PHASE rotation kappa_pp = i y of
// orbital p (complex only; for real orbitals its gradient is identically
// zero and its kappa entry is zero).
std::vector<AdamPair> adamPairIndices(std::size_t n, bool include_diagonal = false);

// The gradient vector fed to Adam: entry I is gradient(p,q) for
// pairs[I] = (p,q) -- `gradient` is orbitalGradient()'s matrix, whose
// lower triangle p >= q is filled (the upper triangle is NOT read).
// Throws if a pair has p < q or is out of range.
template <typename T>
std::vector<T> adamGradientVector(const Matrix<T>& gradient, const std::vector<AdamPair>& pairs);

// The rotation generator kappa (n x n) for a step over `pairs`:
//   real:    kappa_pq = +step_I, kappa_qp = -step_I (p > q); a diagonal
//            pair gives 0 (a real orbital has no phase rotation);
//   complex: kappa_pq = step_I, kappa_qp = -conj(step_I) (p > q), i.e.
//            Re = t, Im = y of the project convention; a diagonal pair
//            gives kappa_pp = i Im(step_I) (anti-Hermitian, so any real
//            part of a diagonal step is dropped).
// Rotate with C_new = C_old * spinorRotationMatrix(kappa) (= exp(-kappa)).
// Throws if step.size() != pairs.size() or a pair has p < q.
Matrix<double> adamKappaMatrix(std::size_t n, const std::vector<AdamPair>& pairs,
                               const std::vector<double>& step);
Matrix<std::complex<double>> adamKappaMatrix(std::size_t n, const std::vector<AdamPair>& pairs,
                                             const std::vector<std::complex<double>>& step);

// ---------------------------------------------------------------------
// One orbital-optimization run (the ADAM branch of opt_orb)
// ---------------------------------------------------------------------

// What a run needs from the method. The problem owns two sets of
// orbitals, the running TRIAL orbitals that every step rotates
// cumulatively (NO_COEF_tmp) and the BEST ones found so far (NO_COEF);
// gradient/energy always refer to the trial orbitals.
template <typename T>
class AdamProblem {
 public:
  virtual ~AdamProblem() = default;

  // Number of gradient entries = number of pairs (Adam only sees a flat
  // vector; the pair list is the caller's).
  virtual std::size_t dimension() const = 0;
  // Energy of the trial orbitals (fixed occupations/RDMs).
  virtual double energy() = 0;
  // Gradient at the trial orbitals, entry per pair: g_pq = dE/dt + i dE/dy
  // (T = double: g_pq = dE/dt) -- adamGradientVector(orbitalGradient(F),
  // pairs). -gradient is the descent direction.
  virtual std::vector<T> gradient() = 0;
  // Rotates the trial orbitals cumulatively:
  //   C_trial <- C_trial * spinorRotationMatrix(adamKappaMatrix(n, pairs, step)).
  virtual void rotate(const std::vector<T>& step) = 0;
  // The current trial orbitals lowered the energy: store them as the best.
  virtual void saveBest() = 0;
  // End of the run: make the best orbitals current again (recompute the
  // integrals in the best basis, as opt_orb does before returning).
  virtual void restoreBest() = 0;
};

struct AdamResult {
  bool improved = false;            // some trial energy went below the starting energy
  bool gradient_converged = false;  // stopped: max |gradient entry| < gradient_tolerance
  bool energy_converged = false;    // stopped: |energy change| < energy_tolerance (after >1 steps)
  bool restart_requested = false;   // no improvement while the energy still moved: the
                                    // learning rate was lowered, more iterations granted
  int iterations = 0;               // ADAM steps taken in this run (icall)
  double energy_start = 0.0;
  double energy_best = 0.0;
  double energy_difference = 0.0;   // last E - E_previous (Ediff)
  double max_gradient = 0.0;        // max |entry| of the last gradient, or of the gradient at
                                    // the last improving iteration if the run improved
                                    // (opt_orb's maxdiff_all)
  double learning_rate = 0.0;       // rate used in THIS run
};

// Drives Adam through one orbital-optimization call and carries the state
// that persists between calls of the macro (occupation/orbital) loop:
// the current learning rate, the extra-iteration allowance and the
// `restart` flag. Per run (opt_orb's imethod == 0 branch):
//   clean moments; icall_max = base + extra (or base if > limit);
//   loop { gradient; if max|g| < tol -> reset rate/extra, stop;
//          step; rotate; energy; if E < E_best -> saveBest;
//          if icall > 1 and |E - E_prev| < tolE -> stop;
//          if icall == icall_max -> stop }
//   if no improvement and |E - E_prev| > tolE ->
//          restart = true, rate *= rate_factor, extra += increment;
//   restoreBest().
template <typename T>
class AdamOptimizer {
 public:
  explicit AdamOptimizer(const AdamOptions& options = {});

  AdamResult run(AdamProblem<T>& problem);

  // True after a run that made no progress although the energy was still
  // moving; the macro loop must not declare convergence then
  // (m_noft_driver: "|E-E_old| < tolE .and. .not. ADAMd%restart").
  bool restartRequested() const { return restart_; }
  double learningRate() const { return learning_rate_; }
  int extraIterations() const { return extra_iterations_; }
  const AdamOptions& options() const { return opt_; }

  // Back to the initial learning rate, no extra iterations, no restart.
  void reset();

 private:
  AdamOptions opt_;
  double learning_rate_;
  int extra_iterations_ = 0;
  bool restart_ = false;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_ADAM_H
