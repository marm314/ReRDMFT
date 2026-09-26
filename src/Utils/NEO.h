#ifndef RERDMFT_UTILS_NEO_H
#define RERDMFT_UTILS_NEO_H

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace rerdmft {

// Norm-Extended Optimization (NEO): a generic, matrix-free, second-order
// restricted-step (trust-region) optimizer -- see doc/NEO.tex (which this
// file follows section by section; "Sec. N" below refers to it). It is
// completely independent of what the parameters MEAN (no orbital,
// occupation or integral knowledge lives here): the caller supplies
//   * the gradient g at the current expansion point (CEP),
//   * a callback returning the Hessian-vector product H*v for any
//     trial vector v (the ONLY way the Hessian is ever touched -- H is
//     never formed or stored), and
//   * for the full macro-iteration driver, the energy at the CEP and at
//     trial points (a NeoProblem, below).
//
// The second-order model is (Sec. 2)
//   q(x) = f0 + Re(g^dagger x) + (1/2) x^dagger H x,
// and the step is the i-th eigenvector (ascending) of the extended
// ("augmented Hessian") matrix
//   L(alpha) = [ 0        alpha g^dagger ]
//              [ alpha g  H              ]      (Hermitian; real symmetric for T=double),
// normalized to z0 = 1, with the step d = x/alpha (Sec. 3): a Newton step
// with the Hessian level-shifted by the self-consistent shift
// lambda = alpha g^dagger x. By Cauchy interlacing (Sec. 3.2) H-lambda has
// EXACTLY i-1 negative eigenvalues, so
//   target_order m (number of negative Hessian eigenvalues at the
//   solution) = 0 -> minimum (i=1), m > 0 -> saddle point of order m
//   (i = m+1), i.e. an "excited state" search: energy MAXIMIZED along m
//   Hessian directions and minimized along the rest. Same algorithm, only
//   the eigenvector index (and the trust-radius acceptance rule,
//   Sec. 5.2) change.
//
// Scalar types: T = double (real parameters, e.g. NON_REL orbitals, or the
// real (t;y) joint rotation parameters of Hessian_opt for complex spinors)
// and T = std::complex<double> (a genuinely complex-linear parameter
// vector with a Hermitian Hessian; all transposes become adjoints,
// Sec. 7). The energy is always real, and so are lambda, alpha and the
// trust radius. NOTE for complex T: the model above is what the solver
// optimizes -- g and H must already be the ones that make
// q(x) = f0 + Re(g^dagger x) + (1/2) x^dagger H x the correct second-order
// expansion in the caller's own complex parametrization (a caller using
// the doubled (kappa; kappa*) form of Sec. 7 is responsible for keeping
// its vectors on the conjugation-structure subspace).
//
// SADDLE-POINT CAVEAT (Sec. 3.2): a Hessian eigenvector orthogonal to g
// (e.g. by symmetry) is also an eigenvector of L with z0 = 0 and defines
// no step; the solver never selects such a root and does not count it in
// the index (roots with |z0| < `decoupled_z0` are skipped). Because the
// Krylov-type starting vectors (1;0) and (0;g) can only generate roots
// coupled to g in the first place, "target_order m" therefore counts the
// negative-curvature directions COUPLED to the gradient; use
// neoLowestHessianEigenpairs (below) at a converged point to verify the
// actual index of the full Hessian.

// Hessian-vector product callback: returns H*v for a vector v of the
// problem dimension n. Called once per new trial vector (never for the
// all-zero vector).
template <typename T>
using NeoHessianVectorFn = std::function<std::vector<T>(const std::vector<T>&)>;

// ---------------------------------------------------------------------
// One NEO step (micro-iterations: Davidson on L(alpha), Sec. 6)
// ---------------------------------------------------------------------

struct NeoStepOptions {
  // Number of negative Hessian eigenvalues at the target stationary point
  // (0 = minimum, m = saddle point of order m). Eigenvector index i = m+1.
  // Must be <= n.
  std::size_t target_order = 0;

  // Davidson convergence: all target roots' residual norms
  // ||L z - theta z|| must fall below
  //   max(1e-13, min(residual_tolerance, residual_relative * ||g||))
  // (Sec. 6.3: micro-tolerance tied to the macro-iteration's accuracy;
  // the relative part keeps quadratic convergence near the solution).
  double residual_tolerance = 1e-6;
  double residual_relative = 0.1;
  // Maximum Davidson iterations per (re)solve at one alpha.
  int max_micro_iterations = 200;
  // Maximum trial-subspace dimension before collapsing to the current
  // Ritz vectors (stored H*b products are rotated, not recomputed).
  // Raised internally to at least 4*(target_order+1).
  std::size_t max_subspace = 40;

  // Restricted step (Sec. 4): after solving with alpha = 1, if
  // ||d|| > radius, alpha > 1 is searched until
  // | ||d_alpha|| - radius | <= step_length_tolerance * radius.
  double step_length_tolerance = 0.1;
  int max_alpha_iterations = 20;
  double max_alpha = 1e6;

  // A root of L with |z0| below decoupled_z0 * min(1, ||g||) (never below
  // 1e-14) is "decoupled" (see the header comment) and is skipped, not
  // counted, when picking the i-th root. Relative to ||g|| because a
  // genuinely coupled root above the lowest has z0 = O(||g||).
  double decoupled_z0 = 1e-8;
  // A new trial vector whose norm after orthogonalization is below this
  // fraction of its norm before is discarded as linearly dependent.
  double linear_dependence = 1e-10;
  // Denominator floor of the diagonal preconditioner (H_pp - theta).
  double preconditioner_floor = 1e-4;
  // Saddle targets (target_order > 0), when a diagonal of H is supplied:
  // add unit vectors on the target_order lowest diagonal elements to the
  // starting space so the lowest roots of L are represented from the
  // beginning.
  bool guess_from_diagonal = true;
  // > 0: DYNAMIC saddle order. Every solve() sets the target order to the number of coupled roots of
  // L(1) below -saddle_cutoff (target_order is then only the caller's expectation and is ignored),
  // i.e. the energy is maximized along every g-coupled Hessian direction more negative than the
  // cutoff and minimized along the rest. This is what a problem with a known energy-scale gap
  // between its negative- and positive-curvature directions needs (the relativistic min-max
  // problem: electron-positron rotations at -O(2 c^2 n_i), everything else >= 0): a fixed
  // target_order counts only directions COUPLED to g (see the SADDLE-POINT CAVEAT above), which
  // symmetry can make far fewer than the Hessian index. Pair it with guess_from_diagonal = false.
  double saddle_cutoff = 0.0;
  // With `sector` set the count is instead the number of leading roots whose Ritz vector lies mostly in the stiff sector.
  // Optional partition of the parameters (size n, nonzero = "stiff" sector; empty = none): every trial vector is
  // added as two sector-PURE vectors. With a dynamic saddle order this keeps the Ritz vectors of the two
  // decoupled sectors (electron-positron rotations at -1e4 versus everything else at O(1)) from mixing, which
  // otherwise gives mixed Ritz pairs with Rayleigh quotients anywhere -- including near 0 -- and makes the
  // "roots below -cutoff" count chase spurious roots.
  std::vector<char> sector;
};

template <typename T>
struct NeoStep {
  std::vector<T> step;        // d = x/alpha, the step in parameter space
  double shift = 0.0;         // lambda = alpha g^dagger x (the level shift)
  double alpha = 1.0;         // norm-extension scaling that was used
  double step_norm = 0.0;     // ||d|| (Euclidean/Hermitian 2-norm)
  // Predicted change q(d) - f0 = Re(g^dagger d) + (1/2) d^dagger H d,
  // evaluated from the stored H*b products (Sec. 3.4/6), so it is
  // consistent with the step actually returned even when it was scaled.
  double predicted_change = 0.0;
  double residual_norm = 0.0;  // largest target-root Davidson residual
  bool converged = false;      // Davidson reached the residual tolerance
  bool restricted = false;     // alpha > 1 was needed (step on the boundary)
  bool scaled = false;         // no alpha found: step simply rescaled to the radius
  bool trivial = false;        // ||g|| ~ 0: zero step, nothing to do
  int micro_iterations = 0;    // Davidson iterations in THIS solve() call
  std::size_t hessian_products = 0;  // cumulative H*v calls by this solver
};

// Solves the NEO eigenproblem for one CEP. Constructed once per CEP: the
// Davidson subspace and every stored H*b product are kept, so calling
// solve() again with a smaller radius after a REJECTED step (Sec. 5) costs
// (almost) no new Hessian products (changing alpha only changes the
// reduced matrix, Sec. 6.1).
template <typename T>
class NeoStepSolver {
 public:
  // `gradient`: g at the CEP (its size fixes n). `hessian_diagonal`:
  // optional real diagonal of H (empty = none); only used to precondition
  // the Davidson corrections and to seed saddle-point searches.
  // `guess_vectors`: optional EXTENDED vectors (size n+1; element 0 is the
  // z0 component) added to the starting space, e.g. the previous
  // macro-iteration's lowestRoots() (Sec. 6.3, step 1).
  NeoStepSolver(std::vector<T> gradient, NeoHessianVectorFn<T> hessian_vector,
                std::vector<double> hessian_diagonal = {}, NeoStepOptions options = {},
                std::vector<std::vector<T>> guess_vectors = {});

  // Step for trust radius `radius` (a value <= 0 means unrestricted:
  // the plain alpha = 1 NEO step). Throws std::runtime_error on
  // inconsistent input.
  NeoStep<T> solve(double radius);

  // Extended eigenvectors (size n+1, unit norm) of L for the target root
  // and all roots below it, from the last solve() -- warm-start material
  // for the next macro-iteration's solver.
  const std::vector<std::vector<T>>& lowestRoots() const { return roots_; }

  std::size_t hessianProducts() const { return products_; }
  // Target order in force (the option's, or the dynamic count when saddle_cutoff > 0).
  std::size_t targetOrder() const { return target_; }

 private:
  struct RitzSet;  // eigenpairs of the reduced matrix at one alpha

  bool addVector(std::vector<T> ext);
  bool addTrial(std::vector<T> ext);  // addVector, split into sector-pure parts when opt_.sector is set
  void initialize();
  bool addUnitVector();
  RitzSet ritz(double alpha) const;
  bool davidson(double alpha, int& micro, double& residual);
  void collapse(const RitzSet& set, const std::vector<std::size_t>& keep);
  double effectiveTolerance() const;
  bool reducedStepNorm(double alpha, double& norm) const;
  double findAlpha(double radius, double alpha_start) const;

  std::size_t n_ = 0;
  std::vector<T> g_;
  double g_norm_ = 0.0;
  NeoHessianVectorFn<T> hvec_;
  std::vector<double> diag_;
  NeoStepOptions opt_;
  std::size_t target_ = 0;
  std::vector<std::vector<T>> guesses_;

  // Orthonormal trial basis in the EXTENDED space (element 0 = z0
  // component), the stored products H*(lower part), g^dagger*(lower part),
  // and the alpha-independent reduced pieces of L(alpha) = alpha C + A:
  //   A_jl = bx_j^dagger H bx_l,   C_jl = conj(b0_j) gx_l + conj(gx_j) b0_l.
  std::vector<std::vector<T>> basis_;
  std::vector<std::vector<T>> hb_;
  std::vector<T> gx_;
  std::vector<std::vector<T>> a_;
  std::vector<std::vector<T>> c_;
  std::vector<std::vector<T>> roots_;
  std::vector<std::size_t> unit_order_;  // coordinates in the order tried as unit vectors
  std::size_t unit_cursor_ = 0;
  std::size_t products_ = 0;
  bool initialized_ = false;
};

// ---------------------------------------------------------------------
// Trust-radius rule (Sec. 5)
// ---------------------------------------------------------------------

struct NeoTrustOptions {
  double max_radius = 0.75;   // never above ~pi/4 (Sec. 5)
  // Minimization (target_order == 0):
  double shrink_factor = 2.0 / 3.0;   // r < 0.25 (and on rejection r < 0)
  double grow_factor = 1.2;           // r > 0.75
  double ratio_low = 0.25;
  double ratio_high = 0.75;
  // Saddle points (target_order > 0): acceptance window symmetric about 1.
  double saddle_grow = 1.2;           // a_h
  double saddle_ratio_min = 0.7;      // r_min
  double saddle_ratio_good = 0.85;    // r_good
};

struct NeoTrustDecision {
  bool accept = false;
  double radius = 0.0;  // radius for the next step (or the retry, if rejected)
};

// The Sec. 5 rule: `ratio` = DeltaE/Delta q of the step just tried, taken
// with radius `radius`.
NeoTrustDecision neoTrustDecision(std::size_t target_order, double ratio, double radius,
                                  const NeoTrustOptions& options = {});

// ---------------------------------------------------------------------
// Macro-iteration driver (Sec. 8) -- the whole optimization
// ---------------------------------------------------------------------

// What NeoOptimize needs from the method. The CEP lives INSIDE the problem
// object (for orbitals: the current coefficients C^(k)); every quantity
// below refers to it, and accept() moves it: C <- C exp(R(d)).
template <typename T>
class NeoProblem {
 public:
  virtual ~NeoProblem() = default;

  virtual std::size_t dimension() const = 0;
  // Energy f0 at the CEP.
  virtual double energy() = 0;
  // Gradient g at the CEP (size dimension()).
  virtual std::vector<T> gradient() = 0;
  // Hessian-vector product H*v at the CEP, INCLUDING the second-order
  // term of the exponential map (Sec. 2) -- e.g. a central difference of
  // the gradient along v evaluated at C exp(R(+-eta v)).
  virtual std::vector<T> hessianVector(const std::vector<T>& v) = 0;
  // Optional real diagonal of H (empty = none), used as preconditioner.
  virtual std::vector<double> hessianDiagonal() { return {}; }
  // Energy at the trial point CEP o exp(d) WITHOUT moving the CEP.
  virtual double trialEnergy(const std::vector<T>& d) = 0;
  // Moves the CEP to CEP o exp(d).
  virtual void accept(const std::vector<T>& d) = 0;
};

struct NeoOptions {
  NeoStepOptions step;
  NeoTrustOptions trust;
  int max_iterations = 100;
  // Converged when max_p |g_p| falls below this.
  double gradient_tolerance = 1e-6;
  // Sec. 5: h_0 = 0.5 (~pi/6).
  double initial_radius = 0.5;
  // Give up on an iteration after this many rejected steps in a row, or
  // when the radius shrinks below min_radius.
  int max_rejections = 20;
  double min_radius = 1e-8;
  // |Delta q| and |Delta E| both below this: the ratio is roundoff noise
  // and the step is accepted with the radius unchanged.
  double ratio_floor = 1e-13;
  // After convergence, compute the lowest (target_order + 1) Hessian
  // eigenvalues (neoLowestHessianEigenpairs) and report how many are
  // negative -- the check that the stationary point really has the
  // requested order (Sec. 8, step 2). Costs extra Hessian products.
  bool verify_index = false;
  double verify_tolerance = 1e-6;
  bool verbose = false;  // one line per macro-iteration on std::cout
  bool progress = false; // one live line per Newton step on stderr (Utils/Progress.h)
};

struct NeoIteration {
  double energy = 0.0;
  double gradient_max = 0.0;
  double radius = 0.0;        // radius of the accepted step
  double step_norm = 0.0;
  double shift = 0.0;
  double alpha = 1.0;
  double ratio = 1.0;         // DeltaE / Delta q of the accepted step
  double predicted_change = 0.0;
  double actual_change = 0.0;
  int rejections = 0;
  int micro_iterations = 0;
  std::size_t hessian_products = 0;  // this macro-iteration only
};

struct NeoResult {
  bool converged = false;
  double energy = 0.0;
  double gradient_max = 0.0;
  int iterations = 0;
  std::size_t hessian_products = 0;
  std::vector<NeoIteration> history;
  // Only with verify_index: lowest (target_order+1) Hessian eigenvalues at
  // the final point and the number of them below -verify_tolerance.
  std::vector<double> lowest_hessian_eigenvalues;
  int negative_eigenvalues = -1;
};

template <typename T>
NeoResult neoOptimize(NeoProblem<T>& problem, const NeoOptions& options = {});

// ---------------------------------------------------------------------
// Lowest Hessian eigenpairs (Hessian-index verification)
// ---------------------------------------------------------------------

struct NeoEigenOptions {
  double residual_tolerance = 1e-6;
  int max_iterations = 200;
  std::size_t max_subspace = 60;  // raised internally to at least 4*n_roots
  double linear_dependence = 1e-10;
  double preconditioner_floor = 1e-4;
  // Deterministic pseudo-random start vectors added to the unit-vector
  // seeds (so a symmetry-orthogonal seed set cannot hide a root).
  std::size_t n_random_guesses = 1;
};

template <typename T>
struct NeoEigenResult {
  std::vector<double> eigenvalues;               // ascending
  std::vector<std::vector<T>> eigenvectors;      // unit norm, size n each
  std::vector<double> residual_norms;
  bool converged = false;
  int iterations = 0;
  std::size_t hessian_products = 0;
};

// The `n_roots` lowest eigenpairs of H, matrix-free (block Davidson with
// the same H*v callback). Like every Davidson it can miss a root whose
// eigenvector is orthogonal to the whole starting space; the random start
// vector(s) make that measure-zero.
template <typename T>
NeoEigenResult<T> neoLowestHessianEigenpairs(std::size_t n,
                                             const NeoHessianVectorFn<T>& hessian_vector,
                                             std::size_t n_roots,
                                             const std::vector<double>& hessian_diagonal = {},
                                             const NeoEigenOptions& options = {});

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_NEO_H
