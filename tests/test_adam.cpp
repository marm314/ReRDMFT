// Standalone unit test of Utils/ADAM.h/.cpp.
// Build/run: make test_adam LIBCINT=/path/to/libcint.a
//
//   * real ADAM against hand-computed values (first step = -lr*sign(g),
//     bias correction, running maximum of the second moment),
//   * complex ADAM: SHARED |g|^2 normalization of Re/Im (unlike two
//     independent real ADAMs), phase of the gradient preserved,
//   * pair list / gradient vector / kappa assembly in the PROJECT's
//     conventions (pairs p > q, kappa_pq = t + iy, kappa_qp = -conj),
//   * a real orbital-rotation toy (E = sum_i d_i <i|A|i>, C_new = C exp(-kappa),
//     gradient g_pq = dE/dt + i dE/dy by finite differences) that ADAM must
//     drive to the eigenvector solution, real and complex,
//   * AdamOptimizer: convergence on real and complex toy problems,
//     restart rule (rate *= 0.2, +20 iterations, reset on convergence),
//     saveBest/restoreBest bookkeeping.
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "ADAM.h"
#include "LinearAlgebra.h"
#include "Matrix.h"
#include "SpinorRotation.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::cout << "  FAIL: " << what << "\n";
  }
}

bool near(double a, double b, double tol = 1e-13) { return std::abs(a - b) <= tol; }
bool near(const C& a, const C& b, double tol = 1e-13) { return std::abs(a - b) <= tol; }

void testRealStep() {
  AdamOptions o;  // b1=0.7, b2=0.9, lr=0.01, eps=1e-15
  Adam<double> adam(2, o);
  // Step 1: m1hat = g, m2hat = g^2  ->  step = -lr g/(|g|+eps) = -lr sign(g).
  std::vector<double> s1 = adam.step({3.0, -0.25});
  check(near(s1[0], -0.01, 1e-12) && near(s1[1], 0.01, 1e-12), "real step 1 = -lr*sign(g)");
  // Step 2 with g = (1, 1): by hand.
  const double g0[2] = {3.0, -0.25}, g1[2] = {1.0, 1.0};
  std::vector<double> s2 = adam.step({1.0, 1.0});
  for (int i = 0; i < 2; ++i) {
    const double m1 = 0.7 * (0.3 * g0[i]) + 0.3 * g1[i];
    const double m2 = 0.9 * (0.1 * g0[i] * g0[i]) + 0.1 * g1[i] * g1[i];
    const double m1h = m1 / (1.0 - 0.49);
    const double m2h = m2 / (1.0 - 0.81);
    const double m2first = g0[i] * g0[i];       // step-1 hat value
    const double vmax = std::max(m2first, m2h);
    check(near(s2[i], -0.01 * m1h / (std::sqrt(vmax) + 1e-15), 1e-13),
          "real step 2 matches hand computation (component " + std::to_string(i) + ")");
  }
  // Component 0: g dropped 3 -> 1, so the running max keeps the LARGE second moment.
  check(std::abs(s2[0]) < 0.01, "running maximum of the second moment shrinks the step");
  check(adam.iterations() == 2, "iteration counter");
  adam.clean();
  check(adam.iterations() == 0, "clean resets the counter");
  std::vector<double> s3 = adam.step({3.0, -0.25});
  check(near(s3[0], s1[0]) && near(s3[1], s1[1]), "clean resets the moments (step 1 reproduced)");
  bool threw = false;
  try { adam.step({1.0}); } catch (const std::exception&) { threw = true; }
  check(threw, "size mismatch throws");
}

void testComplexStep() {
  AdamOptions o;
  Adam<C> adam(3, o);
  std::vector<C> s = adam.step({C(3.0, 4.0), C(5.0, 0.0), C(0.0, 5.0)});
  // Step 1: m1hat = g, m2hat = |g|^2 -> step = -lr g/|g|.
  check(near(s[0], C(-0.006, -0.008), 1e-12), "complex step 1 = -lr g/|g| (shared normalization)");
  check(near(s[1], C(-0.01, 0.0), 1e-12), "purely real complex gradient = real ADAM");
  check(near(s[2], C(0.0, -0.01), 1e-12), "purely imaginary complex gradient rotates only Im");
  // Two independent real ADAMs would give (-0.01,-0.01) for g = 3+4i.
  check(!near(s[0], C(-0.01, -0.01), 1e-6), "NOT two independent real ADAMs");

  // Step 2, by hand, for component 0.
  const C g0(3.0, 4.0), g1(-1.0, 2.0);
  std::vector<C> s2 = adam.step({g1, C(5.0, 0.0), C(0.0, 5.0)});
  const C m1 = 0.7 * (0.3 * g0) + 0.3 * g1;
  const double m2 = 0.9 * (0.1 * std::norm(g0)) + 0.1 * std::norm(g1);
  const C m1h = m1 / (1.0 - 0.49);
  const double m2h = m2 / (1.0 - 0.81);
  const double vmax = std::max(std::norm(g0), m2h);
  check(near(s2[0], -0.01 * m1h / (std::sqrt(vmax) + 1e-15), 1e-13), "complex step 2 matches hand computation");
}

void testKappa() {
  const std::size_t n = 4;
  const std::vector<AdamPair> pairs = adamPairIndices(n);
  check(pairs.size() == 6 && pairs[0] == AdamPair(1, 0) && pairs[1] == AdamPair(2, 0) &&
            pairs[2] == AdamPair(2, 1) && pairs[5] == AdamPair(3, 2),
        "pairs: lower triangle p > q in hessianPairIndices order");
  const std::vector<AdamPair> with_diag = adamPairIndices(n, true);
  check(with_diag.size() == 10 && with_diag[0] == AdamPair(0, 0) && with_diag[1] == AdamPair(1, 0) &&
            with_diag[2] == AdamPair(1, 1) && with_diag[9] == AdamPair(3, 3),
        "pairs: optional diagonal, row order");

  // Gradient vector reads the LOWER triangle only.
  Matrix<C> gm(n, n, C{});
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q <= p; ++q) gm(p, q) = C(p + 0.1 * q, -0.5 * q);
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = p + 1; q < n; ++q) gm(p, q) = C(999.0, 999.0);
  const std::vector<C> gv = adamGradientVector(gm, with_diag);
  check(near(gv[1], gm(1, 0)) && near(gv[2], gm(1, 1)) && near(gv[9], gm(3, 3)),
        "gradient vector picks g(p,q), p >= q");
  bool threw = false;
  try { adamGradientVector(gm, std::vector<AdamPair>{{0, 1}}); } catch (const std::exception&) { threw = true; }
  check(threw, "gradient vector rejects p < q (upper triangle is not filled)");

  // Real kappa: kappa_pq = +step, kappa_qp = -step; diagonal ignored.
  std::vector<double> rs(with_diag.size());
  for (std::size_t i = 0; i < rs.size(); ++i) rs[i] = 0.1 * (i + 1);
  Matrix<double> kr = adamKappaMatrix(n, with_diag, rs);
  bool anti = true;
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) anti &= near(kr(p, q), -kr(q, p));
  check(anti && near(kr(1, 0), 0.2) && near(kr(0, 1), -0.2) && near(kr(2, 1), 0.5) && near(kr(2, 2), 0.0),
        "real kappa: kappa_pq=+t, kappa_qp=-t, antisymmetric, zero diagonal");

  // Complex kappa: kappa_pq = t+iy, kappa_qp = -t+iy (= -conj), i.e. the
  // explicit t and y generators of the project.
  std::vector<C> cs(with_diag.size());
  for (std::size_t i = 0; i < cs.size(); ++i) cs[i] = C(0.1 * (i + 1), -0.03 * (i + 1));
  Matrix<C> kc = adamKappaMatrix(n, with_diag, cs);
  bool herm = true;
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) herm &= near(kc(p, q), -std::conj(kc(q, p)));
  check(herm, "complex kappa: anti-Hermitian");
  const C step10 = cs[1];  // pair (1,0)
  check(near(kc(1, 0), C(step10.real(), step10.imag())) && near(kc(0, 1), C(-step10.real(), step10.imag())),
        "complex kappa: t+iy in (p,q), -t+iy in (q,p)");
  check(near(kc(1, 1), C(0.0, cs[2].imag())), "complex kappa: diagonal keeps only the imaginary part");
  threw = false;
  try { adamKappaMatrix(n, pairs, std::vector<double>(3)); } catch (const std::exception&) { threw = true; }
  check(threw, "kappa size mismatch throws");
}

// Toy problem: x (real or complex) with E = sum a_i |x_i - c_i|^2 (the
// gradient in the ADAM convention is dE/dRe + i dE/dIm = 2 a (x - c)).
// "Orbitals" = x itself; rotate adds the step.
template <typename T>
class Bowl : public AdamProblem<T> {
 public:
  Bowl(std::vector<T> x, std::vector<T> c, std::vector<double> a)
      : trial_(x), best_(x), c_(std::move(c)), a_(std::move(a)) {}
  std::size_t dimension() const override { return trial_.size(); }
  double energy() override {
    double e = 0.0;
    for (std::size_t i = 0; i < trial_.size(); ++i) e += a_[i] * std::norm(std::complex<double>(trial_[i] - c_[i]));
    return e;
  }
  std::vector<T> gradient() override {
    std::vector<T> g(trial_.size());
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = 2.0 * a_[i] * (trial_[i] - c_[i]);
    return g;
  }
  void rotate(const std::vector<T>& s) override {
    for (std::size_t i = 0; i < s.size(); ++i) trial_[i] += s[i];
  }
  void saveBest() override { best_ = trial_; ++saves; }
  void restoreBest() override { trial_ = best_; ++restores; }
  const std::vector<T>& best() const { return best_; }
  const std::vector<T>& trial() const { return trial_; }
  int saves = 0, restores = 0;

 private:
  std::vector<T> trial_, best_, c_;
  std::vector<double> a_;
};

template <typename T>
void testOptimizer(const std::string& label, std::vector<T> x0, std::vector<T> c) {
  AdamOptions o;
  o.learning_rate = 0.05;
  o.energy_tolerance = 1e-14;
  o.gradient_tolerance = 1e-3;
  AdamOptimizer<T> opt(o);
  Bowl<T> problem(x0, c, {1.0, 2.0, 0.5});
  double e_start = problem.energy();
  int runs = 0;
  bool converged = false;
  for (; runs < 200 && !converged; ++runs) {
    AdamResult r = opt.run(problem);
    converged = r.gradient_converged;
    check(r.energy_best <= r.energy_start + 1e-15, label + ": best energy never above the start");
  }
  double dist = 0.0;
  for (std::size_t i = 0; i < c.size(); ++i) dist = std::max(dist, std::abs(problem.best()[i] - c[i]));
  std::cout << label << ": " << runs << " runs, E " << e_start << " -> " << problem.energy()
            << ", max |x-c| = " << dist << "\n";
  check(converged, label + ": gradient convergence reached");
  check(dist < 5e-3, label + ": best orbitals close to the minimum");
  check(problem.restores >= runs, label + ": restoreBest called at the end of every run");
  check(near(problem.energy(), problem.energy()), label + ": consistent energy");
  check(!opt.restartRequested(), label + ": no restart pending after convergence");
}

// Orbital toy: E(C) = sum_i d_i (C^dagger A C)_ii with fixed weights d_i,
// rotations C -> C exp(-kappa). The gradient is the project's g_pq =
// dE/dt + i dE/dy obtained by central differences along the EXPLICIT
// generators (kappa_pq=+t, kappa_qp=-t) and (kappa_pq=kappa_qp=iy).
template <typename T>
Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) { return a * b; }

template <typename T>
class OrbitalToy : public AdamProblem<T> {
 public:
  OrbitalToy(Matrix<T> a, std::vector<double> d)
      : a_(std::move(a)), d_(std::move(d)), n_(a_.rows()), pairs_(adamPairIndices(n_)),
        trial_(n_, n_, T{}), best_(n_, n_, T{}) {
    for (std::size_t i = 0; i < n_; ++i) trial_(i, i) = T(1.0);
    best_ = trial_;
  }
  std::size_t dimension() const override { return pairs_.size(); }
  double energyOf(const Matrix<T>& c) const {
    double e = 0.0;
    const Matrix<T> ac = a_ * c;
    for (std::size_t i = 0; i < n_; ++i) {
      C m{};
      for (std::size_t k = 0; k < n_; ++k) m += std::conj(C(c(k, i))) * C(ac(k, i));
      e += d_[i] * m.real();
    }
    return e;
  }
  double energy() override { return energyOf(trial_); }
  Matrix<T> generator(std::size_t p, std::size_t q, double t, double y) const {
    Matrix<T> k(n_, n_, T{});
    k(p, q) = T(t);
    k(q, p) = T(-t);
    if constexpr (!std::is_same_v<T, double>) {
      k(p, q) += C(0.0, y);
      k(q, p) += C(0.0, y);
    }
    return k;
  }
  double energyAt(std::size_t p, std::size_t q, double t, double y) const {
    return energyOf(trial_ * spinorRotationMatrix(generator(p, q, t, y)));
  }
  std::vector<T> gradient() override {
    const double h = 1e-5;
    std::vector<T> g(pairs_.size());
    for (std::size_t i = 0; i < pairs_.size(); ++i) {
      const auto [p, q] = pairs_[i];
      const double dt = (energyAt(p, q, h, 0) - energyAt(p, q, -h, 0)) / (2 * h);
      if constexpr (std::is_same_v<T, double>) {
        g[i] = dt;
      } else {
        const double dy = (energyAt(p, q, 0, h) - energyAt(p, q, 0, -h)) / (2 * h);
        g[i] = C(dt, dy);
      }
    }
    return g;
  }
  void rotate(const std::vector<T>& step) override {
    trial_ = trial_ * spinorRotationMatrix(adamKappaMatrix(n_, pairs_, step));
  }
  void saveBest() override { best_ = trial_; }
  void restoreBest() override { trial_ = best_; }
  const Matrix<T>& best() const { return best_; }
  const std::vector<AdamPair>& pairs() const { return pairs_; }
  const Matrix<T>& a() const { return a_; }

 private:
  Matrix<T> a_;
  std::vector<double> d_;
  std::size_t n_;
  std::vector<AdamPair> pairs_;
  Matrix<T> trial_, best_;
};

template <typename T>
void testOrbitalRotation(const std::string& label, const Matrix<T>& a) {
  const std::vector<double> d = {1.0, 0.7, 0.4, 0.1};
  OrbitalToy<T> problem(a, d);
  // Analytic check of the toy's gradient (real case): dE/dt_pq = 2 (d_p - d_q) A_pq at C = 1.
  if constexpr (std::is_same_v<T, double>) {
    const std::vector<double> g = problem.gradient();
    double err = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
      const auto [p, q] = problem.pairs()[i];
      err = std::max(err, std::abs(g[i] - 2.0 * (d[p] - d[q]) * a(p, q)));
    }
    check(err < 1e-8, label + ": FD gradient = 2(d_p-d_q)A_pq, error " + std::to_string(err));
  }
  AdamOptions o;
  o.learning_rate = 0.05;
  o.gradient_tolerance = 1e-4;
  o.energy_tolerance = 1e-13;
  AdamOptimizer<T> opt(o);
  const double e_start = problem.energy();
  bool converged = false;
  int runs = 0;
  for (; runs < 400 && !converged; ++runs) converged = opt.run(problem).gradient_converged;
  // Exact minimum: largest weight with the smallest eigenvalue (rearrangement inequality).
  Matrix<C> ac(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) for (std::size_t j = 0; j < a.cols(); ++j) ac(i, j) = a(i, j);
  const std::vector<double> ev = diagonalizeHermitian(ac).eigenvalues;
  double e_min = 0.0;
  for (std::size_t i = 0; i < d.size(); ++i) e_min += d[i] * ev[i];
  const double e_end = problem.energyOf(problem.best());
  std::cout << label << ": " << runs << " runs, E " << e_start << " -> " << e_end << " (exact minimum " << e_min << ")\n";
  check(converged, label + ": gradient converged");
  check(std::abs(e_end - e_min) < 1e-6, label + ": reaches the exact minimum");
}

void testRestart() {
  // Start almost at the minimum but with a huge learning rate: every
  // ADAM step overshoots, no trial energy beats the start while the
  // energy keeps changing -> restart rule.
  AdamOptions o;
  o.learning_rate = 100.0;
  o.energy_tolerance = 1e-12;
  o.gradient_tolerance = 1e-9;
  AdamOptimizer<double> opt(o);
  Bowl<double> problem({0.001, 0.001, 0.001}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  AdamResult r1 = opt.run(problem);
  check(!r1.improved && r1.restart_requested && opt.restartRequested(), "restart requested after a run without improvement");
  check(r1.iterations == 10, "first run does base_iterations = 10 steps");
  check(near(opt.learningRate(), 100.0 * 0.2) && opt.extraIterations() == 20, "learning rate x0.2, +20 iterations");
  check(near(problem.trial()[0], 0.001), "restoreBest returns the (unchanged) best orbitals");
  AdamResult r2 = opt.run(problem);
  check(r2.iterations == 30 && near(r2.learning_rate, 20.0), "second run: 10+20 steps at the reduced rate");
  check(near(opt.learningRate(), 4.0) && opt.extraIterations() == 40, "second failure compounds");
  // A run that converges resets the state.
  Bowl<double> good({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  AdamResult r3 = opt.run(good);
  check(r3.gradient_converged && r3.iterations == 0, "zero gradient converges immediately, no steps");
  check(near(opt.learningRate(), 100.0) && opt.extraIterations() == 0 && !opt.restartRequested(),
        "gradient convergence resets learning rate and extra iterations");
  // icall_max limit fallback: budget above the limit falls back to base.
  AdamOptions o2 = o;
  o2.max_iterations_limit = 25;
  AdamOptimizer<double> opt2(o2);
  Bowl<double> p2({0.001, 0.001, 0.001}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  opt2.run(p2);              // 10 steps, extra -> 20
  AdamResult q = opt2.run(p2);   // 30 > 25 -> falls back to 10
  check(q.iterations == 10, "icall_max above the limit falls back to the base value");
}

}  // namespace

int main() {
  testRealStep();
  testComplexStep();
  testKappa();
  testOptimizer<double>("real bowl", {1.0, -0.7, 0.4}, {0.2, 0.3, -0.1});
  testOptimizer<C>("complex bowl", {C(1.0, 0.5), C(-0.7, 0.2), C(0.4, -0.9)},
                   {C(0.2, -0.1), C(0.3, 0.3), C(-0.1, 0.0)});
  {
    // Symmetric / Hermitian test matrices with well-separated eigenvalues.
    Matrix<double> ar(4, 4);
    Matrix<C> ac(4, 4);
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        const double v = (i == j) ? 1.0 + 0.8 * i : 0.3 / (1.0 + i + j);
        ar(i, j) = v;
        const double im = (i < j) ? 0.15 / (1.0 + i + 2 * j) : (i > j ? -0.15 / (1.0 + j + 2 * i) : 0.0);
        ac(i, j) = C(v, im);
      }
    }
    testOrbitalRotation<double>("real orbital rotation toy", ar);
    testOrbitalRotation<C>("complex orbital rotation toy", ac);
  }
  testRestart();
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
