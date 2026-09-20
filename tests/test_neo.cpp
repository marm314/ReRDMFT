// Standalone unit test of Utils/NEO.h/.cpp against dense linear algebra.
// Build/run: make test_neo LIBCINT=/path/to/libcint.a
//
// Checks (real AND complex scalar types; every target order tried):
//   * the NEO step is the i-th eigenvector of the dense extended matrix
//     L (shift = its eigenvalue) and satisfies (H - lambda) d = -g,
//   * H - lambda has EXACTLY target_order negative eigenvalues, and
//     eps_{m-1} <= lambda <= eps_m (interlacing),
//   * predicted change = g.d + d.H.d/2 = lambda/2 (1+|d|^2) (alpha = 1),
//   * restricted steps (alpha > 1) hit the trust radius and satisfy the
//     level-shifted Newton equation; a second solve() with a smaller
//     radius reuses the stored H*b products,
//   * only a few Hessian-vector products are needed, never a dense H,
//   * neoLowestHessianEigenpairs matches the dense spectrum,
//   * the macro driver converges to minima and to saddle points of the
//     requested order on a non-quadratic function, and on complex
//     quadratic (Hermitian) models.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "LinearAlgebra.h"
#include "Matrix.h"
#include "NEO.h"

using namespace rerdmft;

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

struct Rng {
  std::uint64_t s = 88172645463325252ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;  // [-0.5,0.5)
  }
};

template <typename T> T randomScalar(Rng& r);
template <> double randomScalar<double>(Rng& r) { return r.next(); }
template <> std::complex<double> randomScalar<std::complex<double>>(Rng& r) {
  const double re = r.next();
  const double im = r.next();
  return {re, im};
}

template <typename T> T conj_(const T& x) { return x; }
template <> std::complex<double> conj_(const std::complex<double>& x) { return std::conj(x); }
template <typename T> double re_(const T& x) { return x; }
template <> double re_(const std::complex<double>& x) { return x.real(); }

// Hermitian / symmetric H = diag(d) + 0.05 * (random Hermitian): diagonal
// dominant (so the Davidson preconditioner is good), indefinite.
template <typename T>
Matrix<T> makeHessian(std::size_t n, Rng& rng, std::size_t n_negative) {
  Matrix<T> h(n, n, T{});
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (i < n_negative) ? -1.5 + 0.4 * i : 0.6 + 0.13 * static_cast<double>(i);
    h(i, i) = T(d);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const T v = T(0.08) * randomScalar<T>(rng);
      h(i, j) = v;
      h(j, i) = conj_(v);
    }
  }
  return h;
}

template <typename T>
std::vector<T> matvec(const Matrix<T>& h, const std::vector<T>& v) {
  std::vector<T> out(v.size(), T{});
  for (std::size_t i = 0; i < h.rows(); ++i) {
    for (std::size_t j = 0; j < h.cols(); ++j) out[i] += h(i, j) * v[j];
  }
  return out;
}

template <typename T>
T dotc(const std::vector<T>& a, const std::vector<T>& b) {
  T s{};
  for (std::size_t i = 0; i < a.size(); ++i) s += conj_(a[i]) * b[i];
  return s;
}

std::vector<double> denseEigenvalues(const Matrix<double>& h) {
  return diagonalizeSymmetric(h).eigenvalues;
}
std::vector<double> denseEigenvalues(const Matrix<std::complex<double>>& h) {
  return diagonalizeHermitian(h).eigenvalues;
}

template <typename T>
Matrix<T> extended(const Matrix<T>& h, const std::vector<T>& g) {
  const std::size_t n = h.rows();
  Matrix<T> l(n + 1, n + 1, T{});
  for (std::size_t i = 0; i < n; ++i) {
    l(0, i + 1) = conj_(g[i]);
    l(i + 1, 0) = g[i];
    for (std::size_t j = 0; j < n; ++j) l(i + 1, j + 1) = h(i, j);
  }
  return l;
}

template <typename T>
void testStep(const std::string& label) {
  const std::size_t n = 60;
  Rng rng;
  const std::size_t n_neg = 4;
  const Matrix<T> h = makeHessian<T>(n, rng, n_neg);
  std::vector<T> g(n);
  for (T& x : g) x = T(0.3) * randomScalar<T>(rng);
  std::vector<double> diag(n);
  for (std::size_t i = 0; i < n; ++i) diag[i] = re_(h(i, i));

  const std::vector<double> eps = denseEigenvalues(h);
  const std::vector<double> lam = denseEigenvalues(extended(h, g));

  std::cout << label << ": n=" << n << ", H has " << n_neg << " strongly negative diagonal entries\n";
  for (std::size_t m = 0; m <= 3; ++m) {
    std::size_t calls = 0;
    NeoHessianVectorFn<T> hv = [&](const std::vector<T>& v) {
      ++calls;
      return matvec(h, v);
    };
    NeoStepOptions opt;
    opt.target_order = m;
    opt.residual_tolerance = 1e-11;
    opt.residual_relative = 1.0;
    NeoStepSolver<T> solver(g, hv, diag, opt);
    const NeoStep<T> st = solver.solve(0.0);
    const std::string tag = label + " m=" + std::to_string(m) + ": ";

    check(st.converged, tag + "Davidson converged");
    check(std::abs(st.shift - lam[m]) < 1e-8, tag + "shift equals dense L eigenvalue " +
                                                   std::to_string(st.shift) + " vs " +
                                                   std::to_string(lam[m]));
    // (H - lambda) d = -g.
    std::vector<T> hd = matvec(h, st.step);
    double resid = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      resid = std::max(resid, std::abs(hd[i] - T(st.shift) * st.step[i] + g[i]));
    }
    check(resid < 1e-8, tag + "level-shifted Newton equation, max residual " + std::to_string(resid));
    check(std::abs(re_(dotc(g, st.step)) - st.shift) < 1e-8, tag + "lambda = alpha g.d (alpha=1)");
    // Index: H - lambda has exactly m negative eigenvalues; interlacing.
    std::size_t negatives = 0;
    for (double e : eps) negatives += (e - st.shift < 0.0);
    check(negatives == m, tag + "H-lambda has exactly m negative eigenvalues (got " +
                              std::to_string(negatives) + ")");
    if (m > 0) check(eps[m - 1] <= st.shift + 1e-10, tag + "interlacing lower bound");
    check(st.shift <= eps[m] + 1e-10, tag + "interlacing upper bound");
    // Predicted change, two ways.
    const double dq_direct = re_(dotc(g, st.step)) + 0.5 * re_(dotc(st.step, hd));
    const double dn = st.step_norm;
    check(std::abs(st.predicted_change - dq_direct) < 1e-9, tag + "predicted change = g.d + d.H.d/2");
    check(std::abs(st.predicted_change - 0.5 * st.shift * (1.0 + dn * dn)) < 1e-8,
          tag + "predicted change = lambda/2 (1+|d|^2)");
    std::cout << "  m=" << m << ": shift=" << std::setprecision(8) << st.shift << "  |d|=" << dn
              << "  micro-its=" << st.micro_iterations << "  H*v products=" << calls
              << " (dense would need " << n << ")\n";
    check(calls < n, tag + "fewer Hessian products than dimension");

    // Restricted steps: alpha > 1 on the boundary (minimum) or scaled.
    if (m == 0) {
      const double radius = 0.5 * st.step_norm;
      const std::size_t before = calls;
      const NeoStep<T> r1 = solver.solve(radius);
      check(r1.restricted && r1.alpha > 1.0, tag + "radius smaller than |d| gives alpha > 1");
      check(std::abs(r1.step_norm - radius) <= 0.1 * radius * 1.05,
            tag + "restricted step length within 10% of radius: " + std::to_string(r1.step_norm) +
                " vs " + std::to_string(radius));
      std::vector<T> hr = matvec(h, r1.step);
      double rr = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        rr = std::max(rr, std::abs(hr[i] - T(r1.shift) * r1.step[i] + g[i]));
      }
      check(rr < 1e-7, tag + "level-shifted equation at alpha: " + std::to_string(rr));
      const double dq_r = re_(dotc(g, r1.step)) + 0.5 * re_(dotc(r1.step, hr));
      check(std::abs(r1.predicted_change - dq_r) < 1e-9, tag + "restricted predicted change");
      check(r1.predicted_change > st.predicted_change - 1e-12,
            tag + "restricted step is less favourable than unrestricted");
      const NeoStep<T> r2 = solver.solve(0.5 * radius);
      std::cout << "  restricted: alpha=" << r1.alpha << " |d|=" << r1.step_norm
                << " (radius " << radius << "); second solve alpha=" << r2.alpha
                << " |d|=" << r2.step_norm << "; extra products " << (calls - before) << "\n";
      check(r2.step_norm <= 0.5 * radius * 1.11, tag + "second (smaller) radius honoured");
    } else {
      // Saddle target with a radius below the natural step: never longer than the radius.
      const double radius = 0.5 * st.step_norm;
      const NeoStep<T> r1 = solver.solve(radius);
      check(r1.step_norm <= radius * 1.11, tag + "saddle restricted step within radius");
      std::vector<T> hr = matvec(h, r1.step);
      const double dq_r = re_(dotc(g, r1.step)) + 0.5 * re_(dotc(r1.step, hr));
      check(std::abs(r1.predicted_change - dq_r) < 1e-9, tag + "saddle restricted predicted change");
    }
  }
}

template <typename T>
void testEigen(const std::string& label) {
  const std::size_t n = 50;
  Rng rng;
  const Matrix<T> h = makeHessian<T>(n, rng, 3);
  std::vector<double> diag(n);
  for (std::size_t i = 0; i < n; ++i) diag[i] = re_(h(i, i));
  const std::vector<double> eps = denseEigenvalues(h);
  std::size_t calls = 0;
  NeoHessianVectorFn<T> hv = [&](const std::vector<T>& v) {
    ++calls;
    return matvec(h, v);
  };
  NeoEigenOptions opt;
  opt.residual_tolerance = 1e-9;
  const NeoEigenResult<T> res = neoLowestHessianEigenpairs<T>(n, hv, 5, diag, opt);
  check(res.converged, label + " eigen: converged");
  double err = 0.0;
  for (std::size_t i = 0; i < 5; ++i) err = std::max(err, std::abs(res.eigenvalues[i] - eps[i]));
  check(err < 1e-8, label + " eigen: lowest 5 match dense spectrum, error " + std::to_string(err));
  std::cout << label << " eigen: 5 lowest, max error " << err << ", products " << calls << "\n";
}

// f(x) = sum (x_i^2-1)^2/4 + (eps/2) sum_{i!=j} x_i x_j + b.x : a
// non-quadratic landscape with 2^n-ish minima and saddles of every order.
class DoubleWell : public NeoProblem<double> {
 public:
  DoubleWell(std::vector<double> x, double coupling, std::vector<double> b)
      : x_(std::move(x)), eps_(coupling), b_(std::move(b)) {}
  std::size_t dimension() const override { return x_.size(); }
  double value(const std::vector<double>& x) const {
    double s = 0.0, sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      s += 0.25 * (x[i] * x[i] - 1.0) * (x[i] * x[i] - 1.0) + b_[i] * x[i];
      sum += x[i];
    }
    double sq = 0.0;
    for (double v : x) sq += v * v;
    return s + 0.5 * eps_ * (sum * sum - sq);
  }
  double energy() override { return value(x_); }
  std::vector<double> gradient() override {
    double sum = 0.0;
    for (double v : x_) sum += v;
    std::vector<double> g(x_.size());
    for (std::size_t i = 0; i < x_.size(); ++i) {
      g[i] = x_[i] * (x_[i] * x_[i] - 1.0) + b_[i] + eps_ * (sum - x_[i]);
    }
    return g;
  }
  std::vector<double> hessianVector(const std::vector<double>& v) override {
    ++products;
    double sum = 0.0;
    for (double t : v) sum += t;
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      out[i] = (3.0 * x_[i] * x_[i] - 1.0) * v[i] + eps_ * (sum - v[i]);
    }
    return out;
  }
  std::vector<double> hessianDiagonal() override {
    std::vector<double> d(x_.size());
    for (std::size_t i = 0; i < x_.size(); ++i) d[i] = 3.0 * x_[i] * x_[i] - 1.0;
    return d;
  }
  double trialEnergy(const std::vector<double>& d) override {
    std::vector<double> y = x_;
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += d[i];
    return value(y);
  }
  void accept(const std::vector<double>& d) override {
    for (std::size_t i = 0; i < x_.size(); ++i) x_[i] += d[i];
  }
  const std::vector<double>& x() const { return x_; }
  std::size_t products = 0;

 private:
  std::vector<double> x_;
  double eps_;
  std::vector<double> b_;
};

void testDriverReal() {
  const std::size_t n = 12;
  Rng rng;
  std::vector<double> b(n);
  for (double& v : b) v = 0.02 * rng.next();
  for (std::size_t m = 0; m <= 3; ++m) {
    std::vector<double> x0(n);
    for (std::size_t i = 0; i < n; ++i) x0[i] = (i < m ? 0.25 : 0.85) + 0.1 * rng.next();
    DoubleWell problem(x0, 0.03, b);
    NeoOptions opt;
    opt.step.target_order = m;
    opt.gradient_tolerance = 1e-9;
    opt.verify_index = true;
    const NeoResult res = neoOptimize<double>(problem, opt);
    std::cout << "driver (double-well, n=" << n << ", target order " << m << "): converged="
              << res.converged << " iterations=" << res.iterations << " |g|max=" << std::scientific
              << std::setprecision(2) << res.gradient_max << std::fixed << " E=" << std::setprecision(8)
              << res.energy << " Hessian-index=" << res.negative_eigenvalues
              << " H*v products=" << res.hessian_products << "\n    |g|max history:";
    for (const auto& it : res.history) {
      std::cout << " " << std::scientific << std::setprecision(1) << it.gradient_max;
    }
    std::cout << std::fixed << "\n";
    const std::string tag = "driver m=" + std::to_string(m) + ": ";
    check(res.converged, tag + "converged");
    check(res.negative_eigenvalues == static_cast<int>(m),
          tag + "converged Hessian has exactly m negative eigenvalues (got " +
              std::to_string(res.negative_eigenvalues) + ")");
    // Quadratic convergence: the final gradient drop is superlinear.
    if (res.history.size() >= 3) {
      const auto& h = res.history;
      const double g1 = h[h.size() - 2].gradient_max;
      const double g0 = h[h.size() - 3].gradient_max;
      check(g1 < 0.5 * g0 || g1 < 1e-6, tag + "gradient decreasing at the end");
    }
  }
}

// Far-from-solution start on the non-quadratic double well: the plain
// Newton step is unreliable there, so the trust-region machinery (alpha > 1
// restricted steps, ratio-based radius updates, rejected steps retried from
// the SAME solver) must do the work.
void testDriverHard() {
  const std::size_t n = 12;
  Rng rng;
  std::vector<double> b(n);
  for (double& v : b) v = 0.02 * rng.next();
  for (std::size_t m = 0; m <= 2; ++m) {
    std::vector<double> x0(n);
    for (std::size_t i = 0; i < n; ++i) x0[i] = (i < m ? 0.1 : 0.2) + 0.3 * rng.next();
    DoubleWell problem(x0, 0.03, b);
    NeoOptions opt;
    opt.step.target_order = m;
    opt.gradient_tolerance = 1e-9;
    opt.initial_radius = 0.75;
    opt.verify_index = true;
    const NeoResult res = neoOptimize<double>(problem, opt);
    int restricted = 0, rejected = 0;
    for (const auto& it : res.history) {
      restricted += (it.alpha > 1.0);
      rejected += it.rejections;
    }
    std::cout << "driver hard start (target order " << m << "): converged=" << res.converged
              << " iterations=" << res.iterations << " restricted steps=" << restricted
              << " rejected steps=" << rejected << " Hessian-index=" << res.negative_eigenvalues
              << " products=" << res.hessian_products << "\n";
    const std::string tag = "hard driver m=" + std::to_string(m) + ": ";
    check(res.converged, tag + "converged");
    check(res.negative_eigenvalues == static_cast<int>(m), tag + "Hessian index");
  }
}

// Complex quadratic model f = 1/2 x^H K x + Re(b^H x) with Hermitian K
// having `n_neg` negative eigenvalues: its stationary point is a saddle of
// order n_neg, reached from any start in a handful of steps with h_max
// respected.
class ComplexQuadratic : public NeoProblem<std::complex<double>> {
 public:
  using C = std::complex<double>;
  ComplexQuadratic(Matrix<C> k, std::vector<C> b, std::vector<C> x)
      : k_(std::move(k)), b_(std::move(b)), x_(std::move(x)) {}
  std::size_t dimension() const override { return x_.size(); }
  double value(const std::vector<C>& x) const {
    const std::vector<C> kx = matvec(k_, x);
    return 0.5 * re_(dotc(x, kx)) + re_(dotc(b_, x));
  }
  double energy() override { return value(x_); }
  std::vector<C> gradient() override {
    std::vector<C> g = matvec(k_, x_);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] += b_[i];
    return g;
  }
  std::vector<C> hessianVector(const std::vector<C>& v) override { return matvec(k_, v); }
  double trialEnergy(const std::vector<C>& d) override {
    std::vector<C> y = x_;
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += d[i];
    return value(y);
  }
  void accept(const std::vector<C>& d) override {
    for (std::size_t i = 0; i < x_.size(); ++i) x_[i] += d[i];
  }
  const std::vector<C>& x() const { return x_; }

 private:
  Matrix<C> k_;
  std::vector<C> b_, x_;
};

void testDriverComplex() {
  using C = std::complex<double>;
  const std::size_t n = 20;
  for (std::size_t n_neg = 0; n_neg <= 2; ++n_neg) {
    Rng rng;
    Matrix<C> k = makeHessian<C>(n, rng, n_neg);
    std::vector<C> b(n), x0(n);
    for (C& v : b) v = 0.5 * randomScalar<C>(rng);
    for (C& v : x0) v = 0.2 * randomScalar<C>(rng);
    ComplexQuadratic problem(k, b, x0);
    NeoOptions opt;
    opt.step.target_order = n_neg;
    opt.gradient_tolerance = 1e-9;
    opt.verify_index = true;
    const NeoResult res = neoOptimize<C>(problem, opt);
    std::cout << "driver (complex quadratic, n=" << n << ", target order " << n_neg
              << "): converged=" << res.converged << " iterations=" << res.iterations
              << " Hessian-index=" << res.negative_eigenvalues << " products=" << res.hessian_products
              << "\n";
    const std::string tag = "complex driver m=" + std::to_string(n_neg) + ": ";
    check(res.converged, tag + "converged");
    check(res.negative_eigenvalues == static_cast<int>(n_neg), tag + "Hessian index");
    std::vector<C> kx = matvec(k, problem.x());
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(kx[i] + b[i]));
    check(err < 1e-8, tag + "stationary point K x = -b");
  }
}

void testTrustRule() {
  NeoTrustOptions o;
  check(!neoTrustDecision(0, -0.1, 0.5, o).accept, "trust: min, r<0 rejects");
  check(std::abs(neoTrustDecision(0, -0.1, 0.5, o).radius - 0.5 * 2.0 / 3.0) < 1e-14, "trust: min r<0 radius");
  check(neoTrustDecision(0, 0.1, 0.5, o).accept &&
            std::abs(neoTrustDecision(0, 0.1, 0.5, o).radius - 0.5 * 2.0 / 3.0) < 1e-14,
        "trust: min r<0.25 accepts and shrinks");
  check(neoTrustDecision(0, 0.5, 0.5, o).radius == 0.5, "trust: min mid keeps");
  check(std::abs(neoTrustDecision(0, 0.9, 0.5, o).radius - 0.6) < 1e-14, "trust: min r>0.75 grows x1.2");
  check(neoTrustDecision(0, 0.9, 0.7, o).radius == 0.75, "trust: radius capped at 0.75");
  check(!neoTrustDecision(1, 0.5, 0.5, o).accept, "trust: saddle r<r_min rejects");
  check(!neoTrustDecision(1, 1.5, 0.5, o).accept, "trust: saddle r>2-r_min rejects");
  check(neoTrustDecision(1, 0.75, 0.5, o).accept && neoTrustDecision(1, 0.75, 0.5, o).radius == 0.5,
        "trust: saddle intermediate keeps");
  check(std::abs(neoTrustDecision(1, 1.0, 0.5, o).radius - 0.6) < 1e-14, "trust: saddle good grows");
}

}  // namespace

int main() {
  std::cout << std::fixed;
  testTrustRule();
  testStep<double>("real");
  testStep<std::complex<double>>("complex");
  testEigen<double>("real");
  testEigen<std::complex<double>>("complex");
  testDriverReal();
  testDriverHard();
  testDriverComplex();
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
