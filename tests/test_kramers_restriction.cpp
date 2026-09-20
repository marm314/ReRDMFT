// Standalone unit test of Utils/KramersRestriction.h/.cpp.
// Build/run: make test_kramers LIBCINT=/path/to/libcint.a
//
//   * counts (pairs, fixed pairs, orbits, sizes),
//   * the physical statement: kappa built from an expanded reduced vector
//     obeys kappa(P p,P q) = s_p s_q conj kappa(p,q), and so does
//     U = exp(-kappa) (a generic full vector does NOT -- control),
//   * Q orthonormal, Q Q^T = (1+R)/2, R an involution, projection,
//   * a time-reversal-symmetric Hessian commutes with R and its reduced
//     matrix Q^T H Q has a spectrum contained in the full one; the reduced
//     Hessian-vector product equals Q^T H Q u,
//   * representative parametrization (J^T adjoint, symmetric expansion),
//   * KramersNeoProblem: neoOptimize in the reduced space reaches the
//     stationary point of a quadratic model with EXACTLY symmetric steps
//     (minimum and a saddle of order 1 of the KR sector),
//   * KramersAdamProblem: ADAM on a time-reversal-even orbital toy stays
//     time-reversal symmetric and reaches the minimum,
//   * input validation.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "KramersRestriction.h"
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

struct Rng {
  std::uint64_t s = 1234567891011ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

double sgn(std::size_t i) { return i % 2 == 0 ? 1.0 : -1.0; }
std::size_t part(std::size_t i) { return i ^ std::size_t{1}; }

std::vector<AdamPair> lowerPairs(std::size_t n) { return adamPairIndices(n, false); }

// kappa_pq = t + iy, kappa_qp = -conj(kappa_pq) from a joint [t;y] vector.
Matrix<C> kappaFromJoint(std::size_t n, const std::vector<AdamPair>& pairs, const std::vector<double>& x) {
  Matrix<C> k(n, n, C{});
  const std::size_t m = pairs.size();
  for (std::size_t i = 0; i < m; ++i) {
    const auto [p, q] = pairs[i];
    const C z(x[i], x[m + i]);
    k(p, q) = z;
    k(q, p) = -std::conj(z);
  }
  return k;
}

// max |M(P p,P q) - s_p s_q conj M(p,q)|.
double timeReversalDeviation(const Matrix<C>& m) {
  double dev = 0.0;
  for (std::size_t p = 0; p < m.rows(); ++p) {
    for (std::size_t q = 0; q < m.cols(); ++q) {
      dev = std::max(dev, std::abs(m(part(p), part(q)) - sgn(p) * sgn(q) * std::conj(m(p, q))));
    }
  }
  return dev;
}

std::vector<double> randomVector(std::size_t n, Rng& rng) {
  std::vector<double> v(n);
  for (double& x : v) x = rng.next();
  return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

double maxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

std::vector<double> matvec(const Matrix<double>& h, const std::vector<double>& v) {
  std::vector<double> out(h.rows(), 0.0);
  for (std::size_t i = 0; i < h.rows(); ++i) for (std::size_t j = 0; j < h.cols(); ++j) out[i] += h(i, j) * v[j];
  return out;
}

// R H R^T for a full joint-space matrix.
Matrix<double> timeReverseMatrix(const KramersRestriction& kr, const Matrix<double>& h) {
  const std::size_t n = h.rows();
  Matrix<double> tmp(n, n, 0.0), out(n, n, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    std::vector<double> col(n);
    for (std::size_t i = 0; i < n; ++i) col[i] = h(i, c);
    const auto r = kr.timeReversed(col);
    for (std::size_t i = 0; i < n; ++i) tmp(i, c) = r[i];
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> row(n);
    for (std::size_t c = 0; c < n; ++c) row[c] = tmp(i, c);
    const auto r = kr.timeReversed(row);  // (R applied to the row) = columns transformed by R^T = R
    for (std::size_t c = 0; c < n; ++c) out(i, c) = r[c];
  }
  return out;
}

Matrix<double> symmetricTrHessian(const KramersRestriction& kr, Rng& rng) {
  const std::size_t n = kr.fullSize();
  Matrix<double> a(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) a(i, j) = a(j, i) = rng.next();
  const Matrix<double> ra = timeReverseMatrix(kr, a);
  Matrix<double> h(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) h(i, j) = 0.5 * (a(i, j) + ra(i, j));
  return h;
}

void testStructure() {
  const std::size_t n = 6;
  const auto pairs = lowerPairs(n);
  KramersRestriction kr(n, pairs);
  check(kr.nPairs() == 15 && kr.nFixedPairs() == 3 && kr.nOrbits() == 9, "counts: 15 pairs, 3 within-Kramers-pair, 9 orbits");
  check(kr.fullSize() == 30 && kr.reducedSize() == 18, "sizes: 2m = 30 full, 18 reduced (half + within-pair)");

  Rng rng;
  // Physical relation: kappa from an expanded reduced vector is time-reversal even.
  const auto u = randomVector(kr.reducedSize(), rng);
  for (int rep = 0; rep < 2; ++rep) {
    const auto x = rep == 0 ? kr.expand(u) : kr.expandRepresentative(u);
    const Matrix<C> kappa = kappaFromJoint(n, pairs, x);
    check(timeReversalDeviation(kappa) < 1e-14,
          std::string("kappa(P p,P q) = s s conj kappa(p,q) for ") + (rep == 0 ? "expand" : "expandRepresentative"));
    const Matrix<C> uu = spinorRotationMatrix(kappa);
    check(timeReversalDeviation(uu) < 1e-12, "U = exp(-kappa) also commutes with time reversal");
    check(kr.asymmetry(x) < 1e-15, "expanded vector is R-symmetric");
  }
  // Control: a generic full vector breaks it.
  const auto generic = randomVector(kr.fullSize(), rng);
  check(timeReversalDeviation(kappaFromJoint(n, pairs, generic)) > 1e-2, "control: generic vector violates the relation");
  check(kr.asymmetry(generic) > 1e-2, "control: generic vector has R-asymmetry");

  // Linear algebra of Q and R.
  bool roundtrip = true;
  for (std::size_t j = 0; j < kr.reducedSize(); ++j) {
    std::vector<double> e(kr.reducedSize(), 0.0);
    e[j] = 1.0;
    roundtrip &= maxDiff(kr.contract(kr.expand(e)), e) < 1e-14;
    roundtrip &= std::abs(dot(kr.expand(e), kr.expand(e)) - 1.0) < 1e-14;
  }
  check(roundtrip, "Q^T Q = 1 (contract(expand(e_j)) = e_j, |Q e_j| = 1)");
  const auto rr = kr.timeReversed(kr.timeReversed(generic));
  check(maxDiff(rr, generic) < 1e-14, "R is an involution");
  check(maxDiff(kr.project(generic), kr.expand(kr.contract(generic))) < 1e-14, "Q Q^T x = (x + R x)/2");
  check(maxDiff(kr.project(kr.project(generic)), kr.project(generic)) < 1e-14, "projection is idempotent");
  check(std::abs(dot(kr.timeReversed(generic), kr.timeReversed(generic)) - dot(generic, generic)) < 1e-13,
        "R is orthogonal");

  // Representative parametrization: J^T is the adjoint of J; expansion symmetric.
  const auto v = randomVector(kr.fullSize(), rng);
  check(std::abs(dot(v, kr.expandRepresentative(u)) - dot(kr.contractRepresentative(v), u)) < 1e-13,
        "representative: contractRepresentative = J^T (adjoint of expandRepresentative)");

  // Complex per-pair form.
  std::vector<C> z(kr.nPairs());
  for (C& c : z) c = C(rng.next(), rng.next());
  const auto zz = KramersRestriction::complexFromJoint(KramersRestriction::jointFromComplex(z));
  double zerr = 0.0;
  for (std::size_t i = 0; i < z.size(); ++i) zerr = std::max(zerr, std::abs(z[i] - zz[i]));
  check(zerr < 1e-15, "complex <-> joint round trip");
  check(kramersOccupationDeviation({0.9, 0.9, 0.3, 0.30001, 1.0, 1.0}) > 9e-6 &&
            kramersOccupationDeviation({0.9, 0.9, 0.3, 0.3}) == 0.0, "occupation deviation over Kramers pairs");
}

void testHessian() {
  const std::size_t n = 6;
  KramersRestriction kr(n, lowerPairs(n));
  Rng rng;
  const Matrix<double> h = symmetricTrHessian(kr, rng);
  const Matrix<double> rhr = timeReverseMatrix(kr, h);
  double comm = 0.0;
  for (std::size_t i = 0; i < h.rows(); ++i) for (std::size_t j = 0; j < h.cols(); ++j) comm = std::max(comm, std::abs(h(i, j) - rhr(i, j)));
  check(comm < 1e-14, "symmetrized Hessian: R H R^T = H");

  const Matrix<double> hr = kr.contractMatrix(h);
  const auto full_ev = diagonalizeSymmetric(h).eigenvalues;
  const auto red_ev = diagonalizeSymmetric(hr).eigenvalues;
  bool subset = true;
  for (double e : red_ev) {
    double best = 1e9;
    for (double f : full_ev) best = std::min(best, std::abs(e - f));
    subset &= best < 1e-10;
  }
  check(subset, "reduced Hessian spectrum is contained in the full spectrum");

  const auto u = randomVector(kr.reducedSize(), rng);
  check(maxDiff(kr.contract(matvec(h, kr.expand(u))), matvec(hr, u)) < 1e-13, "reduced H*v = Q^T H Q u");
  // H maps the symmetric sector into itself.
  check(kr.asymmetry(matvec(h, kr.expand(u))) < 1e-13, "H Q u is again R-symmetric");
  // Diagonal preconditioner shape.
  std::vector<double> d(kr.fullSize());
  for (std::size_t i = 0; i < d.size(); ++i) d[i] = h(i, i);
  const auto dr = kr.contractDiagonal(d);
  check(dr.size() == kr.reducedSize(), "contractDiagonal size");
}

// Quadratic model f(x) = 1/2 x^T H x + g^T x over the full joint vector.
class Quadratic : public NeoProblem<double> {
 public:
  Quadratic(Matrix<double> h, std::vector<double> g) : h_(std::move(h)), g_(std::move(g)), x_(g_.size(), 0.0) {}
  std::size_t dimension() const override { return g_.size(); }
  double value(const std::vector<double>& x) const {
    return 0.5 * dot(x, matvec(h_, x)) + dot(g_, x);
  }
  double energy() override { return value(x_); }
  std::vector<double> gradient() override {
    auto g = matvec(h_, x_);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] += g_[i];
    return g;
  }
  std::vector<double> hessianVector(const std::vector<double>& v) override { return matvec(h_, v); }
  std::vector<double> hessianDiagonal() override {
    std::vector<double> d(g_.size());
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = h_(i, i);
    return d;
  }
  double trialEnergy(const std::vector<double>& d) override {
    auto y = x_;
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += d[i];
    return value(y);
  }
  void accept(const std::vector<double>& d) override {
    for (std::size_t i = 0; i < x_.size(); ++i) x_[i] += d[i];
  }
  const std::vector<double>& x() const { return x_; }

 private:
  Matrix<double> h_;
  std::vector<double> g_, x_;
};

void testNeoAdapter() {
  const std::size_t n = 6;
  KramersRestriction kr(n, lowerPairs(n));
  Rng rng;
  Matrix<double> h = symmetricTrHessian(kr, rng);
  const auto red_ev = diagonalizeSymmetric(kr.contractMatrix(h)).eigenvalues;
  for (std::size_t order = 0; order <= 1; ++order) {
    // Shift so the KR-sector Hessian has exactly `order` negative eigenvalues.
    const double shift = order == 0 ? -(red_ev[0] - 1.0) : -(red_ev[0] + red_ev[1]) / 2.0;
    Matrix<double> hs = h;
    for (std::size_t i = 0; i < hs.rows(); ++i) hs(i, i) += shift;
    const auto g = kr.project(randomVector(kr.fullSize(), rng));
    Quadratic full(hs, g);
    KramersNeoProblem reduced(full, kr);
    NeoOptions opt;
    opt.step.target_order = order;
    opt.gradient_tolerance = 1e-10;
    opt.verify_index = true;
    const NeoResult res = neoOptimize<double>(reduced, opt);
    const std::string tag = "NEO in the KR sector, order " + std::to_string(order) + ": ";
    std::cout << "NEO adapter (order " << order << "): converged=" << res.converged << " iterations=" << res.iterations
              << " reduced dimension " << reduced.dimension() << " of " << full.dimension()
              << " Hessian index " << res.negative_eigenvalues << " products " << res.hessian_products << "\n";
    check(res.converged, tag + "converged");
    check(res.negative_eigenvalues == static_cast<int>(order), tag + "KR-sector Hessian index");
    check(kr.asymmetry(full.x()) < 1e-12, tag + "accumulated step is exactly time-reversal symmetric");
    const auto hx = matvec(hs, full.x());
    double resid = 0.0;
    for (std::size_t i = 0; i < hx.size(); ++i) resid = std::max(resid, std::abs(hx[i] + g[i]));
    check(resid < 1e-8, tag + "stationary point H x = -g of the full model");
  }
}

// Time-reversal-even orbital toy for ADAM: E = sum_i d_i (C^dag A C)_ii
// with A(P p,P q) = s s conj A(p,q) and d equal within Kramers pairs.
class OrbitalToy : public AdamProblem<C> {
 public:
  OrbitalToy(Matrix<C> a, std::vector<double> d)
      : a_(std::move(a)), d_(std::move(d)), n_(a_.rows()), pairs_(lowerPairs(n_)), trial_(n_, n_, C{}), best_(n_, n_, C{}) {
    for (std::size_t i = 0; i < n_; ++i) trial_(i, i) = 1.0;
    best_ = trial_;
  }
  std::size_t dimension() const override { return pairs_.size(); }
  double energyOf(const Matrix<C>& c) const {
    const Matrix<C> ac = a_ * c;
    double e = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      C m{};
      for (std::size_t k = 0; k < n_; ++k) m += std::conj(c(k, i)) * ac(k, i);
      e += d_[i] * m.real();
    }
    return e;
  }
  double energy() override { return energyOf(trial_); }
  std::vector<C> gradient() override {
    const double h = 1e-5;
    std::vector<C> g(pairs_.size());
    for (std::size_t i = 0; i < pairs_.size(); ++i) {
      const auto [p, q] = pairs_[i];
      auto e_at = [&](double t, double y) {
        Matrix<C> k(n_, n_, C{});
        k(p, q) = C(t, y);
        k(q, p) = -std::conj(C(t, y));
        return energyOf(trial_ * spinorRotationMatrix(k));
      };
      g[i] = C((e_at(h, 0) - e_at(-h, 0)) / (2 * h), (e_at(0, h) - e_at(0, -h)) / (2 * h));
    }
    return g;
  }
  void rotate(const std::vector<C>& step) override {
    trial_ = trial_ * spinorRotationMatrix(adamKappaMatrix(n_, pairs_, step));
  }
  void saveBest() override { best_ = trial_; }
  void restoreBest() override { trial_ = best_; }
  const Matrix<C>& best() const { return best_; }

 private:
  Matrix<C> a_;
  std::vector<double> d_;
  std::size_t n_;
  std::vector<AdamPair> pairs_;
  Matrix<C> trial_, best_;
};

void testAdamAdapter() {
  const std::size_t n = 6;
  Rng rng;
  Matrix<C> b(n, n, C{});
  for (std::size_t i = 0; i < n; ++i) {
    b(i, i) = C(1.0 + 0.7 * i + 0.3 * rng.next(), 0.0);
    for (std::size_t j = i + 1; j < n; ++j) {
      b(i, j) = C(0.4 * rng.next(), 0.4 * rng.next());
      b(j, i) = std::conj(b(i, j));
    }
  }
  Matrix<C> a(n, n, C{});  // T(B)(p,q) = s_p s_q conj B(P p,P q); A = (B + T(B))/2
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) a(p, q) = 0.5 * (b(p, q) + sgn(p) * sgn(q) * std::conj(b(part(p), part(q))));
  check(timeReversalDeviation(a) < 1e-14, "toy operator A is time-reversal even");
  const std::vector<double> d = {1.0, 1.0, 0.5, 0.5, 0.1, 0.1};
  OrbitalToy full(a, d);
  KramersRestriction kr(n, lowerPairs(n));
  KramersAdamProblem reduced(full, kr);

  AdamOptions o;
  o.learning_rate = 0.05;
  o.gradient_tolerance = 1e-4;
  o.energy_tolerance = 1e-13;
  AdamOptimizer<C> opt(o);
  const double e_start = full.energy();
  bool converged = false;
  int runs = 0;
  for (; runs < 400 && !converged; ++runs) converged = opt.run(reduced).gradient_converged;
  Matrix<C> ac = a;
  const auto ev = diagonalizeHermitian(ac).eigenvalues;
  double e_min = 0.0;
  for (std::size_t i = 0; i < n; ++i) e_min += d[i] * ev[i];  // d descending, eigenvalues ascending
  const double e_end = full.energyOf(full.best());
  std::cout << "ADAM adapter: " << runs << " runs, reduced dimension " << reduced.dimension() << " of "
            << full.dimension() << ", E " << e_start << " -> " << e_end << " (exact minimum " << e_min << ")\n";
  check(converged, "ADAM in the KR sector: gradient converged");
  check(std::abs(e_end - e_min) < 1e-6, "ADAM in the KR sector reaches the exact minimum");
  check(timeReversalDeviation(full.best()) < 1e-12, "rotated orbitals C stay time-reversal symmetric (C(P p,P q) = s s conj C(p,q))");
}

void testValidation() {
  bool threw = false;
  try { KramersRestriction(5, {{1, 0}}); } catch (const std::exception&) { threw = true; }
  check(threw, "odd number of spinors rejected");
  threw = false;
  try { KramersRestriction(4, {{2, 0}}); } catch (const std::exception&) { threw = true; }
  check(threw, "pair list not closed under Kramers partners rejected");
  threw = false;
  try { KramersRestriction(4, {{0, 1}}); } catch (const std::exception&) { threw = true; }
  check(threw, "pair with p < q rejected");
  threw = false;
  try { KramersRestriction(4, {{1, 0}, {1, 0}}); } catch (const std::exception&) { threw = true; }
  check(threw, "repeated pair rejected");
  KramersRestriction kr(4, lowerPairs(4));
  threw = false;
  try { kr.expand(std::vector<double>(3)); } catch (const std::exception&) { threw = true; }
  check(threw, "wrong reduced size rejected");
}

}  // namespace

int main() {
  testStructure();
  testHessian();
  testNeoAdapter();
  testAdamAdapter();
  testValidation();
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
