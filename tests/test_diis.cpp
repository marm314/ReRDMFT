// Unit test of Utils/DIIS.h: Pulay's DIIS with the SCF commutator error F P S - S P F, real and
// complex. Build/run: make test_diis LIBCINT=/path/to/libcint.a
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "DIIS.h"
#include "LinearAlgebra.h"
#include "Matrix.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {

int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; }
}

struct Rng {
  std::uint64_t s = 424242ULL;
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; }
};

inline double cj(double x) { return x; }
inline C cj(const C& x) { return std::conj(x); }

template <typename T> T randomValue(Rng& rng);
template <> double randomValue<double>(Rng& rng) { return rng.next(); }
template <> C randomValue<C>(Rng& rng) { const double a = rng.next(); return C(a, rng.next()); }

template <typename T>
Matrix<T> randomMatrix(std::size_t n, std::size_t m, Rng& rng) {
  Matrix<T> a(n, m);
  for (std::size_t i = 0; i < n * m; ++i) a.data()[i] = randomValue<T>(rng);
  return a;
}

inline Matrix<double> adjoint(const Matrix<double>& a) {
  Matrix<double> t(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) for (std::size_t j = 0; j < a.cols(); ++j) t(j, i) = a(i, j);
  return t;
}
inline Matrix<C> adjoint(const Matrix<C>& a) { return dagger(a); }

template <typename T>
Matrix<T> hermitianPart(const Matrix<T>& a) {
  const Matrix<T> ad = adjoint(a);
  Matrix<T> h(a.rows(), a.cols());
  for (std::size_t i = 0; i < h.rows() * h.cols(); ++i) h.data()[i] = 0.5 * (a.data()[i] + ad.data()[i]);
  return h;
}

template <typename T>
double maxAbs(const Matrix<T>& a) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) m = std::max(m, std::abs(a.data()[i]));
  return m;
}

inline HermitianEigenResult eigh(const Matrix<C>& a) { return diagonalizeHermitian(a); }
inline SymmetricEigenResult eigh(const Matrix<double>& a) { return diagonalizeSymmetric(a); }

// SPD overlap S, Hermitian F, and the closed-shell-like density of the n_occ lowest solutions of
// F C = S C E (Loewdin: X = S^-1/2).
template <typename T>
void scfLikeSystem(std::size_t n, std::size_t n_occ, Rng& rng, Matrix<T>& f, Matrix<T>& s, Matrix<T>& p) {
  const Matrix<T> a = randomMatrix<T>(n, n, rng);
  s = a * adjoint(a);
  for (std::size_t i = 0; i < n; ++i) s(i, i) += T(double(n));
  f = hermitianPart(randomMatrix<T>(n, n, rng));
  const auto es = eigh(s);
  Matrix<T> x(n, n, T{});
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) x(i, j) += es.eigenvectors(i, k) * (1.0 / std::sqrt(es.eigenvalues[k])) * cj(es.eigenvectors(j, k));
  const auto ef = eigh(adjoint(x) * (f * x));
  const Matrix<T> c = x * ef.eigenvectors;
  p = Matrix<T>(n, n, T{});
  for (std::size_t k = 0; k < n_occ; ++k)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) p(i, j) += c(i, k) * cj(c(j, k));
}

template <typename T>
void testCommutator(const std::string& name, Rng& rng) {
  Matrix<T> f, s, p;
  scfLikeSystem<T>(7, 3, rng, f, s, p);
  const auto e0 = scfCommutatorError(f, p, s);
  check(maxAbs(e0) < 1e-12, name + ": commutator vanishes for a density built from the eigenvectors of F C = S C E");
  Matrix<T> f2 = f;
  f2(1, 2) += T(0.1);
  f2(2, 1) += cj(T(0.1));
  check(maxAbs(scfCommutatorError(f2, p, s)) > 1e-3, name + ": commutator is non-zero for an inconsistent F");
  // Anti-Hermitian for Hermitian F, P, S.
  const auto e1 = scfCommutatorError(f2, p, s);
  const auto e1d = adjoint(e1);
  double dev = 0.0;
  for (std::size_t i = 0; i < e1.rows() * e1.cols(); ++i) dev = std::max(dev, std::abs(e1.data()[i] + e1d.data()[i]));
  check(dev < 1e-12, name + ": error F P S - S P F is anti-Hermitian");
}

// Linear fixed point x = M x + c; DIIS with e = g(x) - x, value g(x), next x = sum w_i g(x_i).
template <typename T>
double runDiisFixedPoint(std::size_t dim, std::size_t max_vectors, int iterations, Rng& rng, bool use_diis,
                         double* weight_sum_dev) {
  Matrix<T> m = randomMatrix<T>(dim, dim, rng);
  const double scale = 0.85 / maxAbs(m) / std::sqrt(double(dim));
  for (std::size_t i = 0; i < dim * dim; ++i) m.data()[i] *= scale;
  // make the spectral radius comfortably below 1 but not tiny
  const Matrix<T> c = randomMatrix<T>(dim, 1, rng);
  // exact solution x* = (1 - M)^-1 c via many plain iterations
  Matrix<T> xs(dim, 1, T{});
  for (int k = 0; k < 20000; ++k) xs = m * xs + c;
  Matrix<T> x(dim, 1, T{});
  Diis<T> diis(max_vectors);
  double wdev = 0.0;
  for (int k = 0; k < iterations; ++k) {
    const Matrix<T> gx = m * x + c;
    Matrix<T> err(dim, 1);
    for (std::size_t i = 0; i < dim; ++i) err(i, 0) = gx(i, 0) - x(i, 0);
    if (use_diis) {
      x = diis.extrapolate(err, gx);
      if (!diis.lastWeights().empty()) {
        double sum = 0.0;
        for (double w : diis.lastWeights()) sum += w;
        wdev = std::max(wdev, std::abs(sum - 1.0));
      }
    } else {
      x = gx;
    }
  }
  if (weight_sum_dev) *weight_sum_dev = wdev;
  double dev = 0.0;
  for (std::size_t i = 0; i < dim; ++i) dev = std::max(dev, std::abs(x(i, 0) - xs(i, 0)));
  return dev;
}

}  // namespace

int main() {
  Rng rng;

  std::cout << "SCF commutator error\n";
  testCommutator<double>("real", rng);
  testCommutator<C>("complex", rng);

  std::cout << "DIIS on a linear fixed-point problem\n";
  {
    Rng r1, r2, r3;
    double wdev = 0.0;
    const double plain = runDiisFixedPoint<double>(6, 8, 14, r1, false, nullptr);
    const double diis = runDiisFixedPoint<double>(6, 8, 14, r2, true, &wdev);
    check(diis < 1e-9, "real: DIIS converges in ~dim steps (error " + std::to_string(diis) + ")");
    check(plain > 1e3 * diis, "real: plain iteration is far from converged after the same number of steps");
    check(wdev < 1e-12, "real: weights sum to 1");
    (void)r3;
  }
  {
    // Real weights on a complex problem span a real subspace of dimension 2*dim, so the history
    // must be about twice as long to converge in a finite number of steps.
    Rng r1, r2, r3;
    double wdev = 0.0;
    const double plain = runDiisFixedPoint<C>(6, 8, 16, r1, false, nullptr);
    const double diis8 = runDiisFixedPoint<C>(6, 8, 16, r2, true, &wdev);
    const double diis14 = runDiisFixedPoint<C>(6, 14, 16, r3, true, nullptr);
    check(diis8 < 1e-2 * plain, "complex (real weights, 8 vectors): DIIS beats plain iteration by 100x");
    check(diis14 < 1e-9, "complex (real weights, 14 vectors): converges (error " + std::to_string(diis14) + ")");
    check(wdev < 1e-12, "complex: weights sum to 1");
  }
  {
    // Shorter history (5, the usual choice) still converges the same linear problem.
    Rng r;
    const double diis = runDiisFixedPoint<double>(6, 5, 40, r, true, nullptr);
    check(diis < 1e-9, "real, 5 vectors: converges (error " + std::to_string(diis) + ")");
  }

  std::cout << "Behaviour at the edges\n";
  {
    Diis<double> d(5);
    Matrix<double> v(2, 2, 3.0), e(2, 2, 0.5);
    const auto first = d.extrapolate(e, v);
    check(maxAbs(first) == 3.0 && d.lastWeights().empty(), "first call: nothing to extrapolate, input returned");
    // identical errors -> singular system -> falls back to the input
    Matrix<double> v2(2, 2, 4.0);
    const auto second = d.extrapolate(e, v2);
    check(second(0, 0) == 4.0, "identical errors (singular B): input returned, no crash");
    Matrix<double> ez(2, 2, 0.0);
    Diis<double> d2(5);
    d2.extrapolate(ez, v);
    check(d2.extrapolate(ez, v2)(0, 0) == 4.0, "zero errors: input returned");
    Diis<double> off(1);
    check(off.extrapolate(e, v)(0, 0) == 3.0 && off.size() == 0, "max_vectors < 2 disables DIIS");
    Diis<double> hist(3);
    for (int i = 0; i < 6; ++i) hist.extrapolate(Matrix<double>(2, 2, 1.0 + 0.1 * i), v);
    check(hist.size() == 3, "history is capped at max_vectors");
    hist.clear();
    check(hist.size() == 0, "clear empties the history");
    bool threw = false;
    try { hist.extrapolate(e, v); hist.extrapolate(Matrix<double>(3, 3, 1.0), v); } catch (const std::exception&) { threw = true; }
    check(threw, "shape change refused");
  }
  {
    // Hermiticity: real weights keep an extrapolated Hermitian Fock Hermitian.
    Rng r;
    Diis<C> d(5);
    Matrix<C> last;
    for (int i = 0; i < 4; ++i) {
      Matrix<C> f = hermitianPart(randomMatrix<C>(5, 5, r));
      last = d.extrapolate(randomMatrix<C>(5, 5, r), f);
    }
    const auto ld = adjoint(last);
    double dev = 0.0;
    for (std::size_t i = 0; i < 25; ++i) dev = std::max(dev, std::abs(last.data()[i] - ld.data()[i]));
    check(dev < 1e-13, "extrapolated Hermitian matrices stay Hermitian");
  }

  std::cout << (g_failures == 0 ? "ALL PASSED" : "FAILURES") << " (" << g_checks - g_failures << "/" << g_checks << ")\n";
  return g_failures == 0 ? 0 : 1;
}
