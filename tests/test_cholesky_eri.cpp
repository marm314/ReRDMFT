// Unit test of Utils/CholeskyEri.h, and of the RDMFT model-layer functions with a
// CholeskyEri argument against the same functions with the dense tensor.
// Build/run: make test_cholesky_eri LIBCINT=/path/to/libcint.a
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "CholeskyEri.h"
#include "Cholesky_Decomposition.h"
#include "HartreeExchangeGradient.h"
#include "JK_only.h"
#include "JkOnlyFock.h"
#include "OccupationEnergy.h"
#include "Orb_subspaces.h"
#include "PNOFs.h"
#include "PnofFock.h"
#include "Matrix.h"
#include "SpinorRotation.h"
#include "Tensor4.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {
int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; } }
struct Rng { std::uint64_t s = 20260920ULL; double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; } };

template <typename T> T randomScalar(Rng& r);
template <> double randomScalar<double>(Rng& r) { return r.next(); }
template <> C randomScalar<C>(Rng& r) { const double a = r.next(); const double b = r.next(); return C(a, b); }
double absOf(double x) { return std::abs(x); }
double absOf(const C& x) { return std::abs(x); }
double conjOf(double x) { return x; }
C conjOf(const C& x) { return std::conj(x); }

// Low-rank physical tensor: eri(A,B,C,D) = sum_L V_L(C,A) conj(V_L(B,D)) with `rank` random vectors.
template <typename T>
Tensor4<T> makeTensor(std::size_t n, std::size_t rank, Rng& rng) {
  std::vector<Matrix<T>> vec(rank, Matrix<T>(n, n, T{}));
  for (auto& v : vec) for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) v(i, j) = 0.3 * randomScalar<T>(rng);
  Tensor4<T> t(n, n, n, n, T{});
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    T s{}; for (const auto& v : vec) s += v(c, a) * conjOf(v(b, d)); t(a, b, c, d) = s;
  }
  return t;
}

// Reference dense rotation, physics notation, bra legs (a,b) conjugated:
// eri'(pqrs) = sum conj(U_ap) conj(U_bq) U_cr U_ds eri(abcd), four one-leg transforms.
template <typename T>
Tensor4<T> rotateDense(const Tensor4<T>& e, const Matrix<T>& u) {
  const std::size_t n = e.dim0();
  Tensor4<T> t1(n, n, n, n, T{}), t2(n, n, n, n, T{}), t3(n, n, n, n, T{}), t4(n, n, n, n, T{});
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t s = 0; s < n; ++s) { T x{}; for (std::size_t d = 0; d < n; ++d) x += e(a, b, c, d) * u(d, s); t1(a, b, c, s) = x; }
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) { T x{}; for (std::size_t c = 0; c < n; ++c) x += t1(a, b, c, s) * u(c, r); t2(a, b, r, s) = x; }
  for (std::size_t a = 0; a < n; ++a) for (std::size_t q = 0; q < n; ++q) for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) { T x{}; for (std::size_t b = 0; b < n; ++b) x += t2(a, b, r, s) * conjOf(u(b, q)); t3(a, q, r, s) = x; }
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) { T x{}; for (std::size_t a = 0; a < n; ++a) x += t3(a, q, r, s) * conjOf(u(a, p)); t4(p, q, r, s) = x; }
  return t4;
}

template <typename T>
void testType(const std::string& label) {
  Rng rng;
  const std::size_t n = 8, rank = 6;
  const Tensor4<T> dense = makeTensor<T>(n, rank, rng);
  const CholeskyEri<T> ch = CholeskyEri<T>::fromDense(dense, 1e-12);
  std::cout << label << ": n=" << n << ", true rank " << rank << ", Cholesky vectors " << ch.nVectors() << " (dense n^2 = " << n * n << ")\n";
  check(ch.nVectors() <= rank + 1, label + ": Cholesky rank of the Coulomb grouping equals the true low rank");
  double err = 0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d)
    err = std::max(err, absOf(ch(a, b, c, d) - dense(a, b, c, d)));
  check(err < 1e-10, label + ": elements reproduce the dense tensor, max error " + std::to_string(err));

  // Rotation: V' = U^T V conj(U) must equal the dense rotation (physics notation, bra legs conjugated).
  Matrix<T> kappa(n, n, T{});
  for (std::size_t p = 1; p < n; ++p) for (std::size_t q = 0; q < p; ++q) {
    const T z = 0.2 * randomScalar<T>(rng);
    kappa(p, q) = z; kappa(q, p) = -conjOf(z);
  }
  const Matrix<T> u = spinorRotationMatrix(kappa);
  const CholeskyEri<T> rot = ch.rotated(u);
  const Tensor4<T> dense_rot_eri = rotateDense(dense, u);
  double rerr = 0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d)
    rerr = std::max(rerr, absOf(rot(a, b, c, d) - dense_rot_eri(a, b, c, d)));
  check(rerr < 1e-9, label + ": rotated vectors reproduce the dense rotation, max error " + std::to_string(rerr));
  const Tensor4<T> back = rot.toDense();
  double berr = 0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d)
    berr = std::max(berr, absOf(back(a, b, c, d) - rot(a, b, c, d)));
  check(berr < 1e-15, label + ": toDense equals element access");
  // Hermiticity <ab|cd> = conj <dc|ba> is preserved by the rotation.
  double herr = 0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d)
    herr = std::max(herr, absOf(rot(a, b, c, d) - conjOf(rot(d, c, b, a))));
  check(herr < 1e-12, label + ": rotated tensor keeps <ab|cd> = conj <dc|ba>");
}

}  // namespace

// The RDMFT model layer with a CholeskyEri argument must give the SAME numbers as with the dense
// tensor (the functions only read elements).
template <typename T, typename M>
double matDiff(const Matrix<T>& a, const Matrix<M>& b) { double m = 0; for (std::size_t i = 0; i < a.rows(); ++i) for (std::size_t j = 0; j < a.cols(); ++j) m = std::max(m, std::abs(std::complex<double>(a(i, j)) - std::complex<double>(b(i, j)))); return m; }
double vecDiff(const std::vector<double>& a, const std::vector<double>& b) { double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

template <typename T>
void testModelLayer(const std::string& label) {
  Rng rng;
  const std::size_t n = 12;
  const Tensor4<T> dense = makeTensor<T>(n, 10, rng);
  const CholeskyEri<T> ch = CholeskyEri<T>::fromDense(dense, 1e-12);
  Matrix<T> h(n, n, T{});
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) { h(i, j) = randomScalar<T>(rng); h(j, i) = conjOf(h(i, j)); }
  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (0.5 + rng.next());
  const double tol = 1e-9;
  // JK_only
  {
    const auto f = JkFunctional::kMbb;
    const double e1 = jkFunctionalEnergy(h, dense, occ, f, 6), e2 = jkFunctionalEnergy(h, ch, occ, f, 6);
    check(std::abs(e1 - e2) < tol, label + ": jkFunctionalEnergy dense vs Cholesky");
    check(vecDiff(jkFunctionalGradient(h, dense, occ, f, 6), jkFunctionalGradient(h, ch, occ, f, 6)) < tol, label + ": jkFunctionalGradient");
    check(matDiff(jkFunctionalHessian(h, dense, occ, f, 6), jkFunctionalHessian(h, ch, occ, f, 6)) < tol, label + ": jkFunctionalHessian");
    const auto hc = jkHartreeCoupling(f, occ, 6), xc = jkExchangeCoupling(f, occ, 6);
    check(matDiff(jkOnlyOrbitalGradient(h, dense, occ, hc, xc), jkOnlyOrbitalGradient(h, ch, occ, hc, xc)) < tol, label + ": jkOnlyOrbitalGradient");
    std::vector<std::size_t> pair_of(n);
    for (std::size_t p = 0; p < n; ++p) pair_of[p] = p ^ 1;
    Matrix<double> l1(n, n, 0.0), l2(n, n, 0.0);
    for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) { l1(p, q) = 0.01 * (1 + p + 2 * q); l2(p, q) = -0.02 * (1 + q); }
    const auto fd = hartreeExchangeFockMatrix(h, dense, occ, hc, xc, pair_of, l1, l2), fc = hartreeExchangeFockMatrix(h, ch, occ, hc, xc, pair_of, l1, l2);
    check(matDiff(fd, fc) < tol, label + ": hartreeExchangeFockMatrix (with pair terms)");
    check(std::abs(hartreeExchangeEnergy(h, dense, occ, hc, xc, pair_of, l1, l2) - hartreeExchangeEnergy(h, ch, occ, hc, xc, pair_of, l1, l2)) < tol, label + ": hartreeExchangeEnergy (with pair terms)");
  }
  // PNOF (real only: J/K/L must be real)
  if constexpr (std::is_same_v<T, double>) {
    std::vector<std::size_t> pair_of(n);
    for (std::size_t p = 0; p < n; ++p) pair_of[p] = p ^ 1;
    const auto table = buildOrbitalSubspaces(pair_of, n, 4.0, 2, 2);
    const auto geminals = buildPnofGeminals(table);
    std::vector<double> o(n, 0.0);
    const std::size_t n_core = table.frozen_occupied.size();
    for (std::size_t a = 0; a < n_core; ++a) o[geminals[a].i] = o[geminals[a].ibar] = 1.0;
    for (std::size_t a = n_core; a < geminals.size(); ++a) o[geminals[a].i] = o[geminals[a].ibar] = geminals[a].is_principal ? 0.93 : 0.07;
    for (auto fn : {PnofFunctional::kPnof5, PnofFunctional::kPnof7, PnofFunctional::kGnof}) {
      for (bool rel : {false, true}) {
        const auto t = buildPnofTwoRdm(fn, geminals, o, rel);
        check(std::abs(pnofElectronicEnergy(fn, h, dense, o, geminals, t, rel) - pnofElectronicEnergy(fn, h, ch, o, geminals, t, rel)) < tol, label + ": pnofElectronicEnergy");
        check(vecDiff(pnofOccupationGradient(fn, h, dense, o, geminals, rel), pnofOccupationGradient(fn, h, ch, o, geminals, rel)) < tol, label + ": pnofOccupationGradient");
        check(matDiff(pnofFockMatrix(fn, h, dense, geminals, o, rel), pnofFockMatrix(fn, h, ch, geminals, o, rel)) < tol, label + ": pnofFockMatrix");
      }
    }
    check(matDiff(pnofOccupationHessian(PnofFunctional::kPnof7, h, dense, o, geminals, false), pnofOccupationHessian(PnofFunctional::kPnof7, h, ch, o, geminals, false)) < tol, label + ": pnofOccupationHessian");
    check(matDiff(pnofOccupationHessianFD(PnofFunctional::kGnof, h, dense, o, geminals, false, 1e-6), pnofOccupationHessianFD(PnofFunctional::kGnof, h, ch, o, geminals, false, 1e-6)) < 1e-7, label + ": pnofOccupationHessianFD");
  }
}

// choleskyDecomposeEriChecked: the bra-ket grouping eri(A,B,C,D) = sum_L V_L(A,B) conj(V_L(C,D)) it
// verifies. A well-conditioned low-rank tensor passes on the first (fastest) batch without a retry;
// a noisy rank-deficient one (tiny non-PSD roundoff, like the CO/cc-pVDZ X2C MO integrals) must come
// out within the tolerance whichever batch it finally needed; a genuinely non-PSD tensor throws.
template <typename T>
void testChecked(const std::string& label) {
  Rng rng;
  const std::size_t n = 6, rank = 9;
  std::vector<Matrix<T>> vec(rank, Matrix<T>(n, n, T{}));
  for (auto& v : vec) for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) v(i, j) = 0.3 * randomScalar<T>(rng);
  Tensor4<T> t(n, n, n, n, T{});
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    T x{}; for (const auto& v : vec) x += v(a, b) * conjOf(v(c, d)); t(a, b, c, d) = x;
  }
  CholeskyCheckReport rep;
  const auto vs = choleskyDecomposeEriChecked(t, 1e-10, &rep);
  check(!vs.empty() && vs.size() <= rank + 1, label + ": checked decomposition finds the true low rank");
  check(rep.max_error <= rep.tolerance && !rep.retried && rep.batch_used == 64, label + ": well-conditioned tensor passes on the default batch, error " + std::to_string(rep.max_error));
  // Roundoff-level Hermitian noise on top of the rank-deficient tensor (1e-13, well below any threshold).
  Tensor4<T> noisy = t;
  Rng nrng; nrng.s = 7ULL;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    if (a * n + b > c * n + d) continue;
    const T e = 1e-13 * randomScalar<T>(nrng);
    noisy(a, b, c, d) += e; if (a * n + b != c * n + d) noisy(c, d, a, b) += conjOf(e);
  }
  for (const double thr : {1e-8, 1e-10, 1e-12}) {
    CholeskyCheckReport r2;
    bool threw = false;
    try { choleskyDecomposeEriChecked(noisy, thr, &r2); } catch (const std::exception&) { threw = true; }
    check(!threw, label + ": noisy rank-deficient tensor decomposes within tolerance at threshold " + std::to_string(thr));
  }
  // Not PSD: a large negative diagonal must be rejected (throws), not returned as garbage.
  Tensor4<T> bad = t;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) bad(a, b, a, b) = -1.0;
  bool threw = false;
  try { choleskyDecomposeEriChecked(bad, 1e-10); } catch (const std::exception&) { threw = true; }
  check(threw, label + ": a non-PSD tensor is rejected");
}

int main() {
  try { testChecked<double>("checked (real)"); } catch (const std::exception& e) { std::cout << "checked real threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  try { testChecked<C>("checked (complex)"); } catch (const std::exception& e) { std::cout << "checked complex threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  try { testType<double>("real"); } catch (const std::exception& e) { std::cout << "real threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  try { testType<C>("complex"); } catch (const std::exception& e) { std::cout << "complex threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  try { testModelLayer<double>("model layer (real)"); } catch (const std::exception& e) { std::cout << "model layer real threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  try { testModelLayer<C>("model layer (complex)"); } catch (const std::exception& e) { std::cout << "model layer complex threw: " << e.what() << "\n"; ++g_failures; ++g_checks; }
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
