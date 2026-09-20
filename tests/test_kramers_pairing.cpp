// Unit test of Utils/KramersPairing.h: exact Kramers re-pairing of
// near-degenerate clusters of two-component spinors.
// Build/run: make test_kramers_pairing LIBCINT=/path/to/libcint.a
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "KramersPairing.h"
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
  std::uint64_t s = 987654321ULL;
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; }
};

std::size_t g_nl = 6;

C inner(const Matrix<double>& s, const std::vector<C>& a, const std::vector<C>& b) {
  C sum{};
  for (std::size_t i = 0; i < g_nl; ++i)
    for (std::size_t j = 0; j < g_nl; ++j)
      sum += s(i, j) * (std::conj(a[i]) * b[j] + std::conj(a[g_nl + i]) * b[g_nl + j]);
  return sum;
}
std::vector<C> theta(const std::vector<C>& v) {
  std::vector<C> out(2 * g_nl);
  for (std::size_t i = 0; i < g_nl; ++i) { out[g_nl + i] = std::conj(v[i]); out[i] = -std::conj(v[g_nl + i]); }
  return out;
}
std::vector<C> column(const Matrix<C>& m, std::size_t j) { std::vector<C> c(m.rows()); for (std::size_t i = 0; i < m.rows(); ++i) c[i] = m(i, j); return c; }

// max over pairs of || Theta c_2k - c_2k+1 ||_S
double thetaDeviation(const Matrix<double>& s, const Matrix<C>& c) {
  double dev = 0.0;
  for (std::size_t k = 0; k + 1 < c.cols(); k += 2) {
    const auto t = theta(column(c, k));
    auto o = column(c, k + 1);
    for (std::size_t i = 0; i < t.size(); ++i) o[i] -= t[i];
    dev = std::max(dev, std::sqrt(std::real(inner(s, o, o))));
  }
  return dev;
}
double orthonormalityError(const Matrix<double>& s, const Matrix<C>& c) {
  double err = 0.0;
  for (std::size_t a = 0; a < c.cols(); ++a) for (std::size_t b = 0; b < c.cols(); ++b)
    err = std::max(err, std::abs(inner(s, column(c, a), column(c, b)) - (a == b ? 1.0 : 0.0)));
  return err;
}

}  // namespace

int main() {
  Rng rng;
  const std::size_t nl = g_nl, n2 = 2 * nl;
  // S: random symmetric positive definite.
  Matrix<double> s(nl, nl, 0.0);
  {
    Matrix<double> a(nl, nl, 0.0);
    for (std::size_t i = 0; i < nl; ++i) for (std::size_t j = 0; j < nl; ++j) a(i, j) = rng.next();
    for (std::size_t i = 0; i < nl; ++i) for (std::size_t j = 0; j < nl; ++j) { double v = 0; for (std::size_t k = 0; k < nl; ++k) v += a(i, k) * a(j, k); s(i, j) = v + (i == j ? 1.0 : 0.0); }
  }
  // 5 exact Kramers pairs (v, Theta v), S-orthonormal.
  const std::size_t n_pairs = 5;
  Matrix<C> c(n2, 2 * n_pairs, C{});
  std::size_t col = 0;
  for (std::size_t k = 0; k < n_pairs; ++k) {
    std::vector<C> v(n2);
    for (C& x : v) x = C(rng.next(), rng.next());
    for (std::size_t j = 0; j < col; ++j) { const C o = inner(s, column(c, j), v); const auto cj = column(c, j); for (std::size_t i = 0; i < n2; ++i) v[i] -= o * cj[i]; }
    double nrm = std::sqrt(std::real(inner(s, v, v))); for (C& x : v) x /= nrm;
    const auto w = theta(v);
    for (std::size_t i = 0; i < n2; ++i) { c(i, col) = v[i]; c(i, col + 1) = w[i]; }
    col += 2;
  }
  check(thetaDeviation(s, c) < 1e-13 && orthonormalityError(s, c) < 1e-13, "constructed Kramers pairs are exact and S-orthonormal");

  // Scramble: pairs 2 and 3 (columns 4..7) form a near-degenerate cluster; a random unitary
  // inside it keeps the subspace time-reversal invariant but destroys the pairing.
  // Pair 1 additionally gets a phase on its odd column.
  const std::vector<double> energies = {-1.0, -1.0, -0.5, -0.5, 0.2, 0.2, 0.2 + 1e-7, 0.2 + 1e-7, 0.9, 0.9};
  Matrix<C> scr = c;
  {
    Matrix<C> h(4, 4, C{});
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = i; j < 4; ++j) { h(i, j) = C(rng.next(), i == j ? 0.0 : rng.next()); h(j, i) = std::conj(h(i, j)); }
    const auto u = diagonalizeHermitian(h).eigenvectors;  // random unitary
    for (std::size_t r = 0; r < n2; ++r) for (std::size_t j = 0; j < 4; ++j) { C sum{}; for (std::size_t a = 0; a < 4; ++a) sum += c(r, 4 + a) * u(a, j); scr(r, 4 + j) = sum; }
    const C ph = std::polar(1.0, 0.7);
    for (std::size_t r = 0; r < n2; ++r) scr(r, 3) *= ph;
  }
  check(orthonormalityError(s, scr) < 1e-12, "scrambled set is still orthonormal");
  const double before = thetaDeviation(s, scr);
  check(before > 1e-2, "control: scrambling breaks Theta|2k> = |2k+1> (deviation " + std::to_string(before) + ")");

  // Tolerance too small to see the cluster: in the generic case the global exact enforcement
  // still yields a valid paired set...
  {
    KramersPairingReport rep0;
    const Matrix<C> f = fixKramersPairingLarge(scr, energies, s, 1e-12, &rep0);
    check(rep0.n_multi_pair_clusters == 0 && thetaDeviation(s, f) < 1e-13 && orthonormalityError(s, f) < 1e-13,
          "generic scrambled cluster: exact pairing even without cluster detection");
  }
  // ...but when the even columns of the cluster are linearly dependent on the earlier pairs
  // (columns ordered [v, u, Theta v, Theta u]) only the cluster step can pair them.
  {
    Matrix<C> perm = c;
    for (std::size_t r = 0; r < n2; ++r) {  // columns 4..7 <- [c4, c6, c5, c7]
      perm(r, 4) = c(r, 4); perm(r, 5) = c(r, 6); perm(r, 6) = c(r, 5); perm(r, 7) = c(r, 7);
    }
    check(thetaDeviation(s, perm) > 1e-2, "control: permuted cluster columns are not paired");
    bool threw = false;
    try { fixKramersPairingLarge(perm, energies, s, 1e-12); } catch (const std::exception&) { threw = true; }
    check(threw, "dependent even column without cluster detection is reported, not silently mangled");
    KramersPairingReport rp;
    const Matrix<C> f = fixKramersPairingLarge(perm, energies, s, 1e-4, &rp);
    check(thetaDeviation(s, f) < 1e-13 && orthonormalityError(s, f) < 1e-13,
          "permuted cluster columns are paired exactly once the cluster is detected");
  }

  KramersPairingReport rep;
  const Matrix<C> fixed = fixKramersPairingLarge(scr, energies, s, 1e-4, &rep);
  std::cout << "clusters with >1 pair: " << rep.n_multi_pair_clusters << ", largest " << rep.largest_cluster_pairs
            << " pairs, invariance residual " << rep.max_invariance_residual << ", max column change " << rep.max_column_change << "\n";
  check(rep.n_multi_pair_clusters == 1 && rep.largest_cluster_pairs == 2, "one two-pair cluster detected");
  check(rep.max_invariance_residual < 1e-12, "cluster subspace is time-reversal invariant");
  const double after = thetaDeviation(s, fixed);
  std::cout << "Theta deviation before " << before << ", after " << after << "\n";
  check(after < 1e-12, "after the repair Theta|2k> = |2k+1> exactly for every pair");
  check(orthonormalityError(s, fixed) < 1e-12, "repaired set is S-orthonormal");
  // Span of the cluster preserved: projector P = V V^dagger S before/after.
  double span_err = 0.0;
  for (std::size_t a = 4; a < 8; ++a) {
    // each new column must lie in the old cluster span: |c_new - P c_new|
    auto v = column(fixed, a);
    std::vector<C> pv(n2, C{});
    for (std::size_t b = 4; b < 8; ++b) { const C o = inner(s, column(scr, b), v); const auto cb = column(scr, b); for (std::size_t i = 0; i < n2; ++i) pv[i] += o * cb[i]; }
    for (std::size_t i = 0; i < n2; ++i) v[i] -= pv[i];
    span_err = std::max(span_err, std::sqrt(std::real(inner(s, v, v))));
  }
  check(span_err < 1e-12, "the cluster subspace is unchanged");
  // Non-cluster pairs: even column unchanged, odd column = Theta(even) (phase fixed).
  double even_change = 0.0;
  for (std::size_t k : {0, 2, 8}) for (std::size_t i = 0; i < n2; ++i) even_change = std::max(even_change, std::abs(fixed(i, k) - scr(i, k)));
  check(even_change < 1e-13, "even columns of non-degenerate pairs are unchanged");
  // Noise: a non-invariant perturbation of ~1e-8 (SCF-level) on every column; the exact
  // enforcement must still deliver Theta|2k> = |2k+1> and orthonormality to machine precision.
  {
    Matrix<C> noisy = scr;
    for (std::size_t r = 0; r < n2; ++r) for (std::size_t j = 0; j < noisy.cols(); ++j) noisy(r, j) += 1e-8 * C(rng.next(), rng.next());
    KramersPairingReport nrep;
    const Matrix<C> f = fixKramersPairingLarge(noisy, energies, s, 1e-4, &nrep);
    std::cout << "noisy input: Theta deviation after " << thetaDeviation(s, f) << ", orthonormality error " << orthonormalityError(s, f)
              << ", enforcement change " << nrep.max_enforcement_change << "\n";
    check(thetaDeviation(s, noisy) > 1e-2, "control: noisy scrambled input is not Kramers-paired");
    check(thetaDeviation(s, f) < 1e-13, "noisy input: Theta|2k> = |2k+1> exact after the repair");
    check(orthonormalityError(s, f) < 1e-13, "noisy input: repaired set is S-orthonormal to machine precision");
    check(nrep.max_enforcement_change < 1e-6, "the enforcement only moves the noisy input at the noise level");
  }
  // Orthonormal-coefficient variant (used for the 4-component eigenvectors): identity metric,
  // Theta c = J conj(c) with J = [[0,-1],[1,0]] blocks (unitary, J conj(J) = -1).
  {
    const std::size_t m = 12, half = m / 2;
    Matrix<C> jm(m, m, C{});
    for (std::size_t i = 0; i < half; ++i) { jm(half + i, i) = 1.0; jm(i, half + i) = -1.0; }
    auto th = [&](const std::vector<C>& v) { std::vector<C> o(m, C{}); for (std::size_t a = 0; a < m; ++a) for (std::size_t b = 0; b < m; ++b) o[a] += jm(a, b) * std::conj(v[b]); return o; };
    auto dot = [&](const std::vector<C>& a, const std::vector<C>& b) { C x{}; for (std::size_t i = 0; i < m; ++i) x += std::conj(a[i]) * b[i]; return x; };
    const std::size_t np = 6;
    Matrix<C> co(m, 2 * np, C{});
    std::size_t k = 0;
    for (std::size_t pr = 0; pr < np; ++pr) {
      std::vector<C> v(m);
      for (C& x : v) x = C(rng.next(), rng.next());
      for (std::size_t j = 0; j < k; ++j) { std::vector<C> cj(m); for (std::size_t i = 0; i < m; ++i) cj[i] = co(i, j); const C o = dot(cj, v); for (std::size_t i = 0; i < m; ++i) v[i] -= o * cj[i]; }
      const double nr = std::sqrt(std::real(dot(v, v))); for (C& x : v) x /= nr;
      const auto w = th(v);
      for (std::size_t i = 0; i < m; ++i) { co(i, k) = v[i]; co(i, k + 1) = w[i]; }
      k += 2;
    }
    // Cluster = pairs 1..2 (columns 2..5), scrambled by a random unitary; energies degenerate to 1e-9.
    std::vector<double> en = {-2, -2, -1, -1, -1 + 1e-9, -1 + 1e-9, 0.5, 0.5, 1, 1, 2, 2};
    Matrix<C> sc = co;
    Matrix<C> hh(4, 4, C{});
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = i; j < 4; ++j) { hh(i, j) = C(rng.next(), i == j ? 0.0 : rng.next()); hh(j, i) = std::conj(hh(i, j)); }
    const auto uu = diagonalizeHermitian(hh).eigenvectors;
    for (std::size_t r = 0; r < m; ++r) for (std::size_t j = 0; j < 4; ++j) { C sum{}; for (std::size_t a = 0; a < 4; ++a) sum += co(r, 2 + a) * uu(a, j); sc(r, 2 + j) = sum; }
    for (std::size_t r = 0; r < m; ++r) for (std::size_t j = 0; j < 4; ++j) sc(r, 2 + j) += 1e-9 * C(rng.next(), rng.next());  // noise
    auto pdev = [&](const Matrix<C>& c2) { double d = 0; for (std::size_t q = 0; q + 1 < c2.cols(); q += 2) { std::vector<C> e(m), o(m); for (std::size_t i = 0; i < m; ++i) { e[i] = c2(i, q); o[i] = c2(i, q + 1); } auto t = th(e); for (std::size_t i = 0; i < m; ++i) t[i] -= o[i]; d = std::max(d, std::sqrt(std::real(dot(t, t)))); } return d; };
    KramersPairingReport orep;
    const Matrix<C> fo = fixKramersPairingOrthonormal(sc, en, jm, 1e-4, &orep);
    double orth = 0; for (std::size_t a = 0; a < fo.cols(); ++a) for (std::size_t b = 0; b < fo.cols(); ++b) { std::vector<C> x(m), y(m); for (std::size_t i = 0; i < m; ++i) { x[i] = fo(i, a); y[i] = fo(i, b); } orth = std::max(orth, std::abs(dot(x, y) - (a == b ? 1.0 : 0.0))); }
    std::cout << "orthonormal variant: partner error " << pdev(sc) << " -> " << pdev(fo) << ", reported " << orep.partner_error_before << " -> " << orep.partner_error_after << ", orthonormality " << orth << "\n";
    check(pdev(sc) > 1e-2, "control (orthonormal variant): scrambled cluster is not paired");
    check(pdev(fo) < 1e-13 && orth < 1e-13, "orthonormal variant: exact pairing and orthonormality");
    check(orep.n_multi_pair_clusters == 1 && std::abs(orep.partner_error_before - pdev(sc)) < 1e-12 && orep.partner_error_after < 1e-13,
          "orthonormal variant: report matches (cluster count, partner error before/after)");
    // structure helpers on integrals built from the exact pairs: one-body h = sum_k eps_k (|k><k| + |Pk><Pk|) is TR-even.
    Matrix<C> hmo(m, m, C{});
    for (std::size_t q = 0; q < m; ++q) hmo(q, q) = 0.3 * static_cast<double>(q / 2);
    double sc1 = 0;
    check(kramersOneBodyDeviation(hmo, &sc1) < 1e-15 && sc1 > 0, "kramersOneBodyDeviation: TR-even one-body matrix has zero deviation");
    hmo(0, 1) = C(0.1, 0.0); hmo(1, 0) = C(0.1, 0.0);
    check(kramersOneBodyDeviation(hmo) > 1e-3, "kramersOneBodyDeviation: a Kramers-mixing element is detected");
  }
  // AO one-body helper: a time-reversal-even [alpha; beta] matrix (spin-orbit-like) and a violation.
  {
    const std::size_t nla = 4;
    Matrix<C> m(2 * nla, 2 * nla, C{});
    for (std::size_t i = 0; i < nla; ++i) for (std::size_t j = 0; j < nla; ++j) {
      const C aa(rng.next(), rng.next()), ab(rng.next(), rng.next());
      m(i, j) = aa; m(nla + i, nla + j) = std::conj(aa);
      m(i, nla + j) = ab; m(nla + i, j) = -std::conj(ab);
    }
    double sc2 = 0;
    check(kramersAoOneBodyDeviation(m, &sc2) < 1e-15 && sc2 > 0, "kramersAoOneBodyDeviation: TR-even AO matrix passes");
    m(nla, nla) += C(0.01, 0.0);
    check(kramersAoOneBodyDeviation(m) > 5e-3, "kramersAoOneBodyDeviation: a violation is detected");
  }
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
