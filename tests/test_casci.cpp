// Unit test of MCSCF/Casci.h: the general spinor CASCI engine (real and complex), against
//  (1) an independent brute-force H built by applying a+_p a+_r a_s a_q to determinants literally,
//  (2) exact diagonalization, Hermiticity / sum rules / E = <c|H|c> of the RDMs, real and complex,
//  (3) PySCF CASCI(6,6) for N2 1.1 A (tests/data/n2_cas66_1p1.txt, made by make_n2_cas66.py):
//      energy and spin-traced dm1 / dm2.
// Build/run: make test_casci LIBCINT=/path/to/libcint.a (run from the repository root).
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Casci.h"
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
  std::uint64_t s = 20261017ULL;
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; }
};
inline double cj(double x) { return x; }
inline C cj(const C& x) { return std::conj(x); }
inline double re(double x) { return x; }
inline double re(const C& x) { return x.real(); }
template <typename T> T rnd(Rng& rng);
template <> double rnd<double>(Rng& rng) { return rng.next(); }
template <> C rnd<C>(Rng& rng) { const double a = rng.next(); return C(a, rng.next()); }

// Random Hermitian h and (pq|rs) = sum_P B^P_pq B^P_rs with Hermitian B^P: has (pq|rs) = (rs|pq) = (qp|sr)*.
template <typename T>
CasciHamiltonian<T> randomHamiltonian(std::size_t m, std::size_t n, Rng& rng) {
  CasciHamiltonian<T> h;
  h.n_orbitals = m; h.n_electrons = n; h.core_energy = 0.37;
  h.h = Matrix<T>(m, m);
  for (std::size_t p = 0; p < m; ++p) {
    h.h(p, p) = T(re(rnd<T>(rng)));
    for (std::size_t q = 0; q < p; ++q) { h.h(p, q) = rnd<T>(rng); h.h(q, p) = cj(h.h(p, q)); }
  }
  h.eri.assign(m * m * m * m, T{});
  for (int P = 0; P < 4; ++P) {
    Matrix<T> b(m, m);
    for (std::size_t p = 0; p < m; ++p) {
      b(p, p) = T(0.5 * re(rnd<T>(rng)));
      for (std::size_t q = 0; q < p; ++q) { b(p, q) = T(0.5) * rnd<T>(rng); b(q, p) = cj(b(p, q)); }
    }
    for (std::size_t p = 0; p < m; ++p) for (std::size_t q = 0; q < m; ++q)
      for (std::size_t r = 0; r < m; ++r) for (std::size_t s = 0; s < m; ++s)
        h.eri[((p * m + q) * m + r) * m + s] += b(p, q) * b(r, s);
  }
  return h;
}

// Independent: literal operator application on bitmasks (not the engine's link tables).
bool apply(std::uint64_t& det, int& sign, std::size_t p, bool create) {
  const bool occ = (det >> p) & 1u;
  if (create == occ) return false;
  if (__builtin_popcountll(det & ((std::uint64_t{1} << p) - 1)) & 1) sign = -sign;
  det ^= (std::uint64_t{1} << p);
  return true;
}

template <typename T>
Matrix<T> bruteForceH(const Casci<T>& ci, const CasciHamiltonian<T>& ham) {
  const std::size_t dim = ci.dimension(), m = ci.nOrbitals();
  const std::vector<std::uint64_t>& dets = ci.determinants();
  auto index = [&](std::uint64_t d) { for (std::size_t i = 0; i < dim; ++i) if (dets[i] == d) return i; return dim; };
  Matrix<T> hm(dim, dim, T{});
  for (std::size_t i = 0; i < dim; ++i) {
    for (std::size_t p = 0; p < m; ++p) for (std::size_t q = 0; q < m; ++q) {   // h_pq a+_p a_q
      std::uint64_t d = dets[i]; int sg = 1;
      if (!apply(d, sg, q, false) || !apply(d, sg, p, true)) continue;
      hm(index(d), i) += static_cast<double>(sg) * ham.h(p, q);
    }
    for (std::size_t p = 0; p < m; ++p) for (std::size_t q = 0; q < m; ++q)
      for (std::size_t r = 0; r < m; ++r) for (std::size_t s = 0; s < m; ++s) {  // 1/2 (pq|rs) a+_p a+_r a_s a_q
        std::uint64_t d = dets[i]; int sg = 1;
        if (!apply(d, sg, q, false) || !apply(d, sg, s, false) || !apply(d, sg, r, true) || !apply(d, sg, p, true)) continue;
        hm(index(d), i) += 0.5 * static_cast<double>(sg) * ham.eri[((p * m + q) * m + r) * m + s];
      }
  }
  return hm;
}

Matrix<C> toComplex(const Matrix<double>& a) {
  Matrix<C> b(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) b.data()[i] = a.data()[i];
  return b;
}
Matrix<C> toComplex(const Matrix<C>& a) { return a; }

template <typename T>
void testRandom(const char* label, std::size_t m, std::size_t n) {
  std::cout << "[" << label << "] M = " << m << ", N = " << n << "\n";
  Rng rng;
  const CasciHamiltonian<T> ham = randomHamiltonian<T>(m, n, rng);
  Casci<T> ci(m, n);
  const std::size_t dim = ci.dimension();

  // (1) sigma vs brute-force H, column by column; H Hermitian.
  const Matrix<T> hb = bruteForceH(ci, ham);
  double worst = 0.0, herm = 0.0;
  for (std::size_t j = 0; j < dim; ++j) {
    std::vector<T> e(dim, T{}); e[j] = T(1.0);
    const std::vector<T> col = ci.sigma(ham, e);
    for (std::size_t i = 0; i < dim; ++i) worst = std::max(worst, std::abs(col[i] - hb(i, j)));
  }
  for (std::size_t i = 0; i < dim; ++i) for (std::size_t j = 0; j < dim; ++j) herm = std::max(herm, std::abs(hb(i, j) - cj(hb(j, i))));
  check(worst < 1e-11, "sigma matches the brute-force Hamiltonian (max dev " + std::to_string(worst) + ")");
  check(herm < 1e-11, "brute-force H is Hermitian (" + std::to_string(herm) + ")");
  const std::vector<double> dg = ci.diagonal(ham);
  double dgw = 0.0;
  for (std::size_t i = 0; i < dim; ++i) dgw = std::max(dgw, std::abs(dg[i] - re(hb(i, i))));
  check(dgw < 1e-11, "diagonal() matches H_II (" + std::to_string(dgw) + ")");

  // (2) Davidson vs dense diagonalization.
  const HermitianEigenResult ex = diagonalizeHermitian(toComplex(hb));
  const CasciResult<T> res = ci.solve(ham);
  check(res.converged, "Davidson converged (" + std::to_string(res.iterations) + " iterations, residual " + std::to_string(res.residual_norm) + ")");
  check(std::abs(res.energy - (ex.eigenvalues[0] + ham.core_energy)) < 1e-9, "Davidson energy = dense lowest eigenvalue + core (" +
        std::to_string(res.energy - (ex.eigenvalues[0] + ham.core_energy)) + ")");

  // (3) RDMs of the ground state and of a random normalized vector.
  for (int which = 0; which < 2; ++which) {
    std::vector<T> c = res.ci;
    if (which == 1) { double nn = 0.0; for (T& v : c) { v = rnd<T>(rng); nn += std::norm(C(v)); } for (T& v : c) v /= std::sqrt(nn); }
    Matrix<T> g1; std::vector<T> g2;
    ci.densities(c, g1, g2);
    const std::vector<T> hc = ci.sigma(ham, c);
    T ehc{}; for (std::size_t i = 0; i < dim; ++i) ehc += cj(c[i]) * hc[i];
    T e{};
    for (std::size_t p = 0; p < m; ++p) for (std::size_t q = 0; q < m; ++q) e += ham.h(p, q) * g1(p, q);
    for (std::size_t i = 0; i < m * m * m * m; ++i) e += 0.5 * ham.eri[i] * g2[i];
    const std::string tag = which == 0 ? "ground state" : "random vector";
    check(std::abs(e - ehc) < 1e-10, tag + ": tr(h g1) + 1/2 (pq|rs) g2 = <c|H|c> (" + std::to_string(std::abs(e - ehc)) + ")");
    double tr1 = 0.0, tr2 = 0.0, herm1 = 0.0, sym2 = 0.0, part = 0.0;
    for (std::size_t p = 0; p < m; ++p) {
      tr1 += re(g1(p, p));
      for (std::size_t q = 0; q < m; ++q) herm1 = std::max(herm1, std::abs(g1(p, q) - cj(g1(q, p))));
      for (std::size_t r = 0; r < m; ++r) tr2 += re(g2[((p * m + p) * m + r) * m + r]);
    }
    for (std::size_t p = 0; p < m; ++p) for (std::size_t q = 0; q < m; ++q) {
      T s{}; for (std::size_t r = 0; r < m; ++r) s += g2[((p * m + q) * m + r) * m + r];
      part = std::max(part, std::abs(s - static_cast<double>(n - 1) * g1(p, q)));
      for (std::size_t r = 0; r < m; ++r) for (std::size_t s2 = 0; s2 < m; ++s2) {
        const T a = g2[((p * m + q) * m + r) * m + s2];
        sym2 = std::max(sym2, std::abs(a - g2[((r * m + s2) * m + p) * m + q]));
        sym2 = std::max(sym2, std::abs(a - cj(g2[((q * m + p) * m + s2) * m + r])));
      }
    }
    check(std::abs(tr1 - n) < 1e-11, tag + ": tr(g1) = N");
    check(std::abs(tr2 - static_cast<double>(n * (n - 1))) < 1e-10, tag + ": sum g2[ppqq] = N(N-1)");
    check(herm1 < 1e-11, tag + ": g1 Hermitian");
    check(sym2 < 1e-10, tag + ": g2 symmetric under (pq<->rs) and conjugation (" + std::to_string(sym2) + ")");
    check(part < 1e-10, tag + ": sum_r g2[pqrr] = (N-1) g1 (" + std::to_string(part) + ")");
  }
}

// PySCF CASCI(6,6) reference, spatial orbitals -> spin-orbitals (index 2p + sigma).
void testPyscf() {
  std::cout << "[PySCF] N2 1.1 A CASCI(6,6), spin-orbital engine\n";
  std::ifstream in("tests/data/n2_cas66_1p1.txt");
  if (!in) { check(false, "reference file tests/data/n2_cas66_1p1.txt not found (run from the repository root)"); return; }
  std::size_t ncas, nelec; double ecore, e_ref;
  in >> ncas >> nelec >> ecore >> e_ref;
  std::vector<double> h1(ncas * ncas), h2(ncas * ncas * ncas * ncas), d1(ncas * ncas), d2(h2.size());
  for (double& v : h1) in >> v;
  for (double& v : h2) in >> v;
  for (double& v : d1) in >> v;
  for (double& v : d2) in >> v;
  const std::size_t m = 2 * ncas;
  CasciHamiltonian<double> ham;
  ham.n_orbitals = m; ham.n_electrons = nelec; ham.core_energy = ecore;
  ham.h = Matrix<double>(m, m, 0.0);
  ham.eri.assign(m * m * m * m, 0.0);
  for (std::size_t p = 0; p < ncas; ++p) for (std::size_t q = 0; q < ncas; ++q) for (int s = 0; s < 2; ++s)
    ham.h(2 * p + s, 2 * q + s) = h1[p * ncas + q];
  for (std::size_t p = 0; p < ncas; ++p) for (std::size_t q = 0; q < ncas; ++q)
    for (std::size_t r = 0; r < ncas; ++r) for (std::size_t s = 0; s < ncas; ++s)
      for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b)
        ham.eri[(((2 * p + a) * m + (2 * q + a)) * m + (2 * r + b)) * m + (2 * s + b)] = h2[((p * ncas + q) * ncas + r) * ncas + s];
  Casci<double> ci(m, nelec);
  const CasciResult<double> res = ci.solve(ham);
  std::cout << "  determinants " << ci.dimension() << ", Davidson iterations " << res.iterations << ", residual " << res.residual_norm
            << "\n  E(ours) = " << std::setprecision(12) << res.energy << "  E(PySCF) = " << e_ref << "\n";
  check(res.converged, "Davidson converged");
  check(std::abs(res.energy - e_ref) < 1e-9, "energy matches PySCF (" + std::to_string(res.energy - e_ref) + ")");
  Matrix<double> g1; std::vector<double> g2;
  ci.densities(res.ci, g1, g2);
  double w1 = 0.0, w2 = 0.0;
  for (std::size_t p = 0; p < ncas; ++p) for (std::size_t q = 0; q < ncas; ++q) {
    double s = 0.0; for (int a = 0; a < 2; ++a) s += g1(2 * p + a, 2 * q + a);
    w1 = std::max(w1, std::abs(s - d1[p * ncas + q]));
    for (std::size_t r = 0; r < ncas; ++r) for (std::size_t t = 0; t < ncas; ++t) {
      double v = 0.0;
      for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b)
        v += g2[(((2 * p + a) * m + (2 * q + a)) * m + (2 * r + b)) * m + (2 * t + b)];
      w2 = std::max(w2, std::abs(v - d2[((p * ncas + q) * ncas + r) * ncas + t]));
    }
  }
  check(w1 < 1e-8, "spin-traced dm1 matches PySCF (" + std::to_string(w1) + ")");
  check(w2 < 1e-8, "spin-traced dm2 matches PySCF (" + std::to_string(w2) + ")");
}

}  // namespace

int main() {
  testRandom<double>("real", 6, 3);
  testRandom<C>("complex", 6, 3);
  testRandom<double>("real", 8, 4);
  testRandom<C>("complex", 8, 4);
  testRandom<C>("complex, odd N", 7, 3);
  testPyscf();
  std::cout << (g_failures == 0 ? "ALL " : "FAILURES: ") << g_checks << " checks, " << g_failures << " failed\n";
  return g_failures == 0 ? 0 : 1;
}
