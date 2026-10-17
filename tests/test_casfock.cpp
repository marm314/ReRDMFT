// Unit test of MCSCF/CasFock.h: the CASSCF generalized Fock matrix and energy.
//  * against the dense oracle Hessian_opt/GeneralizedFock.h (full 2-RDM = Wick(D) + CAS cumulant, n^4 tensors),
//  * the energy against MCSCF/Casci.h's core-dressed CI energy,
//  * for every backend: element-access (dense Tensor4), CholeskyEri, RiNonRelSpinMoEri (real), RiMoEri (complex).
// Build/run: make test_casfock LIBCINT=/path/to/libcint.a
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "CasFock.h"
#include "Casci.h"
#include "CholeskyEri.h"
#include "GeneralizedFock.h"
#include "OrbitalGradient.h"
#include "ADAM.h"
#include "SpinorRotation.h"
#include "Matrix.h"
#include "Tensor4.h"

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
inline double re(double x) { return x; }
inline double re(const C& x) { return x.real(); }
template <typename T> T rnd(Rng& rng);
template <> double rnd<double>(Rng& rng) { return rng.next(); }
template <> C rnd<C>(Rng& rng) { const double a = rng.next(); return C(a, rng.next()); }

// Hermitian B_P (n_aux of them): B(P, a*n + c).
template <typename T>
Matrix<T> randomB(std::size_t n_aux, std::size_t n, Rng& rng) {
  Matrix<T> b(n_aux, n * n);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t a = 0; a < n; ++a) {
      b(P, a * n + a) = T(0.4 * re(rnd<T>(rng)));
      for (std::size_t c = 0; c < a; ++c) { b(P, a * n + c) = T(0.4) * rnd<T>(rng); b(P, c * n + a) = cj(b(P, a * n + c)); }
    }
  return b;
}
template <typename T>
Matrix<T> randomHermitian(std::size_t n, Rng& rng) {
  Matrix<T> h(n, n);
  for (std::size_t p = 0; p < n; ++p) {
    h(p, p) = T(re(rnd<T>(rng)));
    for (std::size_t q = 0; q < p; ++q) { h(p, q) = rnd<T>(rng); h(q, p) = cj(h(p, q)); }
  }
  return h;
}

// Plain dense rotation (four leg transforms): h' = U^dagger h U, eri'(pqrs) = sum conj(U_ap) conj(U_bq) U_cr U_ds eri(abcd).
template <typename T>
struct Rotated { Matrix<T> h; Tensor4<T> eri; };
template <typename T>
Rotated<T> rotateDense(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& u) {
  const std::size_t n = h.rows();
  Rotated<T> out{Matrix<T>(n, n, T{}), Tensor4<T>(n, n, n, n, T{})};
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) {
    T s{}; for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) s += cj(u(a, p)) * h(a, b) * u(b, q);
    out.h(p, q) = s;
  }
  Tensor4<T> t1(n, n, n, n, T{}), t2(n, n, n, n, T{});
  for (std::size_t p = 0; p < n; ++p) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    T s{}; for (std::size_t a = 0; a < n; ++a) s += cj(u(a, p)) * eri(a, b, c, d); t1(p, b, c, d) = s; }
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    T s{}; for (std::size_t b = 0; b < n; ++b) s += cj(u(b, q)) * t1(p, b, c, d); t2(p, q, c, d) = s; }
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) for (std::size_t r = 0; r < n; ++r) for (std::size_t d = 0; d < n; ++d) {
    T s{}; for (std::size_t c = 0; c < n; ++c) s += u(c, r) * t2(p, q, c, d); t1(p, q, r, d) = s; }
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) for (std::size_t r = 0; r < n; ++r) for (std::size_t s2 = 0; s2 < n; ++s2) {
    T s{}; for (std::size_t d = 0; d < n; ++d) s += u(d, s2) * t1(p, q, r, d); out.eri(p, q, r, s2) = s; }
  return out;
}

template <typename Eri, typename T>
Tensor4<T> toTensor(const Eri& eri, std::size_t n) {
  Tensor4<T> t(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b)
    for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) t(a, b, c, d) = eri(a, b, c, d);
  return t;
}

double maxDiff(const Matrix<double>& a, const Matrix<double>& b) {
  double w = 0.0;
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) w = std::max(w, std::abs(a.data()[i] - b.data()[i]));
  return w;
}
double maxDiff(const Matrix<C>& a, const Matrix<C>& b) {
  double w = 0.0;
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) w = std::max(w, std::abs(a.data()[i] - b.data()[i]));
  return w;
}

// Dense reference: GeneralizedFock oracle + explicit energy contraction, from a random CI state.
template <typename T>
struct Reference {
  Matrix<T> fock;
  double energy;
  Matrix<T> rdm1;
  std::vector<T> rdm2;
  double ci_energy;  // <c|H_dressed|c> + core from MCSCF/Casci.h
  std::vector<T> state;
};

template <typename T>
double ciEnergy(const Matrix<T>& h, const Tensor4<T>& eri, std::size_t n_core, std::size_t a, std::size_t n_el, const std::vector<T>& c);

template <typename T>
Reference<T> buildReference(const Matrix<T>& h, const Tensor4<T>& eri, std::size_t n_core, std::size_t a, std::size_t n_el, Rng& rng) {
  const std::size_t n = h.rows();
  Casci<T> ci(a, n_el);
  std::vector<T> c(ci.dimension());
  double nn = 0.0;
  for (T& v : c) { v = rnd<T>(rng); nn += std::norm(C(v)); }
  for (T& v : c) v /= std::sqrt(nn);

  Reference<T> ref;
  ci.densities(c, ref.rdm1, ref.rdm2);

  // total D and the full 2-RDM g2[p,q,r,s] = <a+_p a+_r a_s a_q> = D_pq D_rs - D_ps D_rq + lambda_full
  Matrix<T> d(n, n, T{});
  for (std::size_t i = 0; i < n_core; ++i) d(i, i) = T(1.0);
  for (std::size_t t = 0; t < a; ++t) for (std::size_t u = 0; u < a; ++u) d(n_core + t, n_core + u) = ref.rdm1(t, u);
  Tensor4<T> two(n, n, n, n, T{});
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q)
    for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) {
      T g2 = d(p, q) * d(r, s) - d(p, s) * d(r, q);
      if (p >= n_core && p < n_core + a && q >= n_core && q < n_core + a && r >= n_core && r < n_core + a && s >= n_core && s < n_core + a) {
        const std::size_t t = p - n_core, u = q - n_core, v = r - n_core, w = s - n_core;
        g2 += ref.rdm2[((t * a + u) * a + v) * a + w] - ref.rdm1(t, u) * ref.rdm1(v, w) + ref.rdm1(t, w) * ref.rdm1(v, u);
      }
      // two_rdm_phys(p,q,r,s) = 1/2 <a+_p a+_q a_s a_r> = 1/2 g2[p,r,q,s]  -> store at (p,r,q,s)
      two(p, q, r, s) = T(0.5) * g2;   // index meaning here: (p, r_phys? ) fixed below
    }
  // g2[p,q,r,s] stored above at (p,q,r,s); convert: phys(p,q,r,s) = 1/2 g2[p,r,q,s]
  Tensor4<T> phys(n, n, n, n, T{});
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q)
    for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) phys(p, q, r, s) = two(p, r, q, s);
  ref.fock = generalizedFockMatrix(h, eri, d, phys);
  T e{};
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) e += h(p, q) * d(p, q);
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q)
    for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) e += eri(p, q, r, s) * phys(p, q, r, s);
  ref.energy = re(e);

  ref.ci_energy = ciEnergy(h, eri, n_core, a, n_el, c);
  ref.state = c;
  return ref;
}

}  // namespace
namespace {
template <typename T>
double ciEnergy(const Matrix<T>& h, const Tensor4<T>& eri, std::size_t n_core, std::size_t a, std::size_t n_el, const std::vector<T>& c) {
  Casci<T> ci(a, n_el);
  // core-dressed CI Hamiltonian over the active spinors
  CasciHamiltonian<T> ham;
  ham.n_orbitals = a; ham.n_electrons = n_el;
  ham.h = Matrix<T>(a, a);
  for (std::size_t t = 0; t < a; ++t) for (std::size_t u = 0; u < a; ++u) {
    T v = h(n_core + t, n_core + u);
    for (std::size_t i = 0; i < n_core; ++i) v += eri(n_core + t, i, n_core + u, i) - eri(n_core + t, i, i, n_core + u);
    ham.h(t, u) = v;
  }
  ham.eri.assign(a * a * a * a, T{});
  for (std::size_t t = 0; t < a; ++t) for (std::size_t u = 0; u < a; ++u)
    for (std::size_t v = 0; v < a; ++v) for (std::size_t w = 0; w < a; ++w)
      ham.eri[((t * a + u) * a + v) * a + w] = eri(n_core + t, n_core + v, n_core + u, n_core + w);   // (tu|vw) = <tv|uw>
  double ecore = 0.0;
  for (std::size_t i = 0; i < n_core; ++i) {
    ecore += re(h(i, i));
    for (std::size_t j = 0; j < n_core; ++j) ecore += 0.5 * re(eri(i, j, i, j) - eri(i, j, j, i));
  }
  ham.core_energy = ecore;
  const std::vector<T> hc = ci.sigma(ham, c);
  T eh{}; for (std::size_t i = 0; i < c.size(); ++i) eh += cj(c[i]) * hc[i];
  return re(eh) + ecore;
}

std::vector<std::size_t> range(std::size_t first, std::size_t count) {
  std::vector<std::size_t> v(count);
  for (std::size_t i = 0; i < count; ++i) v[i] = first + i;
  return v;
}

template <typename T, typename Eri>
void checkBackend(const std::string& label, const Matrix<T>& h, const Eri& eri, std::size_t n_core, std::size_t a, const Reference<T>& ref) {
  const CasFockResult<T> r = casFock(h, eri, range(0, n_core), range(n_core, a), ref.rdm1, ref.rdm2);
  const double df = maxDiff(r.fock, ref.fock);
  check(df < 1e-10, label + ": GenFock matches the dense oracle (max dev " + std::to_string(df) + ")");
  check(std::abs(r.energy - ref.energy) < 1e-10, label + ": energy matches the dense contraction (" + std::to_string(r.energy - ref.energy) + ")");
  check(std::abs(r.energy - ref.ci_energy) < 1e-10, label + ": energy equals the core-dressed CI energy (" + std::to_string(r.energy - ref.ci_energy) + ")");
}


// Orbital gradient vs a central finite difference of the TRUE CI energy (MCSCF/Casci.h's Hamiltonian, CI vector
// fixed) under C_new = C exp(-kappa): dE/dt = Re g_pq, dE/dy = Im g_pq (kappa_pq = t + i y; OrbitalGradient.h).
template <typename T, typename Eri>
void checkGradient(const std::string& label, const Matrix<T>& h, const Tensor4<T>& eri_dense, const Eri& eri, std::size_t n_core,
                   std::size_t a, std::size_t n_el, const Reference<T>& ref) {
  const std::size_t n = h.rows();
  const CasFockResult<T> r = casFock(h, eri, range(0, n_core), range(n_core, a), ref.rdm1, ref.rdm2);
  const Matrix<T> g = orbitalGradient(r.fock);
  const std::vector<std::pair<std::size_t, std::size_t>> pairs = {{n - 1, 0}, {3, 1}, {n_core + 1, 0}, {n - 1, n_core + 1}, {n_core + 2, n_core}};
  const double dl = 1e-5;
  double worst = 0.0;
  for (const auto& pq : pairs) {
    if (pq.first >= n || pq.second >= pq.first) continue;
    for (int part = 0; part < (std::is_same_v<T, C> ? 2 : 1); ++part) {
      auto energyAt = [&](double delta) {
        std::vector<T> step(1);
        if constexpr (std::is_same_v<T, C>) step[0] = part == 0 ? C(delta, 0.0) : C(0.0, delta);
        else step[0] = delta;
        const Matrix<T> kappa = adamKappaMatrix(n, {pq}, step);
        const Matrix<T> u = spinorRotationMatrix(kappa);
        const Rotated<T> rot = rotateDense(h, eri_dense, u);
        return ciEnergy(rot.h, rot.eri, n_core, a, n_el, ref.state);
      };
      const double fd = (energyAt(dl) - energyAt(-dl)) / (2 * dl);
      const double an = part == 0 ? re(g(pq.first, pq.second)) : std::imag(C(g(pq.first, pq.second)));
      worst = std::max(worst, std::abs(fd - an));
    }
  }
  check(worst < 1e-7, label + ": orbital gradient matches the finite difference of the CI energy (max dev " + std::to_string(worst) + ")");
}

template <typename T>
void testComplexOrReal(const char* name, std::size_t n, std::size_t n_core, std::size_t a, std::size_t n_el) {
  std::cout << "[" << name << "] n = " << n << ", core = " << n_core << ", active = " << a << " (" << n_el << " e)\n";
  Rng rng;
  const Matrix<T> h = randomHermitian<T>(n, rng);
  const Matrix<T> b = randomB<T>(7, n, rng);
  Tensor4<T> eri(n, n, n, n);
  for (std::size_t x = 0; x < n; ++x) for (std::size_t y = 0; y < n; ++y)
    for (std::size_t z = 0; z < n; ++z) for (std::size_t w = 0; w < n; ++w) {
      T s{}; for (std::size_t P = 0; P < b.rows(); ++P) s += b(P, x * n + z) * b(P, y * n + w);
      eri(x, y, z, w) = s;
    }
  const Reference<T> ref = buildReference(h, eri, n_core, a, n_el, rng);
  checkBackend<T>(std::string(name) + " dense Tensor4", h, eri, n_core, a, ref);
  checkGradient<T>(std::string(name) + " dense Tensor4", h, eri, eri, n_core, a, n_el, ref);
  const CholeskyEri<T> chol = CholeskyEri<T>::fromDense(eri, 1e-13);
  std::cout << "  Cholesky vectors: " << chol.nVectors() << "\n";
  checkBackend<T>(std::string(name) + " CholeskyEri", h, chol, n_core, a, ref);
  if constexpr (std::is_same_v<T, C>) {
    const RiMoEri ri(n, b);
    checkBackend<T>(std::string(name) + " RiMoEri", h, ri, n_core, a, ref);
  }
}

// Non-contiguous core / active index lists (NON_REL's [alpha | beta] layout): relabel the orbitals by a permutation and compare
// with the contiguous result; also casciHamiltonian() against the hand-built dressed Hamiltonian's CI energy.
template <typename T>
void testPermuted(const char* name) {
  const std::size_t n = 9, n_core = 2, a = 4, n_el = 2;
  std::cout << "[" << name << ", permuted orbital labels] n = " << n << "\n";
  Rng rng;
  const Matrix<T> h = randomHermitian<T>(n, rng);
  const Matrix<T> b = randomB<T>(6, n, rng);
  Tensor4<T> eri(n, n, n, n);
  for (std::size_t x = 0; x < n; ++x) for (std::size_t y = 0; y < n; ++y)
    for (std::size_t z = 0; z < n; ++z) for (std::size_t w = 0; w < n; ++w) {
      T s{}; for (std::size_t P = 0; P < b.rows(); ++P) s += b(P, x * n + z) * b(P, y * n + w);
      eri(x, y, z, w) = s;
    }
  const Reference<T> ref = buildReference(h, eri, n_core, a, n_el, rng);
  // new label j holds old orbital perm[j]
  const std::vector<std::size_t> perm = {5, 0, 7, 2, 8, 1, 4, 6, 3};
  std::vector<std::size_t> inv(n);
  for (std::size_t j = 0; j < n; ++j) inv[perm[j]] = j;
  Matrix<T> h2(n, n);
  Tensor4<T> eri2(n, n, n, n);
  Matrix<T> b2(b.rows(), n * n);
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) {
    h2(p, q) = h(perm[p], perm[q]);
    for (std::size_t P = 0; P < b.rows(); ++P) b2(P, p * n + q) = b(P, perm[p] * n + perm[q]);
    for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s < n; ++s) eri2(p, q, r, s) = eri(perm[p], perm[q], perm[r], perm[s]);
  }
  std::vector<std::size_t> core2, act2;
  for (std::size_t i = 0; i < n_core; ++i) core2.push_back(inv[i]);
  for (std::size_t t = 0; t < a; ++t) act2.push_back(inv[n_core + t]);
  auto compare = [&](const std::string& label, const CasFockResult<T>& r) {
    double w = 0.0;
    for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) w = std::max(w, std::abs(r.fock(p, q) - ref.fock(perm[p], perm[q])));
    check(w < 1e-10, label + ": GenFock of the relabelled orbitals matches the permuted contiguous result (" + std::to_string(w) + ")");
    check(std::abs(r.energy - ref.energy) < 1e-10, label + ": energy unchanged by relabelling");
  };
  compare("dense", casFock(h2, eri2, core2, act2, ref.rdm1, ref.rdm2));
  compare("Cholesky", casFock(h2, CholeskyEri<T>::fromDense(eri2, 1e-13), core2, act2, ref.rdm1, ref.rdm2));
  if constexpr (std::is_same_v<T, C>) compare("RiMoEri", casFock(h2, RiMoEri(n, b2), core2, act2, ref.rdm1, ref.rdm2));
  // the core-dressed Hamiltonian built from the backends reproduces the reference CI energy
  auto dressed = [&](const std::string& label, const CasciHamiltonian<T>& ham) {
    Casci<T> ci(a, n_el);
    const std::vector<T> hc = ci.sigma(ham, ref.state);
    T eh{}; for (std::size_t i = 0; i < hc.size(); ++i) eh += cj(ref.state[i]) * hc[i];
    check(std::abs(re(eh) + ham.core_energy - ref.ci_energy) < 1e-10, label + ": casciHamiltonian gives the reference CI energy (" +
          std::to_string(re(eh) + ham.core_energy - ref.ci_energy) + ")");
  };
  dressed("dense", casciHamiltonian(h2, eri2, core2, act2, n_el));
  dressed("Cholesky", casciHamiltonian(h2, CholeskyEri<T>::fromDense(eri2, 1e-13), core2, act2, n_el));
  if constexpr (std::is_same_v<T, C>) dressed("RiMoEri", casciHamiltonian(h2, RiMoEri(n, b2), core2, act2, n_el));
}

// NON_REL spin-orbital RI: B real spatial, spin selection rule built into operator() / embeddedB().
void testRiNonRel() {
  const std::size_t ns = 5, n = 2 * ns, n_core = 2, a = 5, n_el = 3;
  std::cout << "[RiNonRelSpinMoEri] n_spatial = " << ns << " (n = " << n << "), core = " << n_core << ", active = " << a << "\n";
  Rng rng;
  const Matrix<double> h = randomHermitian<double>(n, rng);
  const Matrix<double> b = randomB<double>(6, ns, rng);
  const RiNonRelSpinMoEri ri(ns, b);
  const Tensor4<double> eri = toTensor<RiNonRelSpinMoEri, double>(ri, n);
  const Reference<double> ref = buildReference(h, eri, n_core, a, n_el, rng);
  checkBackend<double>("RiNonRelSpinMoEri", h, ri, n_core, a, ref);
  checkBackend<double>("element-access on RiNonRelSpinMoEri (generic path)", h, eri, n_core, a, ref);
}

}  // namespace

int main() {
  testComplexOrReal<double>("real", 9, 2, 5, 3);
  testComplexOrReal<C>("complex", 9, 2, 5, 3);
  testComplexOrReal<C>("complex, no core", 8, 0, 4, 2);
  testComplexOrReal<double>("real, no virtual", 6, 2, 4, 2);
  testRiNonRel();
  testPermuted<double>("real");
  testPermuted<C>("complex");
  std::cout << (g_failures == 0 ? "ALL " : "FAILURES: ") << g_checks << " checks, " << g_failures << " failed\n";
  return g_failures == 0 ? 0 : 1;
}
