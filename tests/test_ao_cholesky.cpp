// Unit test of Utils/AoCholesky.h against brute-force references built directly from the packed AO
// tensor: the NON_REL and X2C Fock matrices from AO Cholesky vectors, and the AO -> MO vector transforms
// (NON_REL closed-shell spin-orbitals, X2C spinors). Build/run: make test_ao_cholesky
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "AoCholesky.h"
#include "ElectronRepulsion.h"
#include "Matrix.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {
int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; } }
struct Rng { std::uint64_t s = 20260926ULL; double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; } };

double maxDiff(const Matrix<double>& a, const Matrix<double>& b) { double w = 0; for (std::size_t i = 0; i < a.rows(); ++i) for (std::size_t j = 0; j < a.cols(); ++j) w = std::max(w, std::abs(a(i, j) - b(i, j))); return w; }
double maxDiff(const Matrix<C>& a, const Matrix<C>& b) { double w = 0; for (std::size_t i = 0; i < a.rows(); ++i) for (std::size_t j = 0; j < a.cols(); ++j) w = std::max(w, std::abs(a(i, j) - b(i, j))); return w; }

// AO tensor of rank `rank`: (pq|rs) = sum_L B_L(p,q) B_L(r,s), B_L real symmetric.
PackedTwoElectronTensor makeAo(std::size_t n, std::size_t rank, Rng& rng) {
  std::vector<Matrix<double>> b(rank, Matrix<double>(n, n, 0.0));
  for (auto& m : b) for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) m(i, j) = m(j, i) = 0.3 * rng.next();
  PackedTwoElectronTensor t(n);
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q <= p; ++q) for (std::size_t r = 0; r < n; ++r) for (std::size_t s = 0; s <= r; ++s) {
    double v = 0; for (const auto& m : b) v += m(p, q) * m(r, s);
    t.set(p, q, r, s, v);
  }
  return t;
}
}  // namespace

int main() {
  Rng rng;
  const std::size_t n = 6, rank = 9;
  const PackedTwoElectronTensor ao_int = makeAo(n, rank, rng);
  CholeskyCheckReport rep;
  const AoCholesky ao = AoCholesky::fromPacked(ao_int, 1e-10, &rep);
  check(ao.nVectors() > 0 && ao.nVectors() <= rank + 1, "AO Cholesky finds the true rank (" + std::to_string(ao.nVectors()) + ")");
  check(rep.max_error <= rep.tolerance, "AO decomposition reproduces the packed tensor, error " + std::to_string(rep.max_error));

  // ---- NON_REL Fock ----
  Matrix<double> h(n, n, 0.0), p(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) { h(i, j) = h(j, i) = rng.next(); p(i, j) = p(j, i) = rng.next(); }
  Matrix<double> f_ref(n, n, 0.0);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t r = 0; r < n; ++r) {
    double hart = 0, exch = 0;
    for (std::size_t q = 0; q < n; ++q) for (std::size_t s = 0; s < n; ++s) { hart += p(s, q) * ao_int(a, r, q, s); exch += p(s, q) * ao_int(a, s, q, r); }
    f_ref(a, r) = h(a, r) + hart - 0.5 * exch;
  }
  check(maxDiff(f_ref, nonRelFockMatrix(h, ao, p)) < 1e-9, "NON_REL Fock from AO Cholesky vectors matches the packed-tensor Fock");

  // ---- X2C Fock (general Hermitian 2n x 2n density, spin structure as x2cFockMatrix) ----
  const std::size_t n2 = 2 * n;
  Matrix<C> hx(n2, n2, C{}), px(n2, n2, C{});
  for (std::size_t i = 0; i < n2; ++i) for (std::size_t j = i; j < n2; ++j) {
    const C v(rng.next(), i == j ? 0.0 : rng.next());
    hx(i, j) = v; hx(j, i) = std::conj(v);
    const C w(rng.next(), i == j ? 0.0 : rng.next());
    px(i, j) = w; px(j, i) = std::conj(w);
  }
  auto spin = [&](std::size_t i) { return i / n; };
  auto ao_of = [&](std::size_t i) { return i % n; };
  // <AB|CD> of the closed-shell spin-orbital tensor = (AC|BD) if spin(A)=spin(C) and spin(B)=spin(D)
  auto eri_spin = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) -> double {
    if (spin(a) != spin(c) || spin(b) != spin(d)) return 0.0;
    return ao_int(ao_of(a), ao_of(c), ao_of(b), ao_of(d));
  };
  Matrix<C> fx_ref(n2, n2, C{});
  for (std::size_t a = 0; a < n2; ++a) for (std::size_t r = 0; r < n2; ++r) {
    C hart{}, exch{};
    for (std::size_t q = 0; q < n2; ++q) for (std::size_t s = 0; s < n2; ++s) { hart += px(s, q) * eri_spin(a, q, r, s); exch += px(s, q) * eri_spin(a, q, s, r); }
    fx_ref(a, r) = hx(a, r) + hart - exch;
  }
  check(maxDiff(fx_ref, x2cFockMatrix(hx, ao, px)) < 1e-9, "X2C Fock from AO Cholesky vectors matches the spin-orbital-tensor Fock");

  // ---- MO vectors: NON_REL spin-orbital ----
  const std::size_t nmo = 5;
  Matrix<double> cr(n, nmo, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < nmo; ++j) cr(i, j) = rng.next();
  const CholeskyEri<double> mo_so = aoCholeskyToMoSpinOrbital(ao, cr);
  check(mo_so.dim0() == 2 * nmo, "NON_REL MO vectors have spin-orbital dimension");
  double worst = 0;
  auto so_spin = [&](std::size_t i) { return i / nmo; };
  for (std::size_t a = 0; a < 2 * nmo; ++a) for (std::size_t b = 0; b < 2 * nmo; ++b) for (std::size_t c = 0; c < 2 * nmo; ++c) for (std::size_t d = 0; d < 2 * nmo; ++d) {
    double ref = 0;
    if (so_spin(a) == so_spin(c) && so_spin(b) == so_spin(d)) {
      const std::size_t pa = a % nmo, pb = b % nmo, pc = c % nmo, pd = d % nmo;
      for (std::size_t m = 0; m < n; ++m) for (std::size_t nn = 0; nn < n; ++nn) for (std::size_t l = 0; l < n; ++l) for (std::size_t s = 0; s < n; ++s)
        ref += cr(m, pa) * cr(nn, pc) * cr(l, pb) * cr(s, pd) * ao_int(m, nn, l, s);   // <pa pb|pc pd> = (pa pc|pb pd)
    }
    worst = std::max(worst, std::abs(mo_so(a, b, c, d) - ref));
  }
  check(worst < 1e-9, "NON_REL spin-orbital MO vectors reproduce the dense MO transform, max err " + std::to_string(worst));

  // ---- MO vectors: X2C spinors ----
  Matrix<C> cc(n2, nmo, C{});
  for (std::size_t i = 0; i < n2; ++i) for (std::size_t j = 0; j < nmo; ++j) cc(i, j) = C(rng.next(), rng.next());
  const CholeskyEri<C> mo_sp = aoCholeskyToMoSpinor(ao, cc);
  worst = 0;
  for (std::size_t a = 0; a < nmo; ++a) for (std::size_t b = 0; b < nmo; ++b) for (std::size_t c = 0; c < nmo; ++c) for (std::size_t d = 0; d < nmo; ++d) {
    C ref{};   // <ab|cd> = sum conj(C_Aa) conj(C_Bb) C_Cc C_Dd <AB|CD>
    for (std::size_t A = 0; A < n2; ++A) for (std::size_t Cc = 0; Cc < n2; ++Cc) {
      if (spin(A) != spin(Cc)) continue;
      for (std::size_t B = 0; B < n2; ++B) for (std::size_t D = 0; D < n2; ++D) {
        if (spin(B) != spin(D)) continue;
        ref += std::conj(cc(A, a)) * std::conj(cc(B, b)) * cc(Cc, c) * cc(D, d) * eri_spin(A, B, Cc, D);
      }
    }
    worst = std::max(worst, std::abs(mo_sp(a, b, c, d) - ref));
  }
  check(worst < 1e-9, "X2C spinor MO vectors reproduce the dense MO transform, max err " + std::to_string(worst));

  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
