// Unit test of Utils/SymmetricEri.h: the unique-element store reproduces a dense tensor that has the
// exchange/Hermitian (complex) or full 8-fold (real) symmetry, for every element, and stores ~1/4 (1/8).
// Build/run: make test_symmetric_eri
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>

#include "SymmetricEri.h"
#include "Tensor4.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {
int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; } }
struct Rng { std::uint64_t s = 424242ULL; double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; } };
double absOf(double x) { return std::abs(x); }
double absOf(const C& x) { return std::abs(x); }
double rnd(Rng& r, double*) { return r.next(); }
C rnd(Rng& r, C*) { const double a = r.next(); return C(a, r.next()); }

// Random tensor with exactly the symmetries the container assumes.
template <typename T>
Tensor4<T> symmetricRandom(std::size_t n, Rng& rng) {
  Tensor4<T> raw(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) raw(a, b, c, d) = rnd(rng, static_cast<T*>(nullptr));
  Tensor4<T> t(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    if constexpr (std::is_same_v<T, double>) {
      // the 8 permutations of (ac|bd) written in physics indices
      t(a, b, c, d) = (raw(a, b, c, d) + raw(c, b, a, d) + raw(a, d, c, b) + raw(c, d, a, b) + raw(b, a, d, c) + raw(b, c, d, a) + raw(d, a, b, c) + raw(d, c, b, a)) / 8.0;
    } else {
      t(a, b, c, d) = (raw(a, b, c, d) + raw(b, a, d, c) + std::conj(raw(c, d, a, b)) + std::conj(raw(d, c, b, a))) / 4.0;
    }
  }
  return t;
}

template <typename T>
void testType(const std::string& label, std::size_t n) {
  Rng rng;
  const Tensor4<T> dense = symmetricRandom<T>(n, rng);
  SymmetricEri<T> sym(n);
  // Fill from ALL elements in scrambled order: every write must agree with its symmetry partners.
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) sym.set(a, b, c, d, dense(a, b, c, d));
  double worst = 0.0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) worst = std::max(worst, absOf(sym(a, b, c, d) - dense(a, b, c, d)));
  check(worst < 1e-15, label + ": n=" + std::to_string(n) + " reproduces every element of the dense tensor (max err " + std::to_string(worst) + ")");
  // Filling from ONE representative per orbit only must already reproduce everything.
  SymmetricEri<T> sparse(n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) if (a <= c) sparse.set(a, b, c, d, dense(a, b, c, d));
  double worst2 = 0.0;
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) worst2 = std::max(worst2, absOf(sparse(a, b, c, d) - dense(a, b, c, d)));
  check(worst2 < 1e-15, label + ": setting only the a<=c elements rebuilds the rest by symmetry (max err " + std::to_string(worst2) + ")");
  const double ratio = static_cast<double>(sym.storedCount()) / static_cast<double>(SymmetricEri<T>::denseCount(n));
  const double limit = (std::is_same_v<T, double> ? 0.125 : 0.25) * (1.0 + 3.0 / static_cast<double>(n)) + 1e-12;
  std::cout << "  " << label << " n=" << n << ": stored " << sym.storedCount() << " of " << SymmetricEri<T>::denseCount(n) << " (" << ratio << ")\n";
  if (n >= 5) check(ratio <= limit, label + ": stores about 1/" + (std::is_same_v<T, double> ? "8" : "4") + " of the dense count");
}
}  // namespace

int main() {
  for (const std::size_t n : {1, 2, 3, 5, 8, 13}) { testType<double>("real", n); testType<C>("complex", n); }
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
