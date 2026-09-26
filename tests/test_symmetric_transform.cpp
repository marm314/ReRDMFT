// Unit test of Utils/SymmetricTransform.h: the slab four-index transform into a SymmetricEri against a
// brute-force dense transform, for real and complex coefficients, real and complex sources.
// Build/run: make test_symmetric_transform
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>

#include "Matrix.h"
#include "SymmetricEri.h"
#include "SymmetricTransform.h"
#include "Tensor4.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {
int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; } }
struct Rng { std::uint64_t s = 99ULL; double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5; } };
double rnd(Rng& r, double*) { return r.next(); }
C rnd(Rng& r, C*) { const double a = r.next(); return C(a, r.next()); }
double cj(double x) { return x; }
C cj(const C& x) { return std::conj(x); }

template <typename S>
Tensor4<S> symmetricRandom(std::size_t n, Rng& rng) {
  Tensor4<S> raw(n, n, n, n), t(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) raw(a, b, c, d) = rnd(rng, static_cast<S*>(nullptr));
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    if constexpr (std::is_same_v<S, double>)
      t(a, b, c, d) = (raw(a, b, c, d) + raw(c, b, a, d) + raw(a, d, c, b) + raw(c, d, a, b) + raw(b, a, d, c) + raw(b, c, d, a) + raw(d, a, b, c) + raw(d, c, b, a)) / 8.0;
    else
      t(a, b, c, d) = (raw(a, b, c, d) + raw(b, a, d, c) + std::conj(raw(c, d, a, b)) + std::conj(raw(d, c, b, a))) / 4.0;
  }
  return t;
}

template <typename S, typename T>
void testCase(const std::string& label, std::size_t n, std::size_t nmo) {
  Rng rng;
  const Tensor4<S> dense = symmetricRandom<S>(n, rng);
  SymmetricEri<S> src(n);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) src.set(a, b, c, d, dense(a, b, c, d));
  Matrix<T> cm(n, nmo);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < nmo; ++j) cm(i, j) = rnd(rng, static_cast<T*>(nullptr));
  const SymmetricEri<T> out = transformToSymmetric<T>(src, n, cm);
  double worst = 0.0, scale = 0.0;
  for (std::size_t p = 0; p < nmo; ++p) for (std::size_t q = 0; q < nmo; ++q) for (std::size_t r = 0; r < nmo; ++r) for (std::size_t s = 0; s < nmo; ++s) {
    T ref{};
    for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d)
      ref += cj(cm(a, p)) * cj(cm(b, q)) * cm(c, r) * cm(d, s) * T(dense(a, b, c, d));
    worst = std::max(worst, std::abs(std::complex<double>(out(p, q, r, s) - ref)));
    scale = std::max(scale, std::abs(std::complex<double>(ref)));
  }
  check(worst <= 1e-12 * std::max(1.0, scale), label + ": n=" + std::to_string(n) + " nmo=" + std::to_string(nmo) + " matches the brute-force transform, max err " + std::to_string(worst));
}
}  // namespace

int main() {
  for (const auto& [n, nmo] : {std::pair<std::size_t, std::size_t>{4, 4}, {5, 3}, {6, 6}, {3, 19}}) {
    testCase<double, double>("real src, real C", n, nmo);
    testCase<double, C>("real src, complex C", n, nmo);
    testCase<C, C>("complex src, complex C", n, nmo);
  }
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
