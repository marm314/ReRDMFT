// Validates Hessian_opt/JkOnlySigmaHessianVector.h's `jkOnlySigmaHessianVector` (Stage 1c of
// [[project-ri-hessian-neo-design]]: a resummed, non-element-by-element way to compute the real
// (T=double) JK_only orbital-rotation Hessian-vector product) against the EXISTING, already
// trusted `jkOnlyHessianElement`-based formula (Hessian_opt/JkOnlyHessian.h), replicating
// Full_opt/FullOptimization.cpp's own `jkOnlyHessianVectorImpl` double loop exactly (same
// symmetrization) as the reference. Two independently-written implementations of the SAME
// mathematical object must agree to float precision if both are correct -- this is the decisive
// check (prior Python-only validation used a from-scratch Python replica of rawJkOnlyG, never the
// actual C++ formula; this test closes that gap by calling the real, compiled, production
// `jkOnlyHessianElement` directly).
//
// Build/run: make test_jkonly_sigma_hessian LIBCINT=/path/to/libcint.a && ./build/test_jkonly_sigma_hessian
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "JkOnlyHessian.h"
#include "JkOnlySigmaHessianVector.h"
#include "Matrix.h"
#include "Tensor4.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 24681357ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

std::vector<std::pair<std::size_t, std::size_t>> lowerPairsLocal(std::size_t n) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < p; ++q) pairs.emplace_back(p, q);
  return pairs;
}

}  // namespace

int main() {
  Rng rng;
  const std::size_t n = 7;  // small, deliberately not matching any real molecule -- pure formula check

  Matrix<double> h(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) h(i, j) = rng.next();

  Tensor4<double> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) eri(a, b, c, d) = rng.next();

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  Matrix<double> hc(n, n), xc(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      hc(i, j) = rng.next();
      xc(i, j) = rng.next();
    }

  const auto pairs = lowerPairsLocal(n);
  const std::size_t npairs = pairs.size();
  std::vector<double> v(npairs);
  for (std::size_t i = 0; i < npairs; ++i) v[i] = rng.next();

  // Reference: the EXISTING, trusted production formula (same symmetrization
  // Full_opt/FullOptimization.cpp's jkOnlyHessianVectorImpl<double,...> performs).
  std::vector<double> w_reference(npairs, 0.0);
  for (std::size_t i = 0; i < npairs; ++i) {
    const auto& [p, q] = pairs[i];
    double acc = 0.0;
    for (std::size_t j = 0; j < npairs; ++j) {
      const auto& [r, s] = pairs[j];
      const double e_ij = jkOnlyHessianElement(h, eri, occ, hc, xc, p, q, r, s);
      const double e_ji = jkOnlyHessianElement(h, eri, occ, hc, xc, r, s, p, q);
      acc += 0.5 * (e_ij + e_ji) * v[j];
    }
    w_reference[i] = acc;
  }

  // Candidate: the new resummed (Stage 1c) formula.
  const std::vector<double> w_sigma = jkOnlySigmaHessianVector(h, eri, occ, hc, xc, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < npairs; ++i) {
    max_diff = std::max(max_diff, std::abs(w_reference[i] - w_sigma[i]));
    max_ref = std::max(max_ref, std::abs(w_reference[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_pairs = " << npairs << "\n";
  std::cout << "max |w_reference| = " << max_ref << "\n";
  std::cout << "max |w_reference - w_sigma| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " jkOnlySigmaHessianVector matches the trusted element-by-element formula\n";
  return pass ? 0 : 1;
}
