// Validates Hessian_opt/JkOnlySigmaHessianVectorRi.h's `jkOnlySigmaJointHessianVectorRi` (the
// COMPLEX RI-direct, B/K-only, never-dense-eri form) against `jkOnlySigmaJointHessianVector` (the
// already-validated generic complex/joint form) fed the SAME eri=B^T B, on a toy COMPLEX RI
// tensor `b` -- confirms the RI reconstruction itself needs no conjugate (unlike the fock terms
// in the HartreeExchange case). See [[project-ri-hessian-neo-design]] Stage 4.
//
// Build/run: make test_jkonly_sigma_hessian_ri_joint LIBCINT=/path/to/libcint.a && ./build/test_jkonly_sigma_hessian_ri_joint
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "JkOnlySigmaHessianVector.h"
#include "JkOnlySigmaHessianVectorRi.h"
#include "Matrix.h"
#include "Tensor4.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 55443322110ULL;
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
  using C = std::complex<double>;
  Rng rng;
  const std::size_t n = 6;
  const std::size_t n_aux = 9;

  Matrix<C> b(n_aux, n * n);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) b(P, p * n + q) = C(rng.next(), rng.next());

  Tensor4<C> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t bb = 0; bb < n; ++bb)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          C acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += b(P, a * n + c) * b(P, bb * n + d);
          eri(a, bb, c, d) = acc;
        }

  Matrix<C> h(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) h(i, j) = C(rng.next(), rng.next());

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  Matrix<double> hc(n, n), xc(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      hc(i, j) = rng.next();
      xc(i, j) = rng.next();
    }

  const auto pairs = lowerPairsLocal(n);
  std::vector<double> v(2 * pairs.size());
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = rng.next();

  const std::vector<double> w_generic = jkOnlySigmaJointHessianVector(h, eri, occ, hc, xc, pairs, v);
  const std::vector<double> w_ri = jkOnlySigmaJointHessianVectorRi(h, b, n, occ, hc, xc, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < w_generic.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(w_generic[i] - w_ri[i]));
    max_ref = std::max(max_ref, std::abs(w_generic[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_aux = " << n_aux << ", n_pairs = " << pairs.size() << " (complex RI-direct joint)\n";
  std::cout << "max |w_generic| = " << max_ref << "\n";
  std::cout << "max |w_generic - w_ri_direct| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " RI-direct complex/joint form matches the generic complex/joint form\n";
  return pass ? 0 : 1;
}
