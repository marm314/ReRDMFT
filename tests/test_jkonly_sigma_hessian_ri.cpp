// Validates Hessian_opt/JkOnlySigmaHessianVectorRi.h (the RI-NATIVE, B/K(v)-only form, built
// ONLY from a toy RI tensor `b`, never a dense eri) against Hessian_opt/JkOnlySigmaHessianVector.h
// (the already-compiled-validated generic form, Stage 1c) fed the SAME `eri = B^T B` (an exact,
// if rank-deficient, factorization -- not an approximation here, just a different representation
// of the identical tensor), per [[project-ri-hessian-neo-design]]'s Stage 1d.
//
// Build/run: make test_jkonly_sigma_hessian_ri LIBCINT=/path/to/libcint.a && ./build/test_jkonly_sigma_hessian_ri
#include <cmath>
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
  std::uint64_t s = 1357924680ULL;
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
  const std::size_t n = 6;
  const std::size_t n_aux = 9;  // deliberately rank-deficient (n_aux < n^2=36)

  Matrix<double> b(n_aux, n * n);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) b(P, p * n + q) = rng.next();

  // Dense eri = B^T B, same convention RiMoEri/RiNonRelSpinMoEri use: eri(A,B,C,D) =
  // sum_P B(P,A,C)*B(P,B,D). An EXACT reconstruction (not an approximation) of this specific eri.
  Tensor4<double> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t bb = 0; bb < n; ++bb)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          double acc = 0.0;
          for (std::size_t P = 0; P < n_aux; ++P) acc += b(P, a * n + c) * b(P, bb * n + d);
          eri(a, bb, c, d) = acc;
        }

  Matrix<double> h(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) h(i, j) = rng.next();

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  Matrix<double> hc(n, n), xc(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      hc(i, j) = rng.next();
      xc(i, j) = rng.next();
    }

  const auto pairs = lowerPairsLocal(n);
  std::vector<double> v(pairs.size());
  for (std::size_t i = 0; i < pairs.size(); ++i) v[i] = rng.next();

  const std::vector<double> w_generic = jkOnlySigmaHessianVector(h, eri, occ, hc, xc, pairs, v);
  const std::vector<double> w_ri = jkOnlySigmaHessianVectorRi(h, b, n, occ, hc, xc, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(w_generic[i] - w_ri[i]));
    max_ref = std::max(max_ref, std::abs(w_generic[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_aux = " << n_aux << ", n_pairs = " << pairs.size() << "\n";
  std::cout << "max |w_generic| = " << max_ref << "\n";
  std::cout << "max |w_generic - w_ri_direct| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " RI-direct (B/K-only) form matches the dense-eri generic form\n";
  return pass ? 0 : 1;
}
