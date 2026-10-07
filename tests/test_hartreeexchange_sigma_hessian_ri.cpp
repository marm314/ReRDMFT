// Validates Hessian_opt/HartreeExchangeSigmaHessianVectorRi.h (RI-native, B/K(v)-only, covers
// PNOF+pCCD) against Hessian_opt/HartreeExchangeSigmaHessianVector.h (the already-validated
// generic form) fed the SAME eri=B^T B, with a non-trivial pair_of so the L1/L2 RI-direct
// reduction is actually exercised. See [[project-ri-hessian-neo-design]] Stage 2d.
//
// Build/run: make test_hartreeexchange_sigma_hessian_ri LIBCINT=/path/to/libcint.a && ./build/test_hartreeexchange_sigma_hessian_ri
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "HartreeExchangeSigmaHessianVector.h"
#include "HartreeExchangeSigmaHessianVectorRi.h"
#include "Matrix.h"
#include "Tensor4.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 864213579ULL;
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
  const std::size_t n = 8;
  const std::size_t n_aux = 11;

  Matrix<double> b(n_aux, n * n);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) b(P, p * n + q) = rng.next();

  Tensor4<double> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t bb = 0; bb < n; ++bb)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          double acc = 0.0;
          for (std::size_t P = 0; P < n_aux; ++P) acc += b(P, a * n + c) * b(P, bb * n + d);
          eri(a, bb, c, d) = acc;
        }

  Matrix<double> h(n, n), fock(n, n), two_rdm_h(n, n), two_rdm_x(n, n), two_rdm_l1(n, n), two_rdm_l2(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      h(i, j) = rng.next();
      fock(i, j) = rng.next();
      two_rdm_h(i, j) = rng.next();
      two_rdm_x(i, j) = rng.next();
      two_rdm_l1(i, j) = rng.next();
      two_rdm_l2(i, j) = rng.next();
    }

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  std::vector<std::size_t> pair_of(n);
  for (std::size_t i = 0; i < n; i += 2) {
    pair_of[i] = i + 1;
    pair_of[i + 1] = i;
  }

  const auto pairs = lowerPairsLocal(n);
  std::vector<double> v(pairs.size());
  for (std::size_t i = 0; i < pairs.size(); ++i) v[i] = rng.next();

  const std::vector<double> w_generic =
      hartreeExchangeSigmaHessianVector(h, eri, occ, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1,
                                         two_rdm_l2, pairs, v);
  const std::vector<double> w_ri = hartreeExchangeSigmaHessianVectorRi(
      h, b, n, occ, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1, two_rdm_l2, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(w_generic[i] - w_ri[i]));
    max_ref = std::max(max_ref, std::abs(w_generic[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_aux = " << n_aux << ", n_pairs = " << pairs.size()
            << " (pair_of = adjacent involution, L1/L2 exercised)\n";
  std::cout << "max |w_generic| = " << max_ref << "\n";
  std::cout << "max |w_generic - w_ri_direct| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " RI-direct (B/K-only) form matches the dense-eri generic form\n";
  return pass ? 0 : 1;
}
