// Validates Hessian_opt/HartreeExchangeSigmaHessianVector.h's `hartreeExchangeSigmaHessianVector`
// ([[project-ri-hessian-neo-design]] Stage 2 -- covers PNOF and pCCD at once, since both are thin
// wrappers around this exact formula) against the EXISTING, already trusted
// `hartreeExchangeHessianElement`-based formula (Hessian_opt/HartreeExchangeHessian.h), replicating
// the same symmetrization Full_opt/FullOptimization.cpp's `pnofHessianVectorImpl`/
// `pccdHessianVectorImpl` perform. Uses a NON-TRIVIAL `pair_of` (adjacent-pair involution) and
// nonzero two_rdm_l1/l2 so the L1/L2 term (the part flagged as riskiest in project memory) is
// actually exercised, not skipped.
//
// Build/run: make test_hartreeexchange_sigma_hessian LIBCINT=/path/to/libcint.a && ./build/test_hartreeexchange_sigma_hessian
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "HartreeExchangeHessian.h"
#include "HartreeExchangeSigmaHessianVector.h"
#include "Matrix.h"
#include "Tensor4.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 97531246801ULL;
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
  const std::size_t n = 8;  // even, for a clean adjacent-pair involution

  Matrix<double> h(n, n), fock(n, n), two_rdm_h(n, n), two_rdm_x(n, n), two_rdm_l1(n, n),
      two_rdm_l2(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      h(i, j) = rng.next();
      fock(i, j) = rng.next();
      two_rdm_h(i, j) = rng.next();
      two_rdm_x(i, j) = rng.next();
      two_rdm_l1(i, j) = rng.next();
      two_rdm_l2(i, j) = rng.next();
    }

  Tensor4<double> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) eri(a, b, c, d) = rng.next();

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  std::vector<std::size_t> pair_of(n);
  for (std::size_t i = 0; i < n; i += 2) {
    pair_of[i] = i + 1;
    pair_of[i + 1] = i;
  }

  const auto pairs = lowerPairsLocal(n);
  const std::size_t npairs = pairs.size();
  std::vector<double> v(npairs);
  for (std::size_t i = 0; i < npairs; ++i) v[i] = rng.next();

  // Reference: the EXISTING, trusted production formula, L1/L2 term included.
  std::vector<double> w_reference(npairs, 0.0);
  for (std::size_t i = 0; i < npairs; ++i) {
    const auto& [p, q] = pairs[i];
    double acc = 0.0;
    for (std::size_t j = 0; j < npairs; ++j) {
      const auto& [r, s] = pairs[j];
      const double e_ij = hartreeExchangeHessianElement(h, eri, occ, two_rdm_h, two_rdm_x, fock, p, q,
                                                         r, s, pair_of, two_rdm_l1, two_rdm_l2);
      const double e_ji = hartreeExchangeHessianElement(h, eri, occ, two_rdm_h, two_rdm_x, fock, r, s,
                                                         p, q, pair_of, two_rdm_l1, two_rdm_l2);
      acc += 0.5 * (e_ij + e_ji) * v[j];
    }
    w_reference[i] = acc;
  }

  const std::vector<double> w_sigma = hartreeExchangeSigmaHessianVector(
      h, eri, occ, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1, two_rdm_l2, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < npairs; ++i) {
    max_diff = std::max(max_diff, std::abs(w_reference[i] - w_sigma[i]));
    max_ref = std::max(max_ref, std::abs(w_reference[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_pairs = " << npairs << " (pair_of = adjacent involution, L1/L2 exercised)\n";
  std::cout << "max |w_reference| = " << max_ref << "\n";
  std::cout << "max |w_reference - w_sigma| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " hartreeExchangeSigmaHessianVector matches the trusted element-by-element formula\n";
  return pass ? 0 : 1;
}
