// Validates Hessian_opt/HartreeExchangeSigmaHessianVector.h's
// `hartreeExchangeSigmaJointHessianVector` (complex/joint [t;y], covers PNOF+pCCD) against the
// existing trusted `hartreeExchangeJointHessianVector`, with a non-trivial pair_of so L1/L2 is
// exercised. See [[project-ri-hessian-neo-design]] Stage 3.
//
// Build/run: make test_hartreeexchange_sigma_hessian_joint LIBCINT=/path/to/libcint.a && ./build/test_hartreeexchange_sigma_hessian_joint
#include <cmath>
#include <complex>
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
  std::uint64_t s = 998877665544ULL;
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
  const std::size_t n = 8;

  Matrix<C> h(n, n), fock(n, n);
  Matrix<double> two_rdm_h(n, n), two_rdm_x(n, n), two_rdm_l1(n, n), two_rdm_l2(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      h(i, j) = C(rng.next(), rng.next());
      fock(i, j) = C(rng.next(), rng.next());
      two_rdm_h(i, j) = rng.next();
      two_rdm_x(i, j) = rng.next();
      two_rdm_l1(i, j) = rng.next();
      two_rdm_l2(i, j) = rng.next();
    }

  Tensor4<C> eri(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) eri(a, b, c, d) = C(rng.next(), rng.next());

  std::vector<double> occ(n);
  for (std::size_t i = 0; i < n; ++i) occ[i] = 0.1 + 0.8 * (rng.next() + 0.5);

  std::vector<std::size_t> pair_of(n);
  for (std::size_t i = 0; i < n; i += 2) {
    pair_of[i] = i + 1;
    pair_of[i + 1] = i;
  }

  const auto pairs = lowerPairsLocal(n);
  std::vector<double> v(2 * pairs.size());
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = rng.next();

  const std::vector<double> w_production = hartreeExchangeJointHessianVector(
      h, eri, occ, two_rdm_h, two_rdm_x, fock, pairs, v, pair_of, two_rdm_l1, two_rdm_l2);
  const std::vector<double> w_sigma = hartreeExchangeSigmaJointHessianVector(
      h, eri, occ, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1, two_rdm_l2, pairs, v);

  double max_diff = 0.0, max_ref = 0.0;
  for (std::size_t i = 0; i < w_production.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(w_production[i] - w_sigma[i]));
    max_ref = std::max(max_ref, std::abs(w_production[i]));
  }

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "n = " << n << ", n_pairs = " << pairs.size()
            << " (complex/joint [t;y], pair_of = adjacent involution, L1/L2 exercised)\n";
  std::cout << "max |w_production| = " << max_ref << "\n";
  std::cout << "max |w_production - w_sigma| = " << max_diff << "\n";

  const bool pass = max_diff < 1e-9 * std::max(1.0, max_ref);
  std::cout << (pass ? "[PASS]" : "[FAIL]")
            << " hartreeExchangeSigmaJointHessianVector matches the trusted joint formula\n";
  return pass ? 0 : 1;
}
