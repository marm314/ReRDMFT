// Unit test of Hessian_opt/PccdFock.h's buildPccdFullTwoRdm: the energy recomputed from
// the UNFOLDED, full-orbital-indexed two_rdm_h/x/l1/l2 matrices via
// HartreeExchangeGradient.h's generic hartreeExchangeEnergy must match
// Occ_opt/pCCD.h's own pccdReferenceEnergy + pccdCorrelationEnergy EXACTLY -- the
// independent cross-check this unfolding (a brand-new derivation, not ported from
// anywhere) needs before it is trusted to drive ADAM/NEO orbital optimization.
// Synthetic NON_REL spin-orbital data: 8 spatial orbitals, interleaved spin-orbitals
// (2k=alpha_k, 2k+1=beta_k, pair_of=i^1), exactly tests/test_pnof_occupation_gradient.cpp's
// own construction -- guarantees the opposite-Kramers exchange K_{p,pbar} is genuinely
// (not just assumed) zero, via real spin selection rather than hand-picked integrals.
// One frozen-core pair, one active-occupied pair, three active-virtual pairs, and two
// untouched deep-virtual pairs -- exercises the core/deep-virtual extensions too.
// Build/run: make test_pccd_fock LIBCINT=/path/to/libcint.a && ./build/test_pccd_fock
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "HartreeExchangeGradient.h"
#include "Matrix.h"
#include "PccdFock.h"
#include "Tensor4.h"
#include "pCCD.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 987654321ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

}  // namespace

int main() {
  const std::size_t ns = 8, n = 2 * ns;
  Rng rng;
  Matrix<double> hs(ns, ns, 0.0);
  for (std::size_t i = 0; i < ns; ++i)
    for (std::size_t j = i; j < ns; ++j) hs(i, j) = hs(j, i) = rng.next();
  std::vector<Matrix<double>> vec(8, Matrix<double>(ns, ns, 0.0));
  for (auto& v : vec)
    for (std::size_t i = 0; i < ns; ++i)
      for (std::size_t j = i; j < ns; ++j) v(i, j) = v(j, i) = 0.4 * rng.next();
  Matrix<double> h(n, n, 0.0);
  Tensor4<double> eri(n, n, n, n, 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      if (p % 2 == q % 2) h(p, q) = hs(p / 2, q / 2);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          if (a % 2 != c % 2 || b % 2 != d % 2) continue;
          double sum = 0.0;
          for (const auto& v : vec) sum += v(a / 2, c / 2) * v(b / 2, d / 2);
          eri(a, b, c, d) = sum;
        }

  // Pairs: geminal 0 (indices 0,1) = frozen core; geminal 1 (2,3) = active occupied;
  // geminals 2,3,4 (4/5, 6/7, 8/9) = active virtual; geminals 5,6 (10/11, 12/13) and the
  // last spatial orbital's pair (14/15) are left as untouched deep virtual.
  const std::size_t n_core = 1, n_occ = 1, n_vir = 3;
  const std::vector<std::size_t> reps = {0, 2, 4, 6, 8};
  const std::vector<std::size_t> bar = {1, 3, 5, 7, 9};

  const auto coeff = buildPccdCoefficients(h, eri, reps, bar, n_core, n_occ, n_vir);

  PccdSettings settings;
  settings.solver = PccdAmplitudeSolver::kNewton;
  settings.max_iterations = 500;
  settings.residual_tolerance = 1e-12;
  Matrix<double> t0(n_occ, n_vir, 0.0);
  const auto t_result = solvePccdTAmplitudes(coeff, t0, settings);
  Matrix<double> z0(n_occ, n_vir, 0.0);
  const auto z_result = solvePccdZAmplitudes(coeff, t_result.t, z0, settings);
  if (!t_result.t_converged || !z_result.z_converged) {
    std::cout << "FAIL: amplitude solve did not converge\n";
    return 1;
  }
  const PccdRdm rdm = buildPccdRdm(t_result.t, z_result.z);

  const double e_ref = pccdReferenceEnergy(coeff);
  const double e_corr = pccdCorrelationEnergy(coeff, t_result.t);
  const double e_pccd = e_ref + e_corr;

  std::vector<double> occupations(n, 0.0);
  occupations[0] = occupations[1] = 1.0;
  occupations[2] = occupations[3] = rdm.n_occ[0];
  for (std::size_t a = 0; a < n_vir; ++a) {
    occupations[4 + 2 * a] = occupations[4 + 2 * a + 1] = rdm.n_vir[a];
  }

  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, n);
  const auto pair_of = buildPccdPairOf(reps, bar, n);
  const double e_full = hartreeExchangeEnergy(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                               pair_of, full.two_rdm_l1, full.two_rdm_l2);
  const double e_no_pair = hartreeExchangeEnergy(h, eri, occupations, full.two_rdm_h,
                                                  full.two_rdm_x, {}, Matrix<double>(), Matrix<double>());
  const double pair_energy_only = e_full - e_no_pair;
  double gd_sum = 0.0;
  const std::size_t base = coeff.n_core, n_active = n_occ + n_vir;
  for (std::size_t p = 0; p < n_active; ++p) {
    for (std::size_t q = 0; q < n_active; ++q) {
      if (p == q) continue;
      gd_sum += coeff.g(base + p, base + q) * rdm.d(p, q);
    }
  }
  std::cout << "pair_energy_only (l1/l2 contrib) = " << pair_energy_only << "\n";
  std::cout << "gd_sum (sum_{p!=q} G_pq D_pq)    = " << gd_sum << "\n";
  std::cout << "ratio pair_energy_only/gd_sum    = " << (pair_energy_only / gd_sum) << "\n";

  // Independent hand-built "all pairs" (core ++ occ ++ vir) reference, extending
  // pccdReferenceEnergy's own sum (core ++ occ only) to also include the active virtual
  // pairs with their amplitude-derived occupations -- should equal e_no_pair exactly if
  // the H/X unfolding (Q_pq) is correct, independent of any core/active bookkeeping.
  std::vector<double> n_all(n_active + n_core, 0.0);
  n_all[0] = 1.0;
  n_all[1] = rdm.n_occ[0];
  for (std::size_t a = 0; a < n_vir; ++a) n_all[2 + a] = rdm.n_vir[a];
  double e_ref_all = 0.0;
  for (std::size_t p = 0; p < n_core + n_active; ++p) e_ref_all += coeff.eps[p] * n_all[p];
  for (std::size_t p = 0; p < n_core + n_active; ++p) {
    for (std::size_t q = 0; q < n_core + n_active; ++q) {
      if (p == q) continue;
      const bool p_active = p >= n_core, q_active = q >= n_core;
      const double w_pq = coeff.w(p, q);
      const double q_pq = (p_active && q_active) ? rdm.q(p - n_core, q - n_core) : n_all[p] * n_all[q];
      e_ref_all += 0.5 * w_pq * q_pq;
    }
  }
  std::cout << "e_ref_all (hand-built, core+occ+vir) = " << e_ref_all << "\n";
  std::cout << "e_no_pair (H/X only via PccdFock)    = " << e_no_pair << "\n";
  std::cout << "diff (e_no_pair - e_ref_all)          = " << (e_no_pair - e_ref_all) << "\n";

  std::cout << std::setprecision(12);
  std::cout << "E_pCCD (ref+corr)      = " << e_pccd << "\n";
  std::cout << "E_full (unfolded RDM)  = " << e_full << "\n";
  std::cout << "diff                   = " << std::scientific << (e_full - e_pccd) << "\n";

  const bool ok = std::abs(e_full - e_pccd) < 1e-8;
  std::cout << (ok ? "ok" : "FAIL") << ": buildPccdFullTwoRdm energy matches pCCD reference\n";
  return ok ? 0 : 1;
}
