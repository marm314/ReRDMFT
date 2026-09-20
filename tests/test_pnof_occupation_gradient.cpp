// Unit test of Occ_opt/PNOFs.h's pnofOccupationGradient against central
// finite differences of pnofElectronicEnergy, for every PNOF functional,
// real (NON_REL) and relativistic (L_ij) form, at occupations covering
// principal occupations from 0.5 to 1 - 1e-5 (GNOF's Pi^inter depends on
// the principal occupation of each subspace, also for the principal
// geminal itself). Synthetic random h/eri: the energy/gradient only read
// eri(i,i,i,i), (i,j,i,j), (i,j,j,i), (ibar,j,j,ibar).
// Build/run: make test_pnof_gradient LIBCINT=/path/to/libcint.a
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Matrix.h"
#include "Orb_subspaces.h"
#include "PNOFs.h"
#include "Tensor4.h"

using namespace rerdmft;

namespace {

int g_failures = 0, g_checks = 0;

struct Rng {
  std::uint64_t s = 42424242ULL;
  double next() {  // [-0.5, 0.5)
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

void run(PnofFunctional functional, const std::string& name, bool relativistic) {
  // Closed-shell spin-orbitals from 8 real spatial orbitals, interleaved (2k = alpha_k, 2k+1 =
  // beta_k, pair_of = i^1): h and <ab|cd> are spin-diagonal (spin(a)=spin(c), spin(b)=spin(d))
  // with real 8-fold-symmetric spatial integrals (pq|rs) = sum_L V_L(pq) V_L(rs), physics
  // <ab|cd> = (ac|bd) -- the pair symmetry and integral symmetry the energy/gradient assume.
  const std::size_t ns = 8, n = 2 * ns;
  Rng rng;
  Matrix<double> hs(ns, ns, 0.0);
  for (std::size_t i = 0; i < ns; ++i) for (std::size_t j = i; j < ns; ++j) hs(i, j) = hs(j, i) = rng.next();
  std::vector<Matrix<double>> vec(8, Matrix<double>(ns, ns, 0.0));
  for (auto& v : vec) for (std::size_t i = 0; i < ns; ++i) for (std::size_t j = i; j < ns; ++j) v(i, j) = v(j, i) = 0.4 * rng.next();
  Matrix<double> h(n, n, 0.0);
  Tensor4<double> eri(n, n, n, n, 0.0);
  for (std::size_t p = 0; p < n; ++p) for (std::size_t q = 0; q < n; ++q) if (p % 2 == q % 2) h(p, q) = hs(p / 2, q / 2);
  for (std::size_t a = 0; a < n; ++a) for (std::size_t b = 0; b < n; ++b) for (std::size_t c = 0; c < n; ++c) for (std::size_t d = 0; d < n; ++d) {
    if (a % 2 != c % 2 || b % 2 != d % 2) continue;
    double sum = 0.0;
    for (const auto& v : vec) sum += v(a / 2, c / 2) * v(b / 2, d / 2);
    eri(a, b, c, d) = sum;
  }

  std::vector<std::size_t> pair_of(n);
  for (std::size_t p = 0; p < n; ++p) pair_of[p] = p ^ 1;
  const auto table = buildOrbitalSubspaces(pair_of, n, 4.0, 2, 3);
  const auto geminals = buildPnofGeminals(table);
  const std::size_t n_core = table.frozen_occupied.size();

  for (double principal : {0.5, 0.9, 0.9972, 0.99999}) {
    // Per subspace: principal + two virtuals sharing the rest 1 : 2.
    std::vector<double> occ(n, 0.0);
    for (std::size_t a = 0; a < n_core; ++a) occ[geminals[a].i] = occ[geminals[a].ibar] = 1.0;
    for (std::size_t a = n_core; a < geminals.size(); ++a) {
      double v;
      if (geminals[a].is_principal) v = principal;
      else { const bool first = (a > 0 && geminals[a - 1].subspace_id == geminals[a].subspace_id && geminals[a - 1].is_principal); v = (1.0 - principal) * (first ? 1.0 / 3.0 : 2.0 / 3.0); }
      occ[geminals[a].i] = occ[geminals[a].ibar] = v;
    }
    const auto analytic = pnofOccupationGradient(functional, h, eri, occ, geminals, relativistic);
    auto energy = [&](const std::vector<double>& o) {
      const auto t = buildPnofTwoRdm(functional, geminals, o, relativistic);
      return pnofElectronicEnergy(functional, h, eri, o, geminals, t, relativistic);
    };
    double worst = 0.0;
    std::size_t worst_a = 0;
    for (std::size_t a = n_core; a < geminals.size(); ++a) {
      const double na = occ[geminals[a].i];
      const double step = 1e-3 * std::min(na, 1.0 - na);
      auto op = occ, om = occ;
      op[geminals[a].i] += step; op[geminals[a].ibar] += step;
      om[geminals[a].i] -= step; om[geminals[a].ibar] -= step;
      const double fd = (energy(op) - energy(om)) / (2.0 * step);
      const double err = std::abs(analytic[a] - fd) / (1.0 + std::abs(fd));
      if (err > worst) { worst = err; worst_a = a; }
    }
    ++g_checks;
    const bool ok = worst < 1e-5;
    if (!ok) ++g_failures;
    std::cout << "  " << (ok ? "[PASS] " : "[FAIL] ") << name << (relativistic ? " (relativistic)" : " (NON_REL)")
              << ", principal n = " << principal << ": max relative |analytic - FD| = " << std::scientific << std::setprecision(2)
              << worst << " (geminal " << worst_a << ")" << std::defaultfloat << "\n";
  }
}

}  // namespace

int main() {
  for (bool rel : {false, true}) {
    run(PnofFunctional::kPnof5, "PNOF5", rel);
    run(PnofFunctional::kPnof7, "PNOF7", rel);
    run(PnofFunctional::kPnof7s, "PNOF7S", rel);
    run(PnofFunctional::kGnof, "GNOF", rel);
  }
  std::cout << "\n" << g_checks - g_failures << " / " << g_checks << " checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
