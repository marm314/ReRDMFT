// Unit test of ElectronRepulsion.h's twoElectronShellQuartet: validates that the shell-batched
// (ni,nj,nk,nl) block, computed via ONE libcint call per shell quartet, reproduces
// twoElectronQuadruplet's own per-AO-quadruplet values exactly (to roundoff) -- the prerequisite
// correctness check before this is ever used in an integral-direct Fock build (SCF_DIRECT_4C).
// Build/run: make test_shell_quartet
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ElectronRepulsion.h"
#include "Integrals.h"
#include "MolecularBasis.h"

using namespace rerdmft;

namespace {
int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}
}  // namespace

int main() {
  // A basis with s, p, d, f shells at distinct centers, exercising every angular momentum whose
  // Cartesian redundancy (d: 6 vs 5, f: 10 vs 7) makes normalizeCartesianBasis's per-component
  // rescaling non-trivial, and genuinely different shell quartets (not all on the same center).
  std::vector<BasisFunction> basis;
  auto addShell = [&](int l, double x, double y, double z, std::vector<double> exps,
                       std::vector<double> coefs) {
    for (const CartesianExponents& c : cartesianComponents(l)) {
      BasisFunction fn;
      fn.element = "X";
      fn.x = x;
      fn.y = y;
      fn.z = z;
      fn.l = l;
      fn.cartesian = c;
      fn.exponents = exps;
      fn.coefficients = coefs;
      basis.push_back(fn);
    }
  };
  addShell(0, 0.0, 0.0, 0.0, {1.2, 0.4}, {0.5, 0.6});
  addShell(1, 0.3, -0.1, 0.2, {0.8}, {1.0});
  addShell(2, -0.2, 0.5, 0.1, {0.55, 0.2}, {0.7, 0.3});
  addShell(3, 0.1, 0.1, -0.4, {0.45}, {1.0});
  addShell(1, 1.1, 0.0, 0.0, {0.9, 0.3}, {0.4, 0.8});

  normalizeCartesianBasis(basis);
  const std::vector<ShellInfo> shells = groupIntoShells(basis);
  std::printf("n_ao = %zu, n_shells = %zu\n", basis.size(), shells.size());
  check(shells.size() == 5, "groupIntoShells finds exactly the 5 shells added");

  // Every shell-quartet combination, every entry, compared against twoElectronQuadruplet.
  double worst = 0.0;
  std::size_t n_entries_checked = 0;
  for (std::size_t a = 0; a < shells.size(); ++a)
    for (std::size_t b = 0; b < shells.size(); ++b)
      for (std::size_t c = 0; c < shells.size(); ++c)
        for (std::size_t d = 0; d < shells.size(); ++d) {
          const ShellQuartetBlock block = twoElectronShellQuartet(basis, shells[a], shells[b], shells[c], shells[d]);
          for (int i = 0; i < block.ni; ++i)
            for (int j = 0; j < block.nj; ++j)
              for (int k = 0; k < block.nk; ++k)
                for (int l = 0; l < block.nl; ++l) {
                  const BasisFunction& p = basis[shells[a].first + static_cast<std::size_t>(i)];
                  const BasisFunction& q = basis[shells[b].first + static_cast<std::size_t>(j)];
                  const BasisFunction& r = basis[shells[c].first + static_cast<std::size_t>(k)];
                  const BasisFunction& s = basis[shells[d].first + static_cast<std::size_t>(l)];
                  const double reference = twoElectronQuadruplet(p, q, r, s);
                  const double got = block(i, j, k, l);
                  worst = std::max(worst, std::abs(got - reference));
                  ++n_entries_checked;
                }
        }
  std::printf("checked %zu individual AO-quadruplet entries across all shell quartets\n", n_entries_checked);
  std::printf("max |shell-quartet - per-AO-quadruplet| = %.3e\n", worst);
  check(worst < 1e-10, "shell-quartet block reproduces twoElectronQuadruplet on every entry");

  // Speedup sanity check: the d-d-d-d quartet (6^4 = 1296 AO combinations) via one shell call vs
  // 1296 separate twoElectronQuadruplet calls.
  const ShellInfo& d_shell = shells[2];
  const auto t0 = std::chrono::steady_clock::now();
  const ShellQuartetBlock dddd = twoElectronShellQuartet(basis, d_shell, d_shell, d_shell, d_shell);
  const auto t1 = std::chrono::steady_clock::now();
  double sum_direct = 0.0;
  for (std::size_t i = 0; i < d_shell.count; ++i)
    for (std::size_t j = 0; j < d_shell.count; ++j)
      for (std::size_t k = 0; k < d_shell.count; ++k)
        for (std::size_t l = 0; l < d_shell.count; ++l)
          sum_direct += twoElectronQuadruplet(basis[d_shell.first + i], basis[d_shell.first + j],
                                               basis[d_shell.first + k], basis[d_shell.first + l]);
  const auto t2 = std::chrono::steady_clock::now();
  const double shell_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double direct_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::printf("d-d-d-d quartet (6^4=1296 entries): shell-batched %.3f ms, per-AO-quadruplet %.3f ms (%.1fx)\n",
              shell_ms, direct_ms, direct_ms / std::max(shell_ms, 1e-9));
  double sum_block = 0.0;
  for (double v : dddd.data) sum_block += v;
  (void)sum_block;

  std::printf("\n%d / %d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
