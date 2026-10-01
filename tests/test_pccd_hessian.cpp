// Unit test of Hessian_opt/PccdHessian.h's pccdHessianElement against a finite difference of
// the pccdFockMatrix-based orbital gradient, SPECIFICALLY for a rotation pair that mixes an
// ACTIVE (core/occ/vir, genuinely correlated) pair-representative index with an UNTOUCHED one
// (not in `reps`/`bar` at all -- e.g. a deep-virtual or, in a real C4_DHF run, a negative-energy
// spinor). This is the one combination nothing else currently tests: the ordinary ADAM/NEO
// examples only ever rotate among ACTIVE indices (no-pair trim excludes the untouched/negative
// branch from rotation entirely), so this is the first direct check of the Hessian for a
// mixed pair -- motivated by FULL_OPTIMIZATION_4C_NEG (the min-max stage, where rotations DO
// include such pairs) diverging after a few otherwise-clean NEO Newton steps.
// Complex T (the min-max stage is always complex spinors); synthetic data built the SAME way as
// tests/test_pccd_fock.cpp's own NON_REL-structured integrals (interleaved spin-orbitals,
// pair_of = i^1, real values embedded as complex<double>) -- simple to get right, and sufficient
// to exercise the complex `hartreeExchangeHessianElement`/`pccdHessianElement` code path.
// Build/run: make test_pccd_hessian LIBCINT=/path/to/libcint.a && ./build/test_pccd_hessian
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "FullOptimization.h"
#include "HartreeExchangeGradient.h"
#include "Matrix.h"
#include "OrbitalGradient.h"
#include "PccdFock.h"
#include "PccdHessian.h"
#include "SpinorRotation.h"
#include "Tensor4.h"
#include "pCCD.h"

using namespace rerdmft;

namespace {

struct Rng {
  std::uint64_t s = 24681357ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

// Antihermitian generator with a single real ("t") rotation angle at (p,q): kappa(p,q)=eps,
// kappa(q,p)=-eps, matching generatorMatrix<T>(n,p,q,t,0) elsewhere in this project.
Matrix<std::complex<double>> realGenerator(std::size_t n, std::size_t p, std::size_t q,
                                            double eps) {
  Matrix<std::complex<double>> k(n, n, std::complex<double>(0.0, 0.0));
  k(p, q) = eps;
  k(q, p) = -eps;
  return k;
}

// Imaginary ("y") rotation angle at (p,q): kappa(p,q) += i*eps, kappa(q,p) += i*eps (SAME sign
// both slots -- matching this project's own generatorMatrix<T>'s y-branch for complex T).
Matrix<std::complex<double>> imagGenerator(std::size_t n, std::size_t p, std::size_t q,
                                            double eps) {
  Matrix<std::complex<double>> k(n, n, std::complex<double>(0.0, 0.0));
  k(p, q) += std::complex<double>(0.0, eps);
  k(q, p) += std::complex<double>(0.0, eps);
  return k;
}

}  // namespace

int main() {
  const std::size_t ns = 4, n = 2 * ns;  // 4 spatial orbitals -> 8 interleaved spin-orbitals
  Rng rng;
  Matrix<double> hs(ns, ns, 0.0);
  for (std::size_t i = 0; i < ns; ++i)
    for (std::size_t j = i; j < ns; ++j) hs(i, j) = hs(j, i) = rng.next();
  std::vector<Matrix<double>> vec(6, Matrix<double>(ns, ns, 0.0));
  for (auto& v : vec)
    for (std::size_t i = 0; i < ns; ++i)
      for (std::size_t j = i; j < ns; ++j) v(i, j) = v(j, i) = 0.4 * rng.next();

  Matrix<std::complex<double>> h(n, n, std::complex<double>(0.0, 0.0));
  Tensor4<std::complex<double>> eri(n, n, n, n, std::complex<double>(0.0, 0.0));
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

  // Pairs 0 (indices 0,1) and 1 (indices 2,3) are the pCCD active window (1 occupied + 1
  // virtual, no frozen core); pairs 2 (4,5) and 3 (6,7) are left UNTOUCHED -- exactly like a
  // trimmed negative-energy branch: not in `reps`/`bar`, occupation 0, no amplitude.
  const std::size_t n_core = 0, n_occ = 1, n_vir = 1;
  const std::vector<std::size_t> reps = {0, 2};
  const std::vector<std::size_t> bar = {1, 3};

  const auto coeff = buildPccdCoefficients(h, eri, reps, bar, n_core, n_occ, n_vir);
  PccdSettings settings;
  settings.solver = PccdAmplitudeSolver::kNewton;
  settings.max_iterations = 500;
  settings.residual_tolerance = 1e-12;
  Matrix<double> t0(n_occ, n_vir, 0.0), z0(n_occ, n_vir, 0.0);
  const auto t_result = solvePccdTAmplitudes(coeff, t0, settings);
  const auto z_result = solvePccdZAmplitudes(coeff, t_result.t, z0, settings);
  if (!t_result.t_converged || !z_result.z_converged) {
    std::cout << "FAIL: amplitude solve did not converge\n";
    return 1;
  }
  const PccdRdm rdm = buildPccdRdm(t_result.t, z_result.z);

  std::vector<double> occupations(n, 0.0);
  occupations[0] = occupations[1] = rdm.n_occ[0];
  occupations[2] = occupations[3] = rdm.n_vir[0];
  // occupations[4..7] left at 0 -- the untouched/"negative-energy-like" pairs.

  const auto fock = pccdFockMatrix(h, eri, reps, bar, n_core, n_occ, n_vir, rdm, occupations);

  int checks = 0, failures = 0;
  const double step = 1e-5;
  // Test every (P, Q) with P in the ACTIVE window (0..3) and Q UNTOUCHED (4..7) -- the one
  // combination the ordinary (no-pair-trimmed) examples never rotate along.
  for (std::size_t P = 0; P < 4; ++P) {
    for (std::size_t Q = 4; Q < 8; ++Q) {
      // kappa(Q,P)=+eps, kappa(P,Q)=-eps -- Q > P, matching this project's own "larger index
      // first" pair convention (hessianPairIndices' p > q pairs; generatorMatrix<T>(n,p,q,t,y)
      // calls elsewhere always pass the larger index as p).
      const auto u_plus = spinorRotationMatrix(realGenerator(n, Q, P, step));
      const auto u_minus = spinorRotationMatrix(realGenerator(n, Q, P, -step));
      const auto rot_plus = rotateIntegralsExact(h, eri, u_plus);
      const auto rot_minus = rotateIntegralsExact(h, eri, u_minus);
      const auto fock_plus =
          pccdFockMatrix(rot_plus.h, rot_plus.eri, reps, bar, n_core, n_occ, n_vir, rdm, occupations);
      const auto fock_minus = pccdFockMatrix(rot_minus.h, rot_minus.eri, reps, bar, n_core, n_occ,
                                             n_vir, rdm, occupations);
      const auto g_plus = orbitalGradient(fock_plus);
      const auto g_minus = orbitalGradient(fock_minus);
      // orbitalGradient only fills the p >= q "lower triangle" (OrbitalGradient.h's own
      // convention); Q > P here, so read g(Q,P), not g(P,Q) (left at its default-constructed 0).
      const std::complex<double> fd = (g_plus(Q, P) - g_minus(Q, P)) / (2.0 * step);
      const std::complex<double> analytic = pccdHessianElement(h, eri, reps, bar, n_core, n_occ,
                                                                n_vir, rdm, occupations, fock, Q, P, Q, P);
      const double err = std::abs(analytic - fd) / (1.0 + std::abs(fd));
      ++checks;
      const bool ok = err < 1e-5;
      if (!ok) ++failures;
      std::cout << (ok ? "ok  " : "FAIL") << " (P=" << P << ",Q=" << Q
                << "): analytic=" << analytic << "  FD=" << fd << "  rel.err=" << std::scientific
                << std::setprecision(2) << err << std::defaultfloat << "\n";

      // Imaginary ("y") rotation: pccdHessianElementImag(Q,P,Q,P) vs a finite difference of
      // g(Q,P) under an IMAGINARY-angle rotation at (Q,P) -- untested by the real-rotation check
      // above, and the actual C4_DHF/X2C min-max stage rotates with BOTH real and imaginary
      // angles throughout.
      {
        const auto uy_plus = spinorRotationMatrix(imagGenerator(n, Q, P, step));
        const auto uy_minus = spinorRotationMatrix(imagGenerator(n, Q, P, -step));
        const auto roty_plus = rotateIntegralsExact(h, eri, uy_plus);
        const auto roty_minus = rotateIntegralsExact(h, eri, uy_minus);
        const auto focky_plus = pccdFockMatrix(roty_plus.h, roty_plus.eri, reps, bar, n_core,
                                               n_occ, n_vir, rdm, occupations);
        const auto focky_minus = pccdFockMatrix(roty_minus.h, roty_minus.eri, reps, bar, n_core,
                                                n_occ, n_vir, rdm, occupations);
        const auto gy_plus = orbitalGradient(focky_plus);
        const auto gy_minus = orbitalGradient(focky_minus);
        // pccdHessianElementImag's own convention (PnofHessian.h's docstring: a REAL
        // d^2E/dy^2-like combination) differs from a raw complex dg/dy by a factor of i --
        // confirmed empirically (FD/analytic = i exactly on the first failing attempt here, a
        // test-comparison-convention issue, not a code bug): compare against -i*dg/dy instead.
        const std::complex<double> fd_y =
            std::complex<double>(0.0, -1.0) * (gy_plus(Q, P) - gy_minus(Q, P)) / (2.0 * step);
        const std::complex<double> analytic_y = pccdHessianElementImag(
            h, eri, reps, bar, n_core, n_occ, n_vir, rdm, occupations, fock, Q, P, Q, P);
        const double err_y = std::abs(analytic_y - fd_y) / (1.0 + std::abs(fd_y));
        ++checks;
        const bool ok_y = err_y < 1e-5;
        if (!ok_y) ++failures;
        std::cout << (ok_y ? "ok  " : "FAIL") << " imag (P=" << P << ",Q=" << Q
                  << "): analytic=" << analytic_y << "  FD=" << fd_y
                  << "  rel.err=" << std::scientific << std::setprecision(2) << err_y
                  << std::defaultfloat << "\n";

        // Mixed: pccdHessianElementMixed(Q,P,Q,P) vs d(Im g(Q,P))/d(real angle t at (Q,P)).
        const std::complex<double> fd_mixed =
            (std::complex<double>(0.0, g_plus(Q, P).imag()) -
             std::complex<double>(0.0, g_minus(Q, P).imag())) /
            (2.0 * step);
        const std::complex<double> analytic_mixed = pccdHessianElementMixed(
            h, eri, reps, bar, n_core, n_occ, n_vir, rdm, occupations, fock, Q, P, Q, P);
        const double err_mixed = std::abs(analytic_mixed - fd_mixed) / (1.0 + std::abs(fd_mixed));
        ++checks;
        const bool ok_mixed = err_mixed < 1e-5;
        if (!ok_mixed) ++failures;
        std::cout << (ok_mixed ? "ok  " : "FAIL") << " mixed (P=" << P << ",Q=" << Q
                  << "): analytic=" << analytic_mixed << "  FD=" << fd_mixed
                  << "  rel.err=" << std::scientific << std::setprecision(2) << err_mixed
                  << std::defaultfloat << "\n";
      }
    }
  }
  // Off-diagonal: Hess_{(Q1,P1),(Q2,P2)} for two DIFFERENT mixed (active, untouched) pairs,
  // sharing no index, the ACTIVE index, or the UNTOUCHED index in turn -- the off-diagonal
  // couplings a Davidson/NEO Hessian-vector product actually needs (not exercised by the
  // diagonal-only check above).
  // Each tuple is {Q1,P1,Q2,P2} with Qi > Pi always (the project's own "larger index first"
  // pair convention, matching the diagonal loop above -- Q1=4,P1=0 means the pair (4,0), i.e.
  // untouched index 4 with active index 0).
  const std::vector<std::array<std::size_t, 4>> cross_cases = {
      {4, 0, 5, 1},  // disjoint
      {4, 0, 5, 0},  // shares the active index (0)
      {4, 0, 4, 1},  // shares the untouched index (4)
      {4, 2, 5, 3},  // disjoint, other active pair
  };
  for (const auto& c : cross_cases) {
    const std::size_t Q1 = c[0], P1 = c[1], Q2 = c[2], P2 = c[3];
    const auto u_plus = spinorRotationMatrix(realGenerator(n, Q2, P2, step));
    const auto u_minus = spinorRotationMatrix(realGenerator(n, Q2, P2, -step));
    const auto rot_plus = rotateIntegralsExact(h, eri, u_plus);
    const auto rot_minus = rotateIntegralsExact(h, eri, u_minus);
    const auto fock_plus =
        pccdFockMatrix(rot_plus.h, rot_plus.eri, reps, bar, n_core, n_occ, n_vir, rdm, occupations);
    const auto fock_minus = pccdFockMatrix(rot_minus.h, rot_minus.eri, reps, bar, n_core, n_occ,
                                           n_vir, rdm, occupations);
    const auto g_plus = orbitalGradient(fock_plus);
    const auto g_minus = orbitalGradient(fock_minus);
    const std::complex<double> fd = (g_plus(Q1, P1) - g_minus(Q1, P1)) / (2.0 * step);
    const std::complex<double> analytic = pccdHessianElement(h, eri, reps, bar, n_core, n_occ,
                                                              n_vir, rdm, occupations, fock, Q2, P2, Q1, P1);
    const double err = std::abs(analytic - fd) / (1.0 + std::abs(fd));
    ++checks;
    const bool ok = err < 1e-5;
    if (!ok) ++failures;
    std::cout << (ok ? "ok  " : "FAIL") << " cross (Q1=" << Q1 << ",P1=" << P1 << ",Q2=" << Q2
              << ",P2=" << P2 << "): analytic=" << analytic << "  FD=" << fd
              << "  rel.err=" << std::scientific << std::setprecision(2) << err
              << std::defaultfloat << "\n";
  }

  std::cout << "\n" << checks << " checks, " << failures << " failed.\n";
  return failures == 0 ? 0 : 1;
}
