// Unit test of Occ_opt/pCCD.h's amplitude residual, its analytic Jacobian, and the
// pair-level RDM/energy formulas:
//   1) pccdTJacobian vs central finite differences of pccdTResidual, for a random active
//      window (several occ/vir pairs) -- the Jacobian was independently re-derived (not
//      ported from the reference Fortran implementation) from doc/kr_pccd.tex's boxed
//      Eq. (tamp); this is the check requested before trusting it for Newton's method.
//   2) The exact 2-pair (one occ, one vir) analytic solution of the t- and z-amplitude
//      equations (both reduce to a plain quadratic/linear scalar equation in this case,
//      re-derived by hand in the implementation's own header comments) vs
//      solvePccdTAmplitudes/solvePccdZAmplitudes (both solvers: NEWTON and LBFGS).
//   3) pccdPairEnergyFromRdm (the RDM-level energy, Eq. (Epair)) vs
//      pccdCorrelationEnergy + the active-occupied-only share of pccdReferenceEnergy, for
//      a random multi-pair active window -- an independent cross-check that
//      buildPccdRdm's D/Q formulas and the amplitude solve agree.
// Build/run: make test_pccd LIBCINT=/path/to/libcint.a && ./build/test_pccd
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Matrix.h"
#include "pCCD.h"

using namespace rerdmft;

namespace {

int g_failures = 0, g_checks = 0;

struct Rng {
  std::uint64_t s = 1234567891ULL;
  double next() {  // [-0.5, 0.5)
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
  }
};

void check(bool ok, const std::string& name) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::cout << "FAIL: " << name << "\n";
  } else {
    std::cout << "ok:   " << name << "\n";
  }
}

// A random PccdCoefficients over n_core=0 core pairs, n_occ occupied, n_vir virtual --
// eps random, g/w random SYMMETRIC with a zero diagonal (exactly what
// buildPccdCoefficients itself produces from real integrals, so the amplitude/RDM math
// below is exercised in the same shape it will see in production).
PccdCoefficients randomCoefficients(std::size_t n_occ, std::size_t n_vir, Rng& rng) {
  const std::size_t n = n_occ + n_vir;
  PccdCoefficients coeff;
  coeff.n_core = 0;
  coeff.n_occ = n_occ;
  coeff.n_vir = n_vir;
  coeff.eps.assign(n, 0.0);
  for (std::size_t p = 0; p < n; ++p) coeff.eps[p] = 1.0 + 2.0 * rng.next();
  coeff.g = Matrix<double>(n, n, 0.0);
  coeff.w = Matrix<double>(n, n, 0.0);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q) {
      const double g_pq = 0.3 * rng.next();
      const double w_pq = 0.3 * rng.next();
      coeff.g(p, q) = coeff.g(q, p) = g_pq;
      coeff.w(p, q) = coeff.w(q, p) = w_pq;
    }
  }
  return coeff;
}

void testJacobianFiniteDifference() {
  Rng rng;
  const std::size_t n_occ = 3, n_vir = 4;
  const PccdCoefficients coeff = randomCoefficients(n_occ, n_vir, rng);
  Matrix<double> t(n_occ, n_vir);
  for (std::size_t i = 0; i < n_occ; ++i)
    for (std::size_t a = 0; a < n_vir; ++a) t(i, a) = 0.1 * rng.next();

  const Matrix<double> analytic = pccdTJacobian(coeff, t);
  const double step = 1e-6;
  double worst = 0.0;
  for (std::size_t k = 0; k < n_occ; ++k) {
    for (std::size_t d = 0; d < n_vir; ++d) {
      Matrix<double> tp = t, tm = t;
      tp(k, d) += step;
      tm(k, d) -= step;
      const Matrix<double> rp = pccdTResidual(coeff, tp);
      const Matrix<double> rm = pccdTResidual(coeff, tm);
      for (std::size_t i = 0; i < n_occ; ++i) {
        for (std::size_t a = 0; a < n_vir; ++a) {
          const double fd = (rp(i, a) - rm(i, a)) / (2.0 * step);
          const double an = analytic(i * n_vir + a, k * n_vir + d);
          const double err = std::abs(an - fd) / (1.0 + std::abs(fd));
          if (err > worst) worst = err;
        }
      }
    }
  }
  std::cout << "  Jacobian vs FD worst relative error: " << std::scientific << worst << "\n";
  check(worst < 1e-6, "pccdTJacobian matches finite differences of pccdTResidual");
}

void testTwoPairAnalytic(PccdAmplitudeSolver solver, const std::string& solver_name) {
  Rng rng;
  PccdCoefficients coeff;
  coeff.n_core = 0;
  coeff.n_occ = 1;
  coeff.n_vir = 1;
  const double eps_i = 1.0 + rng.next(), eps_a = 3.0 + rng.next();
  const double g = 0.2 + 0.1 * rng.next();
  coeff.eps = {eps_i, eps_a};
  coeff.g = Matrix<double>(2, 2, 0.0);
  coeff.w = Matrix<double>(2, 2, 0.0);
  coeff.g(0, 1) = coeff.g(1, 0) = g;

  // G + (eps_a-eps_i) t - G t^2 = 0 -- the root continuously connected to t=0 as G -> 0
  // (the physical pCCD branch): t = [Delta - sqrt(Delta^2+4G^2)] / (2G), Delta=eps_a-eps_i
  // (picking the root with the SAME sign as -Delta/Delta for small G, i.e. -G/Delta).
  const double delta = eps_a - eps_i;
  const double t_exact = (delta - std::sqrt(delta * delta + 4.0 * g * g)) / (2.0 * g);

  PccdSettings settings;
  settings.solver = solver;
  settings.max_iterations = 500;
  settings.residual_tolerance = 1e-12;
  Matrix<double> t0(1, 1, 0.0);
  const auto t_result = solvePccdTAmplitudes(coeff, t0, settings);
  check(t_result.t_converged, "2-pair t-amplitude converged (" + solver_name + ")");
  check(std::abs(t_result.t(0, 0) - t_exact) < 1e-6,
        "2-pair t-amplitude matches exact quadratic root (" + solver_name + ")");

  // z = -G / (Delta - 2 G t), from the SAME reduction (see this file's own header).
  const double z_exact = -g / (delta - 2.0 * g * t_result.t(0, 0));
  Matrix<double> z0(1, 1, 0.0);
  const auto z_result = solvePccdZAmplitudes(coeff, t_result.t, z0, settings);
  check(z_result.z_converged, "2-pair z-amplitude converged (" + solver_name + ")");
  check(std::abs(z_result.z(0, 0) - z_exact) < 1e-6,
        "2-pair z-amplitude matches exact linear solution (" + solver_name + ")");
}

void testPairEnergyCrossCheck() {
  Rng rng;
  const std::size_t n_occ = 2, n_vir = 3;
  const PccdCoefficients coeff = randomCoefficients(n_occ, n_vir, rng);

  PccdSettings settings;
  settings.solver = PccdAmplitudeSolver::kNewton;
  settings.max_iterations = 500;
  settings.residual_tolerance = 1e-12;
  Matrix<double> t0(n_occ, n_vir, 0.0);
  const auto t_result = solvePccdTAmplitudes(coeff, t0, settings);
  check(t_result.t_converged, "multi-pair t-amplitudes converged");
  Matrix<double> z0(n_occ, n_vir, 0.0);
  const auto z_result = solvePccdZAmplitudes(coeff, t_result.t, z0, settings);
  check(z_result.z_converged, "multi-pair z-amplitudes converged");

  const PccdRdm rdm = buildPccdRdm(t_result.t, z_result.z);
  const double e_corr = pccdCorrelationEnergy(coeff, t_result.t);
  // The active-occupied-only share of pccdReferenceEnergy (n_core=0 here, so it IS
  // pccdReferenceEnergy itself).
  const double e_ref_active = pccdReferenceEnergy(coeff);
  const double e_pair = pccdPairEnergyFromRdm(coeff, rdm);
  const double diff = std::abs(e_pair - (e_ref_active + e_corr));
  std::cout << "  E_pair(RDM) = " << std::setprecision(12) << e_pair
            << "   E_ref+E_corr = " << (e_ref_active + e_corr) << "   diff = " << diff << "\n";
  check(diff < 1e-8, "RDM-level energy (Eq. Epair) matches E_ref + E_corr");

  // Trace sum rule (tex doc Eq. (trace)): sum_p n_p + 2*sum_{p!=q} Q_pq == N(N-1)/2, N = 2*n_occ
  // (the NOMINAL electron count of the reference determinant -- holds for ANY t,z, not just
  // converged ones, since it is a pure RDM identity with no integral/orbital dependence at all).
  const std::size_t n_total_pairs = n_occ + n_vir;
  double trace = 0.0;
  for (std::size_t p = 0; p < n_total_pairs; ++p) trace += rdm.d(p, p);  // d(p,p) == n_p
  for (std::size_t p = 0; p < n_total_pairs; ++p) {
    for (std::size_t q = 0; q < n_total_pairs; ++q) {
      if (p == q) continue;
      trace += 2.0 * rdm.q(p, q);
    }
  }
  const double n_electrons = 2.0 * static_cast<double>(n_occ);
  const double expected_trace = 0.5 * n_electrons * (n_electrons - 1.0);
  const double trace_err = std::abs(trace - expected_trace);
  std::cout << "  trace = " << trace << "   N(N-1)/2 = " << expected_trace
            << "   diff = " << trace_err << "\n";
  check(trace_err < 1e-10, "2-RDM trace sum rule (sum n_p + 2 sum Q_pq = N(N-1)/2)");

  // Symmetry: Q_pq must be symmetric (Q_pq == Q_qp, every block); D_pq is explicitly NOT
  // symmetric in general (D_ia != D_ai -- tex doc, Sec. "Density matrices") and is not checked.
  double worst_q_asym = 0.0;
  for (std::size_t p = 0; p < n_total_pairs; ++p) {
    for (std::size_t q = p + 1; q < n_total_pairs; ++q) {
      worst_q_asym = std::max(worst_q_asym, std::abs(rdm.q(p, q) - rdm.q(q, p)));
    }
  }
  std::cout << "  worst |Q_pq - Q_qp| = " << worst_q_asym << "\n";
  check(worst_q_asym < 1e-12, "Q_pq is exactly symmetric");
}

}  // namespace

int main() {
  std::cout << std::setprecision(10);
  testJacobianFiniteDifference();
  testTwoPairAnalytic(PccdAmplitudeSolver::kNewton, "NEWTON");
  testTwoPairAnalytic(PccdAmplitudeSolver::kLbfgs, "LBFGS");
  testPairEnergyCrossCheck();

  std::cout << "\n" << g_checks << " checks, " << g_failures << " failed.\n";
  return g_failures == 0 ? 0 : 1;
}
