#include "pCCD.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <string>
#include <stdexcept>

#include "CholeskyEri.h"
#include "DIIS.h"
#include "LBFGS.h"
#include "LinearAlgebra.h"
#include "Progress.h"
#include "NonRelSpinRiMoEri.h"
#include "RiMoEri.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

namespace {

// Symmetrizes a real matrix in place: M <- (M + M^T)/2.
void symmetrize(Matrix<double>& m) {
  const std::size_t n = m.rows();
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q) {
      const double avg = 0.5 * (m(p, q) + m(q, p));
      m(p, q) = avg;
      m(q, p) = avg;
    }
  }
}

}  // namespace

template <typename T, typename Eri>
PccdCoefficients buildPccdCoefficients(const Matrix<T>& h, const Eri& eri,
                                       const std::vector<std::size_t>& reps,
                                       const std::vector<std::size_t>& bar,
                                       std::size_t n_core, std::size_t n_occ,
                                       std::size_t n_vir) {
  const std::size_t n = n_core + n_occ + n_vir;
  if (reps.size() != n || bar.size() != n) {
    throw std::runtime_error("buildPccdCoefficients: reps/bar size != n_core+n_occ+n_vir");
  }

  PccdCoefficients coeff;
  coeff.n_core = n_core;
  coeff.n_occ = n_occ;
  coeff.n_vir = n_vir;
  coeff.eps.assign(n, 0.0);
  coeff.g = Matrix<double>(n, n, 0.0);
  coeff.w = Matrix<double>(n, n, 0.0);

  // eps_p = 2 h_pp + J_pp (Eq. (coulomb), boxed): the opposite-Kramers piece V_p reduces
  // to the plain self-Coulomb J_pp = (pp|pp) because K_{p\bar p} = 0 identically (tex
  // doc, Eq. (coulrel)) -- no `bar[p]` needed here at all.
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t rp = reps[p];
    coeff.eps[p] = 2.0 * std::real(h(rp, rp)) + std::real(eri(rp, rp, rp, rp));
  }

  // G_pq = K_pq + K_{p\bar q}, W_pq = 4 J_pq - 2 K_pq - 2 K_{p\bar q} (Eq. (coulomb)),
  // for every unordered pair {p,q} of the combined list. eri(a,b,c,d) is this project's
  // own dir(a,b,c,d) = (ac|bd) chemist convention (Occ_opt/OccupationEnergy.cpp's own
  // J_pq = eri(p,q,p,q), K_pq = eri(p,q,q,p)); K_{p\bar q} = (p \bar q|\bar q p) is then
  // eri(p, \bar q, \bar q, p).
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t rp = reps[p];
    for (std::size_t q = p + 1; q < n; ++q) {
      const std::size_t rq = reps[q];
      const std::size_t rqbar = bar[q];
      const double j_pq = std::real(eri(rp, rq, rp, rq));
      const double k_pq = std::real(eri(rp, rq, rq, rp));
      const double k_pqbar = std::real(eri(rp, rqbar, rqbar, rp));
      const double g_pq = k_pq + k_pqbar;
      const double w_pq = 4.0 * j_pq - 2.0 * k_pq - 2.0 * k_pqbar;
      coeff.g(p, q) = g_pq;
      coeff.g(q, p) = g_pq;
      coeff.w(p, q) = w_pq;
      coeff.w(q, p) = w_pq;
    }
  }
  // Guards against any residual roundoff asymmetry (time-reversal symmetry makes G, W
  // exactly symmetric only in exact arithmetic -- tex doc, Sec. "Conventions").
  symmetrize(coeff.g);
  symmetrize(coeff.w);

  return coeff;
}

namespace {

// G_ov, W_ov blocks (n_occ x n_vir) of a combined-list coefficient matrix, i.e. rows
// [n_core, n_core+n_occ) and columns [n_core+n_occ, n_core+n_occ+n_vir) of `m`.
Matrix<double> ovBlock(const Matrix<double>& m, std::size_t n_core, std::size_t n_occ,
                        std::size_t n_vir) {
  Matrix<double> block(n_occ, n_vir);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      block(i, a) = m(n_core + i, n_core + n_occ + a);
    }
  }
  return block;
}

// G_oo, W_oo blocks (n_occ x n_occ), rows/columns both [n_core, n_core+n_occ).
Matrix<double> ooBlock(const Matrix<double>& m, std::size_t n_core, std::size_t n_occ) {
  Matrix<double> block(n_occ, n_occ);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t j = 0; j < n_occ; ++j) {
      block(i, j) = m(n_core + i, n_core + j);
    }
  }
  return block;
}

// G_vv block (n_vir x n_vir), rows/columns both [n_core+n_occ, n_core+n_occ+n_vir).
Matrix<double> vvBlock(const Matrix<double>& m, std::size_t n_core, std::size_t n_occ,
                        std::size_t n_vir) {
  Matrix<double> block(n_vir, n_vir);
  for (std::size_t a = 0; a < n_vir; ++a) {
    for (std::size_t b = 0; b < n_vir; ++b) {
      block(a, b) = m(n_core + n_occ + a, n_core + n_occ + b);
    }
  }
  return block;
}

// eps restricted to the active occupied/virtual window.
std::vector<double> activeEps(const PccdCoefficients& coeff, std::size_t offset,
                               std::size_t count) {
  std::vector<double> e(count);
  for (std::size_t p = 0; p < count; ++p) e[p] = coeff.eps[offset + p];
  return e;
}

// Delta_ia = eps_a - eps_i + sum_{j!=i} (W_aj - W_ij) (Eq. (Delta)), with j ranging over
// EVERY occupied pair -- frozen CORE pairs too, not just the active-occupied ones i, a
// run over. A frozen-core pair has no amplitude of its own (it never appears as an i/a
// index anywhere else), but it still occupies a REAL pair at n=1 that dresses every
// virtual orbital's effective energy via ordinary Coulomb/exchange -- the standard
// frozen-core Fock correction, needed so a's own effective energy properly includes the
// core's mean field (found by an energy-consistency check, tests/test_pccd_fock.cpp,
// against the independently unfolded RDM energy: omitting core from this sum left the
// amplitude equations under-dressed, silently different from the true frozen-core
// problem by the core-virtual W contribution).
Matrix<double> buildDelta(const PccdCoefficients& coeff, const std::vector<double>& eps_vir) {
  const std::size_t n_core = coeff.n_core, n_occ = coeff.n_occ, n_vir = eps_vir.size();
  const std::size_t n_ref = n_core + n_occ;  // every occupied pair (core ++ active-occ)
  Matrix<double> delta(n_occ, n_vir, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    const std::size_t i_global = n_core + i;
    for (std::size_t a = 0; a < n_vir; ++a) {
      const std::size_t a_global = n_core + n_occ + a;
      double sum = 0.0;
      for (std::size_t j_global = 0; j_global < n_ref; ++j_global) {
        if (j_global == i_global) continue;
        sum += coeff.w(a_global, j_global) - coeff.w(i_global, j_global);
      }
      delta(i, a) = eps_vir[a] - coeff.eps[i_global] + sum;
    }
  }
  return delta;
}

struct PccdBlocks {
  Matrix<double> g_oo, g_ov, g_vv, w_oo, w_ov, delta;
  std::vector<double> eps_occ, eps_vir;
};

PccdBlocks buildBlocks(const PccdCoefficients& coeff) {
  PccdBlocks b;
  b.g_oo = ooBlock(coeff.g, coeff.n_core, coeff.n_occ);
  b.g_ov = ovBlock(coeff.g, coeff.n_core, coeff.n_occ, coeff.n_vir);
  b.g_vv = vvBlock(coeff.g, coeff.n_core, coeff.n_occ, coeff.n_vir);
  b.w_oo = ooBlock(coeff.w, coeff.n_core, coeff.n_occ);
  b.w_ov = ovBlock(coeff.w, coeff.n_core, coeff.n_occ, coeff.n_vir);
  b.eps_occ = activeEps(coeff, coeff.n_core, coeff.n_occ);
  b.eps_vir = activeEps(coeff, coeff.n_core + coeff.n_occ, coeff.n_vir);
  b.delta = buildDelta(coeff, b.eps_vir);
  return b;
}

// Sigma^(v)_a = sum_j G_ov(j,a) T(j,a), Sigma^(o)_i = sum_b G_ov(i,b) T(i,b) (Eq. after
// (Delta)).
void buildSigmas(const Matrix<double>& g_ov, const Matrix<double>& t,
                  std::vector<double>& sigma_v, std::vector<double>& sigma_o) {
  const std::size_t n_occ = t.rows(), n_vir = t.cols();
  sigma_v.assign(n_vir, 0.0);
  sigma_o.assign(n_occ, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      const double term = g_ov(i, a) * t(i, a);
      sigma_v[a] += term;
      sigma_o[i] += term;
    }
  }
}

}  // namespace

Matrix<double> pccdTResidual(const PccdCoefficients& coeff, const Matrix<double>& t) {
  const PccdBlocks b = buildBlocks(coeff);
  const std::size_t n_occ = coeff.n_occ, n_vir = coeff.n_vir;
  std::vector<double> sigma_v, sigma_o;
  buildSigmas(b.g_ov, t, sigma_v, sigma_o);

  // Y(j,i) = sum_b G_ov(j,b) T(i,b) = (G_ov * T^T)(j,i).
  Matrix<double> y(n_occ, n_occ, 0.0);
  for (std::size_t j = 0; j < n_occ; ++j) {
    for (std::size_t i = 0; i < n_occ; ++i) {
      double sum = 0.0;
      for (std::size_t bb = 0; bb < n_vir; ++bb) sum += b.g_ov(j, bb) * t(i, bb);
      y(j, i) = sum;
    }
  }

  Matrix<double> r(n_occ, n_vir, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      double val = b.g_ov(i, a);
      val += (b.delta(i, a) - 2.0 * (sigma_v[a] + sigma_o[i]) + 2.0 * b.g_ov(i, a) * t(i, a)) *
             t(i, a);
      for (std::size_t bb = 0; bb < n_vir; ++bb) {
        if (bb == a) continue;
        val += b.g_vv(a, bb) * t(i, bb);
      }
      for (std::size_t j = 0; j < n_occ; ++j) {
        if (j == i) continue;
        val += b.g_oo(i, j) * t(j, a);
      }
      for (std::size_t j = 0; j < n_occ; ++j) val += t(j, a) * y(j, i);
      r(i, a) = val;
    }
  }
  return r;
}

Matrix<double> pccdTJacobian(const PccdCoefficients& coeff, const Matrix<double>& t) {
  const PccdBlocks b = buildBlocks(coeff);
  const std::size_t n_occ = coeff.n_occ, n_vir = coeff.n_vir;
  std::vector<double> sigma_v, sigma_o;
  buildSigmas(b.g_ov, t, sigma_v, sigma_o);

  // P(d,a) = sum_j G_ov(j,d) T(j,a) = (G_ov^T * T)(d,a); note P(a,a) == Sigma^(v)_a.
  Matrix<double> p(n_vir, n_vir, 0.0);
  for (std::size_t d = 0; d < n_vir; ++d) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n_occ; ++j) sum += b.g_ov(j, d) * t(j, a);
      p(d, a) = sum;
    }
  }
  // Q(i,k) = sum_b T(i,b) G_ov(k,b) = (T * G_ov^T)(i,k); Q(i,i) == Sigma^(o)_i.
  Matrix<double> q(n_occ, n_occ, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t k = 0; k < n_occ; ++k) {
      double sum = 0.0;
      for (std::size_t bb = 0; bb < n_vir; ++bb) sum += t(i, bb) * b.g_ov(k, bb);
      q(i, k) = sum;
    }
  }

  const std::size_t dim = n_occ * n_vir;
  Matrix<double> jac(dim, dim, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      const std::size_t row = i * n_vir + a;
      // (i,a),(i,a): Delta_ia - Sigma^(v)_a - Sigma^(o)_i.
      jac(row, row) = b.delta(i, a) - sigma_v[a] - sigma_o[i];
      // (i,a),(i,d), d != a: G_vv(a,d) + P(d,a) - 2 T(i,a) G_ov(i,d).
      for (std::size_t d = 0; d < n_vir; ++d) {
        if (d == a) continue;
        const std::size_t col = i * n_vir + d;
        jac(row, col) = b.g_vv(a, d) + p(d, a) - 2.0 * t(i, a) * b.g_ov(i, d);
      }
      // (i,a),(k,a), k != i: G_oo(i,k) + Q(i,k) - 2 T(i,a) G_ov(k,a).
      for (std::size_t k = 0; k < n_occ; ++k) {
        if (k == i) continue;
        const std::size_t col = k * n_vir + a;
        jac(row, col) = b.g_oo(i, k) + q(i, k) - 2.0 * t(i, a) * b.g_ov(k, a);
      }
    }
  }
  return jac;
}

namespace {

// Flattens/unflattens an n_occ x n_vir matrix row-major, matching pccdTJacobian's own
// (i*n_vir+a) indexing.
std::vector<double> flatten(const Matrix<double>& m) {
  std::vector<double> v(m.rows() * m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t a = 0; a < m.cols(); ++a) v[i * m.cols() + a] = m(i, a);
  }
  return v;
}

Matrix<double> unflatten(const std::vector<double>& v, std::size_t n_occ, std::size_t n_vir) {
  Matrix<double> m(n_occ, n_vir);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) m(i, a) = v[i * n_vir + a];
  }
  return m;
}

double norm2(const std::vector<double>& v) {
  double sum = 0.0;
  for (const double x : v) sum += x * x;
  return std::sqrt(sum);
}

// Levenberg-Marquardt on 0.5 ||R||^2 with the analytic Jacobian (see PccdAmplitudeSolver::kLm); returns the best point reached.
PccdAmplitudeResult solveLevenbergMarquardt(
    const std::function<Matrix<double>(const Matrix<double>&)>& residual,
    const std::function<Matrix<double>(const Matrix<double>&)>& jacobian,
    const Matrix<double>& x0, const PccdSettings& settings) {
  const std::size_t n_occ = x0.rows(), n_vir = x0.cols();
  PccdAmplitudeResult result;
    const std::size_t n = n_occ * n_vir;
    std::vector<double> x = flatten(x0);
    std::vector<double> f = flatten(residual(x0));
    double fnorm = norm2(f);
    double lambda = 1e-3, nu = 2.0;
    int iter = 0;
    while (iter < settings.max_iterations && std::isfinite(fnorm) && fnorm >= settings.residual_tolerance) {
      const Matrix<double> j = jacobian(unflatten(x, n_occ, n_vir));
      Matrix<double> jtj(n, n, 0.0);
      std::vector<double> g(n, 0.0);  // J^T R
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
          double sum = 0.0;
          for (std::size_t k = 0; k < n; ++k) sum += j(k, a) * j(k, b);
          jtj(a, b) = jtj(b, a) = sum;
        }
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) sum += j(k, a) * f[k];
        g[a] = sum;
      }
      bool accepted = false;
      for (int attempt = 0; attempt < 40 && !accepted; ++attempt) {
        Matrix<double> a_mat = jtj;
        for (std::size_t k = 0; k < n; ++k) a_mat(k, k) += lambda * std::max(jtj(k, k), 1e-12);
        const Matrix<double> inv = invert(a_mat);
        std::vector<double> dx(n, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
          double sum = 0.0;
          for (std::size_t c = 0; c < n; ++c) sum += inv(r, c) * g[c];
          dx[r] = -sum;
        }
        std::vector<double> xn(n);
        for (std::size_t k = 0; k < n; ++k) xn[k] = x[k] + dx[k];
        const std::vector<double> fnew = flatten(residual(unflatten(xn, n_occ, n_vir)));
        const double new_norm = norm2(fnew);
        // predicted reduction of 0.5||R||^2 for the damped step: 0.5 dx^T (lambda D dx - J^T R)
        double predicted = 0.0;
        for (std::size_t k = 0; k < n; ++k) predicted += dx[k] * (lambda * std::max(jtj(k, k), 1e-12) * dx[k] - g[k]);
        predicted *= 0.5;
        const double actual = 0.5 * (fnorm * fnorm - new_norm * new_norm);
        if (std::isfinite(new_norm) && predicted > 0.0 && actual > 0.0) {
          const double rho = actual / predicted;
          x = xn;
          f = fnew;
          fnorm = new_norm;
          lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
          nu = 2.0;
          accepted = true;
        } else {
          lambda *= nu;
          nu *= 2.0;
        }
      }
      if (!accepted) break;  // no descent step found
      ++iter;
    }
    result.t = unflatten(x, n_occ, n_vir);
    result.t_converged = std::isfinite(fnorm) && fnorm < settings.residual_tolerance;
    result.t_iterations = iter;
    result.t_residual_norm = fnorm;
    return result;
}

// Generic driver shared by the t- and z-amplitude solves: `residual`/`jacobian` give
// R(x)/dR/dx (R linear in x for z, quadratic for t -- see pCCD.h). kNewton solves
// J dx = -R directly (LinearAlgebra.h's `invert`) each iteration; for a linear R this
// converges in exactly one iteration, which is how `solvePccdZAmplitudes` gets an exact
// one-shot solve for free out of the same code path used for t's genuinely iterative
// Newton-Raphson. kLbfgs minimizes 0.5 ||R(x)||^2 via Utils/LBFGS.h with the exact
// chain-rule gradient J(x)^T R(x).
PccdAmplitudeResult solveResidual(
    const std::function<Matrix<double>(const Matrix<double>&)>& residual,
    const std::function<Matrix<double>(const Matrix<double>&)>& jacobian,
    const Matrix<double>& x0, const PccdSettings& settings) {
  const std::size_t n_occ = x0.rows(), n_vir = x0.cols();
  PccdAmplitudeResult result;
  std::string diis_failure;  // set when DIIS failed and Newton takes over

  if (settings.solver == PccdAmplitudeSolver::kLm) {
    PccdAmplitudeResult lm = solveLevenbergMarquardt(residual, jacobian, x0, settings);
    if (!lm.t_converged) {
      lm.warning = "WARNING: pCCD amplitude solve: LM did not converge (" + std::to_string(lm.t_iterations) + " iterations, |R| = " +
                   std::to_string(lm.t_residual_norm) + "); the unconverged amplitudes are used";
    }
    return lm;
  }

  if (settings.solver == PccdAmplitudeSolver::kKrylov) {
    // Jacobian-free Newton-Krylov (scipy's `krylov`): GMRES on J dx = -R with finite-difference J v, Armijo backtracking.
    const std::size_t n = n_occ * n_vir;
    std::vector<double> x = flatten(x0);
    std::vector<double> f = flatten(residual(x0));
    double fnorm = norm2(f);
    int iter = 0;
    const double rdiff = std::sqrt(std::numeric_limits<double>::epsilon());
    const auto eval = [&](const std::vector<double>& xv) { return flatten(residual(unflatten(xv, n_occ, n_vir))); };
    while (iter < settings.krylov_max_iterations && std::isfinite(fnorm) && fnorm >= settings.residual_tolerance) {
      const double xnorm = norm2(x);
      const auto matvec = [&](const std::vector<double>& v) {
        const double vnorm = norm2(v);
        if (vnorm == 0.0) return std::vector<double>(n, 0.0);
        const double h = rdiff * std::max(1.0, xnorm) / vnorm;
        std::vector<double> xp(n);
        for (std::size_t k = 0; k < n; ++k) xp[k] = x[k] + h * v[k];
        const std::vector<double> fp = eval(xp);
        std::vector<double> jv(n);
        for (std::size_t k = 0; k < n; ++k) jv[k] = (fp[k] - f[k]) / h;
        return jv;
      };
      // GMRES (Arnoldi with modified Gram-Schmidt, Givens rotations), restarted every m vectors; forcing term eta.
      const double eta = std::min(0.1, fnorm);
      const std::size_t m = std::min<std::size_t>(n, 100);
      std::vector<double> dx(n, 0.0);
      std::vector<double> r(n);
      for (std::size_t k = 0; k < n; ++k) r[k] = -f[k];
      const double bnorm = norm2(r);
      for (int restart = 0; restart < 10; ++restart) {
        const double beta = norm2(r);
        if (beta <= eta * bnorm) break;
        std::vector<std::vector<double>> q(1, std::vector<double>(n));
        for (std::size_t k = 0; k < n; ++k) q[0][k] = r[k] / beta;
        std::vector<std::vector<double>> hess;  // hess[j] = column j (length j + 2)
        std::vector<double> cs, sn, g(1, beta);
        std::size_t used = 0;
        for (std::size_t j = 0; j < m; ++j) {
          std::vector<double> w = matvec(q[j]);
          std::vector<double> col(j + 2, 0.0);
          for (std::size_t i = 0; i <= j; ++i) {
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) dot += w[k] * q[i][k];
            col[i] = dot;
            for (std::size_t k = 0; k < n; ++k) w[k] -= dot * q[i][k];
          }
          const double wn = norm2(w);
          col[j + 1] = wn;
          for (std::size_t i = 0; i < j; ++i) {  // apply previous rotations
            const double t1 = cs[i] * col[i] + sn[i] * col[i + 1];
            col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1];
            col[i] = t1;
          }
          const double denom = std::hypot(col[j], col[j + 1]);
          cs.push_back(denom > 0.0 ? col[j] / denom : 1.0);
          sn.push_back(denom > 0.0 ? col[j + 1] / denom : 0.0);
          col[j] = denom;
          g.push_back(-sn[j] * g[j]);
          g[j] = cs[j] * g[j];
          hess.push_back(col);
          used = j + 1;
          if (std::abs(g[j + 1]) <= eta * bnorm || hess[j][j] == 0.0 || wn < 1e-14) break;
          q.push_back(std::vector<double>(n));
          for (std::size_t k = 0; k < n; ++k) q.back()[k] = w[k] / wn;
        }
        std::vector<double> y(used, 0.0);  // back substitution on the rotated Hessenberg
        for (std::size_t ii = used; ii-- > 0;) {
          double sum = g[ii];
          for (std::size_t jj = ii + 1; jj < used; ++jj) sum -= hess[jj][ii] * y[jj];
          y[ii] = hess[ii][ii] != 0.0 ? sum / hess[ii][ii] : 0.0;
        }
        for (std::size_t jj = 0; jj < used; ++jj)
          for (std::size_t k = 0; k < n; ++k) dx[k] += y[jj] * q[jj][k];
        const std::vector<double> jdx = matvec(dx);
        for (std::size_t k = 0; k < n; ++k) r[k] = -f[k] - jdx[k];
      }
      // Armijo backtracking on 0.5 ||R||^2
      double alpha = 1.0;
      std::vector<double> xn(n), fn;
      double fn_norm = 0.0;
      for (int ls = 0; ls < 30; ++ls) {
        for (std::size_t k = 0; k < n; ++k) xn[k] = x[k] + alpha * dx[k];
        fn = eval(xn);
        fn_norm = norm2(fn);
        if (std::isfinite(fn_norm) && fn_norm * fn_norm <= (1.0 - 2e-4 * alpha) * fnorm * fnorm) break;
        alpha *= 0.5;
      }
      x = xn;
      f = fn;
      fnorm = fn_norm;
      ++iter;
    }
    result.t = unflatten(x, n_occ, n_vir);
    result.t_converged = std::isfinite(fnorm) && fnorm < settings.residual_tolerance;
    result.t_iterations = iter;
    result.t_residual_norm = fnorm;
    if (!result.t_converged) {
      throw std::runtime_error("pCCD amplitude solve failed: KRYLOV did not converge (" + std::to_string(iter) +
                               " iterations, |R| = " + std::to_string(fnorm) + ")");
    }
    return result;
  }

  if (settings.solver == PccdAmplitudeSolver::kDiis) {
    const std::size_t n = n_occ * n_vir;
    const Matrix<double> zero(n_occ, n_vir, 0.0);
    const Matrix<double> j0 = jacobian(zero);
    std::vector<double> denominator(n);
    for (std::size_t k = 0; k < n; ++k) {
      denominator[k] = j0(k, k);
      if (std::abs(denominator[k]) < 1e-12) denominator[k] = (denominator[k] < 0.0 ? -1e-12 : 1e-12);
    }
    Matrix<double> x = x0;
    bool cold = true;
    for (std::size_t k = 0; k < n; ++k) cold = cold && x0.data()[k] == 0.0;
    if (cold) {  // MP2-like start x = -R(0) / D
      const std::vector<double> r0 = flatten(residual(zero));
      std::vector<double> start(n);
      for (std::size_t k = 0; k < n; ++k) start[k] = -r0[k] / denominator[k];
      x = unflatten(start, n_occ, n_vir);
    }
    Diis<double> diis(static_cast<std::size_t>(std::max(settings.diis_size, 0)));
    int iter = 0;
    double rnorm = norm2(flatten(residual(x)));
    while (iter < settings.diis_max_iterations && std::isfinite(rnorm) && rnorm >= settings.residual_tolerance) {
      const std::vector<double> xflat = flatten(x), rflat = flatten(residual(x));
      Matrix<double> x_new(1, n), err(1, n);
      for (std::size_t k = 0; k < n; ++k) {
        x_new(0, k) = xflat[k] - rflat[k] / denominator[k];
        err(0, k) = x_new(0, k) - xflat[k];
      }
      const Matrix<double> next = settings.diis_size >= 2 ? diis.extrapolate(err, x_new) : x_new;
      std::vector<double> nflat(n);
      for (std::size_t k = 0; k < n; ++k) nflat[k] = next(0, k);
      x = unflatten(nflat, n_occ, n_vir);
      rnorm = norm2(flatten(residual(x)));
      ++iter;
    }
    result.t = x;
    result.t_converged = std::isfinite(rnorm) && rnorm < settings.residual_tolerance;
    result.t_iterations = iter;
    result.t_residual_norm = rnorm;
    if (result.t_converged) return result;
    // DIIS diverged (NaN or infinity) or reached its iteration limit: Newton from the starting amplitudes.
    diis_failure = std::isfinite(rnorm) ? "DIIS did not converge (" + std::to_string(iter) + " iterations, |R| = " + std::to_string(rnorm) + ")"
                                        : "DIIS diverged (|R| not finite after " + std::to_string(iter) + " iterations)";
    progress("pCCD amplitude solve: " + diis_failure + "; NEWTON reactivated");
  }

  if (settings.solver == PccdAmplitudeSolver::kNewton || !diis_failure.empty()) {
    Matrix<double> x = x0;
    Matrix<double> r = residual(x);
    double rnorm = norm2(flatten(r));
    int iter = 0;
    while (iter < settings.max_iterations && rnorm >= settings.residual_tolerance) {
      const Matrix<double> j = jacobian(x);
      const Matrix<double> j_inv = invert(j);
      std::vector<double> step(n_occ * n_vir, 0.0);
      const std::vector<double> rflat = flatten(r);
      for (std::size_t row = 0; row < n_occ * n_vir; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < n_occ * n_vir; ++col) sum += j_inv(row, col) * rflat[col];
        step[row] = sum;
      }
      const Matrix<double> dx = unflatten(step, n_occ, n_vir);
      for (std::size_t i = 0; i < n_occ; ++i) {
        for (std::size_t a = 0; a < n_vir; ++a) x(i, a) -= dx(i, a);
      }
      r = residual(x);
      rnorm = norm2(flatten(r));
      ++iter;
    }
    result.t = x;
    result.t_converged = rnorm < settings.residual_tolerance;
    result.t_iterations = iter;
    result.t_residual_norm = rnorm;
    if (!result.t_converged && settings.lm_fallback) {  // first solve only: Levenberg-Marquardt from the starting amplitudes
      const std::string newton_text = "NEWTON did not converge (" + std::to_string(iter) + " iterations, |R| = " + std::to_string(rnorm) + ")";
      progress("pCCD amplitude solve: " + newton_text + "; LM reactivated");
      PccdAmplitudeResult lm = solveLevenbergMarquardt(residual, jacobian, x0, settings);
      if (lm.t_converged || lm.t_residual_norm < result.t_residual_norm) result = lm;
      if (!result.t_converged) {
        result.warning = "WARNING: pCCD amplitude solve: " + (diis_failure.empty() ? std::string() : diis_failure + ", ") + newton_text +
                         " and LM did not converge either (" + std::to_string(result.t_iterations) + " iterations, |R| = " +
                         std::to_string(result.t_residual_norm) + "); the unconverged amplitudes are used";
      }
    } else if (!diis_failure.empty() && !result.t_converged) {
      result.warning = "WARNING: pCCD amplitude solve: " + diis_failure + " and the NEWTON fallback did not converge either (" +
                       std::to_string(iter) + " iterations, |R| = " + std::to_string(rnorm) + "); the unconverged amplitudes are used";
    }
    return result;
  }

  // L-BFGS on f(x) = 0.5 ||R(x)||^2, gradient = J(x)^T R(x).
  const LbfgsValueFn value_fn = [&](const std::vector<double>& x) {
    const Matrix<double> r = residual(unflatten(x, n_occ, n_vir));
    const std::vector<double> rflat = flatten(r);
    double sum = 0.0;
    for (const double v : rflat) sum += v * v;
    return 0.5 * sum;
  };
  const LbfgsGradientFn gradient_fn = [&](const std::vector<double>& x) {
    const Matrix<double> xm = unflatten(x, n_occ, n_vir);
    const Matrix<double> r = residual(xm);
    const Matrix<double> j = jacobian(xm);
    const std::vector<double> rflat = flatten(r);
    std::vector<double> grad(n_occ * n_vir, 0.0);
    for (std::size_t col = 0; col < n_occ * n_vir; ++col) {
      double sum = 0.0;
      for (std::size_t row = 0; row < n_occ * n_vir; ++row) sum += j(row, col) * rflat[row];
      grad[col] = sum;
    }
    return grad;
  };
  LbfgsOptions options;
  options.max_iterations = settings.max_iterations;
  // ||grad f||_inf ~ ||J||*||R||, so a residual-norm-scale tolerance on the gradient is
  // the natural translation of `residual_tolerance` into L-BFGS's own stopping test.
  options.gradient_tolerance = settings.residual_tolerance;
  const auto lbfgs_result = solveLbfgs(value_fn, gradient_fn, flatten(x0), options);
  result.t = unflatten(lbfgs_result.x, n_occ, n_vir);
  const Matrix<double> final_r = residual(result.t);
  result.t_residual_norm = norm2(flatten(final_r));
  result.t_converged = lbfgs_result.converged || result.t_residual_norm < settings.residual_tolerance;
  result.t_iterations = lbfgs_result.iterations;
  return result;
}

}  // namespace

void pccdAmplitudesFromOccupations(const std::vector<double>& window_occupations, std::size_t n_occ, std::size_t n_vir,
                                   Matrix<double>& t, Matrix<double>& z) {
  if (window_occupations.size() != n_occ + n_vir) {
    throw std::runtime_error("READ_OCCUPANCIES (pCCD): OCC.in has " + std::to_string(window_occupations.size()) +
                             " line(s) but the amplitude window has " + std::to_string(n_occ) + " occupied + " +
                             std::to_string(n_vir) + " virtual = " + std::to_string(n_occ + n_vir) + " pairs");
  }
  constexpr double kTol = 1e-8;
  t = Matrix<double>(n_occ, n_vir, 0.0);
  z = Matrix<double>(n_occ, n_vir, 0.0);
  for (std::size_t k = 0; k < std::max(n_occ, n_vir); ++k) {
    const bool has_i = k < n_occ, has_a = k < n_vir;
    const double n_i = has_i ? window_occupations[n_occ - 1 - k] : 0.0;
    const double n_a = has_a ? window_occupations[n_occ + k] : 1.0;
    if (has_i && has_a) {
      if (std::abs(n_i + n_a - 1.0) > kTol || n_i < -kTol || n_a < -kTol || n_i < kTol) {
        throw std::runtime_error("READ_OCCUPANCIES (pCCD): pair " + std::to_string(k) + " (occupied HOMO-" + std::to_string(k) +
                                 " with virtual LUMO+" + std::to_string(k) + ") has occupations " + std::to_string(n_i) + " and " +
                                 std::to_string(n_a) + "; pCCD needs n_occupied + n_virtual = 1 (and n_occupied > 0)");
      }
      const std::size_t i = n_occ - 1 - k;
      t(i, k) = -std::sqrt(std::max(0.0, n_a) / n_i);
      z(i, k) = -std::sqrt(std::max(0.0, n_a) * n_i);
    } else if (has_i && std::abs(n_i - 1.0) > kTol) {
      throw std::runtime_error("READ_OCCUPANCIES (pCCD): occupied HOMO-" + std::to_string(k) + " has no virtual partner and needs occupation 1, got " + std::to_string(n_i));
    } else if (has_a && std::abs(n_a) > kTol) {
      throw std::runtime_error("READ_OCCUPANCIES (pCCD): virtual LUMO+" + std::to_string(k) + " has no occupied partner and needs occupation 0, got " + std::to_string(n_a));
    }
  }
}

PccdAmplitudeResult solvePccdTAmplitudes(const PccdCoefficients& coeff, const Matrix<double>& t0,
                                          const PccdSettings& settings) {
  if (!settings.fixed_window_occupations.empty()) {  // READ_OCCUPANCIES: no solve, t built from the occupations
    PccdAmplitudeResult r;
    Matrix<double> z_unused;
    pccdAmplitudesFromOccupations(settings.fixed_window_occupations, coeff.n_occ, coeff.n_vir, r.t, z_unused);
    r.z = z_unused;
    r.t_converged = true;
    return r;
  }
  const auto residual_fn = [&](const Matrix<double>& t) { return pccdTResidual(coeff, t); };
  const auto jacobian_fn = [&](const Matrix<double>& t) { return pccdTJacobian(coeff, t); };
  return solveResidual(residual_fn, jacobian_fn, t0, settings);
}

PccdAmplitudeResult solvePccdZAmplitudes(const PccdCoefficients& coeff, const Matrix<double>& t,
                                          const Matrix<double>& z0, const PccdSettings& settings) {
  // R_z(Z) = G_ov + J(t)^T Z (linear in Z -- the t-Jacobian, evaluated once at the
  // CONVERGED, now-fixed t, is a genuine constant here). See pCCD.h's own header
  // comment on `solvePccdZAmplitudes` for why this is exactly the CC-Lambda equation.
  const std::size_t n_occ = coeff.n_occ, n_vir = coeff.n_vir;
  if (!settings.fixed_window_occupations.empty()) {  // READ_OCCUPANCIES: no solve, z built from the occupations
    PccdAmplitudeResult r;
    Matrix<double> t_unused;
    pccdAmplitudesFromOccupations(settings.fixed_window_occupations, n_occ, n_vir, t_unused, r.z);
    r.t = t;
    r.z_converged = true;
    return r;
  }
  const Matrix<double> j_t = pccdTJacobian(coeff, t);
  const PccdBlocks b = buildBlocks(coeff);

  const auto residual_fn = [&](const Matrix<double>& z) {
    const std::vector<double> zflat = flatten(z);
    std::vector<double> rflat(n_occ * n_vir, 0.0);
    for (std::size_t col = 0; col < n_occ * n_vir; ++col) {
      double sum = 0.0;
      for (std::size_t row = 0; row < n_occ * n_vir; ++row) sum += j_t(row, col) * zflat[row];
      rflat[col] = sum;
    }
    Matrix<double> r = unflatten(rflat, n_occ, n_vir);
    for (std::size_t i = 0; i < n_occ; ++i) {
      for (std::size_t a = 0; a < n_vir; ++a) r(i, a) += b.g_ov(i, a);
    }
    return r;
  };
  // dR_z/dZ == J(t)^T exactly, independent of Z (R_z is linear).
  const auto jacobian_fn = [&](const Matrix<double>&) {
    const std::size_t dim = n_occ * n_vir;
    Matrix<double> jt_transpose(dim, dim, 0.0);
    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = 0; col < dim; ++col) jt_transpose(row, col) = j_t(col, row);
    }
    return jt_transpose;
  };

  // R_z is linear: the LM damping buys nothing, one exact Newton step solves it.
  PccdSettings z_settings = settings;
  if (z_settings.solver == PccdAmplitudeSolver::kLm) z_settings.solver = PccdAmplitudeSolver::kNewton;
  PccdAmplitudeResult result = solveResidual(residual_fn, jacobian_fn, z0, z_settings);
  // solveResidual always writes its answer into `.t` (it is solver-agnostic); move it
  // into `.z` for the caller, matching solvePccdTAmplitudes' own `.t` convention.
  result.z = result.t;
  result.t = Matrix<double>();
  result.z_converged = result.t_converged;
  result.z_iterations = result.t_iterations;
  result.z_residual_norm = result.t_residual_norm;
  result.t_converged = false;
  result.t_iterations = 0;
  result.t_residual_norm = 0.0;
  return result;
}

double pccdReferenceEnergy(const PccdCoefficients& coeff) {
  const std::size_t n_ref = coeff.n_core + coeff.n_occ;
  double energy = 0.0;
  for (std::size_t p = 0; p < n_ref; ++p) energy += coeff.eps[p];
  for (std::size_t p = 0; p < n_ref; ++p) {
    for (std::size_t q = 0; q < n_ref; ++q) {
      if (p == q) continue;
      energy += 0.5 * coeff.w(p, q);
    }
  }
  return energy;
}

double pccdCorrelationEnergy(const PccdCoefficients& coeff, const Matrix<double>& t) {
  const Matrix<double> g_ov = ovBlock(coeff.g, coeff.n_core, coeff.n_occ, coeff.n_vir);
  double energy = 0.0;
  for (std::size_t i = 0; i < coeff.n_occ; ++i) {
    for (std::size_t a = 0; a < coeff.n_vir; ++a) energy += g_ov(i, a) * t(i, a);
  }
  return energy;
}

PccdRdm buildPccdRdm(const Matrix<double>& t, const Matrix<double>& z) {
  const std::size_t n_occ = t.rows(), n_vir = t.cols();
  PccdRdm rdm;
  rdm.n_occ.assign(n_occ, 0.0);
  rdm.n_vir.assign(n_vir, 0.0);

  // M(i,j) = x^j_i = sum_a T(i,a) Z(j,a) = (T * Z^T)(i,j); x_i = M(i,i).
  Matrix<double> m(n_occ, n_occ, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t j = 0; j < n_occ; ++j) {
      double sum = 0.0;
      for (std::size_t a = 0; a < n_vir; ++a) sum += t(i, a) * z(j, a);
      m(i, j) = sum;
    }
  }
  // N(a,b) = x^b_a = sum_i T(i,b) Z(i,a) = (T^T * Z)(a,b); n_a = N(a,a).
  Matrix<double> nn(n_vir, n_vir, 0.0);
  for (std::size_t a = 0; a < n_vir; ++a) {
    for (std::size_t b = 0; b < n_vir; ++b) {
      double sum = 0.0;
      for (std::size_t i = 0; i < n_occ; ++i) sum += t(i, b) * z(i, a);
      nn(a, b) = sum;
    }
  }
  for (std::size_t i = 0; i < n_occ; ++i) rdm.n_occ[i] = 1.0 - m(i, i);
  for (std::size_t a = 0; a < n_vir; ++a) rdm.n_vir[a] = nn(a, a);

  // x^a_i = (M * T)(i,a) (Intermediates, Sec. "Density matrices"): x^a_i =
  // sum_{jb} T(i,b) T(j,a) Z(j,b) = sum_j T(j,a) * (sum_b T(i,b) Z(j,b)) = sum_j T(j,a) M(i,j).
  Matrix<double> x_ia(n_occ, n_vir, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n_occ; ++j) sum += m(i, j) * t(j, a);
      x_ia(i, a) = sum;
    }
  }

  const std::size_t n = n_occ + n_vir;
  rdm.d = Matrix<double>(n, n, 0.0);
  rdm.q = Matrix<double>(n, n, 0.0);
  for (std::size_t i = 0; i < n_occ; ++i) rdm.d(i, i) = rdm.n_occ[i];
  for (std::size_t a = 0; a < n_vir; ++a) rdm.d(n_occ + a, n_occ + a) = rdm.n_vir[a];
  for (std::size_t i = 0; i < n_occ; ++i) rdm.q(i, i) = rdm.n_occ[i];
  for (std::size_t a = 0; a < n_vir; ++a) rdm.q(n_occ + a, n_occ + a) = rdm.n_vir[a];

  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t j = 0; j < n_occ; ++j) {
      if (i == j) continue;
      rdm.d(i, j) = m(i, j);                              // D_ij = x^j_i
      rdm.q(i, j) = 1.0 - m(i, i) - m(j, j);               // Q_ij = 1 - x_i - x_j
    }
  }
  for (std::size_t a = 0; a < n_vir; ++a) {
    for (std::size_t bb = 0; bb < n_vir; ++bb) {
      if (a == bb) continue;
      rdm.d(n_occ + a, n_occ + bb) = nn(a, bb);  // D_ab = x^b_a; Q_ab = 0 (left at 0).
    }
  }
  for (std::size_t i = 0; i < n_occ; ++i) {
    for (std::size_t a = 0; a < n_vir; ++a) {
      rdm.d(n_occ + a, i) = z(i, a);  // D_ai = z^i_a.
      // D_ia = t^a_i + x^a_i - 2 t^a_i (n_a + x_i - t^a_i z^i_a).
      rdm.d(i, n_occ + a) =
          t(i, a) + x_ia(i, a) -
          2.0 * t(i, a) * (rdm.n_vir[a] + m(i, i) - t(i, a) * z(i, a));
      const double q_ia = rdm.n_vir[a] - t(i, a) * z(i, a);  // Q_ia = Q_ai = n_a - t^a_i z^i_a.
      rdm.q(i, n_occ + a) = q_ia;
      rdm.q(n_occ + a, i) = q_ia;
    }
  }
  return rdm;
}

double pccdPairEnergyFromRdm(const PccdCoefficients& coeff, const PccdRdm& rdm) {
  const std::size_t n_occ = coeff.n_occ, n_vir = coeff.n_vir;
  const std::size_t base = coeff.n_core;
  double energy = 0.0;
  for (std::size_t p = 0; p < n_occ; ++p) energy += coeff.eps[base + p] * rdm.n_occ[p];
  for (std::size_t a = 0; a < n_vir; ++a) energy += coeff.eps[base + n_occ + a] * rdm.n_vir[a];

  const std::size_t n = n_occ + n_vir;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      if (p == q) continue;
      const double g_pq = coeff.g(base + p, base + q);
      const double w_pq = coeff.w(base + p, base + q);
      energy += g_pq * rdm.d(p, q) + 0.5 * w_pq * rdm.q(p, q);
    }
  }
  return energy;
}

template PccdCoefficients buildPccdCoefficients(const Matrix<double>&, const Tensor4<double>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<double>&, const CholeskyEri<double>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<double>&, const SymmetricEri<double>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<std::complex<double>>&,
                                                const Tensor4<std::complex<double>>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<std::complex<double>>&,
                                                const CholeskyEri<std::complex<double>>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<std::complex<double>>&,
                                                const SymmetricEri<std::complex<double>>&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
// USE_RI's own FULL_OPTIMIZATION occupation-optimizer extension: the amplitude solve's own
// MO-integral read (model.optimize_occupations, Full_opt/FullOptimization.cpp's makePccdModel)
// is not skippable, so this needs real instantiations too.
template PccdCoefficients buildPccdCoefficients(const Matrix<double>&, const RiNonRelSpinMoEri&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);
template PccdCoefficients buildPccdCoefficients(const Matrix<std::complex<double>>&, const RiMoEri&,
                                                const std::vector<std::size_t>&,
                                                const std::vector<std::size_t>&, std::size_t,
                                                std::size_t, std::size_t);

}  // namespace rerdmft
