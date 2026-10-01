#include "pCCD.h"

#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>

#include "CholeskyEri.h"
#include "LBFGS.h"
#include "LinearAlgebra.h"
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

  if (settings.solver == PccdAmplitudeSolver::kNewton) {
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

PccdAmplitudeResult solvePccdTAmplitudes(const PccdCoefficients& coeff, const Matrix<double>& t0,
                                          const PccdSettings& settings) {
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

  PccdAmplitudeResult result = solveResidual(residual_fn, jacobian_fn, z0, settings);
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

}  // namespace rerdmft
