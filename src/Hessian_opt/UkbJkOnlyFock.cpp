#include "UkbJkOnlyFock.h"

#include <stdexcept>

#include "LinearAlgebra.h"
#include "UkbGenFockPrimitives.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> ukbJkOnlyFockMatrix(const Matrix<C>& h_rkb, const UkbDirectEriSource& eri, const Matrix<C>& c_rkb,
                               const std::vector<double>& occupations, JkFunctional functional, std::size_t f_l,
                               double power_alpha) {
  const std::size_t n_mo = occupations.size();
  if (c_rkb.cols() != n_mo) {
    throw std::runtime_error("ukbJkOnlyFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("ukbJkOnlyFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  // Already-validated, existing coupling matrices (Occ_opt/JK_only.h) -- no per-functional
  // separable-sum decomposition is hand-derived here, see this file's own header comment.
  const Matrix<double> two_rdm_h = jkHartreeCoupling(functional, occupations, f_l, power_alpha);
  const Matrix<double> two_rdm_x = jkExchangeCoupling(functional, occupations, f_l, power_alpha);

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_mo
  const std::size_t n_ukb = c_ukb.rows();

  // One UKB-AO density per output row q, for the Hartree term (weighted by two_rdm_h(:,q), i.e.
  // sum over s) and one for the exchange term (weighted by two_rdm_x(q,:)) -- GenFock's own
  // derivation (project memory project-ukb-direct-genfock-scheme):
  //   D_H^(q)(sigma,lambda) = sum_s two_rdm_h(s,q) * c_ukb(sigma,s) * conj(c_ukb(lambda,s))
  //   D_X^(q)(sigma,lambda) = sum_s two_rdm_x(q,s) * c_ukb(sigma,s) * conj(c_ukb(lambda,s))
  // Built as a single flat list (Hartree densities first, then exchange), fed into ONE
  // ukbGenFockBuild call so the expensive integral pass is paid exactly once for the whole GenFock
  // build, regardless of n_mo.
  std::vector<Matrix<C>> densities(2 * n_mo);
  // Each q owns disjoint output slots -- safe to parallelize; this O(n_mo^2 * n_ukb^2) serial loop
  // was otherwise the dominant single-threaded cost of the whole UKB-direct build (ukbGenFockBuild
  // itself is already parallelized, but this density-building step, which runs before it on every
  // call, was not -- found via CPU-usage profiling).
#pragma omp parallel for schedule(dynamic)
  for (std::size_t q = 0; q < n_mo; ++q) {
    Matrix<C> d_h(n_ukb, n_ukb, C{});
    Matrix<C> d_x(n_ukb, n_ukb, C{});
    for (std::size_t s = 0; s < n_mo; ++s) {
      const double wh = two_rdm_h(s, q);
      const double wx = two_rdm_x(q, s);
      if (wh == 0.0 && wx == 0.0) continue;
      for (std::size_t sigma = 0; sigma < n_ukb; ++sigma) {
        const C c_sigma_s = c_ukb(sigma, s);
        if (c_sigma_s == C{}) continue;
        for (std::size_t lambda = 0; lambda < n_ukb; ++lambda) {
          const C term = c_sigma_s * std::conj(c_ukb(lambda, s));
          if (wh != 0.0) d_h(sigma, lambda) += wh * term;
          if (wx != 0.0) d_x(sigma, lambda) += wx * term;
        }
      }
    }
    densities[q] = std::move(d_h);
    densities[n_mo + q] = std::move(d_x);
  }

  const UkbGenFockBuild build = ukbGenFockBuild(eri.large_basis, eri.small_basis, densities);

  // h in the NO/spinor basis, once: h_mo = c_rkb^dagger h_rkb c_rkb.
  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  Matrix<C> f(n_mo, n_mo, C{});
  const Matrix<C> c_ukb_dagger = dagger(c_ukb);
  // Each q writes only its own column of f -- safe to parallelize (no two iterations ever touch
  // the same (p,q) entry).
#pragma omp parallel for schedule(dynamic)
  for (std::size_t q = 0; q < n_mo; ++q) {
    // [C^dagger . J[D_H^(q)] . C](q,p) for every p -- row q of the small n_mo x n_mo transform.
    const Matrix<C> m_h_row = c_ukb_dagger * (build.j[q] * c_ukb);
    const Matrix<C> m_x_row = c_ukb_dagger * (build.k[n_mo + q] * c_ukb);
    for (std::size_t p = 0; p < n_mo; ++p) {
      f(p, q) = C(occupations[q]) * h_mo(q, p) + m_h_row(q, p) - m_x_row(q, p);
    }
  }
  return f;
}

}  // namespace rerdmft
