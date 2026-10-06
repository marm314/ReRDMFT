#include "UkbPccdFock.h"

#include <stdexcept>

#include "LinearAlgebra.h"
#include "PccdFock.h"
#include "UkbGenFockPrimitives.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> ukbPccdFockMatrix(const Matrix<C>& h_rkb, const UkbDirectEriSource& eri, const Matrix<C>& c_rkb,
                             const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar,
                             std::size_t n_core, std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                             const std::vector<double>& occupations) {
  const std::size_t n_total = occupations.size();
  if (c_rkb.cols() != n_total) {
    throw std::runtime_error("ukbPccdFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("ukbPccdFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  // Already-validated, existing unfold (Hessian_opt/PccdFock.h) -- SAME two_rdm_h/x/l1/l2 shape
  // PnofFock.h's buildPnofFullTwoRdm produces, so the H/X and two_rdm_l1/l2 reductions below are
  // IDENTICAL to UkbPnofFock.cpp's (see project memory project-ukb-direct-genfock-scheme).
  const PccdFullTwoRdm rdm_full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, n_total);
  const std::vector<std::size_t> pair_of = buildPccdPairOf(reps, bar, n_total);

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_total
  const std::size_t n_ukb = c_ukb.rows();

  // Two densities per output row q, not three -- see UkbPnofFock.cpp's own comment: `rdm_full`'s
  // two_rdm_h/x are identical as matrices (buildPccdFullTwoRdm sets both to the SAME value at
  // every assignment, confirmed directly from that file), so Hartree and exchange share ONE
  // density, reading the J/K accumulators off the SAME entry instead of building two numerically
  // identical copies.
  std::vector<Matrix<C>> densities(2 * n_total);
  // Each q owns disjoint output slots -- safe to parallelize; see UkbPnofFock.cpp's own comment
  // (identical structure/reasoning).
#pragma omp parallel for schedule(dynamic)
  for (std::size_t q = 0; q < n_total; ++q) {
    const std::size_t qbar = pair_of[q];
    Matrix<C> d_hx(n_ukb, n_ukb, C{});
    Matrix<C> d_l(n_ukb, n_ukb, C{});
    for (std::size_t s = 0; s < n_total; ++s) {
      const double wh = rdm_full.two_rdm_h(s, q);
      const double wl = rdm_full.two_rdm_l1(s, qbar) + rdm_full.two_rdm_l2(s, q);
      if (wh == 0.0 && wl == 0.0) continue;
      const std::size_t sbar = pair_of[s];
      for (std::size_t mu = 0; mu < n_ukb; ++mu) {
        const C c_mu_s = c_ukb(mu, s);
        const C conj_c_mu_s = std::conj(c_mu_s);
        for (std::size_t lambda = 0; lambda < n_ukb; ++lambda) {
          if (wh != 0.0) d_hx(mu, lambda) += wh * c_mu_s * std::conj(c_ukb(lambda, s));
          if (wl != 0.0) {
            d_l(mu, lambda) += wl * conj_c_mu_s * std::conj(c_ukb(lambda, sbar));
          }
        }
      }
    }
    densities[q] = std::move(d_hx);
    densities[n_total + q] = std::move(d_l);
  }

  const UkbGenFockBuild build = ukbGenFockBuild(eri.large_basis, eri.small_basis, densities);

  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  Matrix<C> c_ukb_t(n_total, n_ukb);
  for (std::size_t a = 0; a < n_total; ++a)
    for (std::size_t b = 0; b < n_ukb; ++b) c_ukb_t(a, b) = c_ukb(b, a);

  Matrix<C> f(n_total, n_total, C{});
  const Matrix<C> c_ukb_dagger = dagger(c_ukb);
  // Each q writes only its own column of f -- safe to parallelize.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t q = 0; q < n_total; ++q) {
    const std::size_t qbar = pair_of[q];
    const Matrix<C> m_h_row = c_ukb_dagger * (build.j[q] * c_ukb);
    const Matrix<C> m_x_row = c_ukb_dagger * (build.k[q] * c_ukb);
    const Matrix<C> m_l_row = c_ukb_t * (build.k[n_total + q] * c_ukb);
    for (std::size_t p = 0; p < n_total; ++p) {
      f(p, q) = C(occupations[q]) * h_mo(q, p) + m_h_row(q, p) - m_x_row(q, p) + C(2.0) * m_l_row(qbar, p);
    }
  }
  return f;
}

}  // namespace rerdmft
