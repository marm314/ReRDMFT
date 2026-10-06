#include "UkbPnofFock.h"

#include <stdexcept>

#include "LinearAlgebra.h"
#include "PnofFock.h"
#include "UkbGenFockPrimitives.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> ukbPnofFockMatrix(PnofFunctional functional, const Matrix<C>& h_rkb, const UkbDirectEriSource& eri,
                             const Matrix<C>& c_rkb, const std::vector<PnofGeminal>& geminals,
                             const std::vector<double>& occupations, bool relativistic) {
  const std::size_t n_total = occupations.size();
  if (c_rkb.cols() != n_total) {
    throw std::runtime_error("ukbPnofFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("ukbPnofFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  // Already-validated, existing unfold (Hessian_opt/PnofFock.h) -- no per-geminal separable-sum
  // decomposition hand-derived here, see this file's own header comment.
  const PnofFullTwoRdm rdm = buildPnofFullTwoRdm(functional, geminals, occupations, n_total, relativistic);
  const std::vector<std::size_t> pair_of = buildPnofPairOf(geminals, n_total);

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_total
  const std::size_t n_ukb = c_ukb.rows();

  // Two densities per output row q, not three: Hartree and exchange share ONE density, not two.
  // buildPnofFullTwoRdm's own code sets two_rdm_h(P,Q)=two_rdm_x(P,Q) (and the mirrored (Q,P))
  // to the IDENTICAL value at every assignment it makes (same-geminal n_a; different-geminal
  // h_x_value) -- confirmed directly from that file, not assumed -- so two_rdm_h(s,q) ==
  // two_rdm_x(q,s) for every (s,q), and the two sums below are literally the same matrix. The
  // generic (pre-this-fix) version built them as two separate, numerically-identical density
  // arrays; this reads both the Hartree (J) and exchange (K) accumulator off the SAME entry
  // instead, halving the density count without any new derivation. The two_rdm_l1/l2 pair-transfer
  // term still needs its OWN density per q (see this file's own header / project memory
  // project-ukb-direct-genfock-scheme for its derivation):
  //   D_HX^(q)(sigma,lambda) := sum_s two_rdm_h(s,q) * c_ukb(sigma,s) * conj(c_ukb(lambda,s))
  //   D_L^(q)(mu,lambda)     := sum_s [two_rdm_l1(s,qbar) + two_rdm_l2(s,q)]
  //                             * conj(c_ukb(mu,s)) * conj(c_ukb(lambda, pair_of[s]))   (qbar = pair_of[q])
  // D_L's own small transform needs K[D_L^(q)] directly (built plain, mu then lambda, no
  // transpose) -- the outer C^T(...)C sandwich below is what carries the asymmetry versus H/X,
  // not the density itself.
  std::vector<Matrix<C>> densities(2 * n_total);
  // Each q owns disjoint output slots (densities[q]/[n_total+q]) and only reads the shared, const
  // c_ukb/rdm/pair_of -- safe to parallelize. O(n_total^2 * n_ukb^2) serial here was otherwise the
  // dominant single-threaded cost of the whole UKB-direct build (found via CPU-usage profiling:
  // ukbGenFockBuild's own sectors are already parallelized, but this density-building step, which
  // runs BEFORE it on every call, was not).
#pragma omp parallel for schedule(dynamic)
  for (std::size_t q = 0; q < n_total; ++q) {
    const std::size_t qbar = pair_of[q];
    Matrix<C> d_hx(n_ukb, n_ukb, C{});
    Matrix<C> d_l(n_ukb, n_ukb, C{});  // D_L^(q) itself -- NO transpose needed, see this file's own
                                        // derivation note below (corrected from an earlier, wrong
                                        // derivation that used K[D_L^(q)^T]: the right symmetry is
                                        // the SIMULTANEOUS bra-swap+ket-swap (A,B|C,D)=(B,A|D,C),
                                        // which gives sum_{A,C}(A,B|C,D)M(A,C) = K[M](B,D) directly)
    for (std::size_t s = 0; s < n_total; ++s) {
      const double wh = rdm.two_rdm_h(s, q);
      const double wl = rdm.two_rdm_l1(s, qbar) + rdm.two_rdm_l2(s, q);
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

  // c_ukb^T (plain transpose, NOT conjugate) -- the two_rdm_l1/l2 term's own small transform uses
  // C^T on both legs, not C^dagger (see this file's header comment).
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
