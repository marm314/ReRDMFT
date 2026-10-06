#include "PccdFock.h"

#include <array>
#include <complex>

#include "CholeskyEri.h"
#include "HartreeExchangeGradient.h"
#include "NonRelSpinRiMoEri.h"
#include "RiMoEri.h"
#include "SymmetricEri.h"

namespace rerdmft {

namespace {

// Q_{a,b} (uniform H/X coupling) and D_{a,b}/D_{b,a} (pair-transfer, generally
// ASYMMETRIC) for two DISTINCT combined-list (core ++ occ ++ vir) local indices a, b --
// the frozen-core extension described in PccdFock.h's own header comment (a core pair
// has x == 0 identically: D to/from it is 0, Q with anything else is the plain
// uncorrelated n_a*n_b product).
struct PairValues {
  double q_ab = 0.0;
  double d_ab = 0.0;  // D_{a,b}
  double d_ba = 0.0;  // D_{b,a}
};

PairValues lookupPair(std::size_t a, std::size_t b, std::size_t n_core, double n_a, double n_b,
                       const PccdRdm& rdm) {
  PairValues v;
  if (a >= n_core && b >= n_core) {
    const std::size_t la = a - n_core, lb = b - n_core;
    v.q_ab = rdm.q(la, lb);
    v.d_ab = rdm.d(la, lb);
    v.d_ba = rdm.d(lb, la);
  } else {
    v.q_ab = n_a * n_b;
    v.d_ab = 0.0;
    v.d_ba = 0.0;
  }
  return v;
}

}  // namespace

PccdFullTwoRdm buildPccdFullTwoRdm(const std::vector<std::size_t>& reps,
                                    const std::vector<std::size_t>& bar, std::size_t n_core,
                                    std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                    const std::vector<double>& occupations, std::size_t n_total) {
  PccdFullTwoRdm result;
  result.two_rdm_h = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_x = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_l1 = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_l2 = Matrix<double>(n_total, n_total, 0.0);

  const std::size_t n_pairs = n_core + n_occ + n_vir;
  for (std::size_t a = 0; a < n_pairs; ++a) {
    const std::size_t i = reps[a];
    const std::size_t ibar = bar[a];
    const double n_a = occupations[i];

    // Same geminal (rep(P)=rep(Q)=a), r=p case of block (a): the two bar-partner
    // combinations both get n_a, no pair-transfer term here (PnofFock.cpp's own
    // identical convention).
    result.two_rdm_h(i, ibar) = n_a;
    result.two_rdm_h(ibar, i) = n_a;
    result.two_rdm_x(i, ibar) = n_a;
    result.two_rdm_x(ibar, i) = n_a;

    for (std::size_t b = a + 1; b < n_pairs; ++b) {
      const std::size_t j = reps[b];
      const std::size_t jbar = bar[b];
      const double n_b = occupations[j];
      const PairValues pv = lookupPair(a, b, n_core, n_a, n_b, rdm);

      const std::array<std::size_t, 2> a_members = {i, ibar};
      const std::array<std::size_t, 2> b_members = {j, jbar};
      for (std::size_t pp = 0; pp < 2; ++pp) {
        for (std::size_t qq = 0; qq < 2; ++qq) {
          const std::size_t P = a_members[pp];
          const std::size_t Q = b_members[qq];
          const bool matching_parity = (pp == qq);

          // Block (b): Q_ab uniform across every bar-parity combination -- see
          // PccdFock.h's own header comment on why pCCD needs no bar-parity branching
          // here (unlike PNOF's NON_REL-only mismatched-exchange special case).
          result.two_rdm_h(P, Q) = pv.q_ab;
          result.two_rdm_h(Q, P) = pv.q_ab;
          result.two_rdm_x(P, Q) = pv.q_ab;
          result.two_rdm_x(Q, P) = pv.q_ab;

          // Block (a), r != p: two_rdm_l1(P,Q) = +-D_{rep(Q),rep(P)}/4 (D_ba/D_ab, NOT
          // symmetric in general, unlike PNOF's Pi_pq). HALF the Gamma element
          // +-D/2 the tex doc gives directly (PccdFock.h's own header comment's naive
          // "direct lookup" derivation) -- exactly PnofFock.cpp's own Pi/4-vs-Pi/2
          // story: every pair-transfer tuple (P,Pbar,Q,Qbar) is visited TWICE by
          // hartreeExchangeEnergy/Fock's own L1+L2 double loop (once as (P,Q)'s L1,
          // once as (Q,Pbar)'s/(P,Qbar)'s L2 via l2=-l1 and the bar-parity sign flip),
          // which together weight the pair term 2x, not 1x -- found empirically
          // (tests/test_pccd_fock.cpp's gd_sum cross-check: the naive D/2 choice gave
          // EXACTLY 2x the correct pair energy), the same factor-of-2 PNOF's own
          // history hit (see PnofFock.cpp's own comment) for the identical reason.
          const double l1_pq = matching_parity ? (pv.d_ba / 4.0) : (-pv.d_ba / 4.0);
          const double l1_qp = matching_parity ? (pv.d_ab / 4.0) : (-pv.d_ab / 4.0);
          result.two_rdm_l1(P, Q) = l1_pq;
          result.two_rdm_l1(Q, P) = l1_qp;
          result.two_rdm_l2(P, Q) = -l1_pq;
          result.two_rdm_l2(Q, P) = -l1_qp;
        }
      }
    }
  }
  return result;
}

std::vector<std::size_t> buildPccdPairOf(const std::vector<std::size_t>& reps,
                                          const std::vector<std::size_t>& bar,
                                          std::size_t n_total) {
  std::vector<std::size_t> pair_of(n_total);
  for (std::size_t p = 0; p < n_total; ++p) pair_of[p] = p;
  for (std::size_t a = 0; a < reps.size(); ++a) {
    pair_of[reps[a]] = bar[a];
    pair_of[bar[a]] = reps[a];
  }
  return pair_of;
}

template <typename T, typename Eri>
Matrix<T> pccdFockMatrix(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                          const std::vector<std::size_t>& bar, std::size_t n_core,
                          std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                          const std::vector<double>& occupations) {
  const std::size_t n = h.rows();
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, n);
  const auto pair_of = buildPccdPairOf(reps, bar, n);
  return hartreeExchangeFockMatrix(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, pair_of,
                                    full.two_rdm_l1, full.two_rdm_l2);
}

template Matrix<double> pccdFockMatrix(const Matrix<double>&, const Tensor4<double>&,
                                        const std::vector<std::size_t>&,
                                        const std::vector<std::size_t>&, std::size_t, std::size_t,
                                        std::size_t, const PccdRdm&, const std::vector<double>&);
template Matrix<double> pccdFockMatrix(const Matrix<double>&, const CholeskyEri<double>&,
                                        const std::vector<std::size_t>&,
                                        const std::vector<std::size_t>&, std::size_t, std::size_t,
                                        std::size_t, const PccdRdm&, const std::vector<double>&);
template Matrix<double> pccdFockMatrix(const Matrix<double>&, const SymmetricEri<double>&,
                                        const std::vector<std::size_t>&,
                                        const std::vector<std::size_t>&, std::size_t, std::size_t,
                                        std::size_t, const PccdRdm&, const std::vector<double>&);
template Matrix<std::complex<double>> pccdFockMatrix(const Matrix<std::complex<double>>&,
                                                      const Tensor4<std::complex<double>>&,
                                                      const std::vector<std::size_t>&,
                                                      const std::vector<std::size_t>&, std::size_t,
                                                      std::size_t, std::size_t, const PccdRdm&,
                                                      const std::vector<double>&);
template Matrix<std::complex<double>> pccdFockMatrix(const Matrix<std::complex<double>>&,
                                                      const CholeskyEri<std::complex<double>>&,
                                                      const std::vector<std::size_t>&,
                                                      const std::vector<std::size_t>&, std::size_t,
                                                      std::size_t, std::size_t, const PccdRdm&,
                                                      const std::vector<double>&);
template Matrix<std::complex<double>> pccdFockMatrix(const Matrix<std::complex<double>>&,
                                                      const SymmetricEri<std::complex<double>>&,
                                                      const std::vector<std::size_t>&,
                                                      const std::vector<std::size_t>&, std::size_t,
                                                      std::size_t, std::size_t, const PccdRdm&,
                                                      const std::vector<double>&);
template Matrix<std::complex<double>> pccdFockMatrix(const Matrix<std::complex<double>>&, const RiMoEri&,
                                                      const std::vector<std::size_t>&,
                                                      const std::vector<std::size_t>&, std::size_t,
                                                      std::size_t, std::size_t, const PccdRdm&,
                                                      const std::vector<double>&);
template Matrix<double> pccdFockMatrix(const Matrix<double>&, const RiNonRelSpinMoEri&,
                                        const std::vector<std::size_t>&,
                                        const std::vector<std::size_t>&, std::size_t, std::size_t,
                                        std::size_t, const PccdRdm&, const std::vector<double>&);

}  // namespace rerdmft
