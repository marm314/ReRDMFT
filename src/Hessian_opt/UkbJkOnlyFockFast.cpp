#include "UkbJkOnlyFockFast.h"

#include <cmath>
#include <stdexcept>

#include "LinearAlgebra.h"
#include "UkbGenFockPrimitives.h"
#include "UkbJkOnlyFock.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;

// Builds sum_s w(s) * c_ukb(:,s) * conj(c_ukb(:,s)) for a per-orbital weight vector w.
Matrix<C> weightedDensity(const Matrix<C>& c_ukb, const std::vector<double>& w) {
  const std::size_t n_ukb = c_ukb.rows(), n_mo = c_ukb.cols();
  Matrix<C> d(n_ukb, n_ukb, C{});
  for (std::size_t s = 0; s < n_mo; ++s) {
    if (w[s] == 0.0) continue;
    for (std::size_t mu = 0; mu < n_ukb; ++mu) {
      const C c_mu_s = c_ukb(mu, s);
      if (c_mu_s == C{}) continue;
      const C ws_cmus = C(w[s]) * c_mu_s;
      for (std::size_t lambda = 0; lambda < n_ukb; ++lambda) d(mu, lambda) += ws_cmus * std::conj(c_ukb(lambda, s));
    }
  }
  return d;
}

}  // namespace

Matrix<C> ukbJkOnlyFockMatrixFast(const Matrix<C>& h_rkb, const UkbDirectEriSource& eri, const Matrix<C>& c_rkb,
                                   const std::vector<double>& occupations, JkFunctional functional,
                                   std::size_t f_l, double power_alpha) {
  // BBC2/GU (diagonal override breaks separability) and ML/MLSIC (the rational form itself has no
  // finite separable sum) are genuinely O(n_mo) by this scheme -- no decomposition exists, fall
  // through to the already-validated generic path unchanged.
  if (functional == JkFunctional::kBbc2 || functional == JkFunctional::kGu ||
      functional == JkFunctional::kMl || functional == JkFunctional::kMlsic) {
    return ukbJkOnlyFockMatrix(h_rkb, eri, c_rkb, occupations, functional, f_l, power_alpha);
  }

  const std::size_t n_mo = occupations.size();
  if (c_rkb.cols() != n_mo) {
    throw std::runtime_error("ukbJkOnlyFockMatrixFast: c_rkb column count must match occupations.size()");
  }

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_mo

  // `n`: the occupation vector itself. `extra`: the SECOND per-orbital weight vector this
  // functional's exchange term needs (none for SD, which reuses `n` for both H and X). See this
  // file's header comment for the per-functional H/X formula each of these feeds.
  std::vector<double> extra;
  bool need_extra = true;
  switch (functional) {
    case JkFunctional::kSd:
      need_extra = false;
      break;
    case JkFunctional::kMbb:
    case JkFunctional::kMullerAs: {
      extra.resize(n_mo);
      for (std::size_t s = 0; s < n_mo; ++s) extra[s] = std::sqrt(occupations[s]);
      break;
    }
    case JkFunctional::kCa: {
      extra.resize(n_mo);
      for (std::size_t s = 0; s < n_mo; ++s) extra[s] = std::sqrt(occupations[s] * (1.0 - occupations[s]));
      break;
    }
    case JkFunctional::kCga: {
      extra.resize(n_mo);
      for (std::size_t s = 0; s < n_mo; ++s) extra[s] = std::sqrt(occupations[s] * (2.0 - occupations[s]));
      break;
    }
    case JkFunctional::kPower: {
      extra.resize(n_mo);
      for (std::size_t s = 0; s < n_mo; ++s) extra[s] = std::pow(occupations[s], power_alpha);
      break;
    }
    default:
      throw std::runtime_error("ukbJkOnlyFockMatrixFast: unhandled fast-path functional");
  }

  std::vector<Matrix<C>> densities;
  densities.push_back(weightedDensity(c_ukb, occupations));
  if (need_extra) densities.push_back(weightedDensity(c_ukb, extra));

  const UkbGenFockBuild build = ukbGenFockBuild(eri.large_basis, eri.small_basis, densities);

  const Matrix<C> c_ukb_dagger = dagger(c_ukb);
  const Matrix<C> m_j_n = c_ukb_dagger * (build.j[0] * c_ukb);
  const Matrix<C> m_k_n = c_ukb_dagger * (build.k[0] * c_ukb);
  Matrix<C> m_j_extra, m_k_extra;
  if (need_extra) {
    m_j_extra = c_ukb_dagger * (build.j[1] * c_ukb);
    m_k_extra = c_ukb_dagger * (build.k[1] * c_ukb);
  }

  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  Matrix<C> f(n_mo, n_mo, C{});
  for (std::size_t q = 0; q < n_mo; ++q) {
    const double n_q = occupations[q];
    double hartree_q = 0.0;      // coefficient of m_j_n(q,p) in the Hartree term
    double hartree_extra_q = 0.0;  // coefficient of m_j_extra(q,p)
    double exch_n_q = 0.0;       // coefficient of m_k_n(q,p) in the exchange term
    double exch_extra_q = 0.0;   // coefficient of m_k_extra(q,p)
    switch (functional) {
      case JkFunctional::kSd:
        hartree_q = n_q;
        exch_n_q = n_q;
        break;
      case JkFunctional::kMbb:
        hartree_q = n_q;
        exch_extra_q = extra[q];  // sqrt(n_q)
        break;
      case JkFunctional::kCa:
        hartree_q = n_q;
        exch_n_q = n_q;
        exch_extra_q = extra[q];  // sqrt(n_q*(1-n_q))
        break;
      case JkFunctional::kCga:
        hartree_q = n_q;
        exch_n_q = 0.5 * n_q;
        exch_extra_q = 0.5 * extra[q];  // 0.5*sqrt(n_q*(2-n_q))
        break;
      case JkFunctional::kPower:
        hartree_q = n_q;
        exch_extra_q = extra[q];  // n_q^alpha
        break;
      case JkFunctional::kMullerAs:
        hartree_q = 0.5 * n_q;
        hartree_extra_q = -0.5 * extra[q];  // -0.5*sqrt(n_q)
        exch_n_q = 0.5 * n_q;
        exch_extra_q = -0.5 * extra[q];
        break;
      default:
        break;
    }
    for (std::size_t p = 0; p < n_mo; ++p) {
      C hartree = C(hartree_q) * m_j_n(q, p);
      if (need_extra) hartree += C(hartree_extra_q) * m_j_extra(q, p);
      C exch = C(exch_n_q) * m_k_n(q, p);
      if (need_extra) exch += C(exch_extra_q) * m_k_extra(q, p);
      f(p, q) = C(n_q) * h_mo(q, p) + hartree - exch;
    }
  }
  return f;
}

}  // namespace rerdmft
