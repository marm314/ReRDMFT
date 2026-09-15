#include "HartreeExchangeHessian.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

}  // namespace

// Derivation (see the header for the statement): starting from
// GeneralizedHessian.h's boxed G_pq,rs,
//   G_pq,rs = delta_qr F_sp + delta_sp F*_rq - h_sp D_rq - h_qr D_ps
//             - 2 sum_uv <uv|pr> Gamma_uvsq
//             + 2 sum_ut <us|pt> Gamma_ruqt - 2 sum_ut <su|pt> Gamma_ruqt
//             + 2 sum_ut <uq|rt> Gamma_pust - 2 sum_uw <uq|wr> Gamma_pusw
//             + 2 sum_wt <sq|wt> Gamma_rpwt,
// substitute D_rq = n_r delta_rq and, for EACH Gamma appearing above,
//   Gamma(A,B,C,D) = (1/2)[two_rdm_h(A,B) delta_AC delta_BD
//                          - two_rdm_x(A,B) delta_AD delta_BC].
// Each double sum collapses to a single sum once one dummy index is
// pinned by a delta (verified term-by-term against the unsimplified
// GeneralizedHessian.h formula on random data, both before and after
// antisymmetrizing -- see feedback/project memory for the scratch
// scripts, not committed):
//   -2 sum_uv <uv|pr> Gamma_uvsq
//     = -<sq|pr> two_rdm_h(s,q) + <qs|pr> two_rdm_x(q,s)
//   +2 sum_ut <us|pt> Gamma_ruqt - 2 sum_ut <su|pt> Gamma_ruqt
//     = delta_rq sum_u two_rdm_h(r,u) [<us|pu> - <su|pu>]
//       - <qs|pr> two_rdm_x(r,q) + <sq|pr> two_rdm_x(r,q)
//   +2 sum_ut <uq|rt> Gamma_pust - 2 sum_uw <uq|wr> Gamma_pusw
//     = delta_ps sum_u two_rdm_h(p,u) [<uq|ru> - <uq|ur>]
//       - <sq|rp> two_rdm_x(p,s) + <sq|pr> two_rdm_x(p,s)
//   +2 sum_wt <sq|wt> Gamma_rpwt
//     = <sq|rp> two_rdm_h(r,p) - <sq|pr> two_rdm_x(r,p)
// Collecting the four leftover (non-delta) eri(s,q,p,r)/eri(s,q,r,p)
// terms (using <qs|pr> = <sq|rp>, the electron-exchange symmetry
// <AB|CD> = <BA|DC>, to reduce to just these two) gives the boxed
// formula below. As in hartreeExchangeFockMatrix, F_sp/F*_rq are
// exactly hartreeExchangeFockMatrix's own F (passed in as `fock`,
// Dyall's free-index convention meaning fock(A,B) == F_tex(B,A)).
template <typename T>
T hartreeExchangeHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t p, std::size_t q,
                                 std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("hartreeExchangeHessianElement: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("hartreeExchangeHessianElement: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error(
        "hartreeExchangeHessianElement: occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error(
        "hartreeExchangeHessianElement: two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error(
        "hartreeExchangeHessianElement: two_rdm_x dimensions inconsistent with h");
  }
  if (fock.rows() != n || fock.cols() != n) {
    throw std::runtime_error(
        "hartreeExchangeHessianElement: fock dimensions inconsistent with h");
  }
  if (p >= n || q >= n || r >= n || s >= n) {
    throw std::runtime_error("hartreeExchangeHessianElement: index out of range");
  }

  auto ftex = [&fock](std::size_t a, std::size_t b) { return fock(b, a); };

  auto rawTerm = [&](std::size_t p_, std::size_t q_, std::size_t r_, std::size_t s_) -> T {
    T term{};
    if (q_ == r_) term += ftex(s_, p_);
    if (s_ == p_) term += conjugate(ftex(r_, q_));
    if (r_ == q_) term -= h(s_, p_) * T(occupations[r_]);
    if (p_ == s_) term -= h(q_, r_) * T(occupations[p_]);

    if (r_ == q_) {
      T sum{};
      for (std::size_t u = 0; u < n; ++u) {
        sum += T(two_rdm_h(r_, u)) * (eri(u, s_, p_, u) - eri(s_, u, p_, u));
      }
      term += sum;
    }
    if (p_ == s_) {
      T sum{};
      for (std::size_t u = 0; u < n; ++u) {
        sum += T(two_rdm_h(p_, u)) * (eri(u, q_, r_, u) - eri(u, q_, u, r_));
      }
      term += sum;
    }

    const T i1 = eri(s_, q_, p_, r_);
    const T i2 = eri(s_, q_, r_, p_);
    term += i1 * T(two_rdm_x(r_, q_) - two_rdm_h(s_, q_) + two_rdm_x(p_, s_) - two_rdm_x(r_, p_));
    term += i2 * T(two_rdm_h(r_, p_) + two_rdm_x(q_, s_) - two_rdm_x(p_, s_) - two_rdm_x(r_, q_));
    return term;
  };

  // Antisymmetrize both pairs, matching GeneralizedHessian.h's own
  // combination -- see that header for why.
  return rawTerm(p, q, r, s) - rawTerm(p, q, s, r) - rawTerm(q, p, r, s) + rawTerm(q, p, s, r);
}

template double hartreeExchangeHessianElement(const Matrix<double>& h,
                                               const Tensor4<double>& eri,
                                               const std::vector<double>& occupations,
                                               const Matrix<double>& two_rdm_h,
                                               const Matrix<double>& two_rdm_x,
                                               const Matrix<double>& fock, std::size_t p,
                                               std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> hartreeExchangeHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s);

}  // namespace rerdmft
