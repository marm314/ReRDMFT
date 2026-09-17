#include "HartreeExchangeHessian.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rerdmft {

namespace {

double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

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
// <AB|CD> = <BA|DC>, to reduce to just these two) gives the term
// below. As in hartreeExchangeFockMatrix, F_sp/F*_rq are exactly
// hartreeExchangeFockMatrix's own F (passed in as `fock`, Dyall's
// free-index convention meaning fock(A,B) == F_tex(B,A)). Shared by
// both hartreeExchangeHessianElement (real-real) and
// hartreeExchangeHessianElementImag (imaginary-imaginary), which only
// differ in how they combine four calls to this same term.
template <typename T>
T rawHartreeExchangeHessianTerm(const Matrix<T>& h, const Tensor4<T>& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t n, std::size_t p_,
                                 std::size_t q_, std::size_t r_, std::size_t s_) {
  auto ftex = [&fock](std::size_t a, std::size_t b) { return fock(b, a); };

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
}

template <typename T>
void checkHartreeExchangeHessianDimensions(const Matrix<T>& h, const Tensor4<T>& eri,
                                            const std::vector<double>& occupations,
                                            const Matrix<double>& two_rdm_h,
                                            const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                            std::size_t n, std::size_t p, std::size_t q,
                                            std::size_t r, std::size_t s, const char* caller) {
  if (h.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error(std::string(caller) + ": occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": two_rdm_x dimensions inconsistent with h");
  }
  if (fock.rows() != n || fock.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": fock dimensions inconsistent with h");
  }
  if (p >= n || q >= n || r >= n || s >= n) {
    throw std::runtime_error(std::string(caller) + ": index out of range");
  }
}

}  // namespace

template <typename T>
T hartreeExchangeHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t p, std::size_t q,
                                 std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElement");

  // Antisymmetrize both pairs, matching GeneralizedHessian.h's own
  // combination -- see that header for why.
  return rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, r,
                                        s) -
         rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, s,
                                        r) -
         rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p, r,
                                        s) +
         rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p, s,
                                        r);
}

template <typename T>
T hartreeExchangeHessianElementImag(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                     std::size_t p, std::size_t q, std::size_t r,
                                     std::size_t s) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElementImag");

  // See GeneralizedHessian.h: ALL FOUR terms add, plus an overall minus
  // sign -- the same combination as generalizedOrbitalHessianElementImag.
  return -(rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                          r, s) +
           rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                          s, r) +
           rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p,
                                          r, s) +
           rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p,
                                          s, r));
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
template double hartreeExchangeHessianElementImag(const Matrix<double>& h,
                                                   const Tensor4<double>& eri,
                                                   const std::vector<double>& occupations,
                                                   const Matrix<double>& two_rdm_h,
                                                   const Matrix<double>& two_rdm_x,
                                                   const Matrix<double>& fock, std::size_t p,
                                                   std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> hartreeExchangeHessianElementImag(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s);

// See the header for the derivation and its numerical validation.
// Writing kappa_pq=t+iy, kappa_qp=-t+iy (t on the FIRST pair) and
// kappa_rs=t'+iy', kappa_sr=-t'+iy' (y' on the SECOND pair) and
// expanding the tex's own quadratic term (1/2) sum kappa kappa G_pq,rs
// isolates the coefficient of t*y' as
// i*[G_rs,pq + G_rs,qp - G_sr,pq - G_sr,qp] -- the "y" pair (r,s) sits
// in the flipping (first) slot of each raw term, (p,q) in the fixed
// (second) slot (compare Hess_pq,rs's (+,-,-,+) pattern and Hess^yy's
// (-,-,-,-) pattern, both with (p,q) in the flipping slot -- this one
// is different, confirmed only after the numerical check below caught
// an earlier version with the roles the other way round). Unlike the
// other two combinations, this one needs an EXPLICIT
// multiplication by the imaginary unit (the t*y'/y*t' coefficients sit
// in the IMAGINARY part of the raw complex expansion, unlike t*t'/y*Y',
// which sit in the real part) -- only meaningful for T =
// std::complex<double> (no "i" exists for T = double, and there is no
// y-direction to mix with t- there anyway), so ONLY that explicit
// instantiation is provided below (deliberately, not a real/double one
// too, unlike every other function in this file).
template <typename T>
T hartreeExchangeHessianElementMixed(const Matrix<T>& h, const Tensor4<T>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                      std::size_t p, std::size_t q, std::size_t r,
                                      std::size_t s) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElementMixed");

  // (r,s), not (p,q), is the pair whose order gets flipped for the
  // sign pattern -- see the comment above.
  return T(0.0, 1.0) *
         (rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, r, s,
                                         p, q) +
          rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, r, s,
                                         q, p) -
          rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, s, r,
                                         p, q) -
          rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, s, r,
                                         q, p));
}

template std::complex<double> hartreeExchangeHessianElementMixed(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s);

std::vector<std::pair<std::size_t, std::size_t>> hessianPairIndices(std::size_t n) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(n * (n - 1) / 2);
  for (std::size_t p = 1; p < n; ++p) {
    for (std::size_t q = 0; q < p; ++q) {
      pairs.emplace_back(p, q);
    }
  }
  return pairs;
}

template <typename T>
Matrix<T> hartreeExchangeHessianMatrix(
    const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const std::size_t n_pairs = pair_indices.size();
  Matrix<T> hess(n_pairs, n_pairs, T{});
  // Each (I,J) owns its own disjoint output position and only reads the
  // shared, const inputs -- safe to parallelize.
#pragma omp parallel for collapse(2)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = 0; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      hess(big_i, big_j) =
          hartreeExchangeHessianElement(h, eri, occupations, two_rdm_h, two_rdm_x, fock, p, q, r, s);
    }
  }
  return hess;
}

template Matrix<double> hartreeExchangeHessianMatrix(
    const Matrix<double>& h, const Tensor4<double>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);
template Matrix<std::complex<double>> hartreeExchangeHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft
