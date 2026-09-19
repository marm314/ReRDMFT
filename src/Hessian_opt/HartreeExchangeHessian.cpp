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

// The L1/L2 (PNOF-style pair-term) contribution to GeneralizedHessian.h's
// own boxed G_pq,rs, on top of rawHartreeExchangeHessianTerm's H/X-only
// one above -- see HartreeExchangeHessian.h's own header for the
// derivation method and validation. Substituting the ansatz's pair
// terms, `two_rdm(A,pair_of(A),C,pair_of(C)) += two_rdm_l1(A,C) +
// two_rdm_l2(A,pair_of(C))` (the two individually-named contributions
// ALWAYS occur together at this same tensor entry -- see
// HartreeExchangeGradient.cpp's own F_pq derivation, which combines
// them the identical way), into each of the boxed formula's six raw
// two-electron sums and collapsing every double sum the delta pattern
// allows (exactly as the base H/X derivation did) gives, term by term
// (using the boxed formula's own six terms in order):
//   -2 sum_uv <uv|pr> Gamma_uvsq
//     -> only when q==pair_of(s): -2 sum_u <u,pair_of(u)|pr>
//        [two_rdm_l1(u,s) + two_rdm_l2(u,q)]
//   +2 sum_ut <us|pt> Gamma_ruqt - 2 sum_ut <su|pt> Gamma_ruqt
//     -> both collapse to u=pair_of(r), t=pair_of(q) UNCONDITIONALLY:
//        [+2<pair_of(r),s|p,pair_of(q)> - 2<s,pair_of(r)|p,pair_of(q)>]
//        * [two_rdm_l1(r,q) + two_rdm_l2(r,pair_of(q))]
//   +2 sum_ut <uq|rt> Gamma_pust - 2 sum_uw <uq|wr> Gamma_pusw
//     -> both collapse to u=pair_of(p), t/w=pair_of(s) UNCONDITIONALLY:
//        [+2<pair_of(p),q|r,pair_of(s)> - 2<pair_of(p),q|pair_of(s),r>]
//        * [two_rdm_l1(p,s) + two_rdm_l2(p,pair_of(s))]
//   +2 sum_wt <sq|wt> Gamma_rpwt
//     -> only when p==pair_of(r): +2 sum_w <sq|w,pair_of(w)>
//        [two_rdm_l1(r,w) + two_rdm_l2(r,pair_of(w))]
// Verified against generalizedOrbitalHessianElement fed an explicit
// dense Tensor4 built from the identical full (H+X+L1+L2) ansatz, on
// random data (T=double and T=complex<double>, all four two_rdm_*
// matrices deliberately unequal/asymmetric/unrelated to any occupation
// formula), to machine precision -- see this file's own validation
// notes (not committed).
template <typename T>
T rawL1L2HessianTerm(const Tensor4<T>& eri, const std::vector<std::size_t>& pair_of,
                      const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2,
                      std::size_t n, std::size_t p_, std::size_t q_, std::size_t r_,
                      std::size_t s_) {
  T term{};

  if (q_ == pair_of[s_]) {
    T sum{};
    for (std::size_t u = 0; u < n; ++u) {
      sum += eri(u, pair_of[u], p_, r_) * T(two_rdm_l1(u, s_) + two_rdm_l2(u, q_));
    }
    term -= T(2.0) * sum;
  }

  {
    const T coeff = T(two_rdm_l1(r_, q_) + two_rdm_l2(r_, pair_of[q_]));
    term += T(2.0) * eri(pair_of[r_], s_, p_, pair_of[q_]) * coeff;
    term -= T(2.0) * eri(s_, pair_of[r_], p_, pair_of[q_]) * coeff;
  }

  {
    const T coeff = T(two_rdm_l1(p_, s_) + two_rdm_l2(p_, pair_of[s_]));
    term += T(2.0) * eri(pair_of[p_], q_, r_, pair_of[s_]) * coeff;
    term -= T(2.0) * eri(pair_of[p_], q_, pair_of[s_], r_) * coeff;
  }

  if (p_ == pair_of[r_]) {
    T sum{};
    for (std::size_t w = 0; w < n; ++w) {
      sum += eri(s_, q_, w, pair_of[w]) * T(two_rdm_l1(r_, w) + two_rdm_l2(r_, pair_of[w]));
    }
    term += T(2.0) * sum;
  }

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

// The full BARE G_pq,rs = H/X part (+ L1/L2 pair part when `pair_of` is
// non-empty) that every Hessian combination below is built from.
template <typename T>
T rawFullBareG(const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occupations,
               const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
               const Matrix<T>& fock, std::size_t n, std::size_t p, std::size_t q, std::size_t r,
               std::size_t s, const std::vector<std::size_t>& pair_of,
               const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2) {
  T g = rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, r,
                                       s);
  if (!pair_of.empty()) g += rawL1L2HessianTerm(eri, pair_of, two_rdm_l1, two_rdm_l2, n, p, q, r, s);
  return g;
}

void checkPairTerms(const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
                     const Matrix<double>& two_rdm_l2, std::size_t n, const char* caller) {
  if (pair_of.empty()) return;
  if (pair_of.size() != n) {
    throw std::runtime_error(std::string(caller) + ": pair_of size inconsistent with h");
  }
  if (two_rdm_l1.rows() != n || two_rdm_l1.cols() != n || two_rdm_l2.rows() != n ||
      two_rdm_l2.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": two_rdm_l1/l2 dimensions inconsistent with h");
  }
}

}  // namespace

template <typename T>
T hartreeExchangeHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                 const Matrix<T>& fock, std::size_t p, std::size_t q,
                                 std::size_t r, std::size_t s,
                                 const std::vector<std::size_t>& pair_of,
                                 const Matrix<double>& two_rdm_l1,
                                 const Matrix<double>& two_rdm_l2) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElement");

  // Antisymmetrize both pairs, matching GeneralizedHessian.h's own
  // combination -- see that header for why.
  T result = rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p,
                                            q, r, s) -
             rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p,
                                            q, s, r) -
             rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q,
                                            p, r, s) +
             rawHartreeExchangeHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q,
                                            p, s, r);

  if (!pair_of.empty()) {
    if (pair_of.size() != n) {
      throw std::runtime_error("hartreeExchangeHessianElement: pair_of size inconsistent with h");
    }
    if (two_rdm_l1.rows() != n || two_rdm_l1.cols() != n) {
      throw std::runtime_error(
          "hartreeExchangeHessianElement: two_rdm_l1 dimensions inconsistent with h");
    }
    if (two_rdm_l2.rows() != n || two_rdm_l2.cols() != n) {
      throw std::runtime_error(
          "hartreeExchangeHessianElement: two_rdm_l2 dimensions inconsistent with h");
    }
    result += rawL1L2HessianTerm(eri, pair_of, two_rdm_l1, two_rdm_l2, n, p, q, r, s) -
              rawL1L2HessianTerm(eri, pair_of, two_rdm_l1, two_rdm_l2, n, p, q, s, r) -
              rawL1L2HessianTerm(eri, pair_of, two_rdm_l1, two_rdm_l2, n, q, p, r, s) +
              rawL1L2HessianTerm(eri, pair_of, two_rdm_l1, two_rdm_l2, n, q, p, s, r);
  }
  return result;
}

template <typename T>
T hartreeExchangeHessianElementImag(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x, const Matrix<T>& fock,
                                     std::size_t p, std::size_t q, std::size_t r, std::size_t s,
                                     const std::vector<std::size_t>& pair_of,
                                     const Matrix<double>& two_rdm_l1,
                                     const Matrix<double>& two_rdm_l2) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElementImag");
  checkPairTerms(pair_of, two_rdm_l1, two_rdm_l2, n, "hartreeExchangeHessianElementImag");
  auto bare = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return rawFullBareG(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, a, b, c, d, pair_of,
                        two_rdm_l1, two_rdm_l2);
  };

  // See GeneralizedHessian.h: ALL FOUR terms add, plus an overall minus
  // sign -- the same combination as generalizedOrbitalHessianElementImag.
  return -(bare(p, q, r, s) + bare(p, q, s, r) + bare(q, p, r, s) + bare(q, p, s, r));
}

template double hartreeExchangeHessianElement(
    const Matrix<double>& h, const Tensor4<double>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    std::size_t p, std::size_t q, std::size_t r, std::size_t s,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2);
template std::complex<double> hartreeExchangeHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2);
template double hartreeExchangeHessianElementImag(
    const Matrix<double>& h, const Tensor4<double>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    std::size_t p, std::size_t q, std::size_t r, std::size_t s,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2);
template std::complex<double> hartreeExchangeHessianElementImag(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2);

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
                                      std::size_t p, std::size_t q, std::size_t r, std::size_t s,
                                      const std::vector<std::size_t>& pair_of,
                                      const Matrix<double>& two_rdm_l1,
                                      const Matrix<double>& two_rdm_l2) {
  const std::size_t n = h.rows();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q,
                                         r, s, "hartreeExchangeHessianElementMixed");
  checkPairTerms(pair_of, two_rdm_l1, two_rdm_l2, n, "hartreeExchangeHessianElementMixed");
  auto bare = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return rawFullBareG(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, a, b, c, d, pair_of,
                        two_rdm_l1, two_rdm_l2);
  };

  // (r,s), not (p,q), is the pair whose order gets flipped for the
  // sign pattern -- see the comment above. In sequential terms this is
  // SS_ty(rs;pq) = d/dy_pq of the t-component gradient of pair (r,s).
  return T(0.0, 1.0) * (bare(r, s, p, q) + bare(r, s, q, p) - bare(s, r, p, q) - bare(s, r, q, p));
}

template std::complex<double> hartreeExchangeHessianElementMixed(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2);

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
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2) {
  const std::size_t n_pairs = pair_indices.size();
  Matrix<T> hess(n_pairs, n_pairs, T{});
  // Each (I,J) owns its own disjoint output position and only reads the
  // shared, const inputs -- safe to parallelize.
#pragma omp parallel for collapse(2)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = 0; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      hess(big_i, big_j) = hartreeExchangeHessianElement(
          h, eri, occupations, two_rdm_h, two_rdm_x, fock, p, q, r, s, pair_of, two_rdm_l1,
          two_rdm_l2);
    }
  }
  return hess;
}

template Matrix<double> hartreeExchangeHessianMatrix(
    const Matrix<double>& h, const Tensor4<double>& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2);
template Matrix<std::complex<double>> hartreeExchangeHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2);

Matrix<double> hartreeExchangeJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const std::size_t n_pairs = pair_indices.size();
  Matrix<double> hess(2 * n_pairs, 2 * n_pairs, 0.0);
  // Each (I,J) owns four disjoint output positions (see the header:
  // the y-t block's (n_pairs+J, I) entry is filled from the SAME
  // t-y value computed for (I,J), not from a separately-derived
  // formula) and only reads the shared, const inputs -- safe to
  // parallelize exactly like hartreeExchangeHessianMatrix.
#pragma omp parallel for collapse(2)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = 0; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      const std::complex<double> tt =
          hartreeExchangeHessianElement(h, eri, occupations, two_rdm_h, two_rdm_x, fock, p, q, r, s);
      const std::complex<double> yy = hartreeExchangeHessianElementImag(
          h, eri, occupations, two_rdm_h, two_rdm_x, fock, p, q, r, s);
      const std::complex<double> ty = hartreeExchangeHessianElementMixed(
          h, eri, occupations, two_rdm_h, two_rdm_x, fock, p, q, r, s);
      hess(big_i, big_j) = tt.real();
      hess(n_pairs + big_i, n_pairs + big_j) = yy.real();
      hess(big_i, n_pairs + big_j) = ty.real();
      hess(n_pairs + big_j, big_i) = ty.real();
    }
  }
  return hess;
}

Matrix<double> hartreeExchangeSymmetricJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2) {
  using C = std::complex<double>;
  const std::size_t n = h.rows();
  const std::size_t n_pairs = pair_indices.size();
  checkHartreeExchangeHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, 0, 0,
                                         0, 0, "hartreeExchangeSymmetricJointHessianMatrix");
  checkPairTerms(pair_of, two_rdm_l1, two_rdm_l2, n, "hartreeExchangeSymmetricJointHessianMatrix");
  Matrix<double> hess(2 * n_pairs, 2 * n_pairs, 0.0);
  const C im(0.0, 1.0);
  // Each unordered (I,J), I <= J, owns four disjoint output positions.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = big_i; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      auto bare = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        return rawFullBareG(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, a, b, c, d,
                            pair_of, two_rdm_l1, two_rdm_l2);
      };
      const C g_pq_rs = bare(p, q, r, s), g_pq_sr = bare(p, q, s, r);
      const C g_qp_rs = bare(q, p, r, s), g_qp_sr = bare(q, p, s, r);
      const C g_rs_pq = bare(r, s, p, q), g_rs_qp = bare(r, s, q, p);
      const C g_sr_pq = bare(s, r, p, q), g_sr_qp = bare(s, r, q, p);

      const double tt_ij = (g_pq_rs - g_pq_sr - g_qp_rs + g_qp_sr).real();
      const double tt_ji = (g_rs_pq - g_rs_qp - g_sr_pq + g_sr_qp).real();
      const double yy_ij = -(g_pq_rs + g_pq_sr + g_qp_rs + g_qp_sr).real();
      const double yy_ji = -(g_rs_pq + g_rs_qp + g_sr_pq + g_sr_qp).real();
      // mixed(t_I, y_J) = (1/2)[SS_ty(I;J) + SS_yt(J;I)]
      const double ty_ij = 0.5 * (im * (g_pq_rs + g_pq_sr - g_qp_rs - g_qp_sr) +
                                   im * (g_rs_pq - g_rs_qp + g_sr_pq - g_sr_qp)).real();
      // mixed(t_J, y_I) = (1/2)[SS_ty(J;I) + SS_yt(I;J)]
      const double ty_ji = 0.5 * (im * (g_rs_pq + g_rs_qp - g_sr_pq - g_sr_qp) +
                                   im * (g_pq_rs - g_pq_sr + g_qp_rs - g_qp_sr)).real();

      hess(big_i, big_j) = hess(big_j, big_i) = 0.5 * (tt_ij + tt_ji);
      hess(n_pairs + big_i, n_pairs + big_j) = hess(n_pairs + big_j, n_pairs + big_i) =
          0.5 * (yy_ij + yy_ji);
      hess(big_i, n_pairs + big_j) = hess(n_pairs + big_j, big_i) = ty_ij;
      hess(big_j, n_pairs + big_i) = hess(n_pairs + big_i, big_j) = ty_ji;
    }
  }
  return hess;
}

}  // namespace rerdmft
