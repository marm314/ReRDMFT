#include "GeneralizedHessian.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

// std::conj(double) promotes to std::complex<double> -- see
// OrbitalGradient.cpp's identical helper for why this matters.
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

// doc/orbital_hessian.tex's "boxed" G_pq,rs (its own Sec. "Full Hessian,
// Fock-matrix form"), evaluated at one (p,q,r,s):
//
//   G_pq,rs = delta_qr F_sp + delta_sp F*_rq - h_sp D_rq - h_qr D_ps
//             - 2 sum_uv <uv|pr> Gamma_uvsq
//             + 2 sum_ut <us|pt> Gamma_ruqt
//             - 2 sum_ut <su|pt> Gamma_ruqt
//             + 2 sum_ut <uq|rt> Gamma_pust
//             - 2 sum_uw <uq|wr> Gamma_pusw
//             + 2 sum_wt <sq|wt> Gamma_rpwt
//
// The tex's own F_ab is built from GeneralizedFock.h's generalizedFockMatrix
// via F_ab = fock(b, a): Dyall's own free-index convention (already
// baked into `fock`, see GeneralizedFock.h) means the tex's own defining
// equation for "F_qp" (its Eq. for the generalized Fock matrix) matches
// this project's `fock(p, q)` with p and q swapped -- confirmed
// numerically (fock(A,B) == F_tex(B,A) to floating-point precision)
// before trusting it here, exactly as `Ftex` below encodes.
template <typename T>
T rawHessianTerm(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                  const Tensor4<T>& two_rdm, const Matrix<T>& fock, std::size_t p, std::size_t q,
                  std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  auto ftex = [&fock](std::size_t a, std::size_t b) { return fock(b, a); };

  T term{};
  if (q == r) term += ftex(s, p);
  if (s == p) term += conjugate(ftex(r, q));
  term -= h(s, p) * d(r, q);
  term -= h(q, r) * d(p, s);

  T sum1{};
  for (std::size_t u = 0; u < n; ++u) {
    for (std::size_t v = 0; v < n; ++v) {
      sum1 += eri(u, v, p, r) * two_rdm(u, v, s, q);
    }
  }
  term -= T(2) * sum1;

  T sum2{};
  T sum3{};
  for (std::size_t u = 0; u < n; ++u) {
    for (std::size_t t = 0; t < n; ++t) {
      sum2 += eri(u, s, p, t) * two_rdm(r, u, q, t);
      sum3 += eri(s, u, p, t) * two_rdm(r, u, q, t);
    }
  }
  term += T(2) * sum2;
  term -= T(2) * sum3;

  T sum4{};
  for (std::size_t u = 0; u < n; ++u) {
    for (std::size_t t = 0; t < n; ++t) {
      sum4 += eri(u, q, r, t) * two_rdm(p, u, s, t);
    }
  }
  term += T(2) * sum4;

  T sum5{};
  for (std::size_t u = 0; u < n; ++u) {
    for (std::size_t w = 0; w < n; ++w) {
      sum5 += eri(u, q, w, r) * two_rdm(p, u, s, w);
    }
  }
  term -= T(2) * sum5;

  T sum6{};
  for (std::size_t w = 0; w < n; ++w) {
    for (std::size_t t = 0; t < n; ++t) {
      sum6 += eri(s, q, w, t) * two_rdm(r, p, w, t);
    }
  }
  term += T(2) * sum6;

  return term;
}

}  // namespace

template <typename T>
T generalizedOrbitalHessianElement(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                    const Tensor4<T>& two_rdm, const Matrix<T>& fock,
                                    std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("generalizedOrbitalHessianElement: h is not square");
  }
  if (d.rows() != n || d.cols() != n) {
    throw std::runtime_error("generalizedOrbitalHessianElement: d dimensions inconsistent with h");
  }
  if (fock.rows() != n || fock.cols() != n) {
    throw std::runtime_error(
        "generalizedOrbitalHessianElement: fock dimensions inconsistent with h");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(
        "generalizedOrbitalHessianElement: eri dimensions inconsistent with h");
  }
  if (two_rdm.dim0() != n || two_rdm.dim1() != n || two_rdm.dim2() != n || two_rdm.dim3() != n) {
    throw std::runtime_error(
        "generalizedOrbitalHessianElement: two_rdm dimensions inconsistent with h");
  }
  if (p >= n || q >= n || r >= n || s >= n) {
    throw std::runtime_error("generalizedOrbitalHessianElement: index out of range");
  }

  // Antisymmetrize both pairs (p,q) and (r,s), matching how
  // OrbitalGradient.h's orbitalGradient antisymmetrizes (p,q) alone --
  // see GeneralizedHessian.h for why this specific 4-term combination
  // is d^2E/dt_pq dt_rs, not the tex's bare G_pq,rs alone.
  return rawHessianTerm(h, eri, d, two_rdm, fock, p, q, r, s) -
         rawHessianTerm(h, eri, d, two_rdm, fock, p, q, s, r) -
         rawHessianTerm(h, eri, d, two_rdm, fock, q, p, r, s) +
         rawHessianTerm(h, eri, d, two_rdm, fock, q, p, s, r);
}

template double generalizedOrbitalHessianElement(const Matrix<double>& h,
                                                  const Tensor4<double>& eri,
                                                  const Matrix<double>& d,
                                                  const Tensor4<double>& two_rdm,
                                                  const Matrix<double>& fock, std::size_t p,
                                                  std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> generalizedOrbitalHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const Matrix<std::complex<double>>& d, const Tensor4<std::complex<double>>& two_rdm,
    const Matrix<std::complex<double>>& fock, std::size_t p, std::size_t q, std::size_t r,
    std::size_t s);

}  // namespace rerdmft
