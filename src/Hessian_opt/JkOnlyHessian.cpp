#include "JkOnlyHessian.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rerdmft {

namespace {

double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

// Forked verbatim from HartreeExchangeHessian.cpp's own
// rawHartreeExchangeHessianTerm (H/X-only path, no L1/L2) -- see that
// file's header comment for the full derivation from
// GeneralizedHessian.h's boxed G_pq,rs. Kept as an independent copy
// (not a shared helper) so future work on JK_only's still-broken
// relativistic Hessian cannot touch PNOF's already-validated formula.
template <typename T>
T rawJkOnlyHessianTerm(const Matrix<T>& h, const Tensor4<T>& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, const Matrix<T>& fock, std::size_t n,
                        std::size_t p_, std::size_t q_, std::size_t r_, std::size_t s_) {
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
void checkJkOnlyHessianDimensions(const Matrix<T>& h, const Tensor4<T>& eri,
                                   const std::vector<double>& occupations,
                                   const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                   const Matrix<T>& fock, std::size_t n, std::size_t p,
                                   std::size_t q, std::size_t r, std::size_t s) {
  const char* caller = "jkOnlyHessianElement";
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
T jkOnlyHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, const Matrix<T>& fock, std::size_t p,
                        std::size_t q, std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  checkJkOnlyHessianDimensions(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, r, s);

  // Antisymmetrize both pairs, matching GeneralizedHessian.h's own
  // combination -- see HartreeExchangeHessian.h for why.
  return rawJkOnlyHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, r, s) -
         rawJkOnlyHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, p, q, s, r) -
         rawJkOnlyHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p, r, s) +
         rawJkOnlyHessianTerm(h, eri, occupations, two_rdm_h, two_rdm_x, fock, n, q, p, s, r);
}

template double jkOnlyHessianElement(const Matrix<double>& h, const Tensor4<double>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
                                      std::size_t p, std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> jkOnlyHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock, std::size_t p,
    std::size_t q, std::size_t r, std::size_t s);

}  // namespace rerdmft
