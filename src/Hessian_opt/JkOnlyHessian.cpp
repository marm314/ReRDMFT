#include "JkOnlyHessian.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rerdmft {

namespace {

// The "bare" second-derivative coefficient G_pq,rs = d C_pq / d kappa_rs
// -- see JkOnlyHessian.h for the derivation and the formula.
template <typename T>
T rawJkOnlyG(const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occupations,
             const Matrix<double>& hc, const Matrix<double>& xc, std::size_t n, std::size_t p,
             std::size_t q, std::size_t r, std::size_t s) {
  T term{};

  if (q == r) term += T(occupations[p] - occupations[q]) * h(s, p);
  if (s == p) term -= T(occupations[p] - occupations[q]) * h(q, r);

  if (q == r) {
    T sum{};
    for (std::size_t t = 0; t < n; ++t) {
      sum += T(hc(p, t) - hc(q, t)) * eri(t, s, t, p) - T(xc(p, t) - xc(q, t)) * eri(s, t, t, p);
    }
    term += sum;
  }
  if (s == p) {
    T sum{};
    for (std::size_t t = 0; t < n; ++t) {
      sum += -T(hc(p, t) - hc(q, t)) * eri(t, q, t, r) + T(xc(p, t) - xc(q, t)) * eri(q, t, t, r);
    }
    term += sum;
  }

  term += T(hc(p, r) - hc(q, r) - hc(p, s) + hc(q, s)) * eri(s, q, r, p);
  term += T(xc(p, s) - xc(p, r) - xc(q, s) + xc(q, r)) * eri(q, s, r, p);
  return term;
}

}  // namespace

template <typename T>
T jkOnlyHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  const char* caller = "jkOnlyHessianElement";
  if (h.cols() != n) throw std::runtime_error(std::string(caller) + ": h is not square");
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error(std::string(caller) + ": occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n || two_rdm_x.rows() != n ||
      two_rdm_x.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": coupling matrices inconsistent with h");
  }
  if (p >= n || q >= n || r >= n || s >= n) {
    throw std::runtime_error(std::string(caller) + ": index out of range");
  }

  return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, p, q, r, s) -
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, p, q, s, r) -
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, q, p, r, s) +
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, q, p, s, r);
}

template double jkOnlyHessianElement(const Matrix<double>& h, const Tensor4<double>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, std::size_t p,
                                      std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> jkOnlyHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q, std::size_t r, std::size_t s);

}  // namespace rerdmft
