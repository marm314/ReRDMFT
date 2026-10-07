#include "RiGemmFock.h"

#include "HartreeExchangeGradient.h"
#include "JkOnlyFock.h"
#include "RiSigmaGemm.h"

#include <cblas.h>

#include <algorithm>
#include <stdexcept>

namespace rerdmft {

namespace {

using C = std::complex<double>;

// C = alpha * op(A) @ op(B) + beta * C wrappers, one per element type.
void gemmTN(std::size_t m, std::size_t n, std::size_t k, const double* a, std::size_t lda, const double* b,
            std::size_t ldb, double* c, std::size_t ldc) {
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              1.0, a, static_cast<int>(lda), b, static_cast<int>(ldb), 1.0, c, static_cast<int>(ldc));
}
void gemmTN(std::size_t m, std::size_t n, std::size_t k, const C* a, std::size_t lda, const C* b, std::size_t ldb,
            C* c, std::size_t ldc) {
  const C one(1.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              &one, a, static_cast<int>(lda), b, static_cast<int>(ldb), &one, c, static_cast<int>(ldc));
}
void gemmNN(std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), 1.0, a, static_cast<int>(k), b, static_cast<int>(n), 0.0, c, static_cast<int>(n));
}
void gemmNN(std::size_t m, std::size_t n, std::size_t k, const C* a, const C* b, C* c) {
  const C one(1.0, 0.0), zero(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), &one, a, static_cast<int>(k), b, static_cast<int>(n), &zero, c,
              static_cast<int>(n));
}

// b: n_aux x n^2 row-major, b[P*n^2 + p*n + q] = B(P,p,q); eri(a,b,c,d) = sum_P B(P,a,c) B(P,b,d).
//
//   F(p,q) = occ_q h(q,p) + sum_s [ h2(s,q) eri(q,s,p,s) - x2(q,s) eri(q,s,s,p) ]
//                          + sum_s 2 eri(s,sbar,qbar,p) (l1(s,qbar) + l2(s,q))      [if pair_of given]
//
//  * h2 term: eri(q,s,p,s) = sum_P B(P,q,p) Bdiag(P,s)  ->  W(P,q) = sum_s Bdiag(P,s) h2(s,q)  (one small
//    GEMM), then F(p,q) += sum_P B(P,q,p) W(P,q).
//  * x2 and L terms are both sum_{P,t} Y(P,t,q) B(P,t,p) with t = s (x2) or t = sbar (L), so they share one
//    GEMM: Y(P*n+t, q) = -x2(q,t) B(P,q,t) + 2 c(pair[t],q) B(P,pair[t],qbar), c(s,q) = l1(s,qbar)+l2(s,q);
//    B itself, viewed as an (n_aux*n) x n matrix, is the other operand. Done in chunks of aux functions so
//    Y never grows to the size of B.
template <typename T>
Matrix<T> riHxFockGemm(const Matrix<T>& h, const T* b, std::size_t n_aux, const std::vector<double>& occ,
                        const Matrix<double>& h2, const Matrix<double>& x2, const std::vector<std::size_t>& pair_of,
                        const Matrix<double>& l1, const Matrix<double>& l2, const char* who) {
  const std::size_t n = h.rows();
  if (h.cols() != n) throw std::runtime_error(std::string(who) + ": h is not square");
  if (occ.size() != n) throw std::runtime_error(std::string(who) + ": occupations size inconsistent with h");
  if (h2.rows() != n || h2.cols() != n || x2.rows() != n || x2.cols() != n)
    throw std::runtime_error(std::string(who) + ": two_rdm_h/two_rdm_x dimensions inconsistent with h");
  const bool have_l = !pair_of.empty();
  if (have_l && (pair_of.size() != n || l1.rows() != n || l1.cols() != n || l2.rows() != n || l2.cols() != n))
    throw std::runtime_error(std::string(who) + ": pair_of/two_rdm_l1/two_rdm_l2 dimensions inconsistent with h");

  const std::size_t n2 = n * n;
  Matrix<T> f(n, n, T{});
  if (n == 0 || n_aux == 0) {
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) f(p, q) = T(occ[q]) * h(q, p);
    return f;
  }

  // h2 term.
  Matrix<T> bdiag(n_aux, n), h2t(n, n), w(n_aux, n);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t s = 0; s < n; ++s) bdiag(P, s) = b[P * n2 + s * n + s];
  for (std::size_t s = 0; s < n; ++s)
    for (std::size_t q = 0; q < n; ++q) h2t(s, q) = T(h2(s, q));
  gemmNN(n_aux, n, n, bdiag.data(), h2t.data(), w.data());
#pragma omp parallel for schedule(static)
  for (std::size_t q = 0; q < n; ++q) {
    std::vector<T> col(n, T{});
    for (std::size_t P = 0; P < n_aux; ++P) {
      const T wpq = w(P, q);
      const T* row = b + P * n2 + q * n;  // B(P,q,p), p contiguous
      for (std::size_t p = 0; p < n; ++p) col[p] += row[p] * wpq;
    }
    for (std::size_t p = 0; p < n; ++p) f(p, q) += col[p];
  }

  // x2 + L terms, chunked over aux functions.
  const std::size_t chunk = std::max<std::size_t>(1, std::min<std::size_t>(n_aux, (std::size_t{1} << 22) / n2));
  std::vector<T> y(chunk * n * n);
  Matrix<T> g(n, n, T{});
  for (std::size_t p0 = 0; p0 < n_aux; p0 += chunk) {
    const std::size_t pc = std::min(chunk, n_aux - p0);
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t pl = 0; pl < pc; ++pl)
      for (std::size_t t = 0; t < n; ++t) {
        const T* bp = b + (p0 + pl) * n2;
        T* yrow = y.data() + (pl * n + t) * n;
        const std::size_t st = have_l ? pair_of[t] : 0;
        for (std::size_t q = 0; q < n; ++q) {
          T val = -T(x2(q, t)) * bp[q * n + t];
          if (have_l) {
            const std::size_t qbar = pair_of[q];
            val += T(2.0 * (l1(st, qbar) + l2(st, q))) * bp[st * n + qbar];
          }
          yrow[q] = val;
        }
      }
    gemmTN(n, n, pc * n, b + p0 * n2, n, y.data(), n, g.data(), n);
  }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) f(p, q) += T(occ[q]) * h(q, p) + g(p, q);
  return f;
}

}  // namespace

Matrix<C> hartreeExchangeFockMatrix(const Matrix<C>& h, const RiMoEri& eri, const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                     const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
                                     const Matrix<double>& two_rdm_l2) {
  if (eri.dim0() != h.rows()) throw std::runtime_error("hartreeExchangeFockMatrix: eri dimensions inconsistent with h");
  if (risigma::useReference())
    return hartreeExchangeFockMatrix<C, RiMoEri>(h, eri, occupations, two_rdm_h, two_rdm_x, pair_of, two_rdm_l1,
                                                  two_rdm_l2);
  return riHxFockGemm<C>(h, eri.b().data(), eri.b().rows(), occupations, two_rdm_h, two_rdm_x, pair_of, two_rdm_l1,
                          two_rdm_l2, "hartreeExchangeFockMatrix");
}

Matrix<double> hartreeExchangeFockMatrix(const Matrix<double>& h, const RiNonRelSpinMoEri& eri,
                                          const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                                          const Matrix<double>& two_rdm_x, const std::vector<std::size_t>& pair_of,
                                          const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2) {
  if (eri.dim0() != h.rows()) throw std::runtime_error("hartreeExchangeFockMatrix: eri dimensions inconsistent with h");
  if (risigma::useReference())
    return hartreeExchangeFockMatrix<double, RiNonRelSpinMoEri>(h, eri, occupations, two_rdm_h, two_rdm_x, pair_of,
                                                                 two_rdm_l1, two_rdm_l2);
  const Matrix<double> b = eri.embeddedB();
  return riHxFockGemm<double>(h, b.data(), b.rows(), occupations, two_rdm_h, two_rdm_x, pair_of, two_rdm_l1,
                               two_rdm_l2, "hartreeExchangeFockMatrix");
}

Matrix<C> jkOnlyFockMatrix(const Matrix<C>& h, const RiMoEri& eri, const std::vector<double>& occupations,
                            const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x) {
  if (eri.dim0() != h.rows()) throw std::runtime_error("jkOnlyFockMatrix: eri dimensions inconsistent with h");
  if (risigma::useReference()) return jkOnlyFockMatrix<C, RiMoEri>(h, eri, occupations, two_rdm_h, two_rdm_x);
  return riHxFockGemm<C>(h, eri.b().data(), eri.b().rows(), occupations, two_rdm_h, two_rdm_x, {}, Matrix<double>(),
                          Matrix<double>(), "jkOnlyFockMatrix");
}

Matrix<double> jkOnlyFockMatrix(const Matrix<double>& h, const RiNonRelSpinMoEri& eri,
                                 const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                                 const Matrix<double>& two_rdm_x) {
  if (eri.dim0() != h.rows()) throw std::runtime_error("jkOnlyFockMatrix: eri dimensions inconsistent with h");
  if (risigma::useReference())
    return jkOnlyFockMatrix<double, RiNonRelSpinMoEri>(h, eri, occupations, two_rdm_h, two_rdm_x);
  const Matrix<double> b = eri.embeddedB();
  return riHxFockGemm<double>(h, b.data(), b.rows(), occupations, two_rdm_h, two_rdm_x, {}, Matrix<double>(),
                               Matrix<double>(), "jkOnlyFockMatrix");
}

}  // namespace rerdmft
