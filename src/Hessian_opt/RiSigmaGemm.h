#ifndef RERDMFT_RISIGMAGEMM_H
#define RERDMFT_RISIGMAGEMM_H

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Matrix.h"

namespace rerdmft {
namespace risigma {

// GEMM engine behind the RI sigma-vector Hessian-vector products (Hessian_opt/*SigmaHessianVectorRi.cpp).
// The RI tensors (B, KB1, KB2) are (n_aux, n^2) row-major, element [P, i*n + j]; viewed as an
// (n_aux*n) x n matrix (row P*n+i, column j) they are exactly what a GEMM wants, so every
// O(n_aux n^3) contraction below is one GEMM per chunk of aux functions -- no strided P-innermost
// dot products and no extra copy of B.
//
// Pair contraction:  target(x,y) += sign * sum_{P,t} F(P,t,x) * G(P,t,y), with each operand
//   value(P,t,x) = [s ? s(P, idx(t,x)) : 1] * [w ? w(t,x) : 1] * [row ? row(P,t) : 1]
// and, when `post` is set, additionally right-multiplied: value(P,t,x) = sum_a v(P,t,a) post(a,x),
// with v built from the same tables indexed (t,a).
template <typename T>
struct Identity {
  using type = T;
};

template <typename T>
struct PairOp {
  const Matrix<T>* s = nullptr;
  std::vector<std::uint32_t> idx;
  std::vector<double> w;
  const Matrix<T>* row = nullptr;
  const Matrix<double>* post = nullptr;
};

template <typename T, typename FI>
PairOp<T> makeOp(std::size_t n, const Matrix<T>* s, FI idx_fn) {
  PairOp<T> op;
  op.s = s;
  if (s) {
    op.idx.resize(n * n);
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t x = 0; x < n; ++x) op.idx[t * n + x] = static_cast<std::uint32_t>(idx_fn(t, x));
  }
  return op;
}

template <typename T, typename FI, typename FW>
PairOp<T> makeOp(std::size_t n, const Matrix<T>* s, FI idx_fn, FW w_fn) {
  PairOp<T> op = makeOp<T>(n, s, idx_fn);
  op.w.resize(n * n);
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t x = 0; x < n; ++x) op.w[t * n + x] = w_fn(t, x);
  return op;
}

template <typename T>
void pairContract(Matrix<T>& target, double sign, std::size_t n, std::size_t n_aux, const PairOp<T>& f,
                  const PairOp<T>& g);

// target(a,b) += sum_P S(P, b*n + a) * (Ua(P,a) + Ub(P,b))   (either of Ua/Ub may be null).
// O(n_aux n^2), streaming over P, parallel over b.
template <typename T>
void diagContract(Matrix<T>& target, std::size_t n, std::size_t n_aux, const Matrix<T>& S,
                  const typename Identity<Matrix<T>>::type* Ua, const typename Identity<Matrix<T>>::type* Ub);

// U(P,x) += alpha * sum_t D[P*n + t] * Wm(x,t)   (U is n_aux x n).
template <typename T>
void accumU(Matrix<T>& U, double alpha, std::size_t n, std::size_t n_aux, const std::vector<T>& D,
            const Matrix<double>& Wm);

// Validation hook: when true, the RI sigma-vector Hessian-vector products use the original
// element-by-element implementations (the oracle the GEMM versions were validated against).
bool& useReferenceFlag();
inline bool useReference() { return useReferenceFlag(); }

}  // namespace risigma
}  // namespace rerdmft

#endif  // RERDMFT_RISIGMAGEMM_H
