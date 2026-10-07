#include "RiSigmaGemm.h"

#include <cstdlib>

namespace rerdmft {
namespace risigma {

namespace {

using C = std::complex<double>;

// C = A @ B (row-major, A m x k, B k x n).
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
// C += A^T @ B (A k x m, B k x n, C m x n).
void gemmTNacc(std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c) {
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              1.0, a, static_cast<int>(m), b, static_cast<int>(n), 1.0, c, static_cast<int>(n));
}
void gemmTNacc(std::size_t m, std::size_t n, std::size_t k, const C* a, const C* b, C* c) {
  const C one(1.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
              &one, a, static_cast<int>(m), b, static_cast<int>(n), &one, c, static_cast<int>(n));
}

// Fills dst (pc*n x n, row P*n+t) with the operand's values for aux functions [p0, p0+pc).
template <typename T>
void fillOp(const PairOp<T>& op, const Matrix<T>* post_t, std::size_t n, std::size_t p0, std::size_t pc, T* dst,
            std::vector<T>& scratch) {
  const std::size_t n2 = n * n;
  T* out = post_t ? scratch.data() : dst;
  const bool has_w = !op.w.empty();
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t pl = 0; pl < pc; ++pl)
    for (std::size_t t = 0; t < n; ++t) {
      const std::size_t P = p0 + pl;
      const T* src = op.s ? op.s->data() + P * n2 : nullptr;
      const T rowf = op.row ? (*op.row)(P, t) : T(1.0);
      T* orow = out + (pl * n + t) * n;
      const std::size_t base = t * n;
      for (std::size_t x = 0; x < n; ++x) {
        T v = src ? src[op.idx[base + x]] : T(1.0);
        if (has_w) v *= op.w[base + x];
        if (op.row) v *= rowf;
        orow[x] = v;
      }
    }
  if (post_t) gemmNN(pc * n, n, n, scratch.data(), post_t->data(), dst);
}

}  // namespace

template <typename T>
void pairContract(Matrix<T>& target, double sign, std::size_t n, std::size_t n_aux, const PairOp<T>& f,
                  const PairOp<T>& g) {
  if (n == 0 || n_aux == 0) return;
  const std::size_t n2 = n * n;
  const std::size_t chunk = std::max<std::size_t>(1, std::min<std::size_t>(n_aux, (std::size_t{1} << 21) / n2));
  std::vector<T> fb(chunk * n2), gb(chunk * n2), scratch;
  Matrix<T> f_post, g_post;
  if (f.post || g.post) scratch.resize(chunk * n2);
  if (f.post) {
    f_post = Matrix<T>(n, n);
    for (std::size_t i = 0; i < n2; ++i) f_post.data()[i] = T((*f.post).data()[i]);
  }
  if (g.post) {
    g_post = Matrix<T>(n, n);
    for (std::size_t i = 0; i < n2; ++i) g_post.data()[i] = T((*g.post).data()[i]);
  }
  Matrix<T> R(n, n, T{});
  for (std::size_t p0 = 0; p0 < n_aux; p0 += chunk) {
    const std::size_t pc = std::min(chunk, n_aux - p0);
    fillOp(f, f.post ? &f_post : nullptr, n, p0, pc, fb.data(), scratch);
    fillOp(g, g.post ? &g_post : nullptr, n, p0, pc, gb.data(), scratch);
    gemmTNacc(n, n, pc * n, fb.data(), gb.data(), R.data());
  }
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y) target(x, y) += T(sign) * R(x, y);
}

template <typename T>
void diagContract(Matrix<T>& target, std::size_t n, std::size_t n_aux, const Matrix<T>& S,
                  const typename Identity<Matrix<T>>::type* Ua, const typename Identity<Matrix<T>>::type* Ub) {
  const std::size_t n2 = n * n;
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n; ++b) {
    std::vector<T> col(n, T{});
    for (std::size_t P = 0; P < n_aux; ++P) {
      const T* s = S.data() + P * n2 + b * n;  // S(P, b*n + a), a contiguous
      const T ub = Ub ? (*Ub)(P, b) : T{};
      if (Ua) {
        const T* ua = Ua->data() + P * n;
        for (std::size_t a = 0; a < n; ++a) col[a] += s[a] * (ua[a] + ub);
      } else {
        for (std::size_t a = 0; a < n; ++a) col[a] += s[a] * ub;
      }
    }
    for (std::size_t a = 0; a < n; ++a) target(a, b) += col[a];
  }
}

template <typename T>
void accumU(Matrix<T>& U, double alpha, std::size_t n, std::size_t n_aux, const std::vector<T>& D,
            const Matrix<double>& Wm) {
#pragma omp parallel for schedule(static)
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t x = 0; x < n; ++x) {
      T acc{};
      for (std::size_t t = 0; t < n; ++t) acc += D[P * n + t] * Wm(x, t);
      U(P, x) += T(alpha) * acc;
    }
}

bool& useReferenceFlag() {
  static bool flag = std::getenv("RERDMFT_RI_HV_REFERENCE") != nullptr;  // A/B switch for validation runs
  return flag;
}

template void pairContract<double>(Matrix<double>&, double, std::size_t, std::size_t, const PairOp<double>&,
                                    const PairOp<double>&);
template void pairContract<C>(Matrix<C>&, double, std::size_t, std::size_t, const PairOp<C>&, const PairOp<C>&);
template void diagContract<double>(Matrix<double>&, std::size_t, std::size_t, const Matrix<double>&,
                                    const Matrix<double>*, const Matrix<double>*);
template void diagContract<C>(Matrix<C>&, std::size_t, std::size_t, const Matrix<C>&, const Matrix<C>*,
                               const Matrix<C>*);
template void accumU<double>(Matrix<double>&, double, std::size_t, std::size_t, const std::vector<double>&,
                              const Matrix<double>&);
template void accumU<C>(Matrix<C>&, double, std::size_t, std::size_t, const std::vector<C>&, const Matrix<double>&);

}  // namespace risigma
}  // namespace rerdmft
