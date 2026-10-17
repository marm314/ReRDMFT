#include "CasFock.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

using Cx = std::complex<double>;

inline double conjugate(double x) { return x; }
inline Cx conjugate(const Cx& x) { return std::conj(x); }

// C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C, row-major with explicit leading dimensions.
enum class Op { N, T, H };  // none, transpose, conjugate transpose (H == T for real)
inline CBLAS_TRANSPOSE cblasOp(Op o, bool is_complex) {
  if (o == Op::N) return CblasNoTrans;
  if (o == Op::T) return CblasTrans;
  return is_complex ? CblasConjTrans : CblasTrans;
}
void gemm(Op oa, Op ob, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta,
          double* c, int ldc) {
  cblas_dgemm(CblasRowMajor, cblasOp(oa, false), cblasOp(ob, false), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm(Op oa, Op ob, int m, int n, int k, double alpha, const Cx* a, int lda, const Cx* b, int ldb, double beta, Cx* c,
          int ldc) {
  const Cx al(alpha, 0.0), be(beta, 0.0);
  cblas_zgemm(CblasRowMajor, cblasOp(oa, true), cblasOp(ob, true), m, n, k, &al, a, lda, b, ldb, &be, c, ldc);
}
// y = alpha * op(A) x + beta y, A (m x n) row-major.
void gemv(bool trans, int m, int n, const double* a, const double* x, double* y) {
  cblas_dgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n, 1.0, a, n, x, 1, 0.0, y, 1);
}
void gemv(bool trans, int m, int n, const Cx* a, const Cx* x, Cx* y) {
  const Cx one(1.0, 0.0), zero(0.0, 0.0);
  cblas_zgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n, &one, a, n, x, 1, &zero, y, 1);
}

// rho = U diag(w) U^dagger, dropping |w| < 1e-14 (the zero eigenvalues of the virtual block).
void lowRankFactors(const Matrix<double>& rho, std::vector<double>& w, Matrix<double>& u) {
  const SymmetricEigenResult e = diagonalizeSymmetric(rho);
  std::vector<std::size_t> keep;
  for (std::size_t k = 0; k < e.eigenvalues.size(); ++k)
    if (std::abs(e.eigenvalues[k]) > 1e-14) keep.push_back(k);
  w.clear();
  u = Matrix<double>(rho.rows(), keep.size());
  for (std::size_t j = 0; j < keep.size(); ++j) {
    w.push_back(e.eigenvalues[keep[j]]);
    for (std::size_t i = 0; i < rho.rows(); ++i) u(i, j) = e.eigenvectors(i, keep[j]);
  }
}
void lowRankFactors(const Matrix<Cx>& rho, std::vector<double>& w, Matrix<Cx>& u) {
  const HermitianEigenResult e = diagonalizeHermitian(rho);
  std::vector<std::size_t> keep;
  for (std::size_t k = 0; k < e.eigenvalues.size(); ++k)
    if (std::abs(e.eigenvalues[k]) > 1e-14) keep.push_back(k);
  w.clear();
  u = Matrix<Cx>(rho.rows(), keep.size());
  for (std::size_t j = 0; j < keep.size(); ++j) {
    w.push_back(e.eigenvalues[keep[j]]);
    for (std::size_t i = 0; i < rho.rows(); ++i) u(i, j) = e.eigenvectors(i, keep[j]);
  }
}

// ------------------------------------------------------------------------------------------------
// RI form: <ab|cd> = sum_P B(P,a*n+c) B(P,b*n+d)   (no conjugation; B_P Hermitian matrices)
// ------------------------------------------------------------------------------------------------
template <typename T>
Matrix<T> riTwoElectronOperator(const Matrix<T>& b, std::size_t n, const Matrix<T>& rho) {
  const std::size_t n_aux = b.rows(), n2 = n * n;
  // Coulomb: G_J(r,p) = sum_P c_P B(P,r*n+p),  c_P = sum_st rho_st B(P,s*n+t)
  std::vector<T> rho_vec(n2);
  for (std::size_t s = 0; s < n; ++s)
    for (std::size_t t = 0; t < n; ++t) rho_vec[s * n + t] = rho(s, t);
  std::vector<T> c(n_aux), jv(n2);
  gemv(false, static_cast<int>(n_aux), static_cast<int>(n2), b.data(), rho_vec.data(), c.data());
  gemv(true, static_cast<int>(n_aux), static_cast<int>(n2), b.data(), c.data(), jv.data());

  // Exchange: K(r,p) = sum_P sum_k w_k [B_P conj(U)](r,k) [U^T B_P](k,p),  rho = U diag(w) U^dagger
  std::vector<double> w;
  Matrix<T> u;
  lowRankFactors(rho, w, u);
  const std::size_t nk = w.size();
  Matrix<T> k_mat(n, n, T{});
  if (nk > 0) {
    Matrix<T> ucj(n, nk), uwt(nk, n);   // conj(U), diag(w) U^T
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < nk; ++k) {
        ucj(i, k) = conjugate(u(i, k));
        uwt(k, i) = static_cast<double>(w[k]) * u(i, k);
      }
    std::vector<T> acc(n2, T{});
    for (std::size_t P = 0; P < n_aux; ++P) {
      const T* bp = b.data() + P * n2;
      Matrix<T> aa(n, nk), cc(nk, n);
      gemm(Op::N, Op::N, static_cast<int>(n), static_cast<int>(nk), static_cast<int>(n), 1.0, bp, static_cast<int>(n), ucj.data(),
           static_cast<int>(nk), 0.0, aa.data(), static_cast<int>(nk));            // A_P(r,k) = sum_t B_P(r,t) conj(U_tk)
      gemm(Op::N, Op::N, static_cast<int>(nk), static_cast<int>(n), static_cast<int>(n), 1.0, uwt.data(), static_cast<int>(n),
           bp, static_cast<int>(n), 0.0, cc.data(), static_cast<int>(n));          // C_P(k,p) = w_k sum_s U_sk B_P(s,p)
      gemm(Op::N, Op::N, static_cast<int>(n), static_cast<int>(n), static_cast<int>(nk), 1.0, aa.data(), static_cast<int>(nk),
           cc.data(), static_cast<int>(n), 1.0, k_mat.data(), static_cast<int>(n));
    }
  }
  Matrix<T> g(n, n);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t p = 0; p < n; ++p) g(r, p) = jv[r * n + p] - k_mat(r, p);
  return g;
}

template <typename T>
Matrix<T> riActiveCumulantFock(const Matrix<T>& b, std::size_t n, const std::vector<std::size_t>& act, const std::vector<T>& lambda) {
  const std::size_t n_aux = b.rows(), n2 = n * n, a = act.size(), a2 = a * a;
  // Bact[P][(s,t)] = B(P, act[s]*n + act[t])
  Matrix<T> bact(n_aux, a2);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t s = 0; s < a; ++s)
      for (std::size_t t = 0; t < a; ++t) bact(P, s * a + t) = b(P, act[s] * n + act[t]);
  // T[P][(r,q)] = sum_{s,t} Bact[P][(s,t)] lambda[s,t,r,q]
  Matrix<T> tm(n_aux, a2);
  gemm(Op::N, Op::N, static_cast<int>(n_aux), static_cast<int>(a2), static_cast<int>(a2), 1.0, bact.data(), static_cast<int>(a2),
       lambda.data(), static_cast<int>(a2), 0.0, tm.data(), static_cast<int>(a2));
  // F(p,q) = sum_r sum_P B(P, act[r]*n + p) T[P][(r,q)]
  Matrix<T> f(n, a, T{});
  for (std::size_t r = 0; r < a; ++r)
    gemm(Op::T, Op::N, static_cast<int>(n), static_cast<int>(a), static_cast<int>(n_aux), 1.0, b.data() + act[r] * n,
         static_cast<int>(n2), tm.data() + r * a, static_cast<int>(a2), 1.0, f.data(), static_cast<int>(a));
  return f;
}

// (tu|vw) = <tv|uw> = sum_P B(P,tu) B(P,vw)
template <typename T>
std::vector<T> riActiveIntegrals(const Matrix<T>& b, std::size_t n, const std::vector<std::size_t>& act) {
  const std::size_t n_aux = b.rows(), a = act.size(), a2 = a * a;
  Matrix<T> ba(n_aux, a2);
  for (std::size_t P = 0; P < n_aux; ++P)
    for (std::size_t t = 0; t < a; ++t)
      for (std::size_t u = 0; u < a; ++u) ba(P, t * a + u) = b(P, act[t] * n + act[u]);
  std::vector<T> out(a2 * a2);
  gemm(Op::T, Op::N, static_cast<int>(a2), static_cast<int>(a2), static_cast<int>(n_aux), 1.0, ba.data(), static_cast<int>(a2), ba.data(),
       static_cast<int>(a2), 0.0, out.data(), static_cast<int>(a2));
  return out;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Cholesky: <ab|cd> = sum_L V_L(c,a) conj(V_L(b,d)),  w[(x*n+y)*nL + L] = V_L(x,y)
// ------------------------------------------------------------------------------------------------
template <typename T>
Matrix<T> casTwoElectronOperator(const CholeskyEri<T>& eri, const Matrix<T>& rho) {
  const std::size_t n = rho.rows(), nl = eri.nVectors(), n2 = n * n;
  const std::vector<T>& w = eri.rawVectors();
  // Coulomb: J(r,p) = sum_L conj(V_L(r,p)) c_L,  c_L = sum_{t,s} rho_st V_L(t,s)
  std::vector<T> rho_t(n2);
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t s = 0; s < n; ++s) rho_t[t * n + s] = rho(s, t);
  std::vector<T> c(nl, T{});
  gemv(true, static_cast<int>(n2), static_cast<int>(nl), w.data(), rho_t.data(), c.data());
  // conj(V).c = conj(V . conj(c))
  std::vector<T> jc(n2), cc(nl);
  for (std::size_t l = 0; l < nl; ++l) cc[l] = conjugate(c[l]);
  gemv(false, static_cast<int>(n2), static_cast<int>(nl), w.data(), cc.data(), jc.data());

  // Exchange: K(r,p) = sum_{L,k} w_k Y_k(r,L) conj(Y_k(p,L)),  Y_k(r,L) = sum_t conj(U_tk) V_L(t,r)
  std::vector<double> wk;
  Matrix<T> u;
  lowRankFactors(rho, wk, u);
  const std::size_t nk = wk.size();
  Matrix<T> k_mat(n, n, T{});
  if (nk > 0) {
    std::vector<T> y(nk * n * nl);
    gemm(Op::H, Op::N, static_cast<int>(nk), static_cast<int>(n * nl), static_cast<int>(n), 1.0, u.data(), static_cast<int>(nk),
         w.data(), static_cast<int>(n * nl), 0.0, y.data(), static_cast<int>(n * nl));
    for (std::size_t k = 0; k < nk; ++k) {
      const T* yk = y.data() + k * n * nl;
      gemm(Op::N, Op::H, static_cast<int>(n), static_cast<int>(n), static_cast<int>(nl), wk[k], yk, static_cast<int>(nl), yk,
           static_cast<int>(nl), 1.0, k_mat.data(), static_cast<int>(n));
    }
  }
  Matrix<T> g(n, n);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t p = 0; p < n; ++p) g(r, p) = conjugate(jc[r * n + p]) - k_mat(r, p);
  return g;
}

template <typename T>
Matrix<T> casActiveCumulantFock(const CholeskyEri<T>& eri, const std::vector<std::size_t>& act, const std::vector<T>& lambda) {
  const std::size_t n = eri.dim0(), nl = eri.nVectors(), a = act.size(), a2 = a * a;
  const std::vector<T>& w = eri.rawVectors();
  // Vat[(t,s)][L] = V_L(act[t], act[s])
  Matrix<T> vat(a2, nl);
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t s = 0; s < a; ++s)
      std::copy_n(w.data() + (act[t] * n + act[s]) * nl, nl, vat.data() + (t * a + s) * nl);
  // lam2[(r,q)][(t,s)] = lambda[s,t,r,q]
  std::vector<T> lam2(a2 * a2);
  for (std::size_t s = 0; s < a; ++s)
    for (std::size_t t = 0; t < a; ++t)
      for (std::size_t r = 0; r < a; ++r)
        for (std::size_t q = 0; q < a; ++q) lam2[(r * a + q) * a2 + (t * a + s)] = lambda[((s * a + t) * a + r) * a + q];
  // Tt[(r,q)][L] = sum_{t,s} lam2[(r,q)][(t,s)] Vat[(t,s)][L]
  Matrix<T> tt(a2, nl);
  gemm(Op::N, Op::N, static_cast<int>(a2), static_cast<int>(nl), static_cast<int>(a2), 1.0, lam2.data(), static_cast<int>(a2),
       vat.data(), static_cast<int>(nl), 0.0, tt.data(), static_cast<int>(nl));
  // Fc(p,q) = sum_r sum_L Wr(p,L) conj(Tt_r(q,L)),  F = conj(Fc),  Wr(p,L) = V_L(act[r], p)
  Matrix<T> fc(n, a, T{});
  for (std::size_t r = 0; r < a; ++r)
    gemm(Op::N, Op::H, static_cast<int>(n), static_cast<int>(a), static_cast<int>(nl), 1.0, w.data() + act[r] * n * nl,
         static_cast<int>(nl), tt.data() + r * a * nl, static_cast<int>(nl), 1.0, fc.data(), static_cast<int>(a));
  for (std::size_t i = 0; i < n * a; ++i) fc.data()[i] = conjugate(fc.data()[i]);
  return fc;
}

// (tu|vw) = <tv|uw> = sum_L V_L(u,t) conj(V_L(v,w))
template <typename T>
std::vector<T> casActiveIntegrals(const CholeskyEri<T>& eri, const std::vector<std::size_t>& act) {
  const std::size_t n = eri.dim0(), nl = eri.nVectors(), a = act.size(), a2 = a * a;
  const std::vector<T>& w = eri.rawVectors();
  Matrix<T> pa(a2, nl), pb(a2, nl);
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t u = 0; u < a; ++u) {
      std::copy_n(w.data() + (act[u] * n + act[t]) * nl, nl, pa.data() + (t * a + u) * nl);   // V_L(u,t)
      std::copy_n(w.data() + (act[t] * n + act[u]) * nl, nl, pb.data() + (t * a + u) * nl);   // V_L(v,w) rows reused below
    }
  std::vector<T> out(a2 * a2);
  gemm(Op::N, Op::H, static_cast<int>(a2), static_cast<int>(a2), static_cast<int>(nl), 1.0, pa.data(), static_cast<int>(nl), pb.data(),
       static_cast<int>(nl), 0.0, out.data(), static_cast<int>(a2));
  return out;
}

template Matrix<double> casTwoElectronOperator(const CholeskyEri<double>&, const Matrix<double>&);
template Matrix<Cx> casTwoElectronOperator(const CholeskyEri<Cx>&, const Matrix<Cx>&);
template Matrix<double> casActiveCumulantFock(const CholeskyEri<double>&, const std::vector<std::size_t>&, const std::vector<double>&);
template Matrix<Cx> casActiveCumulantFock(const CholeskyEri<Cx>&, const std::vector<std::size_t>&, const std::vector<Cx>&);
template std::vector<double> casActiveIntegrals(const CholeskyEri<double>&, const std::vector<std::size_t>&);
template std::vector<Cx> casActiveIntegrals(const CholeskyEri<Cx>&, const std::vector<std::size_t>&);

// ------------------------------------------------------------------------------------------------
// RI backends
// ------------------------------------------------------------------------------------------------
Matrix<double> casTwoElectronOperator(const RiNonRelSpinMoEri& eri, const Matrix<double>& rho) {
  return riTwoElectronOperator(eri.embeddedB(), eri.dim(), rho);
}
Matrix<double> casActiveCumulantFock(const RiNonRelSpinMoEri& eri, const std::vector<std::size_t>& active, const std::vector<double>& lambda) {
  return riActiveCumulantFock(eri.embeddedB(), eri.dim(), active, lambda);
}
std::vector<double> casActiveIntegrals(const RiNonRelSpinMoEri& eri, const std::vector<std::size_t>& active) {
  return riActiveIntegrals(eri.embeddedB(), eri.dim(), active);
}
Matrix<Cx> casTwoElectronOperator(const RiMoEri& eri, const Matrix<Cx>& rho) {
  return riTwoElectronOperator(eri.b(), eri.dim(), rho);
}
Matrix<Cx> casActiveCumulantFock(const RiMoEri& eri, const std::vector<std::size_t>& active, const std::vector<Cx>& lambda) {
  return riActiveCumulantFock(eri.b(), eri.dim(), active, lambda);
}
std::vector<Cx> casActiveIntegrals(const RiMoEri& eri, const std::vector<std::size_t>& active) {
  return riActiveIntegrals(eri.b(), eri.dim(), active);
}

}  // namespace rerdmft
