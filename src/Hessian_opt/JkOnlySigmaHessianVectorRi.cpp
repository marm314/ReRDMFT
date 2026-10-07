#include "JkOnlySigmaHessianVectorRi.h"

#include <complex>
#include <stdexcept>

#include "RiSigmaGemm.h"
#include "RiSigmaHessianToolkit.h"

namespace rerdmft {

namespace {

template <typename T>
struct Tensor3 {
  Tensor3(std::size_t d0, std::size_t d1, std::size_t d2)
      : d0_(d0), d1_(d1), d2_(d2), data_(d0 * d1 * d2, T{}) {}
  T& operator()(std::size_t i, std::size_t j, std::size_t k) { return data_[(i * d1_ + j) * d2_ + k]; }
  T operator()(std::size_t i, std::size_t j, std::size_t k) const { return data_[(i * d1_ + j) * d2_ + k]; }
  std::size_t d0_, d1_, d2_;
  std::vector<T> data_;
};

// Phi(a,b) = sum_{r,s} rawJkOnlyG(a,b,r,s) * K[r,s], built ONLY from `t` (the RI toolkit). `T` is
// `h`/`t`'s own scalar type (double for NON_REL, complex<double> for C4_SPINOR/X2C); `K` is
// always real.
template <typename T>
Matrix<T> buildPhiRiRef(const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hc,
                      const Matrix<double>& xc, const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  Matrix<T> Phi(n, n, T{});

  // T12: commutator, no RI at all.
  {
    Matrix<T> KH(n, n, T{}), HK(n, n, T{});
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        T acc1{}, acc2{};
        for (std::size_t k = 0; k < n; ++k) {
          acc1 += T(K(i, k)) * h(k, j);
          acc2 += h(i, k) * T(K(k, j));
        }
        KH(i, j) = acc1;
        HK(i, j) = acc2;
      }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) Phi(p, q) += T(occ[p] - occ[q]) * (KH(q, p) - HK(q, p));
  }

  // T3 Hc: V_h[t,p,q] = sum_P Bdiag[P,t]*KB1[P,q,p]
  {
    Tensor3<T> V_h(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.Bdiag(P, tt) * t.KB1(P, q, p);
          V_h(tt, p, q) = acc;
        }
    #pragma omp parallel for collapse(2)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t tt = 0; tt < n; ++tt) acc += T(hc(p, tt) - hc(q, tt)) * V_h(tt, p, q);
        Phi(p, q) += acc;
      }
  }
  // T3 Xc: V_x[t,p,q] = sum_P B[P,t,p]*KB1[P,q,t]
  {
    Tensor3<T> V_x(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, tt, p) * t.KB1(P, q, tt);
          V_x(tt, p, q) = acc;
        }
    #pragma omp parallel for collapse(2)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t tt = 0; tt < n; ++tt) acc += T(xc(p, tt) - xc(q, tt)) * V_x(tt, p, q);
        Phi(p, q) -= acc;
      }
  }
  // T4 Hc: W_h[t,p,q] = sum_P Bdiag[P,t]*KB2[P,q,p]
  {
    Tensor3<T> W_h(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.Bdiag(P, tt) * t.KB2(P, q, p);
          W_h(tt, p, q) = acc;
        }
    #pragma omp parallel for collapse(2)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t tt = 0; tt < n; ++tt) acc += T(hc(p, tt) - hc(q, tt)) * W_h(tt, p, q);
        Phi(p, q) -= acc;
      }
  }
  // T4 Xc: W_x[t,p,q] = sum_P B[P,q,t]*KB2[P,t,p]
  {
    Tensor3<T> W_x(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, q, tt) * t.KB2(P, tt, p);
          W_x(tt, p, q) = acc;
        }
    #pragma omp parallel for collapse(2)
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t tt = 0; tt < n; ++tt) acc += T(xc(p, tt) - xc(q, tt)) * W_x(tt, p, q);
        Phi(p, q) += acc;
      }
  }
  // TA: M_A[r,q,p] = sum_P B[P,q,p]*KB1diag[P,r]; M_A2[s,q,p] = sum_P B[P,q,p]*KB2diag[P,s]
  {
    #pragma omp parallel for collapse(2)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t p = 0; p < n; ++p) {
        std::vector<T> row_B(n_aux);
        for (std::size_t P = 0; P < n_aux; ++P) row_B[P] = t.B(P, q, p);
        T acc_total{};
        for (std::size_t r = 0; r < n; ++r) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += row_B[P] * t.KB1diag(P, r);
          acc_total += T(hc(p, r) - hc(q, r)) * acc;
        }
        for (std::size_t s = 0; s < n; ++s) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += row_B[P] * t.KB2diag(P, s);
          acc_total += T(-hc(p, s) + hc(q, s)) * acc;
        }
        Phi(p, q) += acc_total;
      }
  }
  // TB: M_B[s,q,p] = sum_P KB2[P,q,s]*B[P,s,p]; M_B2[r,q,p] = sum_P B[P,q,r]*KB1[P,r,p]
  {
    #pragma omp parallel for collapse(2)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t p = 0; p < n; ++p) {
        T acc_total{};
        for (std::size_t s = 0; s < n; ++s) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.KB2(P, q, s) * t.B(P, s, p);
          acc_total += T(xc(p, s) - xc(q, s)) * acc;
        }
        for (std::size_t r = 0; r < n; ++r) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, q, r) * t.KB1(P, r, p);
          acc_total += T(-xc(p, r) + xc(q, r)) * acc;
        }
        Phi(p, q) += acc_total;
      }
  }

  return Phi;
}

// Psi(c,d) = sum_{a,b} rawJkOnlyG(a,b,c,d) * K[a,b], built ONLY from `t`.
template <typename T>
Matrix<T> buildPsiRiRef(const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hc,
                      const Matrix<double>& xc, const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  Matrix<T> Psi(n, n, T{});

  #pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c) * (occ[a] - occ[c])) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) acc -= T(K(d, b) * (occ[d] - occ[b])) * h(b, c);
      Psi(c, d) += acc;
    }

  // P3: eri_tdta[t,d,a] = sum_P Bdiag[P,t]*B[P,d,a]; eri_dtta[d,t,a] = sum_P B[P,d,t]*B[P,t,a]
  {
    Tensor3<T> eri_tdta(n, n, n), eri_dtta(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t d = 0; d < n; ++d)
        for (std::size_t a = 0; a < n; ++a) {
          T acc1{}, acc2{};
          for (std::size_t P = 0; P < n_aux; ++P) {
            acc1 += t.Bdiag(P, tt) * t.B(P, d, a);
            acc2 += t.B(P, d, tt) * t.B(P, tt, a);
          }
          eri_tdta(tt, d, a) = acc1;
          eri_dtta(d, tt, a) = acc2;
        }
    Matrix<T> term3a(n, n, T{}), term3c(n, n, T{});
    #pragma omp parallel for collapse(2)
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t d = 0; d < n; ++d) {
        T acc_h{}, acc_x{};
        for (std::size_t tt = 0; tt < n; ++tt) {
          acc_h += T(hc(a, tt)) * eri_tdta(tt, d, a);
          acc_x += T(xc(a, tt)) * eri_dtta(d, tt, a);
        }
        term3a(a, d) = acc_h;
        term3c(a, d) = acc_x;
      }
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t a = 0; a < n; ++a) {
          T term3b_pre{}, term3d_pre{};
          for (std::size_t tt = 0; tt < n; ++tt) {
            term3b_pre += T(hc(c, tt)) * eri_tdta(tt, d, a);
            term3d_pre += T(xc(c, tt)) * eri_dtta(d, tt, a);
          }
          acc += T(K(a, c)) * term3a(a, d) - T(K(a, c)) * term3b_pre - T(K(a, c)) * term3c(a, d) +
                 T(K(a, c)) * term3d_pre;
        }
        Psi(c, d) += acc;
      }
  }
  // P4: eri_tbtc[t,b,c] = sum_P Bdiag[P,t]*B[P,b,c]; eri_bttc[b,t,c] = sum_P B[P,b,t]*B[P,t,c]
  {
    Tensor3<T> eri_tbtc(n, n, n), eri_bttc(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t tt = 0; tt < n; ++tt)
      for (std::size_t b = 0; b < n; ++b)
        for (std::size_t c = 0; c < n; ++c) {
          T acc1{}, acc2{};
          for (std::size_t P = 0; P < n_aux; ++P) {
            acc1 += t.Bdiag(P, tt) * t.B(P, b, c);
            acc2 += t.B(P, b, tt) * t.B(P, tt, c);
          }
          eri_tbtc(tt, b, c) = acc1;
          eri_bttc(b, tt, c) = acc2;
        }
    Matrix<T> term4b(n, n, T{}), term4d(n, n, T{});
    #pragma omp parallel for collapse(2)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c) {
        T acc_h{}, acc_x{};
        for (std::size_t tt = 0; tt < n; ++tt) {
          acc_h += T(hc(b, tt)) * eri_tbtc(tt, b, c);
          acc_x += T(xc(b, tt)) * eri_bttc(b, tt, c);
        }
        term4b(b, c) = acc_h;
        term4d(b, c) = acc_x;
      }
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t b = 0; b < n; ++b) {
          T term4a_pre{}, term4c_pre{};
          for (std::size_t tt = 0; tt < n; ++tt) {
            term4a_pre += T(hc(d, tt)) * eri_tbtc(tt, b, c);
            term4c_pre += T(xc(d, tt)) * eri_bttc(b, tt, c);
          }
          acc += T(K(d, b)) * (-(term4a_pre - term4b(b, c)) + (term4c_pre - term4d(b, c)));
        }
        Psi(c, d) += acc;
      }
  }
  // P5: M5a[b,c,d] = sum_P B[P,d,c]*KB2diag[P,b]; M5b[a,c,d] = sum_P B[P,d,c]*KB1diag[P,a]
  {
    Tensor3<T> M5a(n, n, n), M5b(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        for (std::size_t b = 0; b < n; ++b) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, d, c) * t.KB2diag(P, b);
          M5a(b, c, d) = acc;
        }
        for (std::size_t a = 0; a < n; ++a) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, d, c) * t.KB1diag(P, a);
          M5b(a, c, d) = acc;
        }
      }
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t a = 0; a < n; ++a) acc += T(hc(a, c)) * M5b(a, c, d) - T(hc(a, d)) * M5b(a, c, d);
        for (std::size_t b = 0; b < n; ++b) acc += T(-hc(b, c)) * M5a(b, c, d) + T(hc(b, d)) * M5a(b, c, d);
        Psi(c, d) += acc;
      }
  }
  // P6: M6a[b,c,d] = sum_P B[P,b,c]*KB2[P,d,b]; M6b[a,c,d] = sum_P B[P,d,a]*KB1[P,a,c]
  {
    Tensor3<T> M6a(n, n, n), M6b(n, n, n);
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        for (std::size_t b = 0; b < n; ++b) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, b, c) * t.KB2(P, d, b);
          M6a(b, c, d) = acc;
        }
        for (std::size_t a = 0; a < n; ++a) {
          T acc{};
          for (std::size_t P = 0; P < n_aux; ++P) acc += t.B(P, d, a) * t.KB1(P, a, c);
          M6b(a, c, d) = acc;
        }
      }
    #pragma omp parallel for collapse(2)
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t a = 0; a < n; ++a) acc += T(xc(a, d)) * M6b(a, c, d) - T(xc(a, c)) * M6b(a, c, d);
        for (std::size_t b = 0; b < n; ++b) acc += T(-xc(b, d)) * M6a(b, c, d) + T(xc(b, c)) * M6a(b, c, d);
        Psi(c, d) += acc;
      }
  }

  return Psi;
}


// ---- GEMM versions (production); the *Ref functions above stay as the validation oracle.
template <typename T>
Matrix<T> negatedM(const Matrix<T>& m) {
  Matrix<T> r(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows() * m.cols(); ++i) r.data()[i] = -m.data()[i];
  return r;
}

template <typename T>
Matrix<T> buildPhiRi(const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hc,
                      const Matrix<double>& xc, const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return buildPhiRiRef(h, occ, hc, xc, K, t);
  using risigma::accumU;
  using risigma::diagContract;
  using risigma::makeOp;
  using risigma::pairContract;
  using Z = std::size_t;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;
  Matrix<T> Phi(n, n, T{});

  // T12: commutator, no RI at all.
  {
    Matrix<T> KH(n, n, T{}), HK(n, n, T{});
#pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        T acc1{}, acc2{};
        for (std::size_t k = 0; k < n; ++k) {
          acc1 += T(K(i, k)) * h(k, j);
          acc2 += h(i, k) * T(K(k, j));
        }
        KH(i, j) = acc1;
        HK(i, j) = acc2;
      }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) Phi(p, q) += T(occ[p] - occ[q]) * (KH(q, p) - HK(q, p));
  }

  // Uh(P,x) = sum_t Bdiag(P,t) hc(x,t)
  Matrix<T> Uh(n_aux, n, T{});
  accumU(Uh, 1.0, n, n_aux, t.bdiag, hc);
  const Matrix<T> nUh = negatedM(Uh);

  // T3 Hc: sum_P KB1(P,q,p) (Uh(P,p) - Uh(P,q))
  diagContract(Phi, n, n_aux, K1, &Uh, &nUh);
  // T3 Xc: -sum_t (xc(p,t) - xc(q,t)) B(P,t,p) KB1(P,q,t)
  pairContract(Phi, -1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](Z i, Z p) { return i * n + p; }, [&](Z i, Z p) { return xc(p, i); }),
               makeOp<T>(n, &K1, [&](Z i, Z q) { return q * n + i; }));
  pairContract(Phi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z i, Z p) { return i * n + p; }),
               makeOp<T>(n, &K1, [&](Z i, Z q) { return q * n + i; }, [&](Z i, Z q) { return xc(q, i); }));
  // T4 Hc: -sum_P KB2(P,q,p) (Uh(P,p) - Uh(P,q))
  diagContract(Phi, n, n_aux, K2, &nUh, &Uh);
  // T4 Xc: sum_t (xc(p,t) - xc(q,t)) B(P,q,t) KB2(P,t,p)
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](Z i, Z p) { return i * n + p; }, [&](Z i, Z p) { return xc(p, i); }),
               makeOp<T>(n, &Bm, [&](Z i, Z q) { return q * n + i; }));
  pairContract(Phi, -1.0, n, n_aux, makeOp<T>(n, &K2, [&](Z i, Z p) { return i * n + p; }),
               makeOp<T>(n, &Bm, [&](Z i, Z q) { return q * n + i; }, [&](Z i, Z q) { return xc(q, i); }));
  // TA: sum_P B(P,q,p) (V(P,p) - V(P,q)), V(P,x) = sum_r (KB1diag - KB2diag)(P,r) hc(x,r)
  {
    Matrix<T> V(n_aux, n, T{});
    accumU(V, 1.0, n, n_aux, t.kb1diag, hc);
    accumU(V, -1.0, n, n_aux, t.kb2diag, hc);
    const Matrix<T> nV = negatedM(V);
    diagContract(Phi, n, n_aux, Bm, &V, &nV);
  }
  // TB
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](Z i, Z p) { return i * n + p; }, [&](Z i, Z p) { return xc(p, i); }),
               makeOp<T>(n, &K2, [&](Z i, Z q) { return q * n + i; }));
  pairContract(Phi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z i, Z p) { return i * n + p; }),
               makeOp<T>(n, &K2, [&](Z i, Z q) { return q * n + i; }, [&](Z i, Z q) { return xc(q, i); }));
  pairContract(Phi, -1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](Z i, Z p) { return i * n + p; }, [&](Z i, Z p) { return xc(p, i); }),
               makeOp<T>(n, &Bm, [&](Z i, Z q) { return q * n + i; }));
  pairContract(Phi, 1.0, n, n_aux, makeOp<T>(n, &K1, [&](Z i, Z p) { return i * n + p; }),
               makeOp<T>(n, &Bm, [&](Z i, Z q) { return q * n + i; }, [&](Z i, Z q) { return xc(q, i); }));
  return Phi;
}

template <typename T>
Matrix<T> buildPsiRi(const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hc,
                      const Matrix<double>& xc, const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return buildPsiRiRef(h, occ, hc, xc, K, t);
  using risigma::accumU;
  using risigma::diagContract;
  using risigma::makeOp;
  using risigma::pairContract;
  using Z = std::size_t;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;
  Matrix<T> Psi(n, n, T{});

#pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c) * (occ[a] - occ[c])) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) acc -= T(K(d, b) * (occ[d] - occ[b])) * h(b, c);
      Psi(c, d) += acc;
    }

  Matrix<T> Uh(n_aux, n, T{});
  accumU(Uh, 1.0, n, n_aux, t.bdiag, hc);
  const Matrix<T> nUh = negatedM(Uh);

  // P3 (i): sum_{P,a} K(a,c) Uh(P,a) B(P,d,a)
  {
    auto f = makeOp<T>(n, static_cast<const Matrix<T>*>(nullptr), [](Z, Z) { return 0; },
                       [&](Z a, Z c) { return K(a, c); });
    f.row = &Uh;
    pairContract(Psi, 1.0, n, n_aux, f, makeOp<T>(n, &Bm, [&](Z a, Z d) { return d * n + a; }));
  }
  // P3 (ii): -sum_P KB2(P,d,c) Uh(P,c)
  diagContract(Psi, n, n_aux, K2, &nUh, nullptr);
  // P3 (iii): -sum_{P,t} Y_x(P,t,c) B(P,d,t), Y_x(P,t,c) = sum_a B(P,t,a) xc(a,t) K(a,c)
  {
    auto f = makeOp<T>(n, &Bm, [&](Z i, Z a) { return i * n + a; }, [&](Z i, Z a) { return xc(a, i); });
    f.post = &K;
    pairContract(Psi, -1.0, n, n_aux, f, makeOp<T>(n, &Bm, [&](Z i, Z d) { return d * n + i; }));
  }
  // P3 (iv): sum_{P,t} xc(c,t) KB2(P,t,c) B(P,d,t)
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](Z i, Z c) { return i * n + c; }, [&](Z i, Z c) { return xc(c, i); }),
               makeOp<T>(n, &Bm, [&](Z i, Z d) { return d * n + i; }));
  // P4 (i): -sum_P KB1(P,d,c) Uh(P,d)
  diagContract(Psi, n, n_aux, K1, nullptr, &nUh);
  // P4 (ii): sum_{P,b} Uh(P,b) B(P,b,c) K(d,b)
  {
    auto f = makeOp<T>(n, &Bm, [&](Z b, Z c) { return b * n + c; });
    f.row = &Uh;
    auto g = makeOp<T>(n, static_cast<const Matrix<T>*>(nullptr), [](Z, Z) { return 0; },
                       [&](Z b, Z d) { return K(d, b); });
    pairContract(Psi, 1.0, n, n_aux, f, g);
  }
  // P4 (iii): sum_{P,t} xc(d,t) KB1(P,d,t) B(P,t,c)
  pairContract(Psi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z i, Z c) { return i * n + c; }),
               makeOp<T>(n, &K1, [&](Z i, Z d) { return d * n + i; }, [&](Z i, Z d) { return xc(d, i); }));
  // P4 (iv): -sum_{P,t} B(P,t,c) Y2(P,t,d), Y2(P,t,d) = sum_b B(P,b,t) xc(b,t) K(d,b)
  {
    const Matrix<double> Kt = transpose(K);
    auto g = makeOp<T>(n, &Bm, [&](Z i, Z b) { return b * n + i; }, [&](Z i, Z b) { return xc(b, i); });
    g.post = &Kt;
    pairContract(Psi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z i, Z c) { return i * n + c; }), g);
  }
  // P5: sum_P B(P,d,c) (V(P,c) - V(P,d)), V(P,x) = sum_a (KB1diag - KB2diag)(P,a) hc(a,x)
  {
    const Matrix<double> hcT = transpose(hc);
    Matrix<T> V(n_aux, n, T{});
    accumU(V, 1.0, n, n_aux, t.kb1diag, hcT);
    accumU(V, -1.0, n, n_aux, t.kb2diag, hcT);
    const Matrix<T> nV = negatedM(V);
    diagContract(Psi, n, n_aux, Bm, &V, &nV);
  }
  // P6
  pairContract(Psi, 1.0, n, n_aux, makeOp<T>(n, &K1, [&](Z i, Z c) { return i * n + c; }),
               makeOp<T>(n, &Bm, [&](Z i, Z d) { return d * n + i; }, [&](Z i, Z d) { return xc(i, d); }));
  pairContract(Psi, -1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](Z i, Z c) { return i * n + c; }, [&](Z i, Z c) { return xc(i, c); }),
               makeOp<T>(n, &Bm, [&](Z i, Z d) { return d * n + i; }));
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](Z i, Z c) { return i * n + c; }, [&](Z i, Z c) { return xc(i, c); }),
               makeOp<T>(n, &K2, [&](Z i, Z d) { return d * n + i; }));
  pairContract(Psi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z i, Z c) { return i * n + c; }),
               makeOp<T>(n, &K2, [&](Z i, Z d) { return d * n + i; }, [&](Z i, Z d) { return xc(i, d); }));
  return Psi;
}

}  // namespace

std::vector<double> jkOnlySigmaHessianVectorRi(
    const Matrix<double>& h, const Matrix<double>& b, std::size_t n, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v) {
  if (h.rows() != n || h.cols() != n) throw std::runtime_error("jkOnlySigmaHessianVectorRi: h has the wrong shape");
  if (occupations.size() != n) throw std::runtime_error("jkOnlySigmaHessianVectorRi: occupations has the wrong size");
  if (v.size() != pair_indices.size()) throw std::runtime_error("jkOnlySigmaHessianVectorRi: v has the wrong size");

  Matrix<double> K(n, n, 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    K(p, q) = v[i];
    K(q, p) = -v[i];
  }

  const RiSigmaHessianToolkit<double> toolkit = buildRiSigmaHessianToolkit(n, b, K);
  const Matrix<double> Phi = buildPhiRi(h, occupations, two_rdm_h, two_rdm_x, K, toolkit);
  const Matrix<double> Psi = buildPsiRi(h, occupations, two_rdm_h, two_rdm_x, K, toolkit);

  std::vector<double> w(pair_indices.size(), 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    w[i] = 0.5 * ((Phi(p, q) - Phi(q, p)) + (Psi(p, q) - Psi(q, p)));
  }
  return w;
}

std::vector<double> jkOnlySigmaJointHessianVectorRi(
    const Matrix<std::complex<double>>& h, const Matrix<std::complex<double>>& b, std::size_t n,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v) {
  using C = std::complex<double>;
  if (h.rows() != n || h.cols() != n) {
    throw std::runtime_error("jkOnlySigmaJointHessianVectorRi: h has the wrong shape");
  }
  const std::size_t n_pairs = pair_indices.size();
  if (v.size() != 2 * n_pairs) {
    throw std::runtime_error("jkOnlySigmaJointHessianVectorRi: v has the wrong size");
  }

  Matrix<double> K_t(n, n, 0.0), K_y(n, n, 0.0);
  for (std::size_t i = 0; i < n_pairs; ++i) {
    const auto& [p, q] = pair_indices[i];
    K_t(p, q) = v[i];
    K_t(q, p) = -v[i];
    K_y(p, q) = v[n_pairs + i];
    K_y(q, p) = v[n_pairs + i];
  }

  const RiSigmaHessianToolkit<C> toolkit_t = buildRiSigmaHessianToolkit(n, b, K_t);
  const RiSigmaHessianToolkit<C> toolkit_y = buildRiSigmaHessianToolkit(n, b, K_y);
  const Matrix<C> Phi_t = buildPhiRi(h, occupations, two_rdm_h, two_rdm_x, K_t, toolkit_t);
  const Matrix<C> Psi_t = buildPsiRi(h, occupations, two_rdm_h, two_rdm_x, K_t, toolkit_t);
  const Matrix<C> Phi_y = buildPhiRi(h, occupations, two_rdm_h, two_rdm_x, K_y, toolkit_y);
  const Matrix<C> Psi_y = buildPsiRi(h, occupations, two_rdm_h, two_rdm_x, K_y, toolkit_y);
  const C im(0.0, 1.0);

  std::vector<double> w(2 * n_pairs, 0.0);
  for (std::size_t i = 0; i < n_pairs; ++i) {
    const auto& [p, q] = pair_indices[i];
    const double wt = (0.5 * ((Phi_t(p, q) - Phi_t(q, p)) + (Psi_t(p, q) - Psi_t(q, p)))).real() +
                       (0.5 * im * ((Phi_y(p, q) - Phi_y(q, p)) + (Psi_y(p, q) - Psi_y(q, p)))).real();
    const double wy = (0.5 * im * ((Psi_t(p, q) + Psi_t(q, p)) + (Phi_t(p, q) + Phi_t(q, p)))).real() -
                       (0.5 * ((Phi_y(p, q) + Phi_y(q, p)) + (Psi_y(p, q) + Psi_y(q, p)))).real();
    w[i] = wt;
    w[n_pairs + i] = wy;
  }
  return w;
}

}  // namespace rerdmft
