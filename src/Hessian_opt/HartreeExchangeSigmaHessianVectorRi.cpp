#include "HartreeExchangeSigmaHessianVectorRi.h"

#include <complex>
#include <stdexcept>

#include "RiSigmaGemm.h"
#include "RiSigmaHessianToolkit.h"

namespace rerdmft {

namespace {

// Same convention as HartreeExchangeSigmaHessianVector.cpp's own helper -- identity for real T,
// std::conj for complex T. Needed for the SAME reason: `rawHartreeExchangeHessianTerm`'s
// `s_==p_` branch conjugates its fock term.
inline double conjugateT(double x) { return x; }
inline std::complex<double> conjugateT(std::complex<double> x) { return std::conj(x); }

template <typename T>
void addPhiHxRiRef(Matrix<T>& Phi, const Matrix<T>& h, const std::vector<double>& occ,
                 const Matrix<double>& hcpl, const Matrix<double>& xcpl, const Matrix<T>& fock,
                 const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  #pragma omp parallel for collapse(2)
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      T acc{};
      for (std::size_t s = 0; s < n; ++s) acc += T(K(b, s)) * fock(a, s);
      for (std::size_t r = 0; r < n; ++r) acc += conjugateT(fock(b, r)) * T(K(r, a));  // P1b (conjugate)
      T t3{}, t4{};
      for (std::size_t s = 0; s < n; ++s) t3 += T(K(b, s)) * h(s, a);
      for (std::size_t r = 0; r < n; ++r) t4 += h(b, r) * T(K(r, a));
      acc -= T(occ[b]) * t3;
      acc -= T(occ[a]) * t4;
      for (std::size_t u = 0; u < n; ++u) {
        T Ea{}, Eb{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Ea += t.B(P, u, a) * t.KB1(P, b, u);
          Eb += t.Bdiag(P, u) * t.KB1(P, b, a);
        }
        acc += T(hcpl(b, u)) * (Ea - Eb);
      }
      for (std::size_t u = 0; u < n; ++u) {
        T Fa{}, Fb{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Fa += t.B(P, b, u) * t.KB2(P, u, a);
          Fb += t.Bdiag(P, u) * t.KB2(P, b, a);
        }
        acc += T(hcpl(a, u)) * (Fa - Fb);
      }
      std::vector<T> G1(n, T{}), G2(n, T{});
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t P = 0; P < n_aux; ++P) G1[r] += t.B(P, b, r) * t.KB1(P, r, a);
      for (std::size_t s = 0; s < n; ++s)
        for (std::size_t P = 0; P < n_aux; ++P) G2[s] += t.B(P, s, a) * t.KB2(P, b, s);
      for (std::size_t r = 0; r < n; ++r) acc += T(xcpl(r, b)) * G1[r];
      for (std::size_t s = 0; s < n; ++s) acc -= T(hcpl(s, b)) * G2[s];
      for (std::size_t s = 0; s < n; ++s) acc += T(xcpl(a, s)) * G2[s];
      for (std::size_t r = 0; r < n; ++r) acc -= T(xcpl(r, a)) * G1[r];
      std::vector<T> H1(n, T{}), H2(n, T{});
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t P = 0; P < n_aux; ++P) H1[r] += t.B(P, b, a) * t.KB1diag(P, r);
      for (std::size_t s = 0; s < n; ++s)
        for (std::size_t P = 0; P < n_aux; ++P) H2[s] += t.B(P, b, a) * t.KB2diag(P, s);
      for (std::size_t r = 0; r < n; ++r) acc += T(hcpl(r, a)) * H1[r];
      for (std::size_t s = 0; s < n; ++s) acc += T(xcpl(b, s)) * H2[s];
      for (std::size_t s = 0; s < n; ++s) acc -= T(xcpl(a, s)) * H2[s];
      for (std::size_t r = 0; r < n; ++r) acc -= T(xcpl(r, b)) * H1[r];
      Phi(a, b) += acc;
    }
}

template <typename T>
void addPsiHxRiRef(Matrix<T>& Psi, const Matrix<T>& h, const std::vector<double>& occ,
                 const Matrix<double>& hcpl, const Matrix<double>& xcpl, const Matrix<T>& fock,
                 const Matrix<double>& K, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  #pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c)) * fock(a, d);
      for (std::size_t b = 0; b < n; ++b) acc += T(K(d, b)) * conjugateT(fock(b, c));  // Q2 (conjugate)
      T t3{}, t4{};
      for (std::size_t a = 0; a < n; ++a) t3 += T(K(a, c)) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) t4 += T(K(d, b)) * h(b, c);
      acc -= T(occ[c]) * t3;
      acc -= T(occ[d]) * t4;
      for (std::size_t u = 0; u < n; ++u) {
        T Ra{}, Rb{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Ra += t.B(P, d, u) * t.KB2(P, u, c);
          Rb += t.Bdiag(P, u) * t.KB2(P, d, c);
        }
        acc += T(hcpl(c, u)) * (Ra - Rb);
      }
      for (std::size_t u = 0; u < n; ++u) {
        T Sa{}, Sb{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Sa += t.B(P, u, c) * t.KB1(P, d, u);
          Sb += t.Bdiag(P, u) * t.KB1(P, d, c);
        }
        acc += T(hcpl(d, u)) * (Sa - Sb);
      }
      std::vector<T> U1(n, T{}), U2(n, T{});
      for (std::size_t b = 0; b < n; ++b)
        for (std::size_t P = 0; P < n_aux; ++P) U1[b] += t.B(P, b, c) * t.KB2(P, d, b);
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t P = 0; P < n_aux; ++P) U2[a] += t.B(P, d, a) * t.KB1(P, a, c);
      for (std::size_t b = 0; b < n; ++b) acc += T(xcpl(c, b)) * U1[b];
      for (std::size_t b = 0; b < n; ++b) acc -= T(hcpl(d, b)) * U1[b];
      for (std::size_t a = 0; a < n; ++a) acc += T(xcpl(a, d)) * U2[a];
      for (std::size_t a = 0; a < n; ++a) acc -= T(xcpl(c, a)) * U2[a];
      std::vector<T> V1(n, T{}), V2(n, T{});
      for (std::size_t b = 0; b < n; ++b)
        for (std::size_t P = 0; P < n_aux; ++P) V1[b] += t.B(P, d, c) * t.KB2diag(P, b);
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t P = 0; P < n_aux; ++P) V2[a] += t.B(P, d, c) * t.KB1diag(P, a);
      for (std::size_t a = 0; a < n; ++a) acc += T(hcpl(c, a)) * V2[a];
      for (std::size_t b = 0; b < n; ++b) acc += T(xcpl(b, d)) * V1[b];
      for (std::size_t a = 0; a < n; ++a) acc -= T(xcpl(a, d)) * V2[a];
      for (std::size_t b = 0; b < n; ++b) acc -= T(xcpl(c, b)) * V1[b];
      Psi(c, d) += acc;
    }
}

template <typename T>
void addPhiL12RiRef(Matrix<T>& Phi, const std::vector<std::size_t>& pair_of, const Matrix<double>& l1,
                  const Matrix<double>& l2, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  #pragma omp parallel for collapse(2)
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      const std::size_t pb = pair_of[b], pa = pair_of[a];
      T acc{};
      for (std::size_t u = 0; u < n; ++u) {
        T Iu{};
        for (std::size_t P = 0; P < n_aux; ++P) Iu += t.B(P, u, a) * t.KB2(P, pair_of[u], pb);
        acc -= T(2.0 * (l1(u, pb) + l2(u, b))) * Iu;
      }
      for (std::size_t r = 0; r < n; ++r) {
        const double c_rb = l1(r, b) + l2(r, pb);
        T Jr{}, Jr2{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Jr += t.B(P, pair_of[r], a) * t.KB1(P, r, pb);
          Jr2 += t.B(P, pair_of[r], pb) * t.KB1(P, r, a);
        }
        acc += T(2.0 * c_rb) * (Jr - Jr2);
      }
      for (std::size_t s = 0; s < n; ++s) {
        const double d_sa = l1(a, s) + l2(a, pair_of[s]);
        T Ls{}, Ls2{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Ls += t.B(P, b, pair_of[s]) * t.KB2(P, pa, s);
          Ls2 += t.B(P, pa, pair_of[s]) * t.KB2(P, b, s);
        }
        acc += T(2.0 * d_sa) * (Ls - Ls2);
      }
      for (std::size_t w = 0; w < n; ++w) {
        T Mw{};
        for (std::size_t P = 0; P < n_aux; ++P) Mw += t.B(P, b, pair_of[w]) * t.KB1(P, pa, w);
        acc += T(2.0 * (l1(pa, w) + l2(pa, pair_of[w]))) * Mw;
      }
      Phi(a, b) += acc;
    }
}

template <typename T>
void addPsiL12RiRef(Matrix<T>& Psi, const std::vector<std::size_t>& pair_of, const Matrix<double>& l1,
                  const Matrix<double>& l2, const RiSigmaHessianToolkit<T>& t) {
  const std::size_t n = t.n, n_aux = t.n_aux;
  #pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      const std::size_t pd = pair_of[d], pc = pair_of[c];
      T acc{};
      for (std::size_t u = 0; u < n; ++u) {
        T Wu{};
        for (std::size_t P = 0; P < n_aux; ++P) Wu += t.B(P, pair_of[u], c) * t.KB2(P, u, pd);
        acc -= T(2.0 * (l1(u, d) + l2(u, pd))) * Wu;
      }
      for (std::size_t b = 0; b < n; ++b) {
        const double coeff1 = l1(c, b) + l2(c, pair_of[b]);
        T Xb{}, Xb2{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Xb += t.B(P, d, pair_of[b]) * t.KB2(P, pc, b);
          Xb2 += t.B(P, pc, pair_of[b]) * t.KB2(P, d, b);
        }
        acc += T(2.0 * coeff1) * (Xb - Xb2);
      }
      for (std::size_t a = 0; a < n; ++a) {
        const double coeff2 = l1(a, d) + l2(a, pd);
        T Ya{}, Ya2{};
        for (std::size_t P = 0; P < n_aux; ++P) {
          Ya += t.B(P, pair_of[a], c) * t.KB1(P, a, pd);
          Ya2 += t.B(P, pair_of[a], pd) * t.KB1(P, a, c);
        }
        acc += T(2.0 * coeff2) * (Ya - Ya2);
      }
      for (std::size_t w = 0; w < n; ++w) {
        T Zw{};
        for (std::size_t P = 0; P < n_aux; ++P) Zw += t.B(P, d, w) * t.KB1(P, pc, pair_of[w]);
        acc += T(2.0 * (l1(c, w) + l2(c, pair_of[w]))) * Zw;
      }
      Psi(c, d) += acc;
    }
}


// ---- GEMM versions (production). The *Ref functions above are the original element-by-element
// implementations, kept as a validation oracle (see setRiSigmaHessianVectorReference).
template <typename T>
Matrix<T> negated(const Matrix<T>& m) {
  Matrix<T> r(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows() * m.cols(); ++i) r.data()[i] = -m.data()[i];
  return r;
}

template <typename T>
void addPhiHxRi(Matrix<T>& Phi, const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hcpl,
                 const Matrix<double>& xcpl, const Matrix<T>& fock, const Matrix<double>& K,
                 const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return addPhiHxRiRef(Phi, h, occ, hcpl, xcpl, fock, K, t);
  using risigma::accumU;
  using risigma::diagContract;
  using risigma::makeOp;
  using risigma::pairContract;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;

#pragma omp parallel for collapse(2)
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      T acc{};
      for (std::size_t s = 0; s < n; ++s) acc += T(K(b, s)) * fock(a, s);
      for (std::size_t r = 0; r < n; ++r) acc += conjugateT(fock(b, r)) * T(K(r, a));  // P1b (conjugate)
      T t3{}, t4{};
      for (std::size_t s = 0; s < n; ++s) t3 += T(K(b, s)) * h(s, a);
      for (std::size_t r = 0; r < n; ++r) t4 += h(b, r) * T(K(r, a));
      acc -= T(occ[b]) * t3;
      acc -= T(occ[a]) * t4;
      Phi(a, b) += acc;
    }

  // Uh(P,x) = sum_u Bdiag(P,u) hcpl(x,u)
  Matrix<T> Uh(n_aux, n, T{});
  accumU(Uh, 1.0, n, n_aux, t.bdiag, hcpl);
  const Matrix<T> nUh = negated(Uh);

  // Ea: sum_{P,u} hcpl(b,u) B(P,u,a) KB1(P,b,u)
  pairContract(Phi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](std::size_t u, std::size_t a) { return u * n + a; }),
               makeOp<T>(n, &K1, [&](std::size_t u, std::size_t b) { return b * n + u; },
                         [&](std::size_t u, std::size_t b) { return hcpl(b, u); }));
  // -Eb: -sum_P KB1(P,b,a) Uh(P,b)
  diagContract(Phi, n, n_aux, K1, nullptr, &nUh);
  // Fa: sum_{P,u} hcpl(a,u) B(P,b,u) KB2(P,u,a)
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](std::size_t u, std::size_t a) { return u * n + a; },
                         [&](std::size_t u, std::size_t a) { return hcpl(a, u); }),
               makeOp<T>(n, &Bm, [&](std::size_t u, std::size_t b) { return b * n + u; }));
  // -Fb: -sum_P KB2(P,b,a) Uh(P,a)
  diagContract(Phi, n, n_aux, K2, &nUh, nullptr);
  // G1: sum_r (xcpl(r,b) - xcpl(r,a)) B(P,b,r) KB1(P,r,a)
  pairContract(Phi, 1.0, n, n_aux, makeOp<T>(n, &K1, [&](std::size_t r, std::size_t a) { return r * n + a; }),
               makeOp<T>(n, &Bm, [&](std::size_t r, std::size_t b) { return b * n + r; },
                         [&](std::size_t r, std::size_t b) { return xcpl(r, b); }));
  pairContract(Phi, -1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](std::size_t r, std::size_t a) { return r * n + a; },
                         [&](std::size_t r, std::size_t a) { return xcpl(r, a); }),
               makeOp<T>(n, &Bm, [&](std::size_t r, std::size_t b) { return b * n + r; }));
  // G2: sum_s (xcpl(a,s) - hcpl(s,b)) B(P,s,a) KB2(P,b,s)
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](std::size_t s, std::size_t a) { return s * n + a; },
                         [&](std::size_t s, std::size_t a) { return xcpl(a, s); }),
               makeOp<T>(n, &K2, [&](std::size_t s, std::size_t b) { return b * n + s; }));
  pairContract(Phi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](std::size_t s, std::size_t a) { return s * n + a; }),
               makeOp<T>(n, &K2, [&](std::size_t s, std::size_t b) { return b * n + s; },
                         [&](std::size_t s, std::size_t b) { return hcpl(s, b); }));
  // H1/H2: sum_P B(P,b,a) [ UA(P,a) - UB(P,b) + UC(P,b) - UD(P,a) ]
  const Matrix<double> hcplT = transpose(hcpl), xcplT = transpose(xcpl);
  Matrix<T> Ua(n_aux, n, T{}), Ub(n_aux, n, T{});
  accumU(Ua, 1.0, n, n_aux, t.kb1diag, hcplT);   // UA(P,a) = sum_r KB1diag(P,r) hcpl(r,a)
  accumU(Ua, -1.0, n, n_aux, t.kb2diag, xcpl);   // -UD(P,a) = -sum_s KB2diag(P,s) xcpl(a,s)
  accumU(Ub, -1.0, n, n_aux, t.kb1diag, xcplT);  // -UB(P,b) = -sum_r KB1diag(P,r) xcpl(r,b)
  accumU(Ub, 1.0, n, n_aux, t.kb2diag, xcpl);    // UC(P,b) = sum_s KB2diag(P,s) xcpl(b,s)
  diagContract(Phi, n, n_aux, Bm, &Ua, &Ub);
}

template <typename T>
void addPsiHxRi(Matrix<T>& Psi, const Matrix<T>& h, const std::vector<double>& occ, const Matrix<double>& hcpl,
                 const Matrix<double>& xcpl, const Matrix<T>& fock, const Matrix<double>& K,
                 const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return addPsiHxRiRef(Psi, h, occ, hcpl, xcpl, fock, K, t);
  using risigma::accumU;
  using risigma::diagContract;
  using risigma::makeOp;
  using risigma::pairContract;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;

#pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c)) * fock(a, d);
      for (std::size_t b = 0; b < n; ++b) acc += T(K(d, b)) * conjugateT(fock(b, c));  // Q2 (conjugate)
      T t3{}, t4{};
      for (std::size_t a = 0; a < n; ++a) t3 += T(K(a, c)) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) t4 += T(K(d, b)) * h(b, c);
      acc -= T(occ[c]) * t3;
      acc -= T(occ[d]) * t4;
      Psi(c, d) += acc;
    }

  Matrix<T> Uh(n_aux, n, T{});
  accumU(Uh, 1.0, n, n_aux, t.bdiag, hcpl);
  const Matrix<T> nUh = negated(Uh);

  // Ra: sum_{P,u} hcpl(c,u) B(P,d,u) KB2(P,u,c)
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](std::size_t u, std::size_t c) { return u * n + c; },
                         [&](std::size_t u, std::size_t c) { return hcpl(c, u); }),
               makeOp<T>(n, &Bm, [&](std::size_t u, std::size_t d) { return d * n + u; }));
  // -Rb: -sum_P KB2(P,d,c) Uh(P,c)
  diagContract(Psi, n, n_aux, K2, &nUh, nullptr);
  // Sa: sum_{P,u} hcpl(d,u) B(P,u,c) KB1(P,d,u)
  pairContract(Psi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](std::size_t u, std::size_t c) { return u * n + c; }),
               makeOp<T>(n, &K1, [&](std::size_t u, std::size_t d) { return d * n + u; },
                         [&](std::size_t u, std::size_t d) { return hcpl(d, u); }));
  // -Sb: -sum_P KB1(P,d,c) Uh(P,d)
  diagContract(Psi, n, n_aux, K1, nullptr, &nUh);
  // U1: sum_b (xcpl(c,b) - hcpl(d,b)) B(P,b,c) KB2(P,d,b)
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](std::size_t b, std::size_t c) { return b * n + c; },
                         [&](std::size_t b, std::size_t c) { return xcpl(c, b); }),
               makeOp<T>(n, &K2, [&](std::size_t b, std::size_t d) { return d * n + b; }));
  pairContract(Psi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](std::size_t b, std::size_t c) { return b * n + c; }),
               makeOp<T>(n, &K2, [&](std::size_t b, std::size_t d) { return d * n + b; },
                         [&](std::size_t b, std::size_t d) { return hcpl(d, b); }));
  // U2: sum_a (xcpl(a,d) - xcpl(c,a)) B(P,d,a) KB1(P,a,c)
  pairContract(Psi, 1.0, n, n_aux, makeOp<T>(n, &K1, [&](std::size_t a, std::size_t c) { return a * n + c; }),
               makeOp<T>(n, &Bm, [&](std::size_t a, std::size_t d) { return d * n + a; },
                         [&](std::size_t a, std::size_t d) { return xcpl(a, d); }));
  pairContract(Psi, -1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](std::size_t a, std::size_t c) { return a * n + c; },
                         [&](std::size_t a, std::size_t c) { return xcpl(c, a); }),
               makeOp<T>(n, &Bm, [&](std::size_t a, std::size_t d) { return d * n + a; }));
  // V1/V2: sum_P B(P,d,c) [ UE(P,c) - UF(P,d) + UG(P,d) - UH(P,c) ]
  const Matrix<double> xcplT = transpose(xcpl);
  Matrix<T> Ua(n_aux, n, T{}), Ub(n_aux, n, T{});
  accumU(Ua, 1.0, n, n_aux, t.kb1diag, hcpl);   // UE(P,c) = sum_a KB1diag(P,a) hcpl(c,a)
  accumU(Ua, -1.0, n, n_aux, t.kb2diag, xcpl);  // -UH(P,c) = -sum_b KB2diag(P,b) xcpl(c,b)
  accumU(Ub, -1.0, n, n_aux, t.kb1diag, xcplT); // -UF(P,d) = -sum_a KB1diag(P,a) xcpl(a,d)
  accumU(Ub, 1.0, n, n_aux, t.kb2diag, xcplT);  // UG(P,d) = sum_b KB2diag(P,b) xcpl(b,d)
  diagContract(Psi, n, n_aux, Bm, &Ua, &Ub);
}

template <typename T>
void addPhiL12Ri(Matrix<T>& Phi, const std::vector<std::size_t>& pair_of, const Matrix<double>& l1,
                  const Matrix<double>& l2, const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return addPhiL12RiRef(Phi, pair_of, l1, l2, t);
  using risigma::makeOp;
  using risigma::pairContract;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;
  const auto& pr = pair_of;
  using Z = std::size_t;
  // I: -sum_u 2(l1(u,pb)+l2(u,b)) B(P,u,a) KB2(P,pair(u),pb)
  pairContract(Phi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z u, Z a) { return u * n + a; }),
               makeOp<T>(n, &K2, [&](Z u, Z b) { return pr[u] * n + pr[b]; },
                         [&](Z u, Z b) { return 2.0 * (l1(u, pr[b]) + l2(u, b)); }));
  // Jr, Jr2
  pairContract(Phi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z r, Z a) { return pr[r] * n + a; }),
               makeOp<T>(n, &K1, [&](Z r, Z b) { return r * n + pr[b]; },
                         [&](Z r, Z b) { return 2.0 * (l1(r, b) + l2(r, pr[b])); }));
  pairContract(Phi, -1.0, n, n_aux, makeOp<T>(n, &K1, [&](Z r, Z a) { return r * n + a; }),
               makeOp<T>(n, &Bm, [&](Z r, Z b) { return pr[r] * n + pr[b]; },
                         [&](Z r, Z b) { return 2.0 * (l1(r, b) + l2(r, pr[b])); }));
  // Ls, Ls2
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](Z s, Z a) { return pr[a] * n + s; },
                         [&](Z s, Z a) { return 2.0 * (l1(a, s) + l2(a, pr[s])); }),
               makeOp<T>(n, &Bm, [&](Z s, Z b) { return b * n + pr[s]; }));
  pairContract(Phi, -1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](Z s, Z a) { return pr[a] * n + pr[s]; },
                         [&](Z s, Z a) { return 2.0 * (l1(a, s) + l2(a, pr[s])); }),
               makeOp<T>(n, &K2, [&](Z s, Z b) { return b * n + s; }));
  // Mw
  pairContract(Phi, 1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](Z w, Z a) { return pr[a] * n + w; },
                         [&](Z w, Z a) { return 2.0 * (l1(pr[a], w) + l2(pr[a], pr[w])); }),
               makeOp<T>(n, &Bm, [&](Z w, Z b) { return b * n + pr[w]; }));
}

template <typename T>
void addPsiL12Ri(Matrix<T>& Psi, const std::vector<std::size_t>& pair_of, const Matrix<double>& l1,
                  const Matrix<double>& l2, const RiSigmaHessianToolkit<T>& t) {
  if (risigma::useReference()) return addPsiL12RiRef(Psi, pair_of, l1, l2, t);
  using risigma::makeOp;
  using risigma::pairContract;
  const std::size_t n = t.n, n_aux = t.n_aux;
  const Matrix<T>& Bm = t.b();
  const Matrix<T>& K1 = t.kb1;
  const Matrix<T>& K2 = t.kb2;
  const auto& pr = pair_of;
  using Z = std::size_t;
  // Wu
  pairContract(Psi, -1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z u, Z c) { return pr[u] * n + c; }),
               makeOp<T>(n, &K2, [&](Z u, Z d) { return u * n + pr[d]; },
                         [&](Z u, Z d) { return 2.0 * (l1(u, d) + l2(u, pr[d])); }));
  // Xb, Xb2
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &K2, [&](Z b, Z c) { return pr[c] * n + b; },
                         [&](Z b, Z c) { return 2.0 * (l1(c, b) + l2(c, pr[b])); }),
               makeOp<T>(n, &Bm, [&](Z b, Z d) { return d * n + pr[b]; }));
  pairContract(Psi, -1.0, n, n_aux,
               makeOp<T>(n, &Bm, [&](Z b, Z c) { return pr[c] * n + pr[b]; },
                         [&](Z b, Z c) { return 2.0 * (l1(c, b) + l2(c, pr[b])); }),
               makeOp<T>(n, &K2, [&](Z b, Z d) { return d * n + b; }));
  // Ya, Ya2
  pairContract(Psi, 1.0, n, n_aux, makeOp<T>(n, &Bm, [&](Z a, Z c) { return pr[a] * n + c; }),
               makeOp<T>(n, &K1, [&](Z a, Z d) { return a * n + pr[d]; },
                         [&](Z a, Z d) { return 2.0 * (l1(a, d) + l2(a, pr[d])); }));
  pairContract(Psi, -1.0, n, n_aux, makeOp<T>(n, &K1, [&](Z a, Z c) { return a * n + c; }),
               makeOp<T>(n, &Bm, [&](Z a, Z d) { return pr[a] * n + pr[d]; },
                         [&](Z a, Z d) { return 2.0 * (l1(a, d) + l2(a, pr[d])); }));
  // Zw
  pairContract(Psi, 1.0, n, n_aux,
               makeOp<T>(n, &K1, [&](Z w, Z c) { return pr[c] * n + pr[w]; },
                         [&](Z w, Z c) { return 2.0 * (l1(c, w) + l2(c, pr[w])); }),
               makeOp<T>(n, &Bm, [&](Z w, Z d) { return d * n + w; }));
}

}  // namespace

std::vector<double> hartreeExchangeSigmaHessianVectorRi(
    const Matrix<double>& h, const Matrix<double>& b, std::size_t n, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v) {
  if (h.rows() != n || h.cols() != n) {
    throw std::runtime_error("hartreeExchangeSigmaHessianVectorRi: h has the wrong shape");
  }
  if (v.size() != pair_indices.size()) {
    throw std::runtime_error("hartreeExchangeSigmaHessianVectorRi: v has the wrong size");
  }
  if (!pair_of.empty() && pair_of.size() != n) {
    throw std::runtime_error("hartreeExchangeSigmaHessianVectorRi: pair_of has the wrong size");
  }

  Matrix<double> K(n, n, 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    K(p, q) = v[i];
    K(q, p) = -v[i];
  }

  const RiSigmaHessianToolkit<double> toolkit = buildRiSigmaHessianToolkit(n, b, K);

  Matrix<double> Phi(n, n, 0.0), Psi(n, n, 0.0);
  addPhiHxRi(Phi, h, occupations, two_rdm_h, two_rdm_x, fock, K, toolkit);
  addPsiHxRi(Psi, h, occupations, two_rdm_h, two_rdm_x, fock, K, toolkit);
  if (!pair_of.empty()) {
    addPhiL12Ri(Phi, pair_of, two_rdm_l1, two_rdm_l2, toolkit);
    addPsiL12Ri(Psi, pair_of, two_rdm_l1, two_rdm_l2, toolkit);
  }

  std::vector<double> w(pair_indices.size(), 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    w[i] = 0.5 * ((Phi(p, q) - Phi(q, p)) + (Psi(p, q) - Psi(q, p)));
  }
  return w;
}

std::vector<double> hartreeExchangeSigmaJointHessianVectorRi(
    const Matrix<std::complex<double>>& h, const Matrix<std::complex<double>>& b, std::size_t n,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v) {
  using C = std::complex<double>;
  if (h.rows() != n || h.cols() != n) {
    throw std::runtime_error("hartreeExchangeSigmaJointHessianVectorRi: h has the wrong shape");
  }
  const std::size_t n_pairs = pair_indices.size();
  if (v.size() != 2 * n_pairs) {
    throw std::runtime_error("hartreeExchangeSigmaJointHessianVectorRi: v has the wrong size");
  }
  if (!pair_of.empty() && pair_of.size() != n) {
    throw std::runtime_error("hartreeExchangeSigmaJointHessianVectorRi: pair_of has the wrong size");
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

  Matrix<C> Phi_t(n, n, C{}), Psi_t(n, n, C{}), Phi_y(n, n, C{}), Psi_y(n, n, C{});
  addPhiHxRi(Phi_t, h, occupations, two_rdm_h, two_rdm_x, fock, K_t, toolkit_t);
  addPsiHxRi(Psi_t, h, occupations, two_rdm_h, two_rdm_x, fock, K_t, toolkit_t);
  addPhiHxRi(Phi_y, h, occupations, two_rdm_h, two_rdm_x, fock, K_y, toolkit_y);
  addPsiHxRi(Psi_y, h, occupations, two_rdm_h, two_rdm_x, fock, K_y, toolkit_y);
  if (!pair_of.empty()) {
    addPhiL12Ri(Phi_t, pair_of, two_rdm_l1, two_rdm_l2, toolkit_t);
    addPsiL12Ri(Psi_t, pair_of, two_rdm_l1, two_rdm_l2, toolkit_t);
    addPhiL12Ri(Phi_y, pair_of, two_rdm_l1, two_rdm_l2, toolkit_y);
    addPsiL12Ri(Psi_y, pair_of, two_rdm_l1, two_rdm_l2, toolkit_y);
  }
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
