#include "JkOnlySigmaHessianVector.h"

#include <complex>
#include <stdexcept>
#include <string>

#include "CholeskyEri.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

namespace {

// Flat row-major 3-index scratch buffer (dims d0 x d1 x d2) -- avoids pulling in a 3-tensor
// class just for this file's intermediates. Templated on T so the complex (joint [t;y]) case
// below reuses the exact same Phi/Psi code as the real case.
template <typename T>
struct Tensor3 {
  Tensor3(std::size_t d0, std::size_t d1, std::size_t d2)
      : d0_(d0), d1_(d1), d2_(d2), data_(d0 * d1 * d2, T{}) {}
  T& operator()(std::size_t i, std::size_t j, std::size_t k) { return data_[(i * d1_ + j) * d2_ + k]; }
  T operator()(std::size_t i, std::size_t j, std::size_t k) const { return data_[(i * d1_ + j) * d2_ + k]; }
  std::size_t d0_, d1_, d2_;
  std::vector<T> data_;
};

// Phi(a,b) = sum_{r,s} rawG(a,b,r,s) * K[r,s] -- see this file's header for the derivation
// reference. `K` is always REAL (antisymmetric for the real-step/t case, symmetric for the
// imaginary-step/y case, see `jkOnlySigmaJointHessianVector` below) -- `T` is h/eri's own scalar
// type (double for the real-step-only case, complex<double> for the joint case).
template <typename T, typename Eri>
Matrix<T> buildPhi(const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
                    const Matrix<double>& hc, const Matrix<double>& xc, const Matrix<double>& K) {
  const std::size_t n = h.rows();
  Matrix<T> Phi(n, n, T{});

  // T12: (occ[p]-occ[q]) * (K@h - h@K)[q,p] -- the pure commutator term, no eri at all. K is
  // real and h is T -- computed via explicit loops (not Matrix::operator*, which requires equal
  // template types on both operands).
  {
    Matrix<T> KH(n, n, T{}), HK(n, n, T{});
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        T acc1{}, acc2{};
        for (std::size_t k = 0; k < n; ++k) {
          acc1 += K(i, k) * h(k, j);
          acc2 += h(i, k) * K(k, j);
        }
        KH(i, j) = acc1;
        HK(i, j) = acc2;
      }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) Phi(p, q) += T(occ[p] - occ[q]) * (KH(q, p) - HK(q, p));
  }

  // T3 (q==r branch of rawG's sum_t term), Hc half: V_h(t,p,q) = sum_s eri(t,s,t,p)*K[q,s].
  {
    Tensor3<T> eri_diag_tp(n, n, n);  // [t][s][p] = eri(t,s,t,p)
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t s = 0; s < n; ++s)
        for (std::size_t p = 0; p < n; ++p) eri_diag_tp(t, s, p) = static_cast<T>(eri(t, s, t, p));
    Tensor3<T> V_h(n, n, n);  // [t][p][q]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t s = 0; s < n; ++s) acc += eri_diag_tp(t, s, p) * K(q, s);
          V_h(t, p, q) = acc;
        }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t t = 0; t < n; ++t) acc += T(hc(p, t) - hc(q, t)) * V_h(t, p, q);
        Phi(p, q) += acc;
      }
  }
  // T3, Xc half: V_x(t,p,q) = sum_s eri(s,t,t,p)*K[q,s].
  {
    Tensor3<T> eri_diag_stp(n, n, n);  // [s][t][p] = eri(s,t,t,p)
    for (std::size_t s = 0; s < n; ++s)
      for (std::size_t t = 0; t < n; ++t)
        for (std::size_t p = 0; p < n; ++p) eri_diag_stp(s, t, p) = static_cast<T>(eri(s, t, t, p));
    Tensor3<T> V_x(n, n, n);  // [t][p][q]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          T acc{};
          for (std::size_t s = 0; s < n; ++s) acc += eri_diag_stp(s, t, p) * K(q, s);
          V_x(t, p, q) = acc;
        }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t t = 0; t < n; ++t) acc += T(xc(p, t) - xc(q, t)) * V_x(t, p, q);
        Phi(p, q) -= acc;
      }
  }

  // T4 (s==p branch), Hc half: W_h(t,p,q) = sum_r eri(t,q,t,r)*K[r,p].
  {
    Tensor3<T> eri_diag_tqr(n, n, n);  // [t][q][r] = eri(t,q,t,r)
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r) eri_diag_tqr(t, q, r) = static_cast<T>(eri(t, q, t, r));
    Tensor3<T> W_h(n, n, n);  // [t][p][q]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t p = 0; p < n; ++p) {
          T acc{};
          for (std::size_t r = 0; r < n; ++r) acc += eri_diag_tqr(t, q, r) * K(r, p);
          W_h(t, p, q) = acc;
        }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t t = 0; t < n; ++t) acc += T(hc(p, t) - hc(q, t)) * W_h(t, p, q);
        Phi(p, q) -= acc;
      }
  }
  // T4, Xc half: W_x(t,p,q) = sum_r eri(q,t,t,r)*K[r,p].
  {
    Tensor3<T> eri_diag_qttr(n, n, n);  // [q][t][r] = eri(q,t,t,r)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t t = 0; t < n; ++t)
        for (std::size_t r = 0; r < n; ++r) eri_diag_qttr(q, t, r) = static_cast<T>(eri(q, t, t, r));
    Tensor3<T> W_x(n, n, n);  // [t][p][q]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t p = 0; p < n; ++p) {
          T acc{};
          for (std::size_t r = 0; r < n; ++r) acc += eri_diag_qttr(q, t, r) * K(r, p);
          W_x(t, p, q) = acc;
        }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t t = 0; t < n; ++t) acc += T(xc(p, t) - xc(q, t)) * W_x(t, p, q);
        Phi(p, q) += acc;
      }
  }

  // TA (unconditional Hc term): M_A(r,q,p) = sum_s eri(s,q,r,p)*K[r,s]; M_A2(s,q,p) = sum_r eri(s,q,r,p)*K[r,s].
  {
    Tensor3<T> M_A(n, n, n), M_A2(n, n, n);
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t r = 0; r < n; ++r) {
          T acc{};
          for (std::size_t s = 0; s < n; ++s) acc += static_cast<T>(eri(s, q, r, p)) * K(r, s);
          M_A(r, q, p) = acc;
        }
        for (std::size_t s = 0; s < n; ++s) {
          T acc{};
          for (std::size_t r = 0; r < n; ++r) acc += static_cast<T>(eri(s, q, r, p)) * K(r, s);
          M_A2(s, q, p) = acc;
        }
      }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t r = 0; r < n; ++r) acc += T(hc(p, r) - hc(q, r)) * M_A(r, q, p);
        for (std::size_t s = 0; s < n; ++s) acc += T(-hc(p, s) + hc(q, s)) * M_A2(s, q, p);
        Phi(p, q) += acc;
      }
  }

  // TB (unconditional Xc term): M_B(s,q,p) = sum_r eri(q,s,r,p)*K[r,s]; M_B2(r,q,p) = sum_s eri(q,s,r,p)*K[r,s].
  {
    Tensor3<T> M_B(n, n, n), M_B2(n, n, n);
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t s = 0; s < n; ++s) {
          T acc{};
          for (std::size_t r = 0; r < n; ++r) acc += static_cast<T>(eri(q, s, r, p)) * K(r, s);
          M_B(s, q, p) = acc;
        }
        for (std::size_t r = 0; r < n; ++r) {
          T acc{};
          for (std::size_t s = 0; s < n; ++s) acc += static_cast<T>(eri(q, s, r, p)) * K(r, s);
          M_B2(r, q, p) = acc;
        }
      }
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        T acc{};
        for (std::size_t s = 0; s < n; ++s) acc += T(xc(p, s) - xc(q, s)) * M_B(s, q, p);
        for (std::size_t r = 0; r < n; ++r) acc += T(-xc(p, r) + xc(q, r)) * M_B2(r, q, p);
        Phi(p, q) += acc;
      }
  }

  return Phi;
}

// Psi(c,d) = sum_{a,b} rawG(a,b,c,d) * K[a,b] -- the OTHER contraction (first two slots of
// rawG, not symmetric with Phi; see this file's header / project memory).
template <typename T, typename Eri>
Matrix<T> buildPsi(const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
                    const Matrix<double>& hc, const Matrix<double>& xc, const Matrix<double>& K) {
  const std::size_t n = h.rows();
  Matrix<T> Psi(n, n, T{});

  // P1: sum_a K[a,c]*h[d,a]*(occ[a]-occ[c])
  // P2: -sum_b K[d,b]*h[b,c]*(occ[d]-occ[b])
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c) * (occ[a] - occ[c])) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) acc -= T(K(d, b) * (occ[d] - occ[b])) * h(b, c);
      Psi(c, d) += acc;
    }

  // P3 (b==c branch): eri_tdta[t,d,a]=eri(t,d,t,a); term3a[a,d]=sum_t hc(a,t)*eri_tdta(t,d,a);
  // term3b_pre[a,c,d]=sum_t hc(c,t)*eri_tdta(t,d,a); same with xc/eri_dtta for term3c/term3d_pre.
  {
    Tensor3<T> eri_tdta(n, n, n);  // [t][d][a]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t d = 0; d < n; ++d)
        for (std::size_t a = 0; a < n; ++a) eri_tdta(t, d, a) = static_cast<T>(eri(t, d, t, a));
    Tensor3<T> eri_dtta(n, n, n);  // [d][t][a]
    for (std::size_t d = 0; d < n; ++d)
      for (std::size_t t = 0; t < n; ++t)
        for (std::size_t a = 0; a < n; ++a) eri_dtta(d, t, a) = static_cast<T>(eri(d, t, t, a));

    Matrix<T> term3a(n, n, T{}), term3c(n, n, T{});  // [a][d]
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t d = 0; d < n; ++d) {
        T acc_h{}, acc_x{};
        for (std::size_t t = 0; t < n; ++t) {
          acc_h += T(hc(a, t)) * eri_tdta(t, d, a);
          acc_x += T(xc(a, t)) * eri_dtta(d, t, a);
        }
        term3a(a, d) = acc_h;
        term3c(a, d) = acc_x;
      }

    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t a = 0; a < n; ++a) {
          T term3b_pre{}, term3d_pre{};
          for (std::size_t t = 0; t < n; ++t) {
            term3b_pre += T(hc(c, t)) * eri_tdta(t, d, a);
            term3d_pre += T(xc(c, t)) * eri_dtta(d, t, a);
          }
          acc += T(K(a, c)) * term3a(a, d) - T(K(a, c)) * term3b_pre - T(K(a, c)) * term3c(a, d) +
                 T(K(a, c)) * term3d_pre;
        }
        Psi(c, d) += acc;
      }
  }

  // P4 (d==a branch): eri_tbtc[t,b,c]=eri(t,b,t,c); term4a_pre[d,b,c]=sum_t hc(d,t)*eri_tbtc;
  // term4b[b,c]=sum_t hc(b,t)*eri_tbtc; same with xc/eri_bttc for term4c_pre/term4d.
  {
    Tensor3<T> eri_tbtc(n, n, n);  // [t][b][c]
    for (std::size_t t = 0; t < n; ++t)
      for (std::size_t b = 0; b < n; ++b)
        for (std::size_t c = 0; c < n; ++c) eri_tbtc(t, b, c) = static_cast<T>(eri(t, b, t, c));
    Tensor3<T> eri_bttc(n, n, n);  // [b][t][c]
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < n; ++c) eri_bttc(b, t, c) = static_cast<T>(eri(b, t, t, c));

    Matrix<T> term4b(n, n, T{}), term4d(n, n, T{});  // [b][c]
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c) {
        T acc_h{}, acc_x{};
        for (std::size_t t = 0; t < n; ++t) {
          acc_h += T(hc(b, t)) * eri_tbtc(t, b, c);
          acc_x += T(xc(b, t)) * eri_bttc(b, t, c);
        }
        term4b(b, c) = acc_h;
        term4d(b, c) = acc_x;
      }

    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t b = 0; b < n; ++b) {
          T term4a_pre{}, term4c_pre{};
          for (std::size_t t = 0; t < n; ++t) {
            term4a_pre += T(hc(d, t)) * eri_tbtc(t, b, c);
            term4c_pre += T(xc(d, t)) * eri_bttc(b, t, c);
          }
          acc += T(K(d, b)) * (-(term4a_pre - term4b(b, c)) + (term4c_pre - term4d(b, c)));
        }
        Psi(c, d) += acc;
      }
  }

  // P5: M5a[b,c,d]=sum_a eri(d,b,c,a)*K[a,b]; M5b[a,c,d]=sum_b eri(d,b,c,a)*K[a,b].
  {
    Tensor3<T> M5a(n, n, n), M5b(n, n, n);
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        for (std::size_t b = 0; b < n; ++b) {
          T acc{};
          for (std::size_t a = 0; a < n; ++a) acc += static_cast<T>(eri(d, b, c, a)) * K(a, b);
          M5a(b, c, d) = acc;
        }
        for (std::size_t a = 0; a < n; ++a) {
          T acc{};
          for (std::size_t b = 0; b < n; ++b) acc += static_cast<T>(eri(d, b, c, a)) * K(a, b);
          M5b(a, c, d) = acc;
        }
      }
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        T acc{};
        for (std::size_t a = 0; a < n; ++a) acc += T(hc(a, c)) * M5b(a, c, d) - T(hc(a, d)) * M5b(a, c, d);
        for (std::size_t b = 0; b < n; ++b) acc += T(-hc(b, c)) * M5a(b, c, d) + T(hc(b, d)) * M5a(b, c, d);
        Psi(c, d) += acc;
      }
  }

  // P6: M6a[b,c,d]=sum_a eri(b,d,c,a)*K[a,b]; M6b[a,c,d]=sum_b eri(b,d,c,a)*K[a,b].
  {
    Tensor3<T> M6a(n, n, n), M6b(n, n, n);
    for (std::size_t c = 0; c < n; ++c)
      for (std::size_t d = 0; d < n; ++d) {
        for (std::size_t b = 0; b < n; ++b) {
          T acc{};
          for (std::size_t a = 0; a < n; ++a) acc += static_cast<T>(eri(b, d, c, a)) * K(a, b);
          M6a(b, c, d) = acc;
        }
        for (std::size_t a = 0; a < n; ++a) {
          T acc{};
          for (std::size_t b = 0; b < n; ++b) acc += static_cast<T>(eri(b, d, c, a)) * K(a, b);
          M6b(a, c, d) = acc;
        }
      }
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

template <typename Eri>
void checkJkOnlySigmaArgs(std::size_t n, const Eri& eri, const std::vector<double>& occupations,
                           const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                           const char* caller) {
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n || two_rdm_h.rows() != n || two_rdm_x.rows() != n) {
    throw std::runtime_error(std::string(caller) + ": inconsistent input dimensions");
  }
}

}  // namespace

template <typename Eri>
std::vector<double> jkOnlySigmaHessianVector(
    const Matrix<double>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  const std::size_t n = h.rows();
  checkJkOnlySigmaArgs(n, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlySigmaHessianVector");
  if (v.size() != pair_indices.size()) {
    throw std::runtime_error("jkOnlySigmaHessianVector: v has the wrong size");
  }

  Matrix<double> K(n, n, 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    K(p, q) = v[i];
    K(q, p) = -v[i];
  }

  const Matrix<double> Phi = buildPhi(h, eri, occupations, two_rdm_h, two_rdm_x, K);
  const Matrix<double> Psi = buildPsi(h, eri, occupations, two_rdm_h, two_rdm_x, K);

  std::vector<double> w(pair_indices.size(), 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    w[i] = 0.5 * ((Phi(p, q) - Phi(q, p)) + (Psi(p, q) - Psi(q, p)));
  }
  return w;
}

template <typename Eri>
std::vector<double> jkOnlySigmaJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  using C = std::complex<double>;
  const std::size_t n = h.rows();
  checkJkOnlySigmaArgs(n, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlySigmaJointHessianVector");
  const std::size_t n_pairs = pair_indices.size();
  if (v.size() != 2 * n_pairs) {
    throw std::runtime_error("jkOnlySigmaJointHessianVector: v has the wrong size");
  }

  // K_t: antisymmetric (the usual real-step kappa matrix). K_y: SYMMETRIC (imaginary-step
  // convention kappa_pq=kappa_qp=+y_I) -- see this file's header for the derivation reference.
  Matrix<double> K_t(n, n, 0.0), K_y(n, n, 0.0);
  for (std::size_t i = 0; i < n_pairs; ++i) {
    const auto& [p, q] = pair_indices[i];
    K_t(p, q) = v[i];
    K_t(q, p) = -v[i];
    K_y(p, q) = v[n_pairs + i];
    K_y(q, p) = v[n_pairs + i];
  }

  const Matrix<C> Phi_t = buildPhi(h, eri, occupations, two_rdm_h, two_rdm_x, K_t);
  const Matrix<C> Psi_t = buildPsi(h, eri, occupations, two_rdm_h, two_rdm_x, K_t);
  const Matrix<C> Phi_y = buildPhi(h, eri, occupations, two_rdm_h, two_rdm_x, K_y);
  const Matrix<C> Psi_y = buildPsi(h, eri, occupations, two_rdm_h, two_rdm_x, K_y);
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

template std::vector<double> jkOnlySigmaHessianVector(const Matrix<double>&, const Tensor4<double>&,
                                                        const std::vector<double>&, const Matrix<double>&,
                                                        const Matrix<double>&,
                                                        const std::vector<std::pair<std::size_t, std::size_t>>&,
                                                        const std::vector<double>&);
template std::vector<double> jkOnlySigmaHessianVector(
    const Matrix<double>&, const CholeskyEri<double>&, const std::vector<double>&, const Matrix<double>&,
    const Matrix<double>&, const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);

template std::vector<double> jkOnlySigmaJointHessianVector(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&, const std::vector<double>&,
    const Matrix<double>&, const Matrix<double>&, const std::vector<std::pair<std::size_t, std::size_t>>&,
    const std::vector<double>&);
template std::vector<double> jkOnlySigmaJointHessianVector(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&, const std::vector<double>&,
    const Matrix<double>&, const Matrix<double>&, const std::vector<std::pair<std::size_t, std::size_t>>&,
    const std::vector<double>&);

}  // namespace rerdmft
