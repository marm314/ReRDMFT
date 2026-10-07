#include "HartreeExchangeSigmaHessianVector.h"

#include <complex>
#include <stdexcept>
#include <string>

#include "CholeskyEri.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

namespace {

// Same convention as HartreeExchangeHessian.cpp's own `conjugate` helper: identity for real T,
// std::conj for complex T -- needed because `rawHartreeExchangeHessianTerm` applies conjugate()
// to exactly ONE fock term (the `s_==p_` branch), a no-op for T=double but NOT for
// T=complex<double> (caught by test_hartreeexchange_sigma_hessian_joint.cpp failing before this
// fix -- a real bug, not just a missed optimization).
inline double conjugateT(double x) { return x; }
inline std::complex<double> conjugateT(std::complex<double> x) { return std::conj(x); }

// Phi_HX(a,b) = sum_{r,s} rawHartreeExchangeHessianTerm(a,b,r,s) * K[r,s] -- see this file's
// header / project memory for the 8-piece derivation this mirrors exactly. `K` is always REAL
// (antisymmetric for the t-case, symmetric for the y-case, see
// `hartreeExchangeSigmaJointHessianVector`); `T` is h/eri/fock's own scalar type.
template <typename T, typename Eri>
void addPhiHX(Matrix<T>& Phi, const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
               const Matrix<double>& hcpl, const Matrix<double>& xcpl, const Matrix<T>& fock,
               const Matrix<double>& K) {
  const std::size_t n = h.rows();
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      T acc{};
      for (std::size_t s = 0; s < n; ++s) acc += T(K(b, s)) * fock(a, s);        // P1a
      for (std::size_t r = 0; r < n; ++r) acc += conjugateT(fock(b, r)) * T(K(r, a));  // P1b (conjugate, see rawHartreeExchangeHessianTerm's s_==p_ branch)
      T t3{}, t4{};
      for (std::size_t s = 0; s < n; ++s) t3 += T(K(b, s)) * h(s, a);
      for (std::size_t r = 0; r < n; ++r) t4 += h(b, r) * T(K(r, a));
      acc -= T(occ[b]) * t3;                                                    // P3
      acc -= T(occ[a]) * t4;                                                    // P4
      for (std::size_t u = 0; u < n; ++u) {
        T inner{};
        for (std::size_t s = 0; s < n; ++s)
          inner += (static_cast<T>(eri(u, s, a, u)) - static_cast<T>(eri(s, u, a, u))) * T(K(b, s));
        acc += T(hcpl(b, u)) * inner;                                           // P5
      }
      for (std::size_t u = 0; u < n; ++u) {
        T inner{};
        for (std::size_t r = 0; r < n; ++r)
          inner += (static_cast<T>(eri(u, b, r, u)) - static_cast<T>(eri(u, b, u, r))) * T(K(r, a));
        acc += T(hcpl(a, u)) * inner;                                           // P6
      }
      // P7: G1(r) = sum_s eri(s,b,a,r)*K[r,s]; G2(s) = sum_r eri(s,b,a,r)*K[r,s]
      std::vector<T> G1(n, T{}), G2(n, T{});
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const T val = static_cast<T>(eri(s, b, a, r)) * T(K(r, s));
          G1[r] += val;
          G2[s] += val;
        }
      for (std::size_t r = 0; r < n; ++r) acc += T(xcpl(r, b)) * G1[r];
      for (std::size_t s = 0; s < n; ++s) acc -= T(hcpl(s, b)) * G2[s];
      for (std::size_t s = 0; s < n; ++s) acc += T(xcpl(a, s)) * G2[s];
      for (std::size_t r = 0; r < n; ++r) acc -= T(xcpl(r, a)) * G1[r];         // P7
      // P8: H1(r) = sum_s eri(s,b,r,a)*K[r,s]; H2(s) = sum_r eri(s,b,r,a)*K[r,s]
      std::vector<T> H1(n, T{}), H2(n, T{});
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const T val = static_cast<T>(eri(s, b, r, a)) * T(K(r, s));
          H1[r] += val;
          H2[s] += val;
        }
      for (std::size_t r = 0; r < n; ++r) acc += T(hcpl(r, a)) * H1[r];
      for (std::size_t s = 0; s < n; ++s) acc += T(xcpl(b, s)) * H2[s];
      for (std::size_t s = 0; s < n; ++s) acc -= T(xcpl(a, s)) * H2[s];
      for (std::size_t r = 0; r < n; ++r) acc -= T(xcpl(r, b)) * H1[r];         // P8
      Phi(a, b) += acc;
    }
}

// Phi_L12(a,b) = sum_{r,s} rawL1L2HessianTerm(a,b,r,s) * K[r,s] -- the 4-piece Kramers-pair
// reduction. `pair_of` must be non-empty (caller checks).
template <typename T, typename Eri>
void addPhiL12(Matrix<T>& Phi, const Eri& eri, const std::vector<std::size_t>& pair_of,
                const Matrix<double>& l1, const Matrix<double>& l2, const Matrix<double>& K) {
  const std::size_t n = l1.rows();
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      T acc{};
      const std::size_t pb = pair_of[b];
      // L1 (q==pair_of[s] => s=pb): -2*sum_u [l1(u,pb)+l2(u,b)] * Wu(u), Wu(u)=sum_r eri(u,pair_of[u],a,r)*K[r,pb]
      for (std::size_t u = 0; u < n; ++u) {
        T Wu{};
        for (std::size_t r = 0; r < n; ++r) Wu += static_cast<T>(eri(u, pair_of[u], a, r)) * T(K(r, pb));
        acc -= T(2.0 * (l1(u, pb) + l2(u, b))) * Wu;
      }
      // L2 (unconditional): 2*sum_r c(r,b)*(Jr(r)-Jr2(r))
      for (std::size_t r = 0; r < n; ++r) {
        const double c_rb = l1(r, b) + l2(r, pb);
        T Jr{}, Jr2{};
        for (std::size_t s = 0; s < n; ++s) {
          Jr += T(K(r, s)) * static_cast<T>(eri(pair_of[r], s, a, pb));
          Jr2 += T(K(r, s)) * static_cast<T>(eri(s, pair_of[r], a, pb));
        }
        acc += T(2.0 * c_rb) * (Jr - Jr2);
      }
      // L3 (unconditional): 2*sum_s d(s,a)*(Ls(s)-Ls2(s))
      const std::size_t pa = pair_of[a];
      for (std::size_t s = 0; s < n; ++s) {
        const double d_sa = l1(a, s) + l2(a, pair_of[s]);
        T Ls{}, Ls2{};
        for (std::size_t r = 0; r < n; ++r) {
          Ls += T(K(r, s)) * static_cast<T>(eri(pa, b, r, pair_of[s]));
          Ls2 += T(K(r, s)) * static_cast<T>(eri(pa, b, pair_of[s], r));
        }
        acc += T(2.0 * d_sa) * (Ls - Ls2);
      }
      // L4 (p==pair_of[r] => r=pa): 2*sum_w [l1(pa,w)+l2(pa,pair_of[w])] * Mw(w), Mw(w)=sum_s eri(s,b,w,pair_of[w])*K[pa,s]
      for (std::size_t w = 0; w < n; ++w) {
        T Mw{};
        for (std::size_t s = 0; s < n; ++s) Mw += static_cast<T>(eri(s, b, w, pair_of[w])) * T(K(pa, s));
        acc += T(2.0 * (l1(pa, w) + l2(pa, pair_of[w]))) * Mw;
      }
      Phi(a, b) += acc;
    }
}

// Psi_HX(c,d) = sum_{a,b} rawHartreeExchangeHessianTerm(a,b,c,d) * K[a,b] -- the mirror
// contraction (first two slots), independently derived (NOT symmetric with Phi_HX).
template <typename T, typename Eri>
void addPsiHX(Matrix<T>& Psi, const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
               const Matrix<double>& hcpl, const Matrix<double>& xcpl, const Matrix<T>& fock,
               const Matrix<double>& K) {
  const std::size_t n = h.rows();
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      for (std::size_t a = 0; a < n; ++a) acc += T(K(a, c)) * fock(a, d);       // Q1
      for (std::size_t b = 0; b < n; ++b) acc += T(K(d, b)) * conjugateT(fock(b, c));  // Q2 (conjugate, see rawHartreeExchangeHessianTerm's s_==p_ branch)
      T t3{}, t4{};
      for (std::size_t a = 0; a < n; ++a) t3 += T(K(a, c)) * h(d, a);
      for (std::size_t b = 0; b < n; ++b) t4 += T(K(d, b)) * h(b, c);
      acc -= T(occ[c]) * t3;                                                   // Q3
      acc -= T(occ[d]) * t4;                                                   // Q4
      for (std::size_t u = 0; u < n; ++u) {
        T inner{};
        for (std::size_t a = 0; a < n; ++a)
          inner += T(K(a, c)) * (static_cast<T>(eri(u, d, a, u)) - static_cast<T>(eri(d, u, a, u)));
        acc += T(hcpl(c, u)) * inner;                                          // Q5
      }
      for (std::size_t u = 0; u < n; ++u) {
        T inner{};
        for (std::size_t b = 0; b < n; ++b)
          inner += T(K(d, b)) * (static_cast<T>(eri(u, b, c, u)) - static_cast<T>(eri(u, b, u, c)));
        acc += T(hcpl(d, u)) * inner;                                          // Q6
      }
      // Q7: U1(b) = sum_a eri(d,b,a,c)*K[a,b]; U2(a) = sum_b eri(d,b,a,c)*K[a,b]
      std::vector<T> U1(n, T{}), U2(n, T{});
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
          const T val = static_cast<T>(eri(d, b, a, c)) * T(K(a, b));
          U1[b] += val;
          U2[a] += val;
        }
      for (std::size_t b = 0; b < n; ++b) acc += T(xcpl(c, b)) * U1[b];
      for (std::size_t b = 0; b < n; ++b) acc -= T(hcpl(d, b)) * U1[b];
      for (std::size_t a = 0; a < n; ++a) acc += T(xcpl(a, d)) * U2[a];
      for (std::size_t a = 0; a < n; ++a) acc -= T(xcpl(c, a)) * U2[a];        // Q7
      // Q8: V1(b) = sum_a eri(d,b,c,a)*K[a,b]; V2(a) = sum_b eri(d,b,c,a)*K[a,b]
      std::vector<T> V1(n, T{}), V2(n, T{});
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
          const T val = static_cast<T>(eri(d, b, c, a)) * T(K(a, b));
          V1[b] += val;
          V2[a] += val;
        }
      for (std::size_t a = 0; a < n; ++a) acc += T(hcpl(c, a)) * V2[a];
      for (std::size_t b = 0; b < n; ++b) acc += T(xcpl(b, d)) * V1[b];
      for (std::size_t a = 0; a < n; ++a) acc -= T(xcpl(a, d)) * V2[a];
      for (std::size_t b = 0; b < n; ++b) acc -= T(xcpl(c, b)) * V1[b];        // Q8
      Psi(c, d) += acc;
    }
}

// Psi_L12(c,d) = sum_{a,b} rawL1L2HessianTerm(a,b,c,d) * K[a,b].
template <typename T, typename Eri>
void addPsiL12(Matrix<T>& Psi, const Eri& eri, const std::vector<std::size_t>& pair_of,
                const Matrix<double>& l1, const Matrix<double>& l2, const Matrix<double>& K) {
  const std::size_t n = l1.rows();
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) {
      T acc{};
      const std::size_t pd = pair_of[d];
      // R1: -2*sum_u [l1(u,d)+l2(u,pd)] * Wu(u), Wu(u)=sum_a eri(u,pair_of[u],a,c)*K[a,pd]
      for (std::size_t u = 0; u < n; ++u) {
        T Wu{};
        for (std::size_t a = 0; a < n; ++a) Wu += static_cast<T>(eri(u, pair_of[u], a, c)) * T(K(a, pd));
        acc -= T(2.0 * (l1(u, d) + l2(u, pd))) * Wu;
      }
      // R2: 2*sum_b coeff1(b)*(Xb(b)-Xb2(b)), coeff1(b)=l1(c,b)+l2(c,pair_of[b])
      for (std::size_t b = 0; b < n; ++b) {
        const double coeff1 = l1(c, b) + l2(c, pair_of[b]);
        T Xb{}, Xb2{};
        for (std::size_t a = 0; a < n; ++a) {
          Xb += T(K(a, b)) * static_cast<T>(eri(pair_of[c], d, a, pair_of[b]));
          Xb2 += T(K(a, b)) * static_cast<T>(eri(d, pair_of[c], a, pair_of[b]));
        }
        acc += T(2.0 * coeff1) * (Xb - Xb2);
      }
      // R3: 2*sum_a coeff2(a)*(Ya(a)-Ya2(a)), coeff2(a)=l1(a,d)+l2(a,pd)
      const std::size_t pc = pair_of[c];
      for (std::size_t a = 0; a < n; ++a) {
        const double coeff2 = l1(a, d) + l2(a, pd);
        T Ya{}, Ya2{};
        for (std::size_t b = 0; b < n; ++b) {
          Ya += T(K(a, b)) * static_cast<T>(eri(pair_of[a], b, c, pd));
          Ya2 += T(K(a, b)) * static_cast<T>(eri(pair_of[a], b, pd, c));
        }
        acc += T(2.0 * coeff2) * (Ya - Ya2);
      }
      // R4: 2*sum_w [l1(c,w)+l2(c,pair_of[w])] * Zw(w), Zw(w)=sum_b eri(d,b,w,pair_of[w])*K[pc,b]
      for (std::size_t w = 0; w < n; ++w) {
        T Zw{};
        for (std::size_t b = 0; b < n; ++b) Zw += static_cast<T>(eri(d, b, w, pair_of[w])) * T(K(pc, b));
        acc += T(2.0 * (l1(c, w) + l2(c, pair_of[w]))) * Zw;
      }
      Psi(c, d) += acc;
    }
}

template <typename T, typename Eri>
void checkHartreeExchangeSigmaArgs(std::size_t n, const Eri& eri, const std::vector<double>& occupations,
                                    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                                    const Matrix<T>& fock, const std::vector<std::size_t>& pair_of,
                                    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2,
                                    const char* caller) {
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n || two_rdm_h.rows() != n || two_rdm_x.rows() != n || fock.rows() != n) {
    throw std::runtime_error(std::string(caller) + ": inconsistent input dimensions");
  }
  if (!pair_of.empty() && (two_rdm_l1.rows() != n || two_rdm_l2.rows() != n || pair_of.size() != n)) {
    throw std::runtime_error(std::string(caller) + ": pair_of/two_rdm_l1/l2 dimensions inconsistent");
  }
}

}  // namespace

template <typename Eri>
std::vector<double> hartreeExchangeSigmaHessianVector(
    const Matrix<double>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  const std::size_t n = h.rows();
  checkHartreeExchangeSigmaArgs(n, eri, occupations, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1,
                                 two_rdm_l2, "hartreeExchangeSigmaHessianVector");
  if (v.size() != pair_indices.size()) {
    throw std::runtime_error("hartreeExchangeSigmaHessianVector: v has the wrong size");
  }

  Matrix<double> K(n, n, 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    K(p, q) = v[i];
    K(q, p) = -v[i];
  }

  Matrix<double> Phi(n, n, 0.0), Psi(n, n, 0.0);
  addPhiHX(Phi, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K);
  addPsiHX(Psi, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K);
  if (!pair_of.empty()) {
    addPhiL12(Phi, eri, pair_of, two_rdm_l1, two_rdm_l2, K);
    addPsiL12(Psi, eri, pair_of, two_rdm_l1, two_rdm_l2, K);
  }

  std::vector<double> w(pair_indices.size(), 0.0);
  for (std::size_t i = 0; i < pair_indices.size(); ++i) {
    const auto& [p, q] = pair_indices[i];
    w[i] = 0.5 * ((Phi(p, q) - Phi(q, p)) + (Psi(p, q) - Psi(q, p)));
  }
  return w;
}

// Complex/joint [t;y] counterpart -- see HartreeExchangeSigmaHessianVector.h and
// [[project-ri-hessian-neo-design]] Stage 3 (same combination formula as JK_only's
// `jkOnlySigmaJointHessianVector`, confirmed `hartreeExchangeJointHessianVector` uses the
// IDENTICAL TT/YY/TY structure before relying on this).
template <typename Eri>
std::vector<double> hartreeExchangeSigmaJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const Matrix<std::complex<double>>& fock, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  using C = std::complex<double>;
  const std::size_t n = h.rows();
  checkHartreeExchangeSigmaArgs(n, eri, occupations, two_rdm_h, two_rdm_x, fock, pair_of, two_rdm_l1,
                                 two_rdm_l2, "hartreeExchangeSigmaJointHessianVector");
  const std::size_t n_pairs = pair_indices.size();
  if (v.size() != 2 * n_pairs) {
    throw std::runtime_error("hartreeExchangeSigmaJointHessianVector: v has the wrong size");
  }

  Matrix<double> K_t(n, n, 0.0), K_y(n, n, 0.0);
  for (std::size_t i = 0; i < n_pairs; ++i) {
    const auto& [p, q] = pair_indices[i];
    K_t(p, q) = v[i];
    K_t(q, p) = -v[i];
    K_y(p, q) = v[n_pairs + i];
    K_y(q, p) = v[n_pairs + i];
  }

  Matrix<C> Phi_t(n, n, C{}), Psi_t(n, n, C{}), Phi_y(n, n, C{}), Psi_y(n, n, C{});
  addPhiHX(Phi_t, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K_t);
  addPsiHX(Psi_t, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K_t);
  addPhiHX(Phi_y, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K_y);
  addPsiHX(Psi_y, h, eri, occupations, two_rdm_h, two_rdm_x, fock, K_y);
  if (!pair_of.empty()) {
    addPhiL12(Phi_t, eri, pair_of, two_rdm_l1, two_rdm_l2, K_t);
    addPsiL12(Psi_t, eri, pair_of, two_rdm_l1, two_rdm_l2, K_t);
    addPhiL12(Phi_y, eri, pair_of, two_rdm_l1, two_rdm_l2, K_y);
    addPsiL12(Psi_y, eri, pair_of, two_rdm_l1, two_rdm_l2, K_y);
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

template std::vector<double> hartreeExchangeSigmaHessianVector(
    const Matrix<double>&, const Tensor4<double>&, const std::vector<double>&, const Matrix<double>&,
    const Matrix<double>&, const Matrix<double>&, const std::vector<std::size_t>&, const Matrix<double>&,
    const Matrix<double>&, const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);
template std::vector<double> hartreeExchangeSigmaHessianVector(
    const Matrix<double>&, const CholeskyEri<double>&, const std::vector<double>&, const Matrix<double>&,
    const Matrix<double>&, const Matrix<double>&, const std::vector<std::size_t>&, const Matrix<double>&,
    const Matrix<double>&, const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);

template std::vector<double> hartreeExchangeSigmaJointHessianVector(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&, const std::vector<double>&,
    const Matrix<double>&, const Matrix<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::size_t>&, const Matrix<double>&, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);
template std::vector<double> hartreeExchangeSigmaJointHessianVector(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&, const std::vector<double>&,
    const Matrix<double>&, const Matrix<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::size_t>&, const Matrix<double>&, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);

}  // namespace rerdmft
