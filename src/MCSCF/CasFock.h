#ifndef RERDMFT_MCSCF_CASFOCK_H
#define RERDMFT_MCSCF_CASFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Casci.h"
#include "CholeskyEri.h"
#include "Matrix.h"
#include "NonRelSpinRiMoEri.h"
#include "RiMoEri.h"

namespace rerdmft {

// Generalized Fock matrix and electronic energy of a CASSCF state, in the project's GenFock
// convention (Hessian_opt/GeneralizedFock.h): F_pq = sum_r h_rp D_rq + 2 sum_rst <sr|tp> two_rdm_srtq,
// to be fed to orbitalGradient(F) like every other functional.
//
// Orbital sets are INDEX LISTS (they need not be contiguous: NON_REL's spin-orbitals are laid out
// [alpha block | beta block]): `core` = closed-shell core spinors (occupation 1), `active` = the active
// spinors in the order the CI uses them, every other index is virtual (occupation 0). `rdm1`/`rdm2` are the ACTIVE-space CI
// densities in MCSCF/Casci.h's convention,
//   rdm1(t,u) = <a+_t a_u>,   rdm2[t,u,v,w] = <a+_t a+_v a_w a_u>   (flat ((t*A+u)*A+v)*A+w).
// The full 2-RDM is Wick(D) + the active cumulant lambda, with D the total 1-RDM (core 1, active rdm1):
//   lambda[t,u,v,w] = rdm2[t,u,v,w] - D_tu D_vw + D_tw D_vu     (active indices only)
// so, with Fock[D] = h + G[D] and G[rho]_rp = sum_st rho_st (<sr|tp> - <rs|tp>),
//   F_pq = sum_r Fock[D]_rp D_rq + [q active] sum_{s,t,r active} <sr|tp> lambda[s,t,r,q],
//   E    = sum_pq h_pq D_pq + 1/2 sum_pr D_pr G[D]_pr + 1/2 sum_{p active} (lambda part of F)_pp,
// with D_pq = <a+_p a_q> and h_pq = <p|h|q> (the true expectation value; GeneralizedFock.h's comment writes
// h_pq D_qp, which is the same thing only for real or diagonal D -- the complex finite-difference test of
// tests/test_casfock.cpp is what pins the orientation).
// Nothing larger than the n_active^4 CI density is ever formed. Backends (overloads of the two
// primitives below):
//   * any type with dim0() and operator()(a,b,c,d) = <ab|cd>   -- element-access reference, O(n^4..n^5);
//   * CholeskyEri<T>                       -- GEMMs on the vectors, O(N_chol n^2 (n_core+n_active));
//   * RiNonRelSpinMoEri (T=double) and RiMoEri (complex) -- GEMMs on B(P,p,q), O(n_aux n^2 (n_core+n_active)).
// `h` is the BARE one-body Hamiltonian over all n MO spinors; `eri` is physics notation <ab|cd>.
template <typename T>
struct CasFockResult {
  Matrix<T> fock;       // n x n, GenFock convention
  double energy = 0.0;  // electronic energy of the CAS state (no nuclear repulsion)
};

// G[rho]_rp = sum_st rho_st (<sr|tp> - <rs|tp>)   (rho Hermitian, n x n)
template <typename T, typename Eri>
Matrix<T> casTwoElectronOperator(const Eri& eri, const Matrix<T>& rho) {
  const std::size_t n = rho.rows();
  Matrix<T> g(n, n, T{});
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t p = 0; p < n; ++p) {
      T s{};
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) s += rho(a, b) * (eri(a, r, b, p) - eri(r, a, b, p));
      g(r, p) = s;
    }
  return g;
}

// F^lambda(p,q) = sum_{s,t,r in active} <sr|tp> lambda[s,t,r,q], p over all n, q over the active set
// (n x n_active). `lambda` is the flat n_active^4 array lambda[((s*A+t)*A+r)*A+q]; `active` lists the active indices.
template <typename T, typename Eri>
Matrix<T> casActiveCumulantFock(const Eri& eri, const std::vector<std::size_t>& active, const std::vector<T>& lambda) {
  const std::size_t n = eri.dim0(), a = active.size();
  Matrix<T> f(n, a, T{});
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < a; ++q) {
      T s{};
      for (std::size_t is = 0; is < a; ++is)
        for (std::size_t it = 0; it < a; ++it)
          for (std::size_t ir = 0; ir < a; ++ir)
            s += eri(active[is], active[ir], active[it], p) * lambda[((is * a + it) * a + ir) * a + q];
      f(p, q) = s;
    }
  return f;
}

// (tu|vw) = <tv|uw> over the active spinors, chemist notation, flat ((t*A+u)*A+v)*A+w (the integrals of MCSCF/Casci.h).
template <typename T, typename Eri>
std::vector<T> casActiveIntegrals(const Eri& eri, const std::vector<std::size_t>& active) {
  const std::size_t a = active.size();
  std::vector<T> out(a * a * a * a);
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t u = 0; u < a; ++u)
      for (std::size_t v = 0; v < a; ++v)
        for (std::size_t w = 0; w < a; ++w) out[((t * a + u) * a + v) * a + w] = eri(active[t], active[v], active[u], active[w]);
  return out;
}

// Optimized overloads (defined in CasFock.cpp); they win over the generic templates above.
template <typename T>
Matrix<T> casTwoElectronOperator(const CholeskyEri<T>& eri, const Matrix<T>& rho);
template <typename T>
Matrix<T> casActiveCumulantFock(const CholeskyEri<T>& eri, const std::vector<std::size_t>& active, const std::vector<T>& lambda);
template <typename T>
std::vector<T> casActiveIntegrals(const CholeskyEri<T>& eri, const std::vector<std::size_t>& active);
Matrix<double> casTwoElectronOperator(const RiNonRelSpinMoEri& eri, const Matrix<double>& rho);
Matrix<double> casActiveCumulantFock(const RiNonRelSpinMoEri& eri, const std::vector<std::size_t>& active, const std::vector<double>& lambda);
std::vector<double> casActiveIntegrals(const RiNonRelSpinMoEri& eri, const std::vector<std::size_t>& active);
Matrix<std::complex<double>> casTwoElectronOperator(const RiMoEri& eri, const Matrix<std::complex<double>>& rho);
Matrix<std::complex<double>> casActiveCumulantFock(const RiMoEri& eri, const std::vector<std::size_t>& active,
                                                    const std::vector<std::complex<double>>& lambda);
std::vector<std::complex<double>> casActiveIntegrals(const RiMoEri& eri, const std::vector<std::size_t>& active);

// The core-dressed active-space Hamiltonian MCSCF/Casci.h solves: h1eff = h_aa + G[D_core]_aa,
// core_energy = sum_i h_ii + 1/2 sum_i G[D_core]_ii, and the active (tu|vw) block. `n_electrons` = electrons in the active space.
template <typename T, typename Eri>
CasciHamiltonian<T> casciHamiltonian(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& core,
                                     const std::vector<std::size_t>& active, std::size_t n_electrons) {
  const std::size_t n = h.rows(), a = active.size();
  Matrix<T> d_core(n, n, T{});
  for (std::size_t i : core) d_core(i, i) = T(1.0);
  const Matrix<T> g = casTwoElectronOperator(eri, d_core);
  CasciHamiltonian<T> ham;
  ham.n_orbitals = a;
  ham.n_electrons = n_electrons;
  ham.h = Matrix<T>(a, a);
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t u = 0; u < a; ++u) ham.h(t, u) = h(active[t], active[u]) + g(active[t], active[u]);
  double ecore = 0.0;
  for (std::size_t i : core) ecore += std::real(h(i, i)) + 0.5 * std::real(g(i, i));
  ham.core_energy = ecore;
  ham.eri = casActiveIntegrals<T>(eri, active);
  return ham;
}

template <typename T, typename Eri>
CasFockResult<T> casFock(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& core,
                         const std::vector<std::size_t>& active, const Matrix<T>& rdm1, const std::vector<T>& rdm2) {
  const std::size_t n = h.rows(), a = active.size();
  Matrix<T> d(n, n, T{});
  for (std::size_t i : core) d(i, i) = T(1.0);
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t u = 0; u < a; ++u) d(active[t], active[u]) = rdm1(t, u);
  std::vector<long> slot(n, -1);   // position in the active list, -1 for a non-active index
  for (std::size_t t = 0; t < a; ++t) slot[active[t]] = static_cast<long>(t);

  std::vector<T> lambda(a * a * a * a);
  for (std::size_t t = 0; t < a; ++t)
    for (std::size_t u = 0; u < a; ++u)
      for (std::size_t v = 0; v < a; ++v)
        for (std::size_t w = 0; w < a; ++w)
          lambda[((t * a + u) * a + v) * a + w] =
              rdm2[((t * a + u) * a + v) * a + w] - rdm1(t, u) * rdm1(v, w) + rdm1(t, w) * rdm1(v, u);

  const Matrix<T> g = casTwoElectronOperator(eri, d);
  const Matrix<T> fl = casActiveCumulantFock(eri, active, lambda);

  CasFockResult<T> out;
  out.fock = Matrix<T>(n, n, T{});
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      T s{};
      for (std::size_t r = 0; r < n; ++r) s += (h(r, p) + g(r, p)) * d(r, q);
      if (slot[q] >= 0) s += fl(p, static_cast<std::size_t>(slot[q]));
      out.fock(p, q) = s;
    }
  T e{};
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) e += h(p, q) * d(p, q) + T(0.5) * d(p, q) * g(p, q);
  for (std::size_t t = 0; t < a; ++t) e += T(0.5) * fl(active[t], t);
  out.energy = std::real(e);
  return out;
}

}  // namespace rerdmft

#endif
