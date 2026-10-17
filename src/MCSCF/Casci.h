#ifndef RERDMFT_MCSCF_CASCI_H
#define RERDMFT_MCSCF_CASCI_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// General CASCI over an active set of M spinors holding N electrons: the CI space is ALL
// C(M,N) determinants (no Ms, spin or time-reversal restriction), so one engine serves real
// orbitals (T = double, NON_REL spin-orbitals) and complex ones (T = std::complex<double>,
// X2C / 4-component spinors). Kramers-pair details (half-stored CI vector, reduced integral
// lists) are deliberately NOT used: they would only be an efficiency on top of this.
//
// Second-quantized Hamiltonian over the active spinors (chemist's notation):
//   H = core_energy + sum_pq h(p,q) a+_p a_q + 1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q,
//   (pq|rs) = int phi_p*(1) phi_q(1) r12^-1 phi_r*(2) phi_s(2),
// with h Hermitian and (pq|rs) = (rs|pq) = (qp|sr)*. `h` is the CORE-DRESSED one-body
// operator (bare one-body plus the core mean field) and `core_energy` the matching
// constant, exactly like PySCF's h1eff / ecore.
//
// Reduced density matrices (one-index-up convention, E = tr(h gamma) + 1/2 (pq|rs) Gamma):
//   rdm1(p,q)       = < a+_p a_q >
//   rdm2[p,q,r,s]   = < a+_p a+_r a_s a_q >      (flat index ((p*M+q)*M+r)*M+s)
// i.e. PySCF's make_rdm12 convention generalized to complex spinors.
template <typename T>
struct CasciHamiltonian {
  std::size_t n_orbitals = 0;   // M
  std::size_t n_electrons = 0;  // N
  Matrix<T> h;                  // M x M, Hermitian
  std::vector<T> eri;           // M^4, chemist (pq|rs), flat index ((p*M+q)*M+r)*M+s
  double core_energy = 0.0;
};

struct CasciOptions {
  double residual_tolerance = 1e-10;  // Davidson: 2-norm of (H - theta) x
  int max_iterations = 200;
  int max_subspace = 24;              // thick restart (n_keep lowest Ritz vectors) when exceeded
  int n_keep = 4;
  // near-degenerate clusters: stop when the Ritz energy moved less than stagnation_energy over stagnation_window iterations
  // and the residual is below stagnation_residual (0 window = off)
  int stagnation_window = 20;
  double stagnation_energy = 1e-9;
  double stagnation_residual = 1e-5;
  // Complex spinors only: active spinors in adjacent Kramers pairs (2k, 2k+1) with Theta|2k> = |2k+1>, Theta|2k+1> = -|2k>.
  // The start vector and every Davidson correction are projected onto the time-reversal-even subspace (c + Theta c)/2, so
  // near-degenerate states of the other symmetry (e.g. the open-shell multiplet components of a stretched bond) cannot mix in.
  // Requires an even electron count and a time-reversal-invariant Hamiltonian.
  bool time_reversal_projection = false;
  // Start-vector choice used by the CAS model (not by Casci::solve): true = closed-shell symmetric determinant (even S only).
  bool symmetric_start = true;
};

template <typename T>
struct CasciResult {
  double energy = 0.0;       // lowest eigenvalue + core_energy
  std::vector<T> ci;         // normalized, length = number of determinants
  bool converged = false;
  int iterations = 0;
  double residual_norm = 0.0;
};

template <typename T>
class Casci {
 public:
  // Throws if n_orbitals > 64, n_electrons > n_orbitals, or the determinant count exceeds 2^31.
  Casci(std::size_t n_orbitals, std::size_t n_electrons);

  std::size_t dimension() const { return dets_.size(); }
  std::size_t nOrbitals() const { return m_; }
  std::size_t nElectrons() const { return n_; }
  // Occupation bitmask of determinant I (bit p set = spinor p occupied), ascending order.
  const std::vector<std::uint64_t>& determinants() const { return dets_; }

  // (H - core_energy) c, length dimension().
  std::vector<T> sigma(const CasciHamiltonian<T>& ham, const std::vector<T>& c) const;
  // Diagonal elements <I|H - core_energy|I>.
  std::vector<double> diagonal(const CasciHamiltonian<T>& ham) const;

  // Lowest eigenpair by Davidson. `start` (optional, length dimension()) is the initial guess;
  // default is the determinant with the lowest diagonal element.
  CasciResult<T> solve(const CasciHamiltonian<T>& ham, const CasciOptions& options = {},
                       const std::vector<T>* start = nullptr) const;

  // Only the 1-RDM of the state `c`, rdm1(p,q) = < a+_p a_q > (no excited-vector storage; for the natural orbitals / WFN export).
  Matrix<T> oneRdm(const std::vector<T>& c) const;
  // Full 1- and 2-RDM of the state `c` (see the convention above). rdm2 has M^4 entries.
  void densities(const std::vector<T>& c, Matrix<T>& rdm1, std::vector<T>& rdm2) const;

 private:
  struct Link {
    std::uint32_t source;  // determinant I
    std::uint32_t target;  // determinant J = a+_p a_q |I> up to sign
    std::int32_t sign;
  };
  // D[pq*dim + J] = (E_pq c)_J with E_pq = a+_p a_q.
  std::vector<T> excitedVectors(const std::vector<T>& c) const;

  // Theta c for the Kramers pairing above: (Theta c)_J = sign * conj(c_I), J = image of determinant I.
  std::vector<T> timeReversed(const std::vector<T>& c) const;

  std::size_t m_, n_;
  std::vector<std::uint64_t> dets_;
  mutable std::vector<std::uint32_t> tr_target_;  // image determinant of I (built on first use)
  mutable std::vector<std::int8_t> tr_sign_;
  // links_[p*M+q]: every determinant I with q occupied and (p == q or p empty), I -> J
  std::vector<std::vector<Link>> links_;
};

extern template class Casci<double>;
extern template class Casci<std::complex<double>>;

}  // namespace rerdmft

#endif
