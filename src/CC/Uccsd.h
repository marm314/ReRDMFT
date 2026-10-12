#ifndef RERDMFT_CC_UCCSD_H
#define RERDMFT_CC_UCCSD_H

#include <complex>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "CholeskyEri.h"
#include "Matrix.h"
#include "NonRelSpinRiMoEri.h"
#include "RiMoEri.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

// Coupled cluster singles and doubles for complex spin-orbitals / spinors (doc/uccsd.tex): one spin-orbital code,
// valid for UCCSD with complex spatial orbitals (NON_REL), X2C and no-pair 4-component spinors alike. Every
// contraction is a ZGEMM; the ovvv/vvvv two-electron blocks are never stored (they are rebuilt one slab at a time
// from the factorized integrals), so memory is O(o^2 v^2 + N_aux n^2).
using CcScalar = std::complex<double>;

// Source of the two-electron integrals in physics notation <pq|rs>, over the SAME spin-orbital index space as the
// one-electron matrix handed to runUccsd.
class CcIntegrals {
 public:
  virtual ~CcIntegrals() = default;

  // out[((p*|Q| + q)*|R| + r)*|S| + s] = <P[p] Q[q] | R[r] S[s]>.
  virtual void physicsBlock(const std::vector<std::size_t>& P, const std::vector<std::size_t>& Q,
                            const std::vector<std::size_t>& R, const std::vector<std::size_t>& S,
                            CcScalar* out) const = 0;

  // Declares that slab() will be called repeatedly with this `list` (lets a factorized source cache the
  // list x list block of its second factor).
  virtual void prepareSlabs(const std::vector<std::size_t>& /*list*/) const {}

  // out[(q*L + r)*L + s] = <p list[q] | list[r] list[s]>, L = list.size(): one slab of the ovvv (p occupied) or
  // vvvv (p virtual) block.
  virtual void slab(std::size_t p, const std::vector<std::size_t>& list, CcScalar* out) const {
    physicsBlock({p}, list, list, list, out);
  }
};

// The Coulomb kernel as a sum over an auxiliary index: (pq|rs) = sum_P bra_P(p,q) ket_P(r,s). RI: bra == ket == B.
class CcPairFactor {
 public:
  virtual ~CcPairFactor() = default;
  virtual std::size_t nAux() const = 0;
  // out[(P*|I| + i)*|J| + j] = factor_P(I[i], J[j]).
  virtual void gather(const std::vector<std::size_t>& I, const std::vector<std::size_t>& J, CcScalar* out) const = 0;
};

// Integrals contracted through a pair factorization, with ZGEMM. `shift` is subtracted from every index handed
// to the factors (a factor stored over a trimmed orbital range, e.g. the 4C positive-energy block).
class FactoredCcIntegrals : public CcIntegrals {
 public:
  FactoredCcIntegrals(std::shared_ptr<const CcPairFactor> bra, std::shared_ptr<const CcPairFactor> ket,
                      std::size_t shift = 0)
      : bra_(std::move(bra)), ket_(std::move(ket)), shift_(shift) {}

  void physicsBlock(const std::vector<std::size_t>& P, const std::vector<std::size_t>& Q,
                    const std::vector<std::size_t>& R, const std::vector<std::size_t>& S,
                    CcScalar* out) const override;
  void prepareSlabs(const std::vector<std::size_t>& list) const override;
  void slab(std::size_t p, const std::vector<std::size_t>& list, CcScalar* out) const override;

 private:
  std::shared_ptr<const CcPairFactor> bra_, ket_;
  std::size_t shift_;
  mutable std::vector<std::size_t> slab_list_;
  mutable std::vector<CcScalar> slab_ket_;  // ket_P(list[q], list[s]), [P][q][s]
};

// Integrals read element by element (dense Tensor4 / SymmetricEri unique-element store): no factorization,
// each block costs one element access per entry.
template <typename Eri>
class ElementCcIntegrals : public CcIntegrals {
 public:
  ElementCcIntegrals(const Eri& eri, std::size_t shift) : eri_(eri), shift_(shift) {}

  void physicsBlock(const std::vector<std::size_t>& P, const std::vector<std::size_t>& Q,
                    const std::vector<std::size_t>& R, const std::vector<std::size_t>& S,
                    CcScalar* out) const override {
    const std::size_t nq = Q.size(), nr = R.size(), ns = S.size();
#pragma omp parallel for schedule(static) if (P.size() * Q.size() * R.size() * S.size() > (std::size_t(1) << 22))
    for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(P.size()); ++p)
      for (std::size_t q = 0; q < nq; ++q)
        for (std::size_t r = 0; r < nr; ++r)
          for (std::size_t s = 0; s < ns; ++s)
            out[((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns + s] =
                CcScalar(eri_(P[static_cast<std::size_t>(p)] - shift_, Q[q] - shift_, R[r] - shift_, S[s] - shift_));
  }

 private:
  const Eri& eri_;
  std::size_t shift_;
};

// Adapters from every two-electron representation of this code base. The eri object must outlive the result.
std::unique_ptr<CcIntegrals> makeCcIntegrals(const RiMoEri& eri, std::size_t shift = 0);
std::unique_ptr<CcIntegrals> makeCcIntegrals(const RiNonRelSpinMoEri& eri, std::size_t shift = 0);
std::unique_ptr<CcIntegrals> makeCcIntegrals(const CholeskyEri<double>& eri, std::size_t shift = 0);
std::unique_ptr<CcIntegrals> makeCcIntegrals(const CholeskyEri<CcScalar>& eri, std::size_t shift = 0);
template <typename T>
std::unique_ptr<CcIntegrals> makeCcIntegrals(const SymmetricEri<T>& eri, std::size_t shift = 0) {
  return std::make_unique<ElementCcIntegrals<SymmetricEri<T>>>(eri, shift);
}
template <typename T>
std::unique_ptr<CcIntegrals> makeCcIntegrals(const Tensor4<T>& eri, std::size_t shift = 0) {
  return std::make_unique<ElementCcIntegrals<Tensor4<T>>>(eri, shift);
}

struct UccsdSettings {
  int max_iterations = 100;
  double convergence = 1e-8;  // |dE| threshold; the amplitude change must be below 100 * convergence
  int diis_size = 8;          // < 2 disables DIIS
  double level_shift = 0.0;
};

struct UccsdResult {
  bool converged = false;
  int iterations = 0;
  double e_mp2 = 0.0;       // first-iteration (MP2-like) correlation energy
  double e_corr = 0.0;      // Re of the CCSD correlation energy
  double e_corr_imag = 0.0; // Im (should vanish)
  double t1_diagnostic = 0.0;
  double max_fock_asymmetry = 0.0;  // max |F - F^dagger| over the active space (sanity check)
  double max_fock_ov = 0.0;         // max |F_ov| (0 for a converged HF reference)
  std::vector<CcScalar> t1, t2;     // converged amplitudes, row-major [i][a] and [i][j][a][b] over the active spinors
};

// Spin-orbital CCSD on the reference determinant made of occ_frozen + occ_active (all occupied spinors, indices
// into h_mo and into the integral source); only occ_active and vir are correlated, occ_frozen enters the Fock
// matrix. h_mo is the one-electron operator in the MO basis. Progress goes to `log`.
UccsdResult runUccsd(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const std::vector<std::size_t>& occ_frozen,
                     const std::vector<std::size_t>& occ_active, const std::vector<std::size_t>& vir,
                     const UccsdSettings& settings, std::ostream& log);

// Occupied/virtual spinor lists of the three methods.
struct CcOrbitalSpaces {
  std::vector<std::size_t> occ_frozen, occ_active, vir;
  // Position (1-based) of every spinor of the integral/one-electron index space in the output numbering used by the density
  // matrix files (CC_RDM): spinors in Kramers-pair order (p, pbar) = (2k+1, 2k+2); 0 = not part of the correlated/frozen space.
  std::vector<std::size_t> file_index;
};
// Spinors [first, dim) ordered by energy, the lowest n_electrons of them occupied (X2C: first = 0; C4_SPINOR:
// first = number of negative-energy spinors, so only the positive-energy virtuals are correlated).
CcOrbitalSpaces contiguousCcSpaces(std::size_t first, std::size_t dim, std::size_t n_electrons, std::size_t n_frozen);
// NON_REL spin-orbitals in block layout [alpha_0..alpha_{n-1}, beta_0..beta_{n-1}]; n_frozen spin-orbitals
// (even) = n_frozen/2 spatial orbitals per spin.
CcOrbitalSpaces nonRelCcSpaces(std::size_t n_spatial, std::size_t n_electrons, std::size_t n_frozen);

// Convenience wrapper used by main.cpp: converts the one-electron matrix, builds the integral source, runs the CC
// and writes the standard report block (including E_SCF + E_corr) to `log`.
template <typename T, typename Eri>
UccsdResult runUccsdReport(const std::string& label, const Matrix<T>& h_mo, const Eri& eri, std::size_t eri_shift,
                           const CcOrbitalSpaces& spaces, const UccsdSettings& settings, double scf_total_energy,
                           std::ostream& log);

}  // namespace rerdmft

#include "UccsdReport.h"

#endif  // RERDMFT_CC_UCCSD_H
