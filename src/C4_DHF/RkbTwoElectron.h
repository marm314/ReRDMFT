#ifndef RERDMFT_RKBTWOELECTRON_H
#define RERDMFT_RKBTWOELECTRON_H

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"
#include "Tensor4.h"

namespace rerdmft {

// Dense storage for <Spinor_A Spinor_B|Spinor_C Spinor_D> that stores only
// the integrals not related by the electron-exchange symmetry
//   <A B|C D> = <B A|D C>
// (swapping which particle is "1" and which is "2" leaves the Coulomb
// kernel, and hence the integral, unchanged -- true for any spinors, real
// or complex, so it always applies here). Writing x = pairIndex(A,C) for
// the "electron-1 pair" and y = pairIndex(B,D) for the "electron-2 pair",
// this symmetry says the integral only depends on the UNORDERED pair
// {x,y}: exactly the structure of a symmetric matrix, so it is stored the
// same way one would store a symmetric matrix -- a triangular array over
// x<=y, size n^2*(n^2+1)/2 instead of the dense n^4, exactly half.
//
// Hermiticity, <A B|C D> = conj(<C D|A B>), is a second, independent
// symmetry (see rkbTwoElectronIntegrals); it is not folded into this
// storage layout (which would require a more intricate, case-splitting
// index scheme -- fixed points of the "reverse a pair" map behave
// differently from generic pairs -- for a further, and by itself modest,
// 2x on top of this class's already-exact 2x). It is instead checked as a
// runtime invariant (main.cpp, under DEBUG) rather than exploited for
// storage.
class RkbTwoElectronTensor {
 public:
  RkbTwoElectronTensor() = default;
  explicit RkbTwoElectronTensor(std::size_t n)
      : n_(n), pairs_(n * n), data_(pairs_ * (pairs_ + 1) / 2, std::complex<double>(0.0, 0.0)) {}

  // Trusted constructor for IntegralCache.h: takes already-packed raw
  // data (e.g. read back from an on-disk cache file) as-is. Throws if
  // its size does not match what dimension `n` implies, so a corrupt or
  // mismatched cache file cannot silently lead to out-of-bounds reads
  // via operator().
  RkbTwoElectronTensor(std::size_t n, std::vector<std::complex<double>> data)
      : n_(n), pairs_(n * n), data_(std::move(data)) {
    if (data_.size() != pairs_ * (pairs_ + 1) / 2) {
      throw std::runtime_error("RkbTwoElectronTensor: cached data size does not match dimension");
    }
  }

  std::size_t dim() const { return n_; }
  // Complex values actually stored -- half of the dense n^4 count.
  std::size_t storedCount() const { return data_.size(); }
  // Raw packed storage, for IntegralCache.h to write/read directly.
  const std::vector<std::complex<double>>& rawData() const { return data_; }

  std::complex<double> operator()(std::size_t a, std::size_t b, std::size_t c,
                                   std::size_t d) const {
    return data_[triangularIndex(pairIndex(a, c), pairIndex(b, d))];
  }

  // Used only during construction. (a,b,c,d) and its exchange partner
  // (b,a,d,c) share one triangular slot, so it does not matter which of
  // the two the caller passes -- both route to the same storage.
  void set(std::size_t a, std::size_t b, std::size_t c, std::size_t d, std::complex<double> value) {
    data_[triangularIndex(pairIndex(a, c), pairIndex(b, d))] = value;
  }

 private:
  std::size_t pairIndex(std::size_t a, std::size_t c) const { return a * n_ + c; }
  std::size_t triangularIndex(std::size_t x, std::size_t y) const {
    if (x > y) std::swap(x, y);
    return y * (y + 1) / 2 + x;
  }

  std::size_t n_ = 0;
  std::size_t pairs_ = 0;
  std::vector<std::complex<double>> data_;
};

// Builds the full two-electron Coulomb repulsion tensor in the restricted-
// kinetic-balance (RKB) 4-component spinor basis, in PHYSICS notation:
//   <Spinor_A Spinor_B | Spinor_C Spinor_D>
//     = integral integral Spinor_A^*(r1) Spinor_B^*(r2) (1/r12)
//                          Spinor_C(r1)   Spinor_D(r2)   dr1 dr2,
// with Spinor_A, Spinor_C acting on electron 1 and Spinor_B, Spinor_D on
// electron 2 (i.e. <AB|CD> = (AC|BD) in chemist's notation). Each spinor
// index ranges over the same (4*nLarge)-dimensional RKB basis as H_RKB
// (RkbHamiltonian.h): [Large-alpha, Large-beta, RKB-Small-alpha-partner,
// RKB-Small-beta-partner], each block of size nLarge. The result is
// stored in an RkbTwoElectronTensor (electron-exchange symmetry only
// -- see its own comment); Hermiticity is verified, not exploited for
// storage, as a debug-only check (main.cpp).
//
// 1/r12 is a scalar operator (spin- and component-independent), so a
// nonzero <A|Op(r1)|C>-type density requires A and C to be the same
// "flavor" (both Large-alpha, both Large-beta, or both the same RKB-Small
// partner type); cross-flavor combinations are exactly zero and are not
// separately stored. Building the RKB-Small blocks requires the RKB
// coefficients (RkbTransformation.h): a genuine 2-spinor RKB-Small basis
// function has contributions from BOTH the unrestricted-kinetic-balance
// Small-alpha and Small-beta functions, so its two-electron integrals need
// the underlying unrestricted-kinetic-balance Small AO integrals
// transformed through both spin blocks of `rkb_coefficients` and summed.
//
// Computed via three real spatial-AO electron-repulsion tensors
// (ElectronRepulsion.h) -- (Large,Large|Large,Large), (Large,Large|Small,
// Small), and (Small,Small|Small,Small), Small meaning the unrestricted-
// kinetic-balance basis -- each transformed through `rkb_coefficients` on
// whichever legs are Small-type. One of the four RKB-Small(y1),RKB-
// Small(y2) electron-1/electron-2 partner-flavor combinations is skipped
// and recovered by an index permutation of another (see the .cpp), via
// the same electron-exchange symmetry used for storage. Both time and
// memory still scale steeply with basis size, so this remains intended
// for small test systems.
// `use_cholesky` (default true) decomposes the dominant-cost (Small,Small|
// Small,Small) piece via Utils/Cholesky_Decomposition.h's pivoted Cholesky
// decomposition before projecting it into the four RKB-Small(y1),RKB-
// Small(y2) blocks, instead of directly quarter-transforming the full
// (n_small)^4 tensor on each of its two electron-pair legs (see the .cpp
// for the derivation of why this is a legitimate, verified-equivalent
// reformulation). `false` uses the original, unmodified direct-transform
// code path.
RkbTwoElectronTensor rkbTwoElectronIntegrals(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis,
                                              const Matrix<std::complex<double>>& rkb_coefficients,
                                              bool use_cholesky = true, double cholesky_threshold = 1e-10);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTWOELECTRON_H
