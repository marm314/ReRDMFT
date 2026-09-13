#ifndef RERDMFT_ELECTRONREPULSION_H
#define RERDMFT_ELECTRONREPULSION_H

#include <cstddef>
#include <vector>

#include "MolecularBasis.h"
#include "Tensor4.h"

namespace rerdmft {

// Computes the full (real) electron-repulsion tensor in chemist's notation,
//   (pq|rs) = integral integral p(1) q(1) (1/r12) r(2) s(2) dr1 dr2,
// for a single already-normalized cartesian AO basis used on all four
// indices, via libcint's cint2e_cart. Exploits the standard 8-fold
// permutational symmetry of real-orbital ERIs (p<->q, r<->s, (pq)<->(rs))
// to only evaluate one representative of each symmetry-equivalent set --
// but still stores the full dense N^4 result (all 8 equivalent positions
// written out), since this is used as a strided/contiguous intermediate
// by RkbTwoElectron.cpp's leg-transform algorithm. For a real, standalone
// two-electron tensor that is not going to be leg-transformed (e.g.
// NON_REL/NonRelHartreeFock.h's), prefer twoElectronIntegralsPacked,
// which stores only the unique values.
Tensor4<double> twoElectronIntegrals(const std::vector<BasisFunction>& basis);

// Same integral, but with p,q drawn from `basis_pq` and r,s from
// `basis_rs` (generally a different basis, e.g. Large vs unrestricted-
// kinetic-balance Small). Exploits p<->q and r<->s symmetry (not the
// (pq)<->(rs) swap, since the two sides are generally different bases).
Tensor4<double> twoElectronIntegralsCross(const std::vector<BasisFunction>& basis_pq,
                                           const std::vector<BasisFunction>& basis_rs);

// Dense-triangular storage for (pq|rs) that stores only the values not
// related by the full real-orbital 8-fold permutational symmetry: p<->q,
// r<->s, and (pq)<->(rs). Writing PAIR(p,q) for the standard symmetric-
// matrix triangular index over the unordered pair {p,q} (0 <= PAIR < N*
// (N+1)/2), this symmetry means the integral depends only on the
// unordered pair {PAIR(p,q), PAIR(r,s)} -- so it is stored the same way
// one would store a symmetric matrix over PAIR indices: a triangular
// array of size M*(M+1)/2 where M = N*(N+1)/2, instead of the dense N^4
// (e.g. for N=70, M=2485 and this stores ~3.1e6 doubles instead of
// 70^4 = 2.4e7 -- about an 8x reduction).
class PackedTwoElectronTensor {
 public:
  PackedTwoElectronTensor() = default;
  explicit PackedTwoElectronTensor(std::size_t n)
      : n_(n), pairs_(n * (n + 1) / 2), data_(pairs_ * (pairs_ + 1) / 2, 0.0) {}

  std::size_t dim() const { return n_; }
  // Real values actually stored -- about 1/8th of the dense N^4 count.
  std::size_t storedCount() const { return data_.size(); }

  double operator()(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
    return data_[triangularIndex(pairIndex(p, q), pairIndex(r, s))];
  }

  // Used only during construction: (p,q,r,s) and all 7 of its symmetry
  // partners (p<->q, r<->s, (pq)<->(rs)) share one triangular slot, so it
  // does not matter which representative the caller passes.
  void set(std::size_t p, std::size_t q, std::size_t r, std::size_t s, double value) {
    data_[triangularIndex(pairIndex(p, q), pairIndex(r, s))] = value;
  }

 private:
  std::size_t pairIndex(std::size_t p, std::size_t q) const {
    const std::size_t lo = p < q ? p : q;
    const std::size_t hi = p < q ? q : p;
    return hi * (hi + 1) / 2 + lo;
  }
  std::size_t triangularIndex(std::size_t x, std::size_t y) const {
    const std::size_t lo = x < y ? x : y;
    const std::size_t hi = x < y ? y : x;
    return hi * (hi + 1) / 2 + lo;
  }

  std::size_t n_ = 0;
  std::size_t pairs_ = 0;
  std::vector<double> data_;
};

// Builds the packed (pq|rs) tensor for a single basis (see
// PackedTwoElectronTensor) -- the same integral as twoElectronIntegrals,
// via the same libcint evaluation, but exploiting the full 8-fold real-
// orbital symmetry for STORAGE too, not just computation.
PackedTwoElectronTensor twoElectronIntegralsPacked(const std::vector<BasisFunction>& basis);

}  // namespace rerdmft

#endif  // RERDMFT_ELECTRONREPULSION_H
