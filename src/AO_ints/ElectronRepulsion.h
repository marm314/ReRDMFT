#ifndef RERDMFT_ELECTRONREPULSION_H
#define RERDMFT_ELECTRONREPULSION_H

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "MolecularBasis.h"
#include "Tensor4.h"

namespace rerdmft {

// (pq|rs) for one specific quadruplet of individually-normalized cartesian AOs, each placed at its
// own real atomic center -- a minimal, independent 4-shell/4-atom libcint system per call (not
// shell-batched), so it is just as cheap to call this directly for ONE quadruplet at a time as it
// is to loop over many of them building a full tensor. Exposed (out of ElectronRepulsion.cpp's own
// anonymous namespace) specifically so a PairMatrixSource (Utils/Cholesky_Decomposition.h) can
// compute rows of a Coulomb pair matrix ON DEMAND -- see C4_DHF/RkbCholesky.cpp -- without ever
// materializing the full (pq|rs) tensor the ordinary builders below do, for cases where even the
// packed storage of that full tensor (O(n^4/8)) would be too large to hold at once.
double twoElectronQuadruplet(const BasisFunction& p, const BasisFunction& q, const BasisFunction& r,
                              const BasisFunction& s);

// sqrt((pq|pq)) for every unordered pair {p,q} of `basis`, in the standard symmetric-matrix
// triangular index (hi*(hi+1)/2+lo) -- the per-pair "diagonal" the Schwarz prescreen bounds
// |(pq|rs)| <= sqrt_diag[pq] * sqrt_diag[rs] against (see twoElectronIntegralsPacked's own
// comment). O(n^2) quadruplets, cheap enough to precompute once even for a large `basis`. Exposed
// for the same on-demand-PairMatrixSource use as twoElectronQuadruplet above.
std::vector<double> sqrtPairDiagonal(const std::vector<BasisFunction>& basis);

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
//
// `screening_threshold` (default 0.0, off -- every existing caller is unaffected): same Schwarz
// prescreen as twoElectronIntegralsPacked (see its own comment) -- a skipped quadruplet simply
// keeps its default 0.0 in the dense result, which the 8-fold write-out leaves correctly zero at
// all 8 equivalent positions too.
Tensor4<double> twoElectronIntegrals(const std::vector<BasisFunction>& basis,
                                      double screening_threshold = 0.0);

// Same integral, but with p,q drawn from `basis_pq` and r,s from
// `basis_rs` (generally a different basis, e.g. Large vs unrestricted-
// kinetic-balance Small). Exploits p<->q and r<->s symmetry (not the
// (pq)<->(rs) swap, since the two sides are generally different bases).
//
// `screening_threshold`: same Schwarz prescreen as twoElectronIntegralsCrossPacked (default 0.0,
// off); a skipped quadruplet keeps its default 0.0 at all 4 equivalent dense positions.
Tensor4<double> twoElectronIntegralsCross(const std::vector<BasisFunction>& basis_pq,
                                           const std::vector<BasisFunction>& basis_rs,
                                           double screening_threshold = 0.0);

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
//
// `screening_threshold` (default 0.0, i.e. off -- EVERY existing caller is unaffected): when
// positive, a Schwarz/Cauchy-Schwarz prescreen skips evaluating (pq|rs) entirely (leaving it at
// its default 0.0) whenever sqrt((pq|pq)) * sqrt((rs|rs)) < screening_threshold, since
// |(pq|rs)| <= sqrt((pq|pq) * (rs|rs)) for the real Coulomb operator -- the same bound
// Cholesky_Decomposition.h's own pivoted decomposition relies on for its diagonal-only
// convergence check. The n*(n+1)/2 pair-diagonal values (pq|pq) are computed once up front
// (O(n^2) quadruplets, cheap relative to the full O(n^4) loop they gate).
PackedTwoElectronTensor twoElectronIntegralsPacked(const std::vector<BasisFunction>& basis,
                                                    double screening_threshold = 0.0);

// Dense-triangular-x-dense-triangular storage for the CROSS integral (pq|rs) with p,q from one
// basis and r,s from another (e.g. Large vs unrestricted-kinetic-balance Small, as
// twoElectronIntegralsCross computes): p<->q and r<->s are still exact symmetries of the
// integrand (p(1)q(1) = q(1)p(1) regardless of which basis p,q belong to; same for r,s), so only
// M_pq = n_pq*(n_pq+1)/2 unique pq-pairs times M_rs = n_rs*(n_rs+1)/2 unique rs-pairs need
// storing -- UNLIKE PackedTwoElectronTensor, there is no further (pq)<->(rs) swap symmetry here
// (the two sides are generally different bases, so that swap is not even well-typed), hence a
// dense M_pq x M_rs rectangular array rather than a triangular one over a single combined index.
// About a 4x reduction over the dense n_pq^2 x n_rs^2 Tensor4 twoElectronIntegralsCross returns
// (not 8x, since the missing symmetry is exactly the one that gives the other factor of 2).
class CrossPackedTwoElectronTensor {
 public:
  CrossPackedTwoElectronTensor() = default;
  CrossPackedTwoElectronTensor(std::size_t n_pq, std::size_t n_rs)
      : n_pq_(n_pq), n_rs_(n_rs), m_pq_(n_pq * (n_pq + 1) / 2), m_rs_(n_rs * (n_rs + 1) / 2),
        data_(m_pq_ * m_rs_, 0.0) {}

  std::size_t dimPq() const { return n_pq_; }
  std::size_t dimRs() const { return n_rs_; }
  // Real values actually stored -- about 1/4 of the dense n_pq^2 * n_rs^2 count.
  std::size_t storedCount() const { return data_.size(); }

  double operator()(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
    return data_[pairIndexPq(p, q) * m_rs_ + pairIndexRs(r, s)];
  }

  // Used only during construction: (p,q,r,s) and its 3 symmetry partners (p<->q, r<->s) share one
  // slot, so it does not matter which representative the caller passes.
  void set(std::size_t p, std::size_t q, std::size_t r, std::size_t s, double value) {
    data_[pairIndexPq(p, q) * m_rs_ + pairIndexRs(r, s)] = value;
  }

 private:
  static std::size_t pairIndex(std::size_t p, std::size_t q) {
    const std::size_t lo = p < q ? p : q;
    const std::size_t hi = p < q ? q : p;
    return hi * (hi + 1) / 2 + lo;
  }
  std::size_t pairIndexPq(std::size_t p, std::size_t q) const { return pairIndex(p, q); }
  std::size_t pairIndexRs(std::size_t r, std::size_t s) const { return pairIndex(r, s); }

  std::size_t n_pq_ = 0, n_rs_ = 0, m_pq_ = 0, m_rs_ = 0;
  std::vector<double> data_;
};

// Same integral as twoElectronIntegralsCross, via the same libcint evaluation, but exploiting
// p<->q and r<->s symmetry for STORAGE too (see CrossPackedTwoElectronTensor), not just
// computation. Prefer this over twoElectronIntegralsCross whenever the dense, GEMM-strided
// Tensor4 layout is not itself needed (e.g. a Cholesky decomposition's PairMatrixSource, which
// only ever reads individual elements/rows).
//
// `screening_threshold`: same Schwarz prescreen as twoElectronIntegralsPacked (default 0.0, off),
// using each SIDE's own same-basis pair diagonal: |(pq|rs)| <= sqrt((pq|pq)_pq * (rs|rs)_rs)
// holds regardless of pq and rs coming from different bases, since it is a property of the
// Coulomb operator applied to the two (possibly different-basis) charge distributions, not of
// the bases matching.
CrossPackedTwoElectronTensor twoElectronIntegralsCrossPacked(const std::vector<BasisFunction>& basis_pq,
                                                              const std::vector<BasisFunction>& basis_rs,
                                                              double screening_threshold = 0.0);

}  // namespace rerdmft

#endif  // RERDMFT_ELECTRONREPULSION_H
