#ifndef RERDMFT_UTILS_SYMMETRICERI_H
#define RERDMFT_UTILS_SYMMETRICERI_H

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rerdmft {

// Two-electron integrals <ab|cd> (physics notation, spinors a,b on ... see below) stored WITHOUT the
// n^4 redundancy: only the values not related by the permutational symmetries are kept, every other
// element is rebuilt on access (Tensor4's operator() interface: (a,b,c,d) -> value, dim0..dim3()), so
// every routine templated on the integral type works unchanged.
//
// Symmetries used. Group the indices by electron: x = (a,c) (electron 1), y = (b,d) (electron 2), and let
// xT = (c,a), yT = (d,b) be the reversed pairs. For ANY spinors
//   exchange   <ab|cd> = <ba|dc>              i.e.  M[x][y] = M[y][x]
//   Hermitian  <ab|cd> = conj <cd|ab>         i.e.  M[xT][yT] = conj M[x][y]
// (the second follows from the first plus the reality of 1/r12). For REAL orbitals (NON_REL, and the
// closed-shell spin-orbital tensors built from real spatial ones) there is the extra symmetry
// (ac|bd) = (ca|bd), i.e. M[xT][y] = M[x][y] = M[x][yT], giving the full 8-fold real symmetry.
//
// Layout. A pair class is an unordered {(a,c),(c,a)} (index a<=c): C = n(n+1)/2 classes. Every element
// is (up to the symmetries) one of a few values of one class PAIR {X,Y} (X<=Y after using exchange):
//   real T:     1 value per class pair                 -> ~ n^4 / 8 numbers
//   complex T:  2 values per class pair (v1 = M[xr][yr], v2 = M[xr][yrT], xr/yr = representatives with
//               a<=c) -> ~ n^4 / 4 complex numbers; the other orientations follow from the Hermitian
//               relation as conj(v1), conj(v2).
// (When a class is diagonal, a == c, the relations collapse -- v2 = conj v1 or v2 = v1 -- and the second
// slot is simply unused, a waste of O(n^3) numbers.)
//
// set() writes the value of ONE element; because all of its symmetry partners share the same slot(s),
// setting any member fixes the others. The caller is responsible for supplying values that satisfy the
// symmetries (a physical two-electron tensor does); a tensor that only approximately does is
// symmetrized by whichever elements are set last.
template <typename T>
class SymmetricEri {
 public:
  static constexpr bool kComplex = !std::is_same_v<T, double>;

  SymmetricEri() = default;
  explicit SymmetricEri(std::size_t n)
      : n_(n), n_classes_(n * (n + 1) / 2), data_(n_classes_ * (n_classes_ + 1) / 2 * (kComplex ? 2 : 1), T{}) {}

  // Trusted constructor for a cache file: takes already-packed data as-is; throws if its size does not match
  // what dimension `n` implies, so a corrupt or mismatched file cannot lead to out-of-bounds reads.
  SymmetricEri(std::size_t n, std::vector<T> data)
      : n_(n), n_classes_(n * (n + 1) / 2), data_(std::move(data)) {
    if (data_.size() != n_classes_ * (n_classes_ + 1) / 2 * (kComplex ? 2 : 1)) {
      throw std::runtime_error("SymmetricEri: cached data size does not match dimension");
    }
  }

  std::size_t dim0() const { return n_; }
  std::size_t dim1() const { return n_; }
  std::size_t dim2() const { return n_; }
  std::size_t dim3() const { return n_; }
  std::size_t dim() const { return n_; }

  T operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
    const Loc loc = locate(a, b, c, d);
    if constexpr (kComplex) {
      return loc.conj ? std::conj(data_[loc.slot]) : data_[loc.slot];
    } else {
      return data_[loc.slot];
    }
  }

  void set(std::size_t a, std::size_t b, std::size_t c, std::size_t d, T value) {
    const Loc loc = locate(a, b, c, d);
    if constexpr (kComplex) {
      data_[loc.slot] = loc.conj ? std::conj(value) : value;
    } else {
      data_[loc.slot] = value;
    }
  }

  // Numbers actually stored, and what a dense tensor of the same dimension would hold.
  std::size_t storedCount() const { return data_.size(); }
  static std::size_t denseCount(std::size_t n) { return n * n * n * n; }
  // Raw storage (for a cache file).
  const std::vector<T>& rawData() const { return data_; }

 private:
  struct Loc {
    std::size_t slot;
    bool conj;
  };

  static std::size_t classIndex(std::size_t lo, std::size_t hi) { return hi * (hi + 1) / 2 + lo; }

  Loc locate(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
    // x = (a,c), y = (b,d); t* = the pair is stored reversed relative to its representative (lo <= hi); *d = the
    // pair is diagonal (equal indices). Written with selects (no data-dependent branches): this is the hot path of
    // every element access.
    const bool tx = a > c, ty = b > d;
    const std::size_t alo = tx ? c : a, ahi = tx ? a : c;
    const std::size_t blo = ty ? d : b, bhi = ty ? b : d;
    const std::size_t cx = ahi * (ahi + 1) / 2 + alo;
    const std::size_t cy = bhi * (bhi + 1) / 2 + blo;
    const bool sw = cx > cy;  // exchange: M[x][y] = M[y][x]
    const std::size_t big_x = sw ? cy : cx, big_y = sw ? cx : cy;
    const std::size_t base = big_y * (big_y + 1) / 2 + big_x;
    if constexpr (!kComplex) {
      return {base, false};
    } else {
      const bool xd = a == c, yd = b == d;
      const bool t_x = sw ? ty : tx, t_y = sw ? tx : ty;
      const bool d_x = sw ? yd : xd, d_y = sw ? xd : yd;
      // v2 (the second slot) only for two non-diagonal classes with opposite orientations; the conjugate is taken
      // when the pair that carries the orientation is reversed (for a diagonal x that is y).
      const bool second = (t_x != t_y) && !d_x && !d_y;
      const bool conj = d_x ? t_y : t_x;
      return {2 * base + (second ? 1u : 0u), conj};
    }
  }

  std::size_t n_ = 0;
  std::size_t n_classes_ = 0;
  std::vector<T> data_;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_SYMMETRICERI_H
