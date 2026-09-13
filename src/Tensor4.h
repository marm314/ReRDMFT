#ifndef RERDMFT_TENSOR4_H
#define RERDMFT_TENSOR4_H

#include <cstddef>
#include <vector>

namespace rerdmft {

// A minimal dense, row-major (C-order, last index fastest) rank-4 tensor,
// used for two-electron integrals (pq|rs)-type quantities. Mirrors
// Matrix<T>'s style (flat storage, simple accessors) rather than anything
// more elaborate, matching this project's preference for straightforward
// dense containers over cleverness.
template <typename T>
class Tensor4 {
 public:
  Tensor4() = default;
  Tensor4(std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3, T fill = T{})
      : d0_(d0),
        d1_(d1),
        d2_(d2),
        d3_(d3),
        data_(d0 * d1 * d2 * d3, fill) {}

  std::size_t dim0() const { return d0_; }
  std::size_t dim1() const { return d1_; }
  std::size_t dim2() const { return d2_; }
  std::size_t dim3() const { return d3_; }

  T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return data_[((i * d1_ + j) * d2_ + k) * d3_ + l];
  }
  const T& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const {
    return data_[((i * d1_ + j) * d2_ + k) * d3_ + l];
  }

  // Raw access to the flat, row-major (dim0 slowest, dim3 fastest) backing
  // storage -- used by leg-transform algorithms that reshape a tensor as
  // (outer, this_dim, inner) to contract one leg at a time.
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

 private:
  std::size_t d0_ = 0;
  std::size_t d1_ = 0;
  std::size_t d2_ = 0;
  std::size_t d3_ = 0;
  std::vector<T> data_;
};

}  // namespace rerdmft

#endif  // RERDMFT_TENSOR4_H
