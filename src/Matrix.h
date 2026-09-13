#ifndef RERDMFT_MATRIX_H
#define RERDMFT_MATRIX_H

#include <cstddef>
#include <vector>

namespace rerdmft {

// A minimal dense, row-major matrix.
template <typename T>
class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, T fill = T{})
      : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  T& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  const T& operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
  }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<T> data_;
};

}  // namespace rerdmft

#endif  // RERDMFT_MATRIX_H
