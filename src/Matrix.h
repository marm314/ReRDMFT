#ifndef RERDMFT_MATRIX_H
#define RERDMFT_MATRIX_H

#include <complex>
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

template <typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
  Matrix<T> result(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }
  return result;
}

// Requires a.cols() == b.rows().
template <typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
  Matrix<T> result(a.rows(), b.cols(), T{});
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t k = 0; k < a.cols(); ++k) {
      const T a_ik = a(i, k);
      for (std::size_t j = 0; j < b.cols(); ++j) {
        result(i, j) += a_ik * b(k, j);
      }
    }
  }
  return result;
}

// Hermitian adjoint: conjugate and transpose.
inline Matrix<std::complex<double>> dagger(const Matrix<std::complex<double>>& a) {
  Matrix<std::complex<double>> result(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      result(j, i) = std::conj(a(i, j));
    }
  }
  return result;
}

}  // namespace rerdmft

#endif  // RERDMFT_MATRIX_H
