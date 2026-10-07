#ifndef RERDMFT_MATRIX_H
#define RERDMFT_MATRIX_H

#include <cblas.h>

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

  // Raw access to the flat, row-major backing storage -- e.g. for BLAS
  // calls (cblas_dgemm/cblas_zgemm) that need a contiguous buffer
  // directly, mirroring Tensor4<T>'s own data()/const data().
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

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

// BLAS-backed overloads for the two element types actually used in production (double,
// std::complex<double>); the generic template above stays as the fallback for any other T. A
// non-template overload is preferred over the template at equal match quality, so every existing
// `a * b` call picks these up unchanged. The naive triple loop was serial and O(n^3) with no
// blocking -- the dominant non-integral cost of the SCF Fock assembly at Kr scale.
inline Matrix<double> operator*(const Matrix<double>& a, const Matrix<double>& b) {
  Matrix<double> result(a.rows(), b.cols(), 0.0);
  if (a.rows() == 0 || b.cols() == 0 || a.cols() == 0) return result;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows()), static_cast<int>(b.cols()),
              static_cast<int>(a.cols()), 1.0, a.data(), static_cast<int>(a.cols()), b.data(),
              static_cast<int>(b.cols()), 0.0, result.data(), static_cast<int>(b.cols()));
  return result;
}

inline Matrix<std::complex<double>> operator*(const Matrix<std::complex<double>>& a,
                                               const Matrix<std::complex<double>>& b) {
  Matrix<std::complex<double>> result(a.rows(), b.cols(), std::complex<double>{});
  if (a.rows() == 0 || b.cols() == 0 || a.cols() == 0) return result;
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows()), static_cast<int>(b.cols()),
              static_cast<int>(a.cols()), &alpha, a.data(), static_cast<int>(a.cols()), b.data(),
              static_cast<int>(b.cols()), &beta, result.data(), static_cast<int>(b.cols()));
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

// Real transpose -- dagger()'s real-matrix counterpart. Needed explicitly (not just relied on via
// a symmetric X) once a real X (e.g. NON_RELATIVISTIC's own Loewdin matrix) can come back
// RECTANGULAR from a rank-reducing canonical orthogonalization: X^T F X then needs the transpose
// written out, since X is no longer square and "symmetric" stops being a well-formed shortcut.
inline Matrix<double> transpose(const Matrix<double>& a) {
  Matrix<double> result(a.cols(), a.rows());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      result(j, i) = a(i, j);
    }
  }
  return result;
}

}  // namespace rerdmft

#endif  // RERDMFT_MATRIX_H
