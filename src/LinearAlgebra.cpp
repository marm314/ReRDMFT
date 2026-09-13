#include "LinearAlgebra.h"

#include <stdexcept>
#include <vector>

#include <lapacke.h>

namespace rerdmft {

Matrix<double> invert(const Matrix<double>& a) {
  if (a.rows() != a.cols()) {
    throw std::runtime_error("invert: matrix is not square");
  }
  const lapack_int n = static_cast<lapack_int>(a.rows());

  std::vector<double> data(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (lapack_int i = 0; i < n; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      data[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] =
          a(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    }
  }

  std::vector<lapack_int> ipiv(static_cast<std::size_t>(n));
  lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invert: LAPACKE_dgetrf failed (matrix is singular or invalid)");
  }
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, data.data(), n, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("invert: LAPACKE_dgetri failed (matrix is singular)");
  }

  Matrix<double> result(a.rows(), a.cols());
  for (lapack_int i = 0; i < n; ++i) {
    for (lapack_int j = 0; j < n; ++j) {
      result(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
          data[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
    }
  }
  return result;
}

}  // namespace rerdmft
