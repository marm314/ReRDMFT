#include "SingleDeterminantRdm.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

template <typename T>
Tensor4<T> singleDeterminantTwoRdm(const Matrix<T>& d) {
  const std::size_t n = d.rows();
  if (d.cols() != n) {
    throw std::runtime_error("singleDeterminantTwoRdm: d is not square");
  }

  Tensor4<T> two_rdm(n, n, n, n, T{});
  // Each (p,q) owns its own disjoint output slice and only reads the
  // shared, const d -- safe to parallelize.
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          two_rdm(p, q, r, s) = T(0.5) * (d(p, r) * d(q, s) - d(p, s) * d(q, r));
        }
      }
    }
  }
  return two_rdm;
}

template Tensor4<double> singleDeterminantTwoRdm(const Matrix<double>& d);
template Tensor4<std::complex<double>> singleDeterminantTwoRdm(
    const Matrix<std::complex<double>>& d);

}  // namespace rerdmft
