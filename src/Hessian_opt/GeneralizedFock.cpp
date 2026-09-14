#include "GeneralizedFock.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

template <typename T>
Matrix<T> generalizedFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                 const Tensor4<T>& gamma) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("generalizedFockMatrix: h is not square");
  }
  if (d.rows() != n || d.cols() != n) {
    throw std::runtime_error("generalizedFockMatrix: D dimensions are inconsistent with h");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("generalizedFockMatrix: eri dimensions are inconsistent with h");
  }
  if (gamma.dim0() != n || gamma.dim1() != n || gamma.dim2() != n || gamma.dim3() != n) {
    throw std::runtime_error("generalizedFockMatrix: Gamma dimensions are inconsistent with h");
  }

  Matrix<T> fock(n, n, T{});
  // Each (p,q) owns its own disjoint output position and only reads the
  // shared, const h/eri/d/gamma -- safe to parallelize. one_electron and
  // two_electron are declared inside the loop body, so each thread
  // accumulates into its own local variable -- no reduction needed.
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T one_electron{};
      for (std::size_t r = 0; r < n; ++r) {
        one_electron += h(p, r) * d(r, q);
      }

      T two_electron{};
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          for (std::size_t t = 0; t < n; ++t) {
            two_electron += eri(p, r, s, t) * gamma(q, r, s, t);
          }
        }
      }

      fock(p, q) = one_electron + two_electron;
    }
  }
  return fock;
}

template Matrix<double> generalizedFockMatrix(const Matrix<double>& h,
                                               const Tensor4<double>& eri, const Matrix<double>& d,
                                               const Tensor4<double>& gamma);
template Matrix<std::complex<double>> generalizedFockMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const Matrix<std::complex<double>>& d, const Tensor4<std::complex<double>>& gamma);

}  // namespace rerdmft
