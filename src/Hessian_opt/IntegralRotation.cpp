#include "IntegralRotation.h"

#include <complex>
#include <stdexcept>

#include "Cholesky_Decomposition.h"

namespace rerdmft {

namespace {

// std::conj(double) returns std::complex<double>, not double -- these
// overloads keep the T = double instantiation real (see
// Hessian_opt/OrbitalGradient.cpp for the same, established pattern).
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

}  // namespace

template <typename T>
RotatedIntegrals<T> rotateIntegrals(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const Matrix<T>& u) {
  const std::size_t n = h.rows();
  if (h.cols() != n || u.rows() != n || u.cols() != n) {
    throw std::runtime_error("rotateIntegrals: inconsistent input dimensions");
  }

  RotatedIntegrals<T> result;

  // h_rot = U^dagger h U.
  Matrix<T> uh(n, n, T{});
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t q = 0; q < n; ++q) {
      T sum{};
      for (std::size_t b = 0; b < n; ++b) sum += h(a, b) * u(b, q);
      uh(a, q) = sum;
    }
  }
  result.h = Matrix<T>(n, n, T{});
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T sum{};
      for (std::size_t a = 0; a < n; ++a) sum += conjugate(u(a, p)) * uh(a, q);
      result.h(p, q) = sum;
    }
  }

  result.eri = choleskyTransformEri(eri, u);
  return result;
}

template RotatedIntegrals<double> rotateIntegrals(const Matrix<double>&, const Tensor4<double>&,
                                                   const Matrix<double>&);
template RotatedIntegrals<std::complex<double>> rotateIntegrals(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const Matrix<std::complex<double>>&);

}  // namespace rerdmft
