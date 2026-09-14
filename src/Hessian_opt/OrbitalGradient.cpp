#include "OrbitalGradient.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

// std::conj(double) returns std::complex<double>, not double (it
// promotes real arguments to complex per the standard's "additional
// overloads") -- these overloads keep orbitalGradient's T = double
// instantiation real, matching every other Matrix<double> in this
// project, rather than silently complexifying it.
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

}  // namespace

template <typename T>
Matrix<T> orbitalGradient(const Matrix<T>& fock) {
  const std::size_t n = fock.rows();
  if (fock.cols() != n) {
    throw std::runtime_error("orbitalGradient: fock is not square");
  }

  Matrix<T> g(n, n, T{});
  // Each p owns rows q=0..p (a disjoint set of output positions across
  // threads) and only reads the shared, const fock -- safe to
  // parallelize. Only p >= q is computed -- see OrbitalGradient.h. The
  // factor of 2 is deliberate (see OrbitalGradient.h) -- NOT Dyall's
  // own unscaled Eq. 8.32.
#pragma omp parallel for
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q <= p; ++q) {
      g(p, q) = T(2.0) * (fock(q, p) - conjugate(fock(p, q)));
    }
  }
  return g;
}

template Matrix<double> orbitalGradient(const Matrix<double>& fock);
template Matrix<std::complex<double>> orbitalGradient(const Matrix<std::complex<double>>& fock);

}  // namespace rerdmft
