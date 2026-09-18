#include "OrbitalRotationFiniteDifference.h"

#include <complex>
#include <stdexcept>

#include "IntegralRotation.h"
#include "SpinorRotation.h"

namespace rerdmft {

namespace {

double realPart(double x) { return x; }
double realPart(std::complex<double> x) { return x.real(); }

// std::conj(double) returns std::complex<double>, not double (see
// Hessian_opt/OrbitalGradient.cpp for the same, established pattern).
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

}  // namespace

template <typename T>
OrbitalRotationGradientCheck<T> orbitalRotationGradientCheck(const Matrix<T>& h,
                                                               const Tensor4<T>& eri,
                                                               const Matrix<T>& gradient,
                                                               const RdmftEnergyFn<T>& energy_fn,
                                                               std::size_t p, std::size_t q,
                                                               double step) {
  const std::size_t n = h.rows();
  if (h.cols() != n || gradient.rows() != n || gradient.cols() != n || p >= n || q >= n ||
      p == q) {
    throw std::runtime_error("orbitalRotationGradientCheck: inconsistent input dimensions");
  }

  auto buildKappa = [&](double t) {
    Matrix<T> kappa(n, n, T{});
    kappa(p, q) = T(t);
    kappa(q, p) = T(-t);
    return kappa;
  };

  const auto rot_plus = rotateIntegrals(h, eri, spinorRotationMatrix(buildKappa(step)));
  const auto rot_minus = rotateIntegrals(h, eri, spinorRotationMatrix(buildKappa(-step)));
  const double e_plus = energy_fn(rot_plus.h, rot_plus.eri);
  const double e_minus = energy_fn(rot_minus.h, rot_minus.eri);

  // OrbitalGradient.h's own gradient only stores the p >= q "lower
  // triangle" (g(p,q) for p < q is left at its default-constructed
  // zero, by that file's own documented convention) -- reconstruct via
  // g_pq = -conj(g_qp) when the caller asks for an upper-triangle entry,
  // rather than silently reading the placeholder zero.
  const T analytic_pq = (p >= q) ? gradient(p, q) : -conjugate(gradient(q, p));

  OrbitalRotationGradientCheck<T> result;
  result.finite_difference = (e_plus - e_minus) / (2.0 * step);
  result.analytic = realPart(analytic_pq);
  result.abs_diff = std::abs(result.analytic - result.finite_difference);
  return result;
}

template OrbitalRotationGradientCheck<double> orbitalRotationGradientCheck(
    const Matrix<double>&, const Tensor4<double>&, const Matrix<double>&,
    const RdmftEnergyFn<double>&, std::size_t, std::size_t, double);
template OrbitalRotationGradientCheck<std::complex<double>> orbitalRotationGradientCheck(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const Matrix<std::complex<double>>&, const RdmftEnergyFn<std::complex<double>>&, std::size_t,
    std::size_t, double);

}  // namespace rerdmft
