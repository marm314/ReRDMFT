#include "CholeskyEri.h"
#include "SymmetricEri.h"
#include "JkOnlyFock.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

// Forked verbatim from HartreeExchangeGradient.cpp's own H/X-only path
// (pair_of empty) -- see that file's header comment for the full
// derivation from Dyall & Faegri Eq. (8.30). Kept as an independent
// copy (not a shared helper) so future work building/testing a
// generalized Fock for JK_only cannot touch PNOF's already-validated
// formula.
template <typename T>
Matrix<T> jkOnlyFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                            const std::vector<double>& occupations,
                            const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("jkOnlyFockMatrix: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("jkOnlyFockMatrix: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error("jkOnlyFockMatrix: occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error("jkOnlyFockMatrix: two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("jkOnlyFockMatrix: two_rdm_x dimensions inconsistent with h");
  }

  // Each (p,q) owns its own disjoint output position and only reads the
  // shared, const inputs -- safe to parallelize.
  Matrix<T> f(n, n, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T sum{};
      for (std::size_t s = 0; s < n; ++s) {
        sum += T(two_rdm_h(s, q)) * eri(q, s, p, s) - T(two_rdm_x(q, s)) * eri(q, s, s, p);
      }
      f(p, q) = T(occupations[q]) * h(q, p) + sum;
    }
  }
  return f;
}

namespace {
double conjugateValue(double x) { return x; }
std::complex<double> conjugateValue(std::complex<double> x) { return std::conj(x); }
}  // namespace

template <typename T, typename Eri>
Matrix<T> jkOnlyOrbitalGradient(const Matrix<T>& h, const Eri& eri,
                                 const std::vector<double>& occupations,
                                 const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("jkOnlyOrbitalGradient: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("jkOnlyOrbitalGradient: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n || two_rdm_h.rows() != n || two_rdm_h.cols() != n ||
      two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("jkOnlyOrbitalGradient: occupations/couplings inconsistent with h");
  }

  Matrix<T> g(n, n, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      if (q > p) continue;
      T c = T(occupations[p] - occupations[q]) * h(q, p);
      for (std::size_t t = 0; t < n; ++t) {
        c += T(two_rdm_h(p, t) - two_rdm_h(q, t)) * eri(t, q, t, p) -
             T(two_rdm_x(p, t) - two_rdm_x(q, t)) * eri(q, t, t, p);
      }
      g(p, q) = T(2.0) * conjugateValue(c);
    }
  }
  return g;
}

template Matrix<double> jkOnlyOrbitalGradient(const Matrix<double>&, const Tensor4<double>&,
                                               const std::vector<double>&, const Matrix<double>&,
                                               const Matrix<double>&);
template Matrix<double> jkOnlyOrbitalGradient(const Matrix<double>&, const CholeskyEri<double>&,
                                               const std::vector<double>&, const Matrix<double>&,
                                               const Matrix<double>&);
template Matrix<double> jkOnlyOrbitalGradient(const Matrix<double>&, const SymmetricEri<double>&,
                                               const std::vector<double>&, const Matrix<double>&,
                                               const Matrix<double>&);
template Matrix<std::complex<double>> jkOnlyOrbitalGradient(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&);
template Matrix<std::complex<double>> jkOnlyOrbitalGradient(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&);
template Matrix<std::complex<double>> jkOnlyOrbitalGradient(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&);

template <typename T, typename Eri>
double jkOnlyEnergy(const Matrix<T>& h, const Eri& eri,
                     const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                     const Matrix<double>& two_rdm_x) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("jkOnlyEnergy: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("jkOnlyEnergy: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error("jkOnlyEnergy: occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error("jkOnlyEnergy: two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("jkOnlyEnergy: two_rdm_x dimensions inconsistent with h");
  }

  double energy = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    energy += occupations[p] * std::real(h(p, p));
  }

  double two_electron = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : two_electron)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      two_electron += std::real(eri(p, q, p, q)) * two_rdm_h(p, q) -
                      std::real(eri(p, q, q, p)) * two_rdm_x(p, q);
    }
  }
  energy += 0.5 * two_electron;
  return energy;
}

template <typename T>
Tensor4<T> jkOnlyDenseTwoRdm(std::size_t n, const Matrix<double>& two_rdm_h,
                              const Matrix<double>& two_rdm_x) {
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error("jkOnlyDenseTwoRdm: two_rdm_h dimensions inconsistent with n");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("jkOnlyDenseTwoRdm: two_rdm_x dimensions inconsistent with n");
  }
  Tensor4<T> gamma(n, n, n, n, T{});
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      gamma(p, q, p, q) += T(0.5 * two_rdm_h(p, q));
      gamma(p, q, q, p) -= T(0.5 * two_rdm_x(p, q));
    }
  }
  return gamma;
}

template Tensor4<double> jkOnlyDenseTwoRdm(std::size_t n, const Matrix<double>& two_rdm_h,
                                            const Matrix<double>& two_rdm_x);
template Tensor4<std::complex<double>> jkOnlyDenseTwoRdm(std::size_t n,
                                                           const Matrix<double>& two_rdm_h,
                                                           const Matrix<double>& two_rdm_x);

template double jkOnlyEnergy(const Matrix<double>& h, const CholeskyEri<double>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);
template double jkOnlyEnergy(const Matrix<double>& h, const SymmetricEri<double>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);
template double jkOnlyEnergy(const Matrix<std::complex<double>>& h,
                              const CholeskyEri<std::complex<double>>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);
template double jkOnlyEnergy(const Matrix<std::complex<double>>& h,
                              const SymmetricEri<std::complex<double>>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);
template double jkOnlyEnergy(const Matrix<double>& h, const Tensor4<double>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);
template double jkOnlyEnergy(const Matrix<std::complex<double>>& h,
                              const Tensor4<std::complex<double>>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);

template Matrix<double> jkOnlyFockMatrix(const Matrix<double>& h, const Tensor4<double>& eri,
                                          const std::vector<double>& occupations,
                                          const Matrix<double>& two_rdm_h,
                                          const Matrix<double>& two_rdm_x);
template Matrix<std::complex<double>> jkOnlyFockMatrix(const Matrix<std::complex<double>>& h,
                                                         const Tensor4<std::complex<double>>& eri,
                                                         const std::vector<double>& occupations,
                                                         const Matrix<double>& two_rdm_h,
                                                         const Matrix<double>& two_rdm_x);

}  // namespace rerdmft
