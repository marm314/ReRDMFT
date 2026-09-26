#include "CholeskyEri.h"
#include "SymmetricEri.h"
#include "OccupationEnergy.h"

#include <complex>
#include <stdexcept>
#include <string>

namespace rerdmft {

namespace {

template <typename T, typename Eri>
void checkDimensions(const Matrix<T>& h, const Eri& eri,
                      const std::vector<double>& occupations, const char* caller) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error(std::string(caller) + ": occupations size inconsistent with h");
  }
}

}  // namespace

template <typename T, typename Eri>
double jkFunctionalEnergy(const Matrix<T>& h, const Eri& eri,
                           const std::vector<double>& occupations, JkFunctional functional,
                           std::size_t f_l, double power_alpha) {
  checkDimensions(h, eri, occupations, "jkFunctionalEnergy");
  const std::size_t n = h.rows();

  double energy = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    energy += occupations[p] * std::real(h(p, p));
  }

  double two_electron = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : two_electron)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      const double j_pq = std::real(eri(p, q, p, q));
      const double k_pq = std::real(eri(p, q, q, p));
      const double f_h =
          jkHartreeFunction(functional, occupations[p], occupations[q], p, q, f_l, power_alpha);
      const double f_x = jkExchangeFunction(functional, occupations[p], occupations[q], p, q,
                                             f_l, power_alpha);
      two_electron += j_pq * f_h - k_pq * f_x;
    }
  }
  energy += 0.5 * two_electron;
  return energy;
}

template <typename T, typename Eri>
std::vector<double> jkFunctionalGradient(const Matrix<T>& h, const Eri& eri,
                                          const std::vector<double>& occupations,
                                          JkFunctional functional, std::size_t f_l,
                                          double power_alpha) {
  checkDimensions(h, eri, occupations, "jkFunctionalGradient");
  const std::size_t n = h.rows();

  std::vector<double> gradient(n, 0.0);
#pragma omp parallel for
  for (std::size_t r = 0; r < n; ++r) {
    double sum = std::real(h(r, r));
    for (std::size_t q = 0; q < n; ++q) {
      const double j_rq = std::real(eri(r, q, r, q));
      const double k_rq = std::real(eri(r, q, q, r));
      const double d1_h = jkHartreeFunctionD1(functional, occupations[r], occupations[q], r, q,
                                               f_l, power_alpha);
      const double d1_x =
          jkExchangeFunctionD1(functional, occupations[r], occupations[q], r, q, f_l, power_alpha);
      sum += j_rq * d1_h - k_rq * d1_x;
    }
    gradient[r] = sum;
  }
  return gradient;
}

template <typename T, typename Eri>
Matrix<double> jkFunctionalHessian(const Matrix<T>& h, const Eri& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l,
                                    double power_alpha) {
  checkDimensions(h, eri, occupations, "jkFunctionalHessian");
  const std::size_t n = h.rows();

  Matrix<double> hessian(n, n, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t s = 0; s < n; ++s) {
      const double j_rs = std::real(eri(r, s, r, s));
      const double k_rs = std::real(eri(r, s, s, r));
      const double d12_h = jkHartreeFunctionD12(functional, occupations[r], occupations[s], r, s,
                                                 f_l, power_alpha);
      const double d12_x = jkExchangeFunctionD12(functional, occupations[r], occupations[s], r, s,
                                                  f_l, power_alpha);
      double value = j_rs * d12_h - k_rs * d12_x;
      if (r == s) {
        double diag_sum = 0.0;
        for (std::size_t q = 0; q < n; ++q) {
          const double j_rq = std::real(eri(r, q, r, q));
          const double k_rq = std::real(eri(r, q, q, r));
          const double d11_h = jkHartreeFunctionD11(functional, occupations[r], occupations[q], r,
                                                      q, f_l, power_alpha);
          const double d11_x = jkExchangeFunctionD11(functional, occupations[r], occupations[q], r,
                                                       q, f_l, power_alpha);
          diag_sum += j_rq * d11_h - k_rq * d11_x;
        }
        value += diag_sum;
      }
      hessian(r, s) = value;
    }
  }
  return hessian;
}

template double jkFunctionalEnergy(const Matrix<double>& h, const Tensor4<double>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);
template double jkFunctionalEnergy(const Matrix<double>& h, const CholeskyEri<double>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);
template double jkFunctionalEnergy(const Matrix<double>& h, const SymmetricEri<double>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);
template double jkFunctionalEnergy(const Matrix<std::complex<double>>& h,
                                    const Tensor4<std::complex<double>>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);
template double jkFunctionalEnergy(const Matrix<std::complex<double>>& h,
                                    const CholeskyEri<std::complex<double>>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);
template double jkFunctionalEnergy(const Matrix<std::complex<double>>& h,
                                    const SymmetricEri<std::complex<double>>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l, double power_alpha);

template std::vector<double> jkFunctionalGradient(const Matrix<double>& h,
                                                   const Tensor4<double>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);
template std::vector<double> jkFunctionalGradient(const Matrix<double>& h,
                                                   const CholeskyEri<double>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);
template std::vector<double> jkFunctionalGradient(const Matrix<double>& h,
                                                   const SymmetricEri<double>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);
template std::vector<double> jkFunctionalGradient(const Matrix<std::complex<double>>& h,
                                                   const Tensor4<std::complex<double>>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);
template std::vector<double> jkFunctionalGradient(const Matrix<std::complex<double>>& h,
                                                   const CholeskyEri<std::complex<double>>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);
template std::vector<double> jkFunctionalGradient(const Matrix<std::complex<double>>& h,
                                                   const SymmetricEri<std::complex<double>>& eri,
                                                   const std::vector<double>& occupations,
                                                   JkFunctional functional, std::size_t f_l,
                                                   double power_alpha);

template Matrix<double> jkFunctionalHessian(const Matrix<double>& h, const Tensor4<double>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);
template Matrix<double> jkFunctionalHessian(const Matrix<double>& h, const CholeskyEri<double>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);
template Matrix<double> jkFunctionalHessian(const Matrix<double>& h, const SymmetricEri<double>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);
template Matrix<double> jkFunctionalHessian(const Matrix<std::complex<double>>& h,
                                             const Tensor4<std::complex<double>>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);
template Matrix<double> jkFunctionalHessian(const Matrix<std::complex<double>>& h,
                                             const CholeskyEri<std::complex<double>>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);
template Matrix<double> jkFunctionalHessian(const Matrix<std::complex<double>>& h,
                                             const SymmetricEri<std::complex<double>>& eri,
                                             const std::vector<double>& occupations,
                                             JkFunctional functional, std::size_t f_l,
                                             double power_alpha);

}  // namespace rerdmft
