#include "PnofHessian.h"

#include <complex>

#include "HartreeExchangeHessian.h"

namespace rerdmft {

template <typename T>
T pnofHessianElement(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                      const std::vector<PnofGeminal>& geminals,
                      const std::vector<double>& occupations, bool relativistic,
                      const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                      std::size_t s) {
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, h.rows(), relativistic);
  const auto pair_of = buildPnofPairOf(geminals, h.rows());
  return hartreeExchangeHessianElement(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, fock,
                                        p, q, r, s, pair_of, full.two_rdm_l1, full.two_rdm_l2);
}

template <typename T>
Matrix<T> pnofHessianMatrix(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                             const std::vector<PnofGeminal>& geminals,
                             const std::vector<double>& occupations, bool relativistic,
                             const Matrix<T>& fock,
                             const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, h.rows(), relativistic);
  const auto pair_of = buildPnofPairOf(geminals, h.rows());
  return hartreeExchangeHessianMatrix(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, fock,
                                       pair_indices, pair_of, full.two_rdm_l1, full.two_rdm_l2);
}

template <typename T>
T pnofHessianElementImag(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                          const std::vector<PnofGeminal>& geminals,
                          const std::vector<double>& occupations, bool relativistic,
                          const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s) {
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, h.rows(), relativistic);
  const auto pair_of = buildPnofPairOf(geminals, h.rows());
  return hartreeExchangeHessianElementImag(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                            fock, p, q, r, s, pair_of, full.two_rdm_l1,
                                            full.two_rdm_l2);
}

template <typename T>
T pnofHessianElementMixed(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                           const std::vector<PnofGeminal>& geminals,
                           const std::vector<double>& occupations, bool relativistic,
                           const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s) {
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, h.rows(), relativistic);
  const auto pair_of = buildPnofPairOf(geminals, h.rows());
  return hartreeExchangeHessianElementMixed(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                             fock, p, q, r, s, pair_of, full.two_rdm_l1,
                                             full.two_rdm_l2);
}

Matrix<double> pnofJointHessianMatrix(
    PnofFunctional functional, const Matrix<std::complex<double>>& h,
    const Tensor4<std::complex<double>>& eri, const std::vector<PnofGeminal>& geminals,
    const std::vector<double>& occupations, bool relativistic,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, h.rows(), relativistic);
  const auto pair_of = buildPnofPairOf(geminals, h.rows());
  return hartreeExchangeSymmetricJointHessianMatrix(h, eri, occupations, full.two_rdm_h,
                                                     full.two_rdm_x, fock, pair_indices, pair_of,
                                                     full.two_rdm_l1, full.two_rdm_l2);
}

template std::complex<double> pnofHessianElementImag(
    PnofFunctional, const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool,
    const Matrix<std::complex<double>>&, std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pnofHessianElementMixed(
    PnofFunctional, const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool,
    const Matrix<std::complex<double>>&, std::size_t, std::size_t, std::size_t, std::size_t);

template double pnofHessianElement(PnofFunctional, const Matrix<double>&, const Tensor4<double>&,
                                    const std::vector<PnofGeminal>&, const std::vector<double>&,
                                    bool, const Matrix<double>&, std::size_t, std::size_t,
                                    std::size_t, std::size_t);
template std::complex<double> pnofHessianElement(
    PnofFunctional, const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool,
    const Matrix<std::complex<double>>&, std::size_t, std::size_t, std::size_t, std::size_t);

template Matrix<double> pnofHessianMatrix(
    PnofFunctional, const Matrix<double>&, const Tensor4<double>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);
template Matrix<std::complex<double>> pnofHessianMatrix(
    PnofFunctional, const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool,
    const Matrix<std::complex<double>>&, const std::vector<std::pair<std::size_t, std::size_t>>&);

}  // namespace rerdmft
