#include "PccdHessian.h"

#include <complex>

#include "CholeskyEri.h"
#include "HartreeExchangeHessian.h"
#include "SymmetricEri.h"

namespace rerdmft {

template <typename T, typename Eri>
T pccdHessianElement(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                      const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                      std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                      const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                      std::size_t s) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeHessianElement(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, fock, p,
                                        q, r, s, pair_of, full.two_rdm_l1, full.two_rdm_l2);
}

template <typename T>
Matrix<T> pccdHessianMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                             const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar,
                             std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
                             const PccdRdm& rdm, const std::vector<double>& occupations,
                             const Matrix<T>& fock,
                             const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeHessianMatrix(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, fock,
                                       pair_indices, pair_of, full.two_rdm_l1, full.two_rdm_l2);
}

template <typename T, typename Eri>
T pccdHessianElementImag(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                          const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                          std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                          const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeHessianElementImag(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, fock,
                                            p, q, r, s, pair_of, full.two_rdm_l1, full.two_rdm_l2);
}

template <typename T, typename Eri>
T pccdHessianElementMixed(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                           const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                           std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                           const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeHessianElementMixed(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                             fock, p, q, r, s, pair_of, full.two_rdm_l1,
                                             full.two_rdm_l2);
}

Matrix<double> pccdJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar, std::size_t n_core,
    std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeSymmetricJointHessianMatrix(h, eri, occupations, full.two_rdm_h,
                                                     full.two_rdm_x, fock, pair_indices, pair_of,
                                                     full.two_rdm_l1, full.two_rdm_l2);
}

template <typename Eri>
std::vector<double> pccdJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<std::size_t>& reps,
    const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
    const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeJointHessianVector(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                           fock, pair_indices, v, pair_of, full.two_rdm_l1,
                                           full.two_rdm_l2);
}

template <typename Eri>
std::vector<double> pccdJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<std::size_t>& reps,
    const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
    const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const auto full = buildPccdFullTwoRdm(reps, bar, n_core, n_occ, n_vir, rdm, occupations, h.rows());
  const auto pair_of = buildPccdPairOf(reps, bar, h.rows());
  return hartreeExchangeJointHessianDiagonal(h, eri, occupations, full.two_rdm_h, full.two_rdm_x,
                                             fock, pair_indices, pair_of, full.two_rdm_l1,
                                             full.two_rdm_l2);
}

template std::complex<double> pccdHessianElementImag(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElementImag(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElementImag(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElementMixed(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElementMixed(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElementMixed(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);

template double pccdHessianElement(const Matrix<double>&, const Tensor4<double>&,
                                    const std::vector<std::size_t>&, const std::vector<std::size_t>&,
                                    std::size_t, std::size_t, std::size_t, const PccdRdm&,
                                    const std::vector<double>&, const Matrix<double>&, std::size_t,
                                    std::size_t, std::size_t, std::size_t);
template double pccdHessianElement(const Matrix<double>&, const CholeskyEri<double>&,
                                    const std::vector<std::size_t>&, const std::vector<std::size_t>&,
                                    std::size_t, std::size_t, std::size_t, const PccdRdm&,
                                    const std::vector<double>&, const Matrix<double>&, std::size_t,
                                    std::size_t, std::size_t, std::size_t);
template double pccdHessianElement(const Matrix<double>&, const SymmetricEri<double>&,
                                    const std::vector<std::size_t>&, const std::vector<std::size_t>&,
                                    std::size_t, std::size_t, std::size_t, const PccdRdm&,
                                    const std::vector<double>&, const Matrix<double>&, std::size_t,
                                    std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElement(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElement(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);
template std::complex<double> pccdHessianElement(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    std::size_t, std::size_t, std::size_t, std::size_t);

template Matrix<double> pccdHessianMatrix(
    const Matrix<double>&, const Tensor4<double>&, const std::vector<std::size_t>&,
    const std::vector<std::size_t>&, std::size_t, std::size_t, std::size_t, const PccdRdm&,
    const std::vector<double>&, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);
template Matrix<std::complex<double>> pccdHessianMatrix(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);

template std::vector<double> pccdJointHessianVector(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);
template std::vector<double> pccdJointHessianVector(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);
template std::vector<double> pccdJointHessianVector(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&, const std::vector<double>&);

template std::vector<double> pccdJointHessianDiagonal(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);
template std::vector<double> pccdJointHessianDiagonal(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);
template std::vector<double> pccdJointHessianDiagonal(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t, std::size_t,
    std::size_t, const PccdRdm&, const std::vector<double>&, const Matrix<std::complex<double>>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);

}  // namespace rerdmft
