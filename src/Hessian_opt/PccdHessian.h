#ifndef RERDMFT_PCCDHESSIAN_H
#define RERDMFT_PCCDHESSIAN_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "PccdFock.h"
#include "Tensor4.h"
#include "pCCD.h"

namespace rerdmft {

// The pCCD counterpart of Hessian_opt/PnofHessian.h -- mirrors it function-for-function,
// substituting buildPccdFullTwoRdm/buildPccdPairOf/pccdFockMatrix for
// buildPnofFullTwoRdm/buildPnofPairOf/pnofFockMatrix. No new Hessian derivation at all:
// Hessian_opt/HartreeExchangeHessian.h's own formula is ALREADY generic over any
// (two_rdm_h, two_rdm_x, two_rdm_l1, two_rdm_l2) ansatz (PccdFock.h's own unfolding of
// pCCD's n_p/D_pq/Q_pq already produces exactly that ansatz, the SAME one PNOF's unfolding
// does) -- so every function here is a thin wrapper, exactly how PnofHessian.h reuses the
// SAME underlying machinery for PNOF with no PNOF-specific Hessian code of its own either.
// `rdm` is `buildPccdRdm(t, z)`'s own output (the converged amplitudes at the CURRENT
// orbitals, held fixed during orbital rotation -- see Full_opt/FullOptimization.cpp's
// makePccdModel, which caches it); `fock` MUST be `pccdFockMatrix(...)`'s own output.
template <typename T, typename Eri>
T pccdHessianElement(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                      const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                      std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                      const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                      std::size_t s);

template <typename T>
Matrix<T> pccdHessianMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                             const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar,
                             std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
                             const PccdRdm& rdm, const std::vector<double>& occupations,
                             const Matrix<T>& fock,
                             const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

// Complex spinors only (X2C/C4_DHF) -- see PnofHessian.h's own convention note on
// pnofHessianElementImag/Mixed for exactly what these measure off orbital stationarity.
template <typename T, typename Eri>
T pccdHessianElementImag(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                          const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                          std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                          const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s);
template <typename T, typename Eri>
T pccdHessianElementMixed(const Matrix<T>& h, const Eri& eri, const std::vector<std::size_t>& reps,
                           const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ,
                           std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
                           const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s);

template <typename Eri>
std::vector<double> pccdJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<std::size_t>& reps,
    const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
    const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

template <typename Eri>
std::vector<double> pccdJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<std::size_t>& reps,
    const std::vector<std::size_t>& bar, std::size_t n_core, std::size_t n_occ, std::size_t n_vir,
    const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

Matrix<double> pccdJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar, std::size_t n_core,
    std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm, const std::vector<double>& occupations,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft

#endif  // RERDMFT_PCCDHESSIAN_H
