#include "RkbMoTransform.h"

#include "SymmetricTransform.h"

#include <cblas.h>

#include <complex>
#include <cstddef>
#include <stdexcept>

#include "Cholesky_Decomposition.h"
#include "Matrix.h"

namespace rerdmft {

namespace {

// Unpacks the packed, physics-notation RkbTwoElectronTensor into a
// dense Tensor4<complex<double>> -- needed once, up front, so the leg-
// transforms below can operate on contiguous memory via BLAS ZGEMM
// instead of the packed accessor's triangular-index lookups. O(n^4)
// work, negligible next to the O(n^5) leg transforms.
Tensor4<std::complex<double>> densify(const RkbTwoElectronTensor& packed) {
  const std::size_t n = packed.dim();
  Tensor4<std::complex<double>> dense(n, n, n, n);
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          dense(p, q, r, s) = packed(p, q, r, s);
        }
      }
    }
  }
  return dense;
}

// T1(p,B,C,D) = sum_A conj(c(A,p)) * ao(A,B,C,D) -- transforms the
// first (bra) leg. Viewing `ao` as a dense (n_ao) x (n_ao^3) row-major
// matrix and `c` as (n_ao x n_mo), this is T1 = conj(c)^T @ ao -- ONE
// ZGEMM with CblasConjTrans covering the whole tensor (the contracted
// index A is already the tensor's outermost dimension).
Tensor4<std::complex<double>> transformLeg1(const Tensor4<std::complex<double>>& ao,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_ao = ao.dim0();
  const std::size_t n_mo = c.cols();
  const std::size_t inner = n_ao * n_ao * n_ao;
  Tensor4<std::complex<double>> result(n_mo, n_ao, n_ao, n_ao, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(n_mo),
              static_cast<int>(inner), static_cast<int>(n_ao), &alpha, c.data(),
              static_cast<int>(n_mo), ao.data(), static_cast<int>(inner), &beta, result.data(),
              static_cast<int>(inner));
  return result;
}

// T2(p,q,C,D) = sum_B conj(c(B,q)) * t1(p,B,C,D) -- second (bra) leg,
// also conjugated. The contracted index B is now the SECOND dimension,
// so this batches over the (contiguous) outer index p: T2[p] =
// conj(c)^T @ T1[p], each slice (n_ao) x (n_ao^2).
Tensor4<std::complex<double>> transformLeg2(const Tensor4<std::complex<double>>& t1,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t1.dim0();
  const std::size_t n_ao = t1.dim1();
  const std::size_t n_mo = c.cols();
  const std::size_t inner = n_ao * n_ao;
  const std::size_t slice_in = n_ao * inner;
  const std::size_t slice_out = n_mo * inner;
  Tensor4<std::complex<double>> result(n_mo1, n_mo, n_ao, n_ao, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  for (std::size_t p = 0; p < n_mo1; ++p) {
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(n_mo),
                static_cast<int>(inner), static_cast<int>(n_ao), &alpha, c.data(),
                static_cast<int>(n_mo), t1.data() + p * slice_in, static_cast<int>(inner), &beta,
                result.data() + p * slice_out, static_cast<int>(inner));
  }
  return result;
}

// T3(p,q,r,D) = sum_C c(C,r) * t2(p,q,C,D) -- first ket leg, NOT
// conjugated. The contracted index C is the THIRD dimension, so this
// batches over the (contiguous) combined (p,q) index: T3[p,q] = c^T @
// T2[p,q], each slice (n_ao) x (n_ao).
Tensor4<std::complex<double>> transformLeg3(const Tensor4<std::complex<double>>& t2,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t2.dim0();
  const std::size_t n_mo2 = t2.dim1();
  const std::size_t n_ao = t2.dim2();
  const std::size_t n_mo = c.cols();
  const std::size_t outer = n_mo1 * n_mo2;
  const std::size_t slice_in = n_ao * n_ao;
  const std::size_t slice_out = n_mo * n_ao;
  Tensor4<std::complex<double>> result(n_mo1, n_mo2, n_mo, n_ao, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  for (std::size_t pq = 0; pq < outer; ++pq) {
    cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_mo),
                static_cast<int>(n_ao), static_cast<int>(n_ao), &alpha, c.data(),
                static_cast<int>(n_mo), t2.data() + pq * slice_in, static_cast<int>(n_ao), &beta,
                result.data() + pq * slice_out, static_cast<int>(n_ao));
  }
  return result;
}

// T4(p,q,r,s) = sum_D c(D,s) * t3(p,q,r,D) -- second ket leg, NOT
// conjugated. T4(p,q,r,s) == <p q|r s>_MO directly (physics notation).
// Viewing t3 as a dense (n_mo1*n_mo2*n_mo3) x n_ao row-major matrix
// (its own natural layout) and c as (n_ao x n_mo), this is ONE ZGEMM,
// T4 = t3 @ c (the contracted index D is the tensor's innermost
// dimension, so no transpose or batching needed).
Tensor4<std::complex<double>> transformLeg4(const Tensor4<std::complex<double>>& t3,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t3.dim0();
  const std::size_t n_mo2 = t3.dim1();
  const std::size_t n_mo3 = t3.dim2();
  const std::size_t n_ao = t3.dim3();
  const std::size_t n_mo = c.cols();
  const std::size_t outer = n_mo1 * n_mo2 * n_mo3;
  Tensor4<std::complex<double>> result(n_mo1, n_mo2, n_mo3, n_mo, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(outer),
              static_cast<int>(n_mo), static_cast<int>(n_ao), &alpha, t3.data(),
              static_cast<int>(n_ao), c.data(), static_cast<int>(n_mo), &beta, result.data(),
              static_cast<int>(n_mo));
  return result;
}

}  // namespace

Matrix<std::complex<double>> rkbMoOneElectronTransform(const Matrix<std::complex<double>>& h_rkb,
                                                        const Matrix<std::complex<double>>& c_dhf) {
  if (h_rkb.rows() != h_rkb.cols()) {
    throw std::runtime_error("rkbMoOneElectronTransform: h_rkb is not square");
  }
  if (c_dhf.rows() != h_rkb.rows()) {
    throw std::runtime_error("rkbMoOneElectronTransform: c_dhf row count does not match h_rkb");
  }
  return dagger(c_dhf) * (h_rkb * c_dhf);
}

Tensor4<std::complex<double>> rkbMoTwoElectronTransformPhysics(
    const RkbTwoElectronTensor& eri_ao_physics, const Matrix<std::complex<double>>& c_dhf) {
  if (c_dhf.rows() != eri_ao_physics.dim()) {
    throw std::runtime_error(
        "rkbMoTwoElectronTransformPhysics: c_dhf row count does not match eri_ao_physics's "
        "dimension");
  }
  const Tensor4<std::complex<double>> ao_dense = densify(eri_ao_physics);
  const Tensor4<std::complex<double>> t1 = transformLeg1(ao_dense, c_dhf);
  const Tensor4<std::complex<double>> t2 = transformLeg2(t1, c_dhf);
  const Tensor4<std::complex<double>> t3 = transformLeg3(t2, c_dhf);
  return transformLeg4(t3, c_dhf);
}

Tensor4<std::complex<double>> rkbMoTwoElectronTransformPhysicsCholesky(
    const RkbTwoElectronTensor& eri_ao_physics, const Matrix<std::complex<double>>& c_dhf,
    double threshold) {
  if (c_dhf.rows() != eri_ao_physics.dim()) {
    throw std::runtime_error(
        "rkbMoTwoElectronTransformPhysicsCholesky: c_dhf row count does not match "
        "eri_ao_physics's dimension");
  }
  const Tensor4<std::complex<double>> ao_dense = densify(eri_ao_physics);
  return choleskyTransformEri(ao_dense, c_dhf, threshold);
}

Matrix<std::complex<double>> occupiedPositiveEnergyDensity(std::size_t rkb_dim, int n_electrons) {
  if (rkb_dim % 2 != 0) {
    throw std::runtime_error("occupiedPositiveEnergyDensity: rkb_dim must be even");
  }
  const std::size_t n_negative = rkb_dim / 2;
  if (n_electrons < 0 || n_negative + static_cast<std::size_t>(n_electrons) > rkb_dim) {
    throw std::runtime_error(
        "occupiedPositiveEnergyDensity: n_electrons exceeds the number of positive-energy "
        "states");
  }
  Matrix<std::complex<double>> d(rkb_dim, rkb_dim, std::complex<double>(0.0, 0.0));
  for (std::size_t p = n_negative; p < n_negative + static_cast<std::size_t>(n_electrons); ++p) {
    d(p, p) = std::complex<double>(1.0, 0.0);
  }
  return d;
}

SymmetricEri<std::complex<double>> rkbMoTwoElectronSymmetric(const RkbTwoElectronTensor& eri_ao_physics,
                                                              const Matrix<std::complex<double>>& c_dhf) {
  if (c_dhf.rows() != eri_ao_physics.dim()) {
    throw std::runtime_error("rkbMoTwoElectronSymmetric: c_dhf row count does not match the RKB dimension");
  }
  return transformToSymmetric<std::complex<double>>(eri_ao_physics, eri_ao_physics.dim(), c_dhf);
}

}  // namespace rerdmft
