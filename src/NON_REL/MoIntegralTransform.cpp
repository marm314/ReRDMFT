#include "MoIntegralTransform.h"

#include <cblas.h>

#include <cstddef>
#include <stdexcept>

#include "Cholesky_Decomposition.h"

namespace rerdmft {

namespace {

// Unpacks a PackedTwoElectronTensor (chemist notation, 8-fold-symmetry-
// unique storage) into a dense Tensor4<double> -- needed once, up
// front, so the leg-transforms below can operate on contiguous memory
// via BLAS DGEMM instead of the packed accessor's triangular-index
// lookups. O(n^4) work, negligible next to the O(n^5) leg transforms.
Tensor4<double> densify(const PackedTwoElectronTensor& packed) {
  const std::size_t n = packed.dim();
  Tensor4<double> dense(n, n, n, n);
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

// T1(p,B,C,D) = sum_A c(A,p) * ao(A,B,C,D). Viewing `ao` as a dense
// (n_ao) x (n_ao^3) row-major matrix and `c` as (n_ao x n_mo), this is
// exactly T1 = c^T @ ao -- ONE DGEMM covering the whole tensor (the
// contracted index A is already the tensor's outermost dimension, so
// no batching is needed).
Tensor4<double> transformLeg1(const Tensor4<double>& ao, const Matrix<double>& c) {
  const std::size_t n_ao = ao.dim0();
  const std::size_t n_mo = c.cols();
  const std::size_t inner = n_ao * n_ao * n_ao;
  Tensor4<double> result(n_mo, n_ao, n_ao, n_ao, 0.0);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_mo),
              static_cast<int>(inner), static_cast<int>(n_ao), 1.0, c.data(),
              static_cast<int>(n_mo), ao.data(), static_cast<int>(inner), 0.0, result.data(),
              static_cast<int>(inner));
  return result;
}

// T2(p,q,C,D) = sum_B c(B,q) * t1(p,B,C,D). The contracted index B is
// now the SECOND dimension of a 4-index tensor, so this is a batch of
// n_mo1 independent DGEMMs, one per (contiguous, since p is the
// slowest-varying index) p-slice: T2[p] = c^T @ T1[p], each slice
// viewed as (n_ao) x (n_ao^2).
Tensor4<double> transformLeg2(const Tensor4<double>& t1, const Matrix<double>& c) {
  const std::size_t n_mo1 = t1.dim0();
  const std::size_t n_ao = t1.dim1();
  const std::size_t n_mo = c.cols();
  const std::size_t inner = n_ao * n_ao;
  const std::size_t slice_in = n_ao * inner;
  const std::size_t slice_out = n_mo * inner;
  Tensor4<double> result(n_mo1, n_mo, n_ao, n_ao, 0.0);
  for (std::size_t p = 0; p < n_mo1; ++p) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_mo),
                static_cast<int>(inner), static_cast<int>(n_ao), 1.0, c.data(),
                static_cast<int>(n_mo), t1.data() + p * slice_in, static_cast<int>(inner), 0.0,
                result.data() + p * slice_out, static_cast<int>(inner));
  }
  return result;
}

// T3(p,q,r,D) = sum_C c(C,r) * t2(p,q,C,D). The contracted index C is
// the THIRD dimension, so this batches over the (contiguous) combined
// (p,q) index: T3[p,q] = c^T @ T2[p,q], each slice (n_ao) x (n_ao).
Tensor4<double> transformLeg3(const Tensor4<double>& t2, const Matrix<double>& c) {
  const std::size_t n_mo1 = t2.dim0();
  const std::size_t n_mo2 = t2.dim1();
  const std::size_t n_ao = t2.dim2();
  const std::size_t n_mo = c.cols();
  const std::size_t outer = n_mo1 * n_mo2;
  const std::size_t slice_in = n_ao * n_ao;
  const std::size_t slice_out = n_mo * n_ao;
  Tensor4<double> result(n_mo1, n_mo2, n_mo, n_ao, 0.0);
  for (std::size_t pq = 0; pq < outer; ++pq) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_mo),
                static_cast<int>(n_ao), static_cast<int>(n_ao), 1.0, c.data(),
                static_cast<int>(n_mo), t2.data() + pq * slice_in, static_cast<int>(n_ao), 0.0,
                result.data() + pq * slice_out, static_cast<int>(n_ao));
  }
  return result;
}

// T4(p,q,r,s) = sum_D c(D,s) * t3(p,q,r,D) -- the fully-transformed,
// still CHEMIST-notation MO tensor: T4(p,q,r,s) == (p q|r s)_MO.
// Viewing t3 as a dense (n_mo1*n_mo2*n_mo3) x n_ao row-major matrix
// (its own natural layout) and c as (n_ao x n_mo), this is exactly
// T4 = t3 @ c -- again ONE DGEMM (the contracted index D is the
// tensor's innermost dimension, so no transpose or batching needed).
Tensor4<double> transformLeg4(const Tensor4<double>& t3, const Matrix<double>& c) {
  const std::size_t n_mo1 = t3.dim0();
  const std::size_t n_mo2 = t3.dim1();
  const std::size_t n_mo3 = t3.dim2();
  const std::size_t n_ao = t3.dim3();
  const std::size_t n_mo = c.cols();
  const std::size_t outer = n_mo1 * n_mo2 * n_mo3;
  Tensor4<double> result(n_mo1, n_mo2, n_mo3, n_mo, 0.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(outer),
              static_cast<int>(n_mo), static_cast<int>(n_ao), 1.0, t3.data(),
              static_cast<int>(n_ao), c.data(), static_cast<int>(n_mo), 0.0, result.data(),
              static_cast<int>(n_mo));
  return result;
}

}  // namespace

Matrix<double> moOneElectronTransform(const Matrix<double>& h_ao, const Matrix<double>& c) {
  const std::size_t n_ao = h_ao.rows();
  if (h_ao.cols() != n_ao) {
    throw std::runtime_error("moOneElectronTransform: h_ao is not square");
  }
  if (c.rows() != n_ao) {
    throw std::runtime_error("moOneElectronTransform: c row count does not match h_ao");
  }
  const std::size_t n_mo = c.cols();

  // temp = h_ao @ c  (n_ao x n_ao) @ (n_ao x n_mo) -> (n_ao x n_mo)
  Matrix<double> temp(n_ao, n_mo, 0.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_ao),
              static_cast<int>(n_mo), static_cast<int>(n_ao), 1.0, h_ao.data(),
              static_cast<int>(n_ao), c.data(), static_cast<int>(n_mo), 0.0, temp.data(),
              static_cast<int>(n_mo));
  // h_mo = c^T @ temp  (n_mo x n_ao) @ (n_ao x n_mo) -> (n_mo x n_mo)
  Matrix<double> h_mo(n_mo, n_mo, 0.0);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_mo),
              static_cast<int>(n_mo), static_cast<int>(n_ao), 1.0, c.data(),
              static_cast<int>(n_mo), temp.data(), static_cast<int>(n_mo), 0.0, h_mo.data(),
              static_cast<int>(n_mo));
  return h_mo;
}

Tensor4<double> moTwoElectronTransformPhysics(const PackedTwoElectronTensor& eri_ao_chemist,
                                               const Matrix<double>& c) {
  if (c.rows() != eri_ao_chemist.dim()) {
    throw std::runtime_error(
        "moTwoElectronTransformPhysics: c row count does not match eri_ao_chemist's dimension");
  }
  const std::size_t n_mo = c.cols();

  const Tensor4<double> ao_dense = densify(eri_ao_chemist);
  const Tensor4<double> t1 = transformLeg1(ao_dense, c);
  const Tensor4<double> t2 = transformLeg2(t1, c);
  const Tensor4<double> t3 = transformLeg3(t2, c);
  const Tensor4<double> chemist_mo = transformLeg4(t3, c);  // chemist_mo(p,q,r,s) == (pq|rs)_MO

  // physics<p q|r s>_MO = chemist_mo(p,r,q,s) (physics<A B|C D> =
  // chemist(A,C,B,D)).
  Tensor4<double> physics_mo(n_mo, n_mo, n_mo, n_mo, 0.0);
  for (std::size_t p = 0; p < n_mo; ++p) {
    for (std::size_t q = 0; q < n_mo; ++q) {
      for (std::size_t r = 0; r < n_mo; ++r) {
        for (std::size_t s = 0; s < n_mo; ++s) {
          physics_mo(p, q, r, s) = chemist_mo(p, r, q, s);
        }
      }
    }
  }
  return physics_mo;
}

Tensor4<double> moTwoElectronTransformPhysicsCholesky(const PackedTwoElectronTensor& eri_ao_chemist,
                                                       const Matrix<double>& c, double threshold) {
  if (c.rows() != eri_ao_chemist.dim()) {
    throw std::runtime_error(
        "moTwoElectronTransformPhysicsCholesky: c row count does not match eri_ao_chemist's "
        "dimension");
  }
  const std::size_t n_mo = c.cols();
  const Tensor4<double> ao_dense = densify(eri_ao_chemist);
  const Tensor4<double> chemist_mo = choleskyTransformEri(ao_dense, c, threshold);

  Tensor4<double> physics_mo(n_mo, n_mo, n_mo, n_mo, 0.0);
  for (std::size_t p = 0; p < n_mo; ++p) {
    for (std::size_t q = 0; q < n_mo; ++q) {
      for (std::size_t r = 0; r < n_mo; ++r) {
        for (std::size_t s = 0; s < n_mo; ++s) {
          physics_mo(p, q, r, s) = chemist_mo(p, r, q, s);
        }
      }
    }
  }
  return physics_mo;
}

}  // namespace rerdmft
