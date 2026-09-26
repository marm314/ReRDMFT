#include "X2C_MoTransform.h"

#include "SymmetricTransform.h"

#include <cblas.h>

#include <complex>
#include <cstddef>
#include <stdexcept>

#include "Cholesky_Decomposition.h"

namespace rerdmft {

namespace {

// Converts the dense REAL AO tensor into a dense COMPLEX one -- needed
// once, up front, so the leg-transforms below can contract it against
// the complex `c_matrix` via BLAS ZGEMM. O(n^4) work, negligible next
// to the O(n^5) leg transforms.
Tensor4<std::complex<double>> toComplex(const Tensor4<double>& real) {
  const std::size_t n = real.dim0();
  Tensor4<std::complex<double>> result(n, n, n, n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          result(p, q, r, s) = std::complex<double>(real(p, q, r, s), 0.0);
        }
      }
    }
  }
  return result;
}

// Same four leg-transform helpers as C4_DHF/RkbMoTransform.cpp,
// reimplemented independently here (see that file's own comment on why
// this project does not share this code across contexts) -- see there
// for the full derivation of each step; only the docs are trimmed here
// to avoid duplicating that explanation verbatim.
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

Matrix<std::complex<double>> x2cMoOneElectronTransform(const Matrix<std::complex<double>>& h_x2c,
                                                        const Matrix<std::complex<double>>& c_matrix) {
  if (h_x2c.rows() != h_x2c.cols()) {
    throw std::runtime_error("x2cMoOneElectronTransform: h_x2c is not square");
  }
  if (c_matrix.rows() != h_x2c.rows()) {
    throw std::runtime_error("x2cMoOneElectronTransform: c_matrix row count does not match h_x2c");
  }
  return dagger(c_matrix) * (h_x2c * c_matrix);
}

Tensor4<std::complex<double>> x2cMoTwoElectronTransformPhysics(
    const Tensor4<double>& eri_ao_physics, const Matrix<std::complex<double>>& c_matrix) {
  if (c_matrix.rows() != eri_ao_physics.dim0()) {
    throw std::runtime_error(
        "x2cMoTwoElectronTransformPhysics: c_matrix row count does not match eri_ao_physics's "
        "dimension");
  }
  const Tensor4<std::complex<double>> ao_complex = toComplex(eri_ao_physics);
  const Tensor4<std::complex<double>> t1 = transformLeg1(ao_complex, c_matrix);
  const Tensor4<std::complex<double>> t2 = transformLeg2(t1, c_matrix);
  const Tensor4<std::complex<double>> t3 = transformLeg3(t2, c_matrix);
  return transformLeg4(t3, c_matrix);
}

Tensor4<std::complex<double>> x2cMoTwoElectronTransformPhysicsCholesky(
    const Tensor4<double>& eri_ao_physics, const Matrix<std::complex<double>>& c_matrix,
    double threshold) {
  if (c_matrix.rows() != eri_ao_physics.dim0()) {
    throw std::runtime_error(
        "x2cMoTwoElectronTransformPhysicsCholesky: c_matrix row count does not match "
        "eri_ao_physics's dimension");
  }
  return choleskyTransformEriMixed(eri_ao_physics, c_matrix, threshold);
}

namespace {
// <AB|CD> of the closed-shell spin-orbital tensor over the [alpha; beta] Large AO basis, from the packed
// chemist spatial integrals: (AC|BD) if spin(A) = spin(C) and spin(B) = spin(D), else 0.
struct SpinBlockSource {
  const PackedTwoElectronTensor& e;
  std::size_t n;
  double operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
    if (a / n != c / n || b / n != d / n) return 0.0;
    return e(a % n, c % n, b % n, d % n);
  }
};
}  // namespace

SymmetricEri<std::complex<double>> x2cMoTwoElectronSymmetric(const PackedTwoElectronTensor& eri,
                                                              const Matrix<std::complex<double>>& c) {
  if (c.rows() != 2 * eri.dim()) {
    throw std::runtime_error("x2cMoTwoElectronSymmetric: C must have 2*n_AO rows ([alpha; beta] blocks)");
  }
  return transformToSymmetric<std::complex<double>>(SpinBlockSource{eri, eri.dim()}, 2 * eri.dim(), c);
}

}  // namespace rerdmft
