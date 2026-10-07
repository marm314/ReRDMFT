#include "RiSigmaHessianToolkit.h"

#include <cblas.h>

#include <complex>
#include <stdexcept>

namespace rerdmft {

namespace {
// C = A @ B for square n x n row-major blocks.
void gemmNN(std::size_t n, const double* a, const double* b, double* c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
              1.0, a, static_cast<int>(n), b, static_cast<int>(n), 0.0, c, static_cast<int>(n));
}
void gemmNN(std::size_t n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* c) {
  const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n), static_cast<int>(n),
              static_cast<int>(n), &one, a, static_cast<int>(n), b, static_cast<int>(n), &zero, c,
              static_cast<int>(n));
}
}  // namespace

template <typename T>
RiSigmaHessianToolkit<T> buildRiSigmaHessianToolkit(std::size_t n, const Matrix<T>& b,
                                                     const Matrix<double>& K) {
  if (b.cols() != n * n) throw std::runtime_error("buildRiSigmaHessianToolkit: b has the wrong shape");
  if (K.rows() != n || K.cols() != n) throw std::runtime_error("buildRiSigmaHessianToolkit: K has the wrong shape");

  RiSigmaHessianToolkit<T> t;
  t.n = n;
  t.n_aux = b.rows();
  t.bp = &b;
  t.kb1 = Matrix<T>(t.n_aux, n * n, T{});
  t.kb2 = Matrix<T>(t.n_aux, n * n, T{});
  t.bdiag.assign(t.n_aux * n, T{});
  t.kb1diag.assign(t.n_aux * n, T{});
  t.kb2diag.assign(t.n_aux * n, T{});

  Matrix<T> Kt(n, n);
  for (std::size_t i = 0; i < n * n; ++i) Kt.data()[i] = T(K.data()[i]);

  // KB1[P] = K @ B_P and KB2[P] = B_P @ K: one pair of n x n GEMMs per aux function.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t P = 0; P < t.n_aux; ++P) {
    const T* slice = b.data() + P * n * n;
    gemmNN(n, Kt.data(), slice, t.kb1.data() + P * n * n);
    gemmNN(n, slice, Kt.data(), t.kb2.data() + P * n * n);
    for (std::size_t x = 0; x < n; ++x) {
      t.bdiag[P * n + x] = b(P, x * n + x);
      t.kb1diag[P * n + x] = t.kb1(P, x * n + x);
      t.kb2diag[P * n + x] = t.kb2(P, x * n + x);
    }
  }
  return t;
}

template RiSigmaHessianToolkit<double> buildRiSigmaHessianToolkit(std::size_t, const Matrix<double>&,
                                                                   const Matrix<double>&);
template RiSigmaHessianToolkit<std::complex<double>> buildRiSigmaHessianToolkit(
    std::size_t, const Matrix<std::complex<double>>&, const Matrix<double>&);

}  // namespace rerdmft
