#include "Cholesky_Decomposition.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rerdmft {

namespace {

double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

double realPart(double x) { return x; }
double realPart(std::complex<double> x) { return x.real(); }

// Elementwise complex conjugate of a matrix (NOT a transpose) -- the
// real overload is a no-op copy, needed so choleskyTransformVectors<T>
// compiles identically for both T.
Matrix<double> conjMatrix(const Matrix<double>& m) { return m; }
Matrix<std::complex<double>> conjMatrix(const Matrix<std::complex<double>>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) result(i, j) = std::conj(m(i, j));
  }
  return result;
}

// V' = C^T * (V * C) via two DGEMMs, replacing Matrix.h's generic
// (naive, unvectorized, per-vector-heap-allocating) operator* -- called
// once per Cholesky vector by choleskyTransformVectors below. For
// T = double, conj(C) = C, so `c_conj` is unused here (kept as a
// parameter purely so both overloads share one call site, selected by
// ordinary overload resolution on T).
Matrix<double> transformVectorBlas(const Matrix<double>& v, const Matrix<double>& c,
                                    const Matrix<double>& /*c_conj*/) {
  const std::size_t n_old = c.rows();
  const std::size_t n_new = c.cols();
  Matrix<double> temp(n_old, n_new, 0.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_old),
              static_cast<int>(n_new), static_cast<int>(n_old), 1.0, v.data(),
              static_cast<int>(n_old), c.data(), static_cast<int>(n_new), 0.0, temp.data(),
              static_cast<int>(n_new));
  Matrix<double> result(n_new, n_new, 0.0);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_new),
              static_cast<int>(n_new), static_cast<int>(n_old), 1.0, c.data(),
              static_cast<int>(n_new), temp.data(), static_cast<int>(n_new), 0.0, result.data(),
              static_cast<int>(n_new));
  return result;
}

// V' = C^dagger * (V * conj(C)) via two ZGEMMs -- CblasConjTrans applies
// dagger(C) directly from the ORIGINAL (unconjugated) `c`, so this needs
// `c_conj` only for the first product, exactly matching
// choleskyTransformVectors's documented formula.
Matrix<std::complex<double>> transformVectorBlas(const Matrix<std::complex<double>>& v,
                                                  const Matrix<std::complex<double>>& c,
                                                  const Matrix<std::complex<double>>& c_conj) {
  const std::size_t n_old = c.rows();
  const std::size_t n_new = c.cols();
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  Matrix<std::complex<double>> temp(n_old, n_new, std::complex<double>(0.0, 0.0));
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_old),
              static_cast<int>(n_new), static_cast<int>(n_old), &alpha, v.data(),
              static_cast<int>(n_old), c_conj.data(), static_cast<int>(n_new), &beta, temp.data(),
              static_cast<int>(n_new));
  Matrix<std::complex<double>> result(n_new, n_new, std::complex<double>(0.0, 0.0));
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(n_new),
              static_cast<int>(n_new), static_cast<int>(n_old), &alpha, c.data(),
              static_cast<int>(n_new), temp.data(), static_cast<int>(n_new), &beta, result.data(),
              static_cast<int>(n_new));
  return result;
}

// Rebuilds eri(a,b,c,d) = sum_L V_L(a,b)*conj(V_L(c,d)) as ONE GEMM
// instead of Nchol*n^4 naive nested loops: flatten every vector into a
// row of an (Nchol x n^2) matrix `Lmat`, then the whole reconstruction
// is the Gram-like product Lmat^T * conj(Lmat) -- a single (n^2 x n^2)
// = (n^2 x Nchol) @ (Nchol x n^2) DGEMM/ZGEMM. Tensor4's own row-major
// layout ((a,b,c,d) contiguous as ((a*n+b)*n+c)*n+d) is EXACTLY a flat
// (n^2 x n^2) row-major matrix over the combined (a,b)/(c,d) pair
// indices, so the GEMM can write directly into `eri.data()` with no
// separate unflattening step.
Tensor4<double> reconstructBlas(const std::vector<Matrix<double>>& vectors) {
  const std::size_t n = vectors.front().rows();
  const std::size_t n2 = n * n;
  const std::size_t nchol = vectors.size();
  std::vector<double> lmat(nchol * n2);
  for (std::size_t k = 0; k < nchol; ++k) {
    std::copy(vectors[k].data(), vectors[k].data() + n2, lmat.data() + k * n2);
  }
  Tensor4<double> eri(n, n, n, n, 0.0);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n2), static_cast<int>(n2),
              static_cast<int>(nchol), 1.0, lmat.data(), static_cast<int>(n2), lmat.data(),
              static_cast<int>(n2), 0.0, eri.data(), static_cast<int>(n2));
  return eri;
}

Tensor4<std::complex<double>> reconstructBlas(
    const std::vector<Matrix<std::complex<double>>>& vectors) {
  const std::size_t n = vectors.front().rows();
  const std::size_t n2 = n * n;
  const std::size_t nchol = vectors.size();
  std::vector<std::complex<double>> lmat(nchol * n2), lmat_conj(nchol * n2);
  for (std::size_t k = 0; k < nchol; ++k) {
    std::copy(vectors[k].data(), vectors[k].data() + n2, lmat.data() + k * n2);
  }
  for (std::size_t i = 0; i < nchol * n2; ++i) lmat_conj[i] = std::conj(lmat[i]);
  Tensor4<std::complex<double>> eri(n, n, n, n, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n2), static_cast<int>(n2),
              static_cast<int>(nchol), &alpha, lmat.data(), static_cast<int>(n2),
              lmat_conj.data(), static_cast<int>(n2), &beta, eri.data(), static_cast<int>(n2));
  return eri;
}

// Subtracts conj(Lmat)^T @ pivot_col from `row` (length n2) via ONE GEMV,
// replacing choleskyDecomposeEri's own O(Nchol_so_far) scalar loop over
// previously-found vectors per residual point. `lmat` stores the
// Nchol_so_far already-found vectors as its ROWS (row-major, each row one
// vector flattened over the (C,D) pair -- exactly Tensor4's own pair-index
// flattening), and `pivot_col` holds each of those vectors' own value at
// the current pivot (A*,B*) -- i.e. lmat's PIVOT-th COLUMN, hence "Lmat-
// vs-pivot-column GEMV". CblasTrans/CblasConjTrans applied to `lmat`
// computes exactly y(C,D) = sum_k Lmat(k,C,D) * pivot_col(k) (conjugated
// on Lmat for T = complex<double>, a no-op for T = double), i.e. the
// residual's sum_k V_k(A*,B*)*conj(V_k(C,D)) term for every (C,D) at once
// -- turning a badly-cache-behaved scalar loop over separate heap-
// allocated Matrix objects into one BLAS2 call over one contiguous buffer.
void subtractConjTransGemv(const double* lmat, std::size_t nchol, std::size_t n2,
                            const double* pivot_col, double* row) {
  if (nchol == 0) return;
  std::vector<double> y(n2, 0.0);
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(nchol), static_cast<int>(n2), 1.0, lmat,
              static_cast<int>(n2), pivot_col, 1, 0.0, y.data(), 1);
  for (std::size_t i = 0; i < n2; ++i) row[i] -= y[i];
}

void subtractConjTransGemv(const std::complex<double>* lmat, std::size_t nchol, std::size_t n2,
                            const std::complex<double>* pivot_col, std::complex<double>* row) {
  if (nchol == 0) return;
  std::vector<std::complex<double>> y(n2, std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemv(CblasRowMajor, CblasConjTrans, static_cast<int>(nchol), static_cast<int>(n2), &alpha,
              lmat, static_cast<int>(n2), pivot_col, 1, &beta, y.data(), 1);
  for (std::size_t i = 0; i < n2; ++i) row[i] -= y[i];
}

}  // namespace

template <typename T>
std::vector<Matrix<T>> choleskyDecomposeEri(const Tensor4<T>& eri, double threshold,
                                             std::size_t max_vectors) {
  const std::size_t n = eri.dim0();
  if (eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("choleskyDecomposeEri: eri is not square in all four dimensions");
  }
  const std::size_t n2 = n * n;
  // Small numerical-noise allowance for a residual diagonal that should
  // be exactly >= 0 (Hermitian PSD matrix) but can drift slightly
  // negative from floating-point cancellation once it is already tiny.
  constexpr double kNegativeDiagonalTolerance = 1e-10;

  // Residual diagonal D[(A,B)] = eri(A,B,A,B), flattened as index
  // A*n+B -- (A,B) is the BRA pair (see this file's own header
  // comment for why this grouping, not the "same electron" (A,C)
  // grouping used elsewhere, is the one that gives a genuine Hermitian
  // PSD matrix).
  std::vector<double> diag(n2);
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      diag[a * n + b] = realPart(eri(a, b, a, b));
    }
  }

  // Cholesky vectors found so far, stacked as ROWS of one growing (Nchol x
  // n2) contiguous buffer -- see subtractConjTransGemv above for why this
  // replaces an O(Nchol_so_far) scalar loop per residual point with a
  // single GEMV per iteration.
  std::vector<T> lmat;
  std::vector<T> row(n2);
  const std::size_t max_iterations = (max_vectors > 0) ? std::min(max_vectors, n2) : n2;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    std::size_t pivot = 0;
    double pivot_value = diag[0];
    for (std::size_t i = 1; i < n2; ++i) {
      if (diag[i] > pivot_value) {
        pivot_value = diag[i];
        pivot = i;
      }
    }
    if (pivot_value < threshold) break;
    if (pivot_value < -kNegativeDiagonalTolerance) {
      throw std::runtime_error(
          "choleskyDecomposeEri: encountered a significantly negative residual diagonal -- eri "
          "is not Hermitian positive semi-definite");
    }
    const std::size_t a_star = pivot / n;
    const std::size_t b_star = pivot % n;
    const double inv_sqrt_pivot = 1.0 / std::sqrt(pivot_value);

    for (std::size_t c = 0; c < n; ++c) {
      for (std::size_t d = 0; d < n; ++d) {
        row[c * n + d] = eri(a_star, b_star, c, d);
      }
    }

    const std::size_t nchol_so_far = lmat.size() / n2;
    if (nchol_so_far > 0) {
      // pivot_col[k] = the k-th already-found vector's own value at
      // (A*,B*) -- lmat's column `pivot` (pivot == a_star*n+b_star by
      // construction).
      std::vector<T> pivot_col(nchol_so_far);
      for (std::size_t k = 0; k < nchol_so_far; ++k) {
        pivot_col[k] = lmat[k * n2 + pivot];
      }
      subtractConjTransGemv(lmat.data(), nchol_so_far, n2, pivot_col.data(), row.data());
    }

    // The defining sum eri(A,B,C,D) = sum_k V_k(A,B)*conj(V_k(C,D)) gives,
    // at the pivot row (a*,b*): residual(C,D) = V_k(a*,b*) *
    // conj(V_k(C,D)) for the NEW vector k. Solving for V_k(C,D) itself
    // (not conj(V_k(C,D))) needs an EXTRA conjugate here on top of
    // dividing by the (real) sqrt(pivot) -- verified by direct hand
    // substitution (and the numerical test that caught its absence)
    // before trusting this, since it is easy to get backwards for a
    // genuinely complex decomposition (real T makes conjugate() a no-op,
    // silently hiding the bug there).
    for (std::size_t i = 0; i < n2; ++i) {
      row[i] = conjugate(row[i]) * T(inv_sqrt_pivot);
      diag[i] -= realPart(row[i] * conjugate(row[i]));
    }
    lmat.insert(lmat.end(), row.begin(), row.end());
  }

  const std::size_t nchol_final = lmat.size() / n2;
  std::vector<Matrix<T>> vectors;
  vectors.reserve(nchol_final);
  for (std::size_t k = 0; k < nchol_final; ++k) {
    Matrix<T> v(n, n);
    std::copy(lmat.data() + k * n2, lmat.data() + (k + 1) * n2, v.data());
    vectors.push_back(std::move(v));
  }
  return vectors;
}

template <typename T>
std::vector<Matrix<T>> choleskyTransformVectors(const std::vector<Matrix<T>>& vectors,
                                                 const Matrix<T>& c) {
  const Matrix<T> c_conj = conjMatrix(c);
  std::vector<Matrix<T>> result;
  result.reserve(vectors.size());
  for (const auto& v : vectors) {
    result.push_back(transformVectorBlas(v, c, c_conj));
  }
  return result;
}

template <typename T>
Tensor4<T> choleskyReconstructEri(const std::vector<Matrix<T>>& vectors) {
  if (vectors.empty()) {
    throw std::runtime_error("choleskyReconstructEri: no Cholesky vectors given");
  }
  return reconstructBlas(vectors);
}

template <typename T>
Tensor4<T> choleskyTransformEri(const Tensor4<T>& eri, const Matrix<T>& c, double threshold) {
  const auto vectors = choleskyDecomposeEri(eri, threshold);
  const auto transformed = choleskyTransformVectors(vectors, c);
  return choleskyReconstructEri(transformed);
}

namespace {
std::vector<Matrix<std::complex<double>>> promoteVectorsToComplex(
    const std::vector<Matrix<double>>& vectors) {
  std::vector<Matrix<std::complex<double>>> result;
  result.reserve(vectors.size());
  for (const auto& v : vectors) {
    Matrix<std::complex<double>> cv(v.rows(), v.cols());
    for (std::size_t i = 0; i < v.rows(); ++i) {
      for (std::size_t j = 0; j < v.cols(); ++j) cv(i, j) = std::complex<double>(v(i, j), 0.0);
    }
    result.push_back(std::move(cv));
  }
  return result;
}
}  // namespace

Tensor4<std::complex<double>> choleskyTransformEriMixed(const Tensor4<double>& eri,
                                                         const Matrix<std::complex<double>>& c,
                                                         double threshold) {
  const auto real_vectors = choleskyDecomposeEri(eri, threshold);
  const auto complex_vectors = promoteVectorsToComplex(real_vectors);
  const auto transformed = choleskyTransformVectors(complex_vectors, c);
  return choleskyReconstructEri(transformed);
}

template std::vector<Matrix<double>> choleskyDecomposeEri(const Tensor4<double>&, double,
                                                            std::size_t);
template std::vector<Matrix<std::complex<double>>> choleskyDecomposeEri(
    const Tensor4<std::complex<double>>&, double, std::size_t);
template std::vector<Matrix<double>> choleskyTransformVectors(const std::vector<Matrix<double>>&,
                                                                const Matrix<double>&);
template std::vector<Matrix<std::complex<double>>> choleskyTransformVectors(
    const std::vector<Matrix<std::complex<double>>>&, const Matrix<std::complex<double>>&);
template Tensor4<double> choleskyReconstructEri(const std::vector<Matrix<double>>&);
template Tensor4<std::complex<double>> choleskyReconstructEri(
    const std::vector<Matrix<std::complex<double>>>&);
template Tensor4<double> choleskyTransformEri(const Tensor4<double>&, const Matrix<double>&,
                                               double);
template Tensor4<std::complex<double>> choleskyTransformEri(const Tensor4<std::complex<double>>&,
                                                              const Matrix<std::complex<double>>&,
                                                              double);

}  // namespace rerdmft
