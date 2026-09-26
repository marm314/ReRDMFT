#include "Cholesky_Decomposition.h"

#include "ElectronRepulsion.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
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
// vs-pivot-column" correction. Computes exactly
// row(C,D) -= sum_k conj(Lmat(k,C,D)) * pivot_col(k) for every (C,D) at
// once, i.e. the residual's sum_k V_k(A*,B*)*conj(V_k(C,D)) term.
//
// Deliberately a PLAIN scalar loop, not a BLAS2 call (as an earlier
// version of this function was, back when it was called once per pivot
// against the FULL, growing set of every vector found so far): once the
// batched algorithm below made `nchol` here always small and bounded (at
// most kMaxQualified, the within-one-batch correction only -- every
// PRIOR batch's contribution is handled separately, via ONE larger GEMM
// per batch, see subtractConjTransGemm), calling into OpenBLAS
// thousands of times (once per pivot) for such a tiny operand became the
// actual bottleneck: OpenBLAS's own per-call thread-pool dispatch
// overhead, roughly constant per call, ends up dominating the tiny
// amount of real work in each call, an effect that only shows up at
// realistic system sizes (confirmed by a live progress trace on a
// pathologically slow real example before this was caught -- a small
// synthetic test never has enough total pivots for the effect to be
// visible in wall-clock time). A plain loop has no such per-call
// overhead and still vectorizes fine under -O2 for a fixed, bounded
// `nchol`.
void subtractConjTransGemv(const double* lmat, std::size_t nchol, std::size_t n2,
                            const double* pivot_col, double* row) {
  for (std::size_t k = 0; k < nchol; ++k) {
    const double pk = pivot_col[k];
    const double* lrow = lmat + k * n2;
    for (std::size_t p = 0; p < n2; ++p) row[p] -= lrow[p] * pk;
  }
}

void subtractConjTransGemv(const std::complex<double>* lmat, std::size_t nchol, std::size_t n2,
                            const std::complex<double>* pivot_col, std::complex<double>* row) {
  for (std::size_t k = 0; k < nchol; ++k) {
    const std::complex<double> pk = pivot_col[k];
    const std::complex<double>* lrow = lmat + k * n2;
    for (std::size_t p = 0; p < n2; ++p) row[p] -= std::conj(lrow[p]) * pk;
  }
}

// Batched generalization of subtractConjTransGemv above, to a WHOLE
// qualified batch of `batch_size` candidate pivots at once (eT's own
// "efficient algorithm", Folkestad/Kjonstad/Koch, J. Chem. Phys. 150,
// 194112 (2019), Eq. 10 -- see choleskyDecomposeEri's own header comment
// for the full correspondence): `lmat_q` holds, for each of the
// `batch_size` candidates, that PRIOR-ROUND vector's own value at that
// candidate (an (nchol x batch_size) matrix, `lmat_q(k,j)` =
// V_k(candidate_j) -- i.e. `batch_size` "pivot_col" vectors stacked as
// columns), and `mtilde` ((batch_size x n2), ROW-MAJOR, one candidate's
// full (C,D) row PER ROW -- deliberately NOT (n2 x batch_size), so both
// filling it from the dense ERI tensor and reading a single candidate's
// row back out are contiguous, cache-friendly copies rather than
// strided ones) is updated in place, for every candidate row at once,
// via ONE GEMM instead of `batch_size` separate GEMVs. This is what
// turns the aggregate cost of subtracting every PRIOR vector's
// contribution from O(Nchol^2 * n^2) (one GEMV of growing size per
// pivot) into O(Nchol^2 * n^2 / batch_size) (one GEMM per batch of
// `batch_size` pivots) -- the actual algorithmic improvement the eT
// paper reports, not merely a constant-factor BLAS efficiency gain.
//
// CBLAS has no single "conjugate without transpose" operation, so
// conjugating `lmat` itself (up to Nchol*n2 elements, the LARGE operand)
// is avoided: instead this computes temp = ConjTrans(lmat_q) @ lmat
// (conjugating the SMALL, batch_size*nchol operand via ConjTrans) and
// then subtracts conj(temp) (an O(batch_size*n2) elementwise pass,
// applied once per batch, not once per pivot) -- since
// conj(ConjTrans(lmat_q) @ lmat) = lmat_q^T @ conj(lmat), the quantity
// this function actually needs (a no-op conjugate on the whole
// expression for T = double, exactly like every other conjugate() call
// in this file).
void subtractConjTransGemm(const double* lmat, std::size_t nchol, std::size_t n2,
                            const double* lmat_q, std::size_t batch_size, double* mtilde) {
  if (nchol == 0) return;
  std::vector<double> temp(batch_size * n2);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(batch_size),
              static_cast<int>(n2), static_cast<int>(nchol), 1.0, lmat_q,
              static_cast<int>(batch_size), lmat, static_cast<int>(n2), 0.0, temp.data(),
              static_cast<int>(n2));
  for (std::size_t i = 0; i < batch_size * n2; ++i) mtilde[i] -= temp[i];
}

void subtractConjTransGemm(const std::complex<double>* lmat, std::size_t nchol, std::size_t n2,
                            const std::complex<double>* lmat_q, std::size_t batch_size,
                            std::complex<double>* mtilde) {
  if (nchol == 0) return;
  std::vector<std::complex<double>> temp(batch_size * n2);
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(batch_size),
              static_cast<int>(n2), static_cast<int>(nchol), &alpha, lmat_q,
              static_cast<int>(batch_size), lmat, static_cast<int>(n2), &beta, temp.data(),
              static_cast<int>(n2));
  for (std::size_t i = 0; i < batch_size * n2; ++i) mtilde[i] -= std::conj(temp[i]);
}

// Uniform access to the tensor being decomposed: a dense Tensor4 (bra pair (A,B), ket pair (C,D)), or
// the real 8-fold-packed AO tensor (chemist notation (AB|CD), which IS that grouping) so the AO
// integrals can be decomposed without ever being expanded to a dense n^4 array.
template <typename T>
std::size_t srcDim(const Tensor4<T>& e) { return e.dim0(); }
std::size_t srcDim(const PackedTwoElectronTensor& e) { return e.dim(); }

// out[c*n+d] = eri(a,b,c,d) for the bra pair q = a*n+b (one row of the pair matrix).
template <typename T>
void copyRow(const Tensor4<T>& e, std::size_t q, std::size_t n, T* out) {
  const T* row = e.data() + q * n * n;
  std::copy(row, row + n * n, out);
}
void copyRow(const PackedTwoElectronTensor& e, std::size_t q, std::size_t n, double* out) {
  const std::size_t a = q / n, b = q % n;
  for (std::size_t c = 0; c < n; ++c)
    for (std::size_t d = 0; d < n; ++d) out[c * n + d] = e(a, b, c, d);
}

// eri element addressed by the flat index (A*n+B)*n^2 + (C*n+D).
template <typename T>
T elementAt(const Tensor4<T>& e, std::size_t flat, std::size_t) { return e.data()[flat]; }
double elementAt(const PackedTwoElectronTensor& e, std::size_t flat, std::size_t n) {
  const std::size_t n2 = n * n;
  const std::size_t ab = flat / n2, cd = flat % n2;
  return e(ab / n, ab % n, cd / n, cd % n);
}

}  // namespace

namespace {
// The pivoted Cholesky algorithm over an abstract Hermitian PSD PAIR MATRIX (size n2 x n2): a dense
// or packed integral tensor grouped as (bra pair, ket pair), or an implicit matrix such as the
// combined {LL} u {SS} Coulomb matrix of the 4-component basis (C4_DHF/RkbCholesky.h).
template <typename T>
FlatVectors<T> decomposeFlat(const PairMatrixSource<T>& src, double threshold, std::size_t max_vectors,
                             std::size_t max_batch) {
  const std::size_t n2 = src.size();
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
  for (std::size_t i = 0; i < n2; ++i) diag[i] = src.diagonal(i);

  // eT's own "efficient algorithm" (Folkestad, Kjonstad, Koch, J. Chem.
  // Phys. 150, 194112 (2019), Eqs. (8)-(14)): rather than picking ONE
  // pivot per outer iteration and subtracting every PRIOR vector's
  // contribution from its row via a single GEMV (this file's own
  // earlier approach, whose aggregate cost is O(Nchol^2*n^2) since the
  // k-th pivot's GEMV already costs O(k*n^2)), pick a whole BATCH of
  // "qualified" candidate pivots at once -- every index within a factor
  // `kSpanFactor` of the current largest diagonal, up to `kMaxQualified`
  // of them -- and subtract every PRIOR BATCH's vectors' contribution
  // from ALL of them together via ONE GEMM (subtractConjTransGemm
  // above). This is a genuine algorithmic improvement, not just a BLAS
  // efficiency trick: it cuts the aggregate cost of that subtraction
  // step to O(Nchol^2*n^2 / kMaxQualified). Only the (bounded, cheap)
  // correction for vectors already built EARLIER IN THE SAME BATCH still
  // needs a per-pivot GEMV (against just that batch's own, at most
  // kMaxQualified, vectors -- reusing subtractConjTransGemv unchanged,
  // pointed at the tail of `lmat`).
  //
  // Deliberately NOT adopted from eT: their algorithm also permanently
  // drops (treats as exactly zero) any vector's value at AO pairs whose
  // OWN diagonal has already fallen below the threshold, to save memory
  // at the ~80000-AO scale their paper targets. This project always
  // keeps every Cholesky vector's FULL n-by-n value (every (C,D) pair,
  // not just currently-"active" ones), so choleskyReconstructEri still
  // reproduces the original tensor to within `threshold` EVERYWHERE, not
  // just at the pivot points -- unnecessary for the system sizes this
  // project targets, and avoiding it keeps this function's own accuracy
  // guarantee unchanged from before this batching was added.
  //
  // `kSpanFactor` matches the eT paper's own example value; a smaller
  // batch/looser span factor makes this closer to strict one-pivot-at-
  // a-time greedy selection (marginally fewer total vectors, less
  // speedup), a larger one trades a small increase in Nchol for more
  // speedup -- both remain exact to `threshold` regardless, since pivot
  // ORDER only affects Nchol (efficiency), never the accuracy of the
  // resulting factorization.
  const std::size_t kMaxQualified = std::max<std::size_t>(1, max_batch);
  constexpr double kSpanFactor = 1e-2;

  std::vector<T> lmat;  // (nchol_so_far x n2), rows = vectors found so far
  const std::size_t max_iterations = (max_vectors > 0) ? std::min(max_vectors, n2) : n2;
  std::size_t nchol_so_far = 0;

  std::vector<std::size_t> candidates;
  std::vector<T> mtilde;
  std::vector<T> lmat_q;
  std::vector<double> qdiag;
  std::vector<T> row(n2);
  std::vector<T> pivot_col;

  while (nchol_so_far < max_iterations) {
    std::size_t dmax_idx = 0;
    for (std::size_t i = 1; i < n2; ++i) {
      if (diag[i] > diag[dmax_idx]) dmax_idx = i;
    }
    const double dmax = diag[dmax_idx];
    if (dmax < -kNegativeDiagonalTolerance) {
      throw std::runtime_error(
          "choleskyDecomposeEri: encountered a significantly negative residual diagonal -- eri "
          "is not Hermitian positive semi-definite");
    }
    if (dmax < threshold) break;

    // Qualified batch: every currently-significant index within
    // kSpanFactor of dmax, largest first, capped at kMaxQualified and at
    // the remaining vector budget.
    candidates.clear();
    for (std::size_t i = 0; i < n2; ++i) {
      if (diag[i] >= kSpanFactor * dmax) candidates.push_back(i);
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](std::size_t a, std::size_t b) { return diag[a] > diag[b]; });
    const std::size_t remaining_budget = max_iterations - nchol_so_far;
    const std::size_t batch_size = std::min({candidates.size(), kMaxQualified, remaining_budget});
    candidates.resize(batch_size);

    // Mtilde(j,p) = eri(candidates[j], p) for every p in 0..n2 (the FULL
    // pair-index range, per this function's own header comment above),
    // stored ROW-MAJOR as (batch_size x n2) -- one candidate's full
    // (C,D) row per row of `mtilde`, so this fill is a plain contiguous
    // copy straight out of the dense tensor's own row-major storage
    // (eri.data() + q*n2 is exactly eri(A*,B*,:,:) flattened, since
    // (C,D) are the tensor's fastest-varying dimensions and q already
    // IS the flat (A*,B*) pair index) -- no per-element index
    // arithmetic, unlike an (n2 x batch_size) layout, which would need
    // either a division per element or a strided write. candidates[j]
    // (the PIVOT/batch index) goes in the tensor's FIRST pair of slots
    // and p (the eventual (C,D) output index) in the SECOND, exactly
    // matching the original single-pivot algorithm's own
    // `eri(a_star, b_star, c, d)` read order -- getting this backwards
    // is invisible for T = double (eri(P,Q) == eri(Q,P) trivially there,
    // since conjugate() is a no-op) but wrong for T = complex<double>
    // (eri(P,Q) = conj(eri(Q,P)) in general) -- caught by a dedicated
    // complex numerical test, exactly the kind of bug this file's own
    // header comment already warns about for the conjugate step below.
    mtilde.assign(batch_size * n2, T(0));
    for (std::size_t j = 0; j < batch_size; ++j) {
      src.row(candidates[j], mtilde.data() + j * n2);
    }
    if (nchol_so_far > 0) {
      lmat_q.assign(nchol_so_far * batch_size, T(0));
      for (std::size_t k = 0; k < nchol_so_far; ++k) {
        for (std::size_t j = 0; j < batch_size; ++j) {
          lmat_q[k * batch_size + j] = lmat[k * n2 + candidates[j]];
        }
      }
      subtractConjTransGemm(lmat.data(), nchol_so_far, n2, lmat_q.data(), batch_size,
                             mtilde.data());
    }

    // Sequential in-batch construction (Eqs. (11)-(13)): each pivot only
    // needs correcting for vectors already built EARLIER IN THIS SAME
    // BATCH (a cheap GEMV against at most kMaxQualified rows), since
    // every PRIOR batch's contribution is already subtracted into
    // `mtilde` above.
    qdiag.resize(batch_size);
    for (std::size_t j = 0; j < batch_size; ++j) qdiag[j] = diag[candidates[j]];
    std::vector<bool> built(batch_size, false);
    const std::size_t batch_start = nchol_so_far;
    std::size_t built_count = 0;
    // Largest CURRENT residual diagonal over all (A,B) (not just this batch's candidates); refreshed
    // after every vector below. A pivot much smaller than it must NOT be taken (see the stop test
    // in the loop): dividing by sqrt of a small pivot amplifies the roundoff in the tensor's rows
    // by 1/sqrt(pivot), which the batch's frozen candidate list would otherwise allow.
    double current_max = dmax;

    while (built_count < batch_size && nchol_so_far < max_iterations) {
      std::size_t jstar = batch_size;
      for (std::size_t j = 0; j < batch_size; ++j) {
        if (!built[j] && (jstar == batch_size || qdiag[j] > qdiag[jstar])) jstar = j;
      }
      if (jstar == batch_size) break;
      if (qdiag[jstar] < -kNegativeDiagonalTolerance) {
        throw std::runtime_error(
            "choleskyDecomposeEri: encountered a significantly negative residual diagonal -- eri "
            "is not Hermitian positive semi-definite");
      }
      if (qdiag[jstar] < threshold) break;
      // Batch qualification is decided at the batch START from the then-largest diagonal; the
      // diagonal keeps shrinking as the batch's vectors are built, so re-test against the CURRENT
      // largest one and end the batch (the outer loop re-qualifies from scratch) once the best
      // remaining candidate is no longer within kSpanFactor of it. Without this the batch drifts
      // to greedy-violating small pivots and loses accuracy at tight thresholds (CO/cc-pVDZ X2C:
      // threshold 1e-10 gave a reconstruction error of 8e-8, and 1e-14 an error of ~6).
      if (built_count > 0 && qdiag[jstar] < kSpanFactor * current_max) break;

      std::copy(mtilde.data() + jstar * n2, mtilde.data() + jstar * n2 + n2, row.data());
      if (built_count > 0) {
        pivot_col.resize(built_count);
        for (std::size_t c = 0; c < built_count; ++c) {
          pivot_col[c] = lmat[(batch_start + c) * n2 + candidates[jstar]];
        }
        subtractConjTransGemv(lmat.data() + batch_start * n2, built_count, n2, pivot_col.data(),
                               row.data());
      }

      // Same conjugate-then-normalize convention as the original
      // single-pivot algorithm (see its own comment, preserved from
      // before this batching was added): solving V_k(C,D) itself out of
      // the defining sum's conj(V_k(C,D)) factor needs this extra
      // conjugate on top of dividing by sqrt(pivot).
      const double inv_sqrt_pivot = 1.0 / std::sqrt(qdiag[jstar]);
      current_max = -std::numeric_limits<double>::infinity();
      for (std::size_t p = 0; p < n2; ++p) {
        row[p] = conjugate(row[p]) * T(inv_sqrt_pivot);
        diag[p] -= realPart(row[p] * conjugate(row[p]));
        current_max = std::max(current_max, diag[p]);
      }
      lmat.insert(lmat.end(), row.begin(), row.end());
      built[jstar] = true;
      ++built_count;
      ++nchol_so_far;
      for (std::size_t j = 0; j < batch_size; ++j) {
        if (!built[j]) qdiag[j] = diag[candidates[j]];
      }
    }
    if (built_count == 0) break;  // safety net; should not trigger since dmax's own index always qualifies
  }

  FlatVectors<T> flat;
  flat.pair_dim = n2;
  flat.count = lmat.size() / n2;
  flat.data = std::move(lmat);
  return flat;
}

// Adapter: a square tensor (Tensor4 or the packed AO tensor) grouped as bra pair (A,B), ket pair (C,D).
template <typename T, typename Src>
class TensorPairs : public PairMatrixSource<T> {
 public:
  explicit TensorPairs(const Src& e) : e_(e), n_(srcDim(e)) {}
  std::size_t size() const override { return n_ * n_; }
  double diagonal(std::size_t i) const override { return realPart(e_(i / n_, i % n_, i / n_, i % n_)); }
  void row(std::size_t i, T* out) const override { copyRow(e_, i, n_, out); }
  T at(std::size_t i, std::size_t j) const override { return elementAt(e_, i * n_ * n_ + j, n_); }

 private:
  const Src& e_;
  std::size_t n_;
};

template <typename T>
std::vector<Matrix<T>> flatToMatrices(const FlatVectors<T>& flat, std::size_t n) {
  std::vector<Matrix<T>> vectors;
  vectors.reserve(flat.count);
  for (std::size_t k = 0; k < flat.count; ++k) {
    Matrix<T> v(n, n);
    std::copy(flat.data.data() + k * flat.pair_dim, flat.data.data() + (k + 1) * flat.pair_dim, v.data());
    vectors.push_back(std::move(v));
  }
  return vectors;
}

template <typename T, typename Src>
std::vector<Matrix<T>> decomposeImpl(const Src& eri, double threshold, std::size_t max_vectors,
                                     std::size_t max_batch) {
  const TensorPairs<T, Src> pairs(eri);
  return flatToMatrices(decomposeFlat<T>(pairs, threshold, max_vectors, max_batch), srcDim(eri));
}
}  // namespace

template <typename T>
std::vector<Matrix<T>> choleskyDecomposeEri(const Tensor4<T>& eri, double threshold,
                                             std::size_t max_vectors, std::size_t max_batch) {
  if (eri.dim1() != eri.dim0() || eri.dim2() != eri.dim0() || eri.dim3() != eri.dim0()) {
    throw std::runtime_error("choleskyDecomposeEri: eri is not square in all four dimensions");
  }
  return decomposeImpl<T>(eri, threshold, max_vectors, max_batch);
}

namespace {
// max |sum_L V_L(i) conj(V_L(j)) - M(i,j)| over a strided sample of at most `max_samples` elements of the
// pair matrix (all of them for a small one). The stride is made coprime to the pair dimension so it cannot
// alias with the index structure. The (pair_dim x nchol) vector table makes each sample one contiguous dot
// product.
template <typename T>
double sampledPairError(const PairMatrixSource<T>& src, const FlatVectors<T>& v, std::size_t max_samples) {
  const std::size_t n2 = v.pair_dim;
  const std::size_t nchol = v.count;
  std::vector<T> w(n2 * nchol);
  for (std::size_t l = 0; l < nchol; ++l) {
    const T* row = v.data.data() + l * n2;
    for (std::size_t ab = 0; ab < n2; ++ab) w[ab * nchol + l] = row[ab];
  }
  const std::size_t total = n2 * n2;
  std::size_t stride = total > max_samples ? total / max_samples + 1 : 1;
  while (stride > 1 && std::gcd(stride, n2) != 1) ++stride;
  const std::size_t n_samples = (total + stride - 1) / stride;
  double worst = 0.0;
#pragma omp parallel for reduction(max : worst) schedule(static)
  for (std::size_t k = 0; k < n_samples; ++k) {
    const std::size_t flat = k * stride;
    const std::size_t i = flat / n2, j = flat % n2;
    const T* left = &w[i * nchol];
    const T* right = &w[j * nchol];
    T sum{};
    for (std::size_t l = 0; l < nchol; ++l) sum += left[l] * conjugate(right[l]);
    worst = std::max(worst, std::abs(std::complex<double>(sum - src.at(i, j))));
  }
  return worst;
}

template <typename T>
FlatVectors<T> checkedFlat(const PairMatrixSource<T>& src, double threshold, CholeskyCheckReport* report) {
  const double tolerance = 100.0 * threshold + 1e-9;
  constexpr std::size_t kSamples = 4000000;
  FlatVectors<T> vectors;
  double worst = 0.0;
  std::size_t used = 0;
  bool retried = false;
  for (const std::size_t batch : {std::size_t{64}, std::size_t{8}, std::size_t{1}}) {
    vectors = decomposeFlat<T>(src, threshold, 0, batch);
    worst = sampledPairError<T>(src, vectors, kSamples);
    used = batch;
    if (worst <= tolerance) break;
    retried = true;
    std::cerr << "warning: Cholesky decomposition (batch " << batch << ", threshold " << threshold
              << ") reconstructs the integrals only to " << worst << " (tolerance " << tolerance << ")"
              << (batch > 1 ? "; retrying with a smaller batch\n" : "\n");
  }
  if (report != nullptr) {
    report->batch_used = used;
    report->n_vectors = vectors.count;
    report->max_error = worst;
    report->tolerance = tolerance;
    report->retried = retried;
  }
  if (worst > tolerance) {
    throw std::runtime_error(
        "choleskyDecomposeEriChecked: the Cholesky vectors do not reproduce the two-electron integrals "
        "(max error " + std::to_string(worst) + " > tolerance " + std::to_string(tolerance) +
        ") even with a one-pivot-at-a-time decomposition");
  }
  return vectors;
}

template <typename T, typename Src>
std::vector<Matrix<T>> checkedImpl(const Src& eri, double threshold, CholeskyCheckReport* report) {
  const TensorPairs<T, Src> pairs(eri);
  return flatToMatrices(checkedFlat<T>(pairs, threshold, report), srcDim(eri));
}
}  // namespace

template <typename T>
FlatVectors<T> choleskyDecomposePairsChecked(const PairMatrixSource<T>& src, double threshold,
                                              CholeskyCheckReport* report) {
  return checkedFlat<T>(src, threshold, report);
}

template <typename T>
std::vector<Matrix<T>> choleskyDecomposeEriChecked(const Tensor4<T>& eri, double threshold,
                                                    CholeskyCheckReport* report) {
  return checkedImpl<T>(eri, threshold, report);
}

std::vector<Matrix<double>> choleskyDecomposeEriChecked(const PackedTwoElectronTensor& eri,
                                                         double threshold, CholeskyCheckReport* report) {
  return checkedImpl<double>(eri, threshold, report);
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
  const auto vectors = choleskyDecomposeEriChecked(eri, threshold);
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
  const auto real_vectors = choleskyDecomposeEriChecked(eri, threshold);
  const auto complex_vectors = promoteVectorsToComplex(real_vectors);
  const auto transformed = choleskyTransformVectors(complex_vectors, c);
  return choleskyReconstructEri(transformed);
}

template FlatVectors<double> choleskyDecomposePairsChecked(const PairMatrixSource<double>&, double,
                                                          CholeskyCheckReport*);
template FlatVectors<std::complex<double>> choleskyDecomposePairsChecked(
    const PairMatrixSource<std::complex<double>>&, double, CholeskyCheckReport*);
template std::vector<Matrix<double>> choleskyDecomposeEriChecked(const Tensor4<double>&, double,
                                                                   CholeskyCheckReport*);
template std::vector<Matrix<std::complex<double>>> choleskyDecomposeEriChecked(
    const Tensor4<std::complex<double>>&, double, CholeskyCheckReport*);
template std::vector<Matrix<double>> choleskyDecomposeEri(const Tensor4<double>&, double,
                                                            std::size_t, std::size_t);
template std::vector<Matrix<std::complex<double>>> choleskyDecomposeEri(
    const Tensor4<std::complex<double>>&, double, std::size_t, std::size_t);
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
