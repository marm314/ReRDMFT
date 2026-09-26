#ifndef RERDMFT_UTILS_SYMMETRICTRANSFORM_H
#define RERDMFT_UTILS_SYMMETRICTRANSFORM_H

#include <cblas.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "BlasThreads.h"
#include "Matrix.h"
#include "SymmetricEri.h"

namespace rerdmft {

// Physics-notation four-index transform of ANY element-access source into a SymmetricEri, processed in SLABS
// over the first MO index p so that no dense n^4 array is ever formed (memory O(n^3) per thread):
//   <pq|rs>_MO = sum_{abcd} conj(C_ap) conj(C_bq) C_cr C_ds  <ab|cd>_src
// `src(a,b,c,d)` returns the AO/basis integral (real or complex), `c` is the n_src x n_mo coefficient matrix
// (real or complex). Serves both the AO -> MO transform and the rotation of integrals by a unitary U (src = the
// integrals themselves, c = U). Only the elements a SymmetricEri stores are computed and written (first index p
// <= third index r; the store rebuilds the rest by symmetry), so the cost is about half a full transform.
//
// Cost O(n_mo n_src^4) source accesses for the first leg plus O(n^5) BLAS for the other three. `src` must have the
// exchange/Hermitian symmetry SymmetricEri assumes (a physical two-electron tensor does).
namespace symtransform_detail {

inline void gemm(bool trans_a, bool conj_a, int m, int n, int k, const double* a, int lda, const double* b, int ldb,
                 double* c, int ldc) {
  (void)conj_a;
  cblas_dgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}
inline void gemm(bool trans_a, bool conj_a, int m, int n, int k, const std::complex<double>* a, int lda,
                 const std::complex<double>* b, int ldb, std::complex<double>* c, int ldc) {
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, trans_a ? (conj_a ? CblasConjTrans : CblasTrans) : CblasNoTrans, CblasNoTrans, m, n, k,
              &alpha, a, lda, b, ldb, &beta, c, ldc);
}
// C(m x n) += A(m x k) B(k x n), row-major, no transposes.
inline void gemmAcc(int m, int n, int k, const double* a, int lda, const double* b, int ldb, double* c, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 1.0, c, ldc);
}
inline void gemmAcc(int m, int n, int k, const std::complex<double>* a, int lda, const std::complex<double>* b, int ldb,
                    std::complex<double>* c, int ldc) {
  const std::complex<double> alpha(1.0, 0.0), beta(1.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
inline double conjugate(double x) { return x; }
inline std::complex<double> conjugate(const std::complex<double>& x) { return std::conj(x); }

}  // namespace symtransform_detail

template <typename T, typename Src>
SymmetricEri<T> transformToSymmetric(const Src& src, std::size_t n_src, const Matrix<T>& c) {
  using symtransform_detail::conjugate;
  using symtransform_detail::gemm;
  const std::size_t nmo = c.cols();
  SymmetricEri<T> out(nmo);
  const int ns = static_cast<int>(n_src), nm = static_cast<int>(nmo);
  // C^dagger and C^T as explicit matrices for the BLAS calls: cd(q,b) = conj(C(b,q)), ct(r,c) = C(c,r).
  Matrix<T> cd(nmo, n_src), ct(nmo, n_src);
  for (std::size_t a = 0; a < n_src; ++a)
    for (std::size_t q = 0; q < nmo; ++q) {
      cd(q, a) = conjugate(c(a, q));
      ct(q, a) = c(a, q);
    }
  // The first leg is done for a BLOCK of `bp` output orbitals p at once: chunks of `ac` source slabs S_a(b,c,d) are
  // extracted from `src` once per block (n^4 accesses per block instead of per orbital) and contracted with a GEMM,
  // t1(p, bcd) += sum_a conj(C(a,p)) S_a(bcd). Block sizes are set by a memory budget (both buffers together stay
  // around 400 MB at most).
  const std::size_t n3 = n_src * n_src * n_src;
  const std::size_t budget = 200u * 1024u * 1024u / (sizeof(T) * std::max<std::size_t>(1, n3));
  const std::size_t bp = std::max<std::size_t>(1, std::min<std::size_t>(16, budget));
  const std::size_t ac = bp;
  std::vector<T> t1(bp * n3), slab(ac * n3);
  for (std::size_t p0 = 0; p0 < nmo; p0 += bp) {
    const std::size_t np = std::min(bp, nmo - p0);
    std::fill(t1.begin(), t1.begin() + np * n3, T{});
    for (std::size_t a0 = 0; a0 < n_src; a0 += ac) {
      const std::size_t na = std::min(ac, n_src - a0);
#pragma omp parallel for collapse(2) schedule(static)
      for (std::size_t a = 0; a < na; ++a)
        for (std::size_t b = 0; b < n_src; ++b) {
          T* row = &slab[(a * n_src + b) * n_src * n_src];
          for (std::size_t cc = 0; cc < n_src; ++cc)
            for (std::size_t d = 0; d < n_src; ++d) row[cc * n_src + d] = T(src(a0 + a, b, cc, d));
        }
      // t1(p, bcd) += sum_a cd(p0+p, a0+a) slab(a, bcd)
      // Column-chunked so every thread runs a serial GEMM on its own chunk (OpenBLAS' own threading of a product this
      // thin -- m = block of orbitals, k = source chunk -- costs far more than it gains; see BlasThreads.h).
      {
        const SerialBlasScope serial_blas;
        constexpr std::size_t kChunks = 48;
        const std::size_t chunk = (n3 + kChunks - 1) / kChunks;
#pragma omp parallel for schedule(static)
        for (std::size_t ch = 0; ch < kChunks; ++ch) {
          const std::size_t c0 = ch * chunk;
          if (c0 >= n3) continue;
          const std::size_t cn = std::min(chunk, n3 - c0);
          symtransform_detail::gemmAcc(static_cast<int>(np), static_cast<int>(cn), static_cast<int>(na),
                                       cd.data() + p0 * n_src + a0, ns, slab.data() + c0, static_cast<int>(n3),
                                       t1.data() + c0, static_cast<int>(n3));
        }
      }
    }
    std::vector<std::vector<T>> slabs(np);  // per-orbital results of this block (written in order below)
    {
      const SerialBlasScope serial_blas;  // the parallel region supplies the parallelism (see BlasThreads.h)
#pragma omp parallel
      {
        std::vector<T> t2(nmo * n_src * n_src), t3;
#pragma omp for schedule(dynamic)
        for (std::size_t pl = 0; pl < np; ++pl) {
          const std::size_t p = p0 + pl;
          // leg 2: t2(q, c d) = sum_b conj(C(b,q)) t1(b, c d)      [ (nmo x n) (n x n^2) ]
          gemm(false, false, nm, ns * ns, ns, cd.data(), ns, t1.data() + pl * n3, ns * ns, t2.data(), ns * ns);
          // legs 3 and 4 for the r >= p slab only: t3(q, r, d) = sum_c C(c,r) t2(q, c, d)
          const std::size_t r0 = p, nr = nmo - p;
          t3.assign(nmo * nr * n_src, T{});
          slabs[pl].assign(nmo * nr * nmo, T{});
          for (std::size_t q = 0; q < nmo; ++q)
            gemm(false, false, static_cast<int>(nr), ns, ns, ct.data() + r0 * n_src, ns, t2.data() + q * n_src * n_src, ns,
                 t3.data() + q * nr * n_src, ns);
          // t4(q r, s) = sum_d t3(q r, d) C(d, s)   [ (nmo nr x n) (n x nmo) ]
          gemm(false, false, static_cast<int>(nmo * nr), nm, ns, t3.data(), ns, c.data(), nm, slabs[pl].data(), nm);
        }
      }
    }
    // Write the block's slabs in orbital order: the symmetric partners of an element are computed independently and
    // differ by roundoff, so which value a shared slot keeps must not depend on thread timing (reproducible results).
    for (std::size_t pl = 0; pl < np; ++pl) {
      const std::size_t p = p0 + pl, r0 = p, nr = nmo - p;
      for (std::size_t q = 0; q < nmo; ++q)
        for (std::size_t r = 0; r < nr; ++r)
          for (std::size_t s = 0; s < nmo; ++s) out.set(p, q, r0 + r, s, slabs[pl][(q * nr + r) * nmo + s]);
      std::vector<T>().swap(slabs[pl]);
    }
  }
  return out;
}

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_SYMMETRICTRANSFORM_H
