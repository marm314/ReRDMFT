#include "RkbTwoElectron.h"

#include <cblas.h>

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "BlasThreads.h"
#include "Cholesky_Decomposition.h"
#include "ElectronRepulsion.h"

namespace rerdmft {

namespace {

Matrix<std::complex<double>> subBlock(const Matrix<std::complex<double>>& c, std::size_t row_offset,
                                       std::size_t n_rows, std::size_t col_offset,
                                       std::size_t n_cols) {
  Matrix<std::complex<double>> block(n_rows, n_cols);
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_cols; ++j) {
      block(i, j) = c(row_offset + i, col_offset + j);
    }
  }
  return block;
}

Matrix<std::complex<double>> conjMatrix(const Matrix<std::complex<double>>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) result(i, j) = std::conj(m(i, j));
  }
  return result;
}

// Contracts one leg of a rank-4 tensor against `matrix` (new_dim x old_dim):
//   result(..., i_new, ...) = sum_k matrix(i_new, k) * src(..., k, ...).
// Tensor4's row-major (dim0 slowest, dim3 fastest) layout means the flat
// array can always be reshaped as (outer, this_leg, inner), with outer the
// product of dims before `leg` and inner the product of dims after it.
// `matrix` is already given as (new_dim x old_dim) -- no transpose is
// needed for a "matrix @ slice" contraction, which is used directly
// whenever inner > 1 (looping over the `outer` contiguous slices; a
// single loop iteration when outer==1, i.e. leg 0); when the contracted
// axis is the tensor's LAST dimension (inner==1, i.e. leg 3), src is
// reshaped as (outer x old_dim) instead and multiplied by matrix^T from
// the right in one call. Implemented via BLAS GEMM (cblas_zgemm for an
// already-complex src; for a real src, cblas_dgemm applied twice --
// against matrix's real and imaginary parts separately -- which is
// cheaper than upcasting the [generally much larger] src tensor to
// complex first). Verified against the original naive nested-loop
// implementation on random non-square test data (all 4 legs, both real
// and complex src) to floating-point precision before this rewrite.
Tensor4<std::complex<double>> transformLegImpl(const Tensor4<std::complex<double>>& src, int leg,
                                                const Matrix<std::complex<double>>& matrix) {
  const std::size_t dims[4] = {src.dim0(), src.dim1(), src.dim2(), src.dim3()};
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (int i = 0; i < leg; ++i) outer *= dims[static_cast<std::size_t>(i)];
  for (int i = leg + 1; i < 4; ++i) inner *= dims[static_cast<std::size_t>(i)];
  const std::size_t old_dim = dims[static_cast<std::size_t>(leg)];
  const std::size_t new_dim = matrix.rows();

  std::size_t new_dims[4] = {dims[0], dims[1], dims[2], dims[3]};
  new_dims[static_cast<std::size_t>(leg)] = new_dim;
  Tensor4<std::complex<double>> result(new_dims[0], new_dims[1], new_dims[2], new_dims[3],
                                        std::complex<double>(0.0, 0.0));
  const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);

  if (inner == 1) {
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(outer),
                static_cast<int>(new_dim), static_cast<int>(old_dim), &alpha, src.data(),
                static_cast<int>(old_dim), matrix.data(), static_cast<int>(old_dim), &beta,
                result.data(), static_cast<int>(new_dim));
  } else {
    for (std::size_t o = 0; o < outer; ++o) {
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(new_dim),
                  static_cast<int>(inner), static_cast<int>(old_dim), &alpha, matrix.data(),
                  static_cast<int>(old_dim), src.data() + o * old_dim * inner,
                  static_cast<int>(inner), &beta, result.data() + o * new_dim * inner,
                  static_cast<int>(inner));
    }
  }
  return result;
}

Tensor4<std::complex<double>> transformLegImpl(const Tensor4<double>& src, int leg,
                                                const Matrix<std::complex<double>>& matrix) {
  const std::size_t dims[4] = {src.dim0(), src.dim1(), src.dim2(), src.dim3()};
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (int i = 0; i < leg; ++i) outer *= dims[static_cast<std::size_t>(i)];
  for (int i = leg + 1; i < 4; ++i) inner *= dims[static_cast<std::size_t>(i)];
  const std::size_t old_dim = dims[static_cast<std::size_t>(leg)];
  const std::size_t new_dim = matrix.rows();

  Matrix<double> matrix_re(new_dim, old_dim), matrix_im(new_dim, old_dim);
  for (std::size_t i = 0; i < new_dim; ++i) {
    for (std::size_t j = 0; j < old_dim; ++j) {
      matrix_re(i, j) = matrix(i, j).real();
      matrix_im(i, j) = matrix(i, j).imag();
    }
  }

  std::size_t new_dims[4] = {dims[0], dims[1], dims[2], dims[3]};
  new_dims[static_cast<std::size_t>(leg)] = new_dim;
  const std::size_t total_new = new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3];
  std::vector<double> result_re(total_new, 0.0), result_im(total_new, 0.0);

  if (inner == 1) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(outer),
                static_cast<int>(new_dim), static_cast<int>(old_dim), 1.0, src.data(),
                static_cast<int>(old_dim), matrix_re.data(), static_cast<int>(old_dim), 0.0,
                result_re.data(), static_cast<int>(new_dim));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(outer),
                static_cast<int>(new_dim), static_cast<int>(old_dim), 1.0, src.data(),
                static_cast<int>(old_dim), matrix_im.data(), static_cast<int>(old_dim), 0.0,
                result_im.data(), static_cast<int>(new_dim));
  } else {
    for (std::size_t o = 0; o < outer; ++o) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(new_dim),
                  static_cast<int>(inner), static_cast<int>(old_dim), 1.0, matrix_re.data(),
                  static_cast<int>(old_dim), src.data() + o * old_dim * inner,
                  static_cast<int>(inner), 0.0, result_re.data() + o * new_dim * inner,
                  static_cast<int>(inner));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(new_dim),
                  static_cast<int>(inner), static_cast<int>(old_dim), 1.0, matrix_im.data(),
                  static_cast<int>(old_dim), src.data() + o * old_dim * inner,
                  static_cast<int>(inner), 0.0, result_im.data() + o * new_dim * inner,
                  static_cast<int>(inner));
    }
  }

  Tensor4<std::complex<double>> result(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
  std::complex<double>* out = result.data();
#pragma omp parallel for
  for (std::size_t i = 0; i < total_new; ++i) {
    out[i] = std::complex<double>(result_re[i], result_im[i]);
  }
  return result;
}

template <typename SrcT>
Tensor4<std::complex<double>> transformLeg(const Tensor4<SrcT>& src, int leg,
                                            const Matrix<std::complex<double>>& matrix) {
  return transformLegImpl(src, leg, matrix);
}

void addInPlace(Tensor4<std::complex<double>>& total, const Tensor4<std::complex<double>>& add) {
  const std::size_t len = total.dim0() * total.dim1() * total.dim2() * total.dim3();
  std::complex<double>* t = total.data();
  const std::complex<double>* s = add.data();
  for (std::size_t i = 0; i < len; ++i) t[i] += s[i];
}

// Transforms a same-electron leg pair (leg_bra, leg_ket -- either (0,1) or
// (2,3)) from the unrestricted-kinetic-balance Small spin-orbital space
// into a single RKB-Small "partner" flavor: the one built from
// rkb_coefficients' row range [row_offset, row_offset+nLarge). Sums
// internally over the Small basis's own alpha/beta spin block (columns
// [0,nSmall) and [nSmall,2*nSmall) of rkb_coefficients), since a genuine
// RKB-Small spinor mixes both -- see RkbTwoElectron.h.
template <typename SrcT>
Tensor4<std::complex<double>> transformPairToRkbSmall(
    const Tensor4<SrcT>& src, int leg_bra, int leg_ket,
    const Matrix<std::complex<double>>& rkb_coefficients, std::size_t row_offset,
    std::size_t n_large, std::size_t n_small) {
  Tensor4<std::complex<double>> total;
  bool have_total = false;
  for (std::size_t spin_offset : {std::size_t{0}, n_small}) {
    const Matrix<std::complex<double>> c_spin =
        subBlock(rkb_coefficients, row_offset, n_large, spin_offset, n_small);
    const Matrix<std::complex<double>> c_spin_conj = conjMatrix(c_spin);

    Tensor4<std::complex<double>> transformed =
        transformLeg(transformLeg(src, leg_bra, c_spin_conj), leg_ket, c_spin);
    if (!have_total) {
      total = std::move(transformed);
      have_total = true;
    } else {
      addInPlace(total, transformed);
    }
  }
  return total;
}

// The RKB-Small(y1),RKB-Small(y2) blocks of the two-electron tensor,
//   out[y1][y2](a,c,b,d) = sum_{s1,s2} sum_{k,t,l,u} conj(c1(a,k)) c1(c,t) conj(c2(b,l)) c2(d,u) (kt|lu),
// with c1 = the (y1, spin s1) block of the RKB coefficients (n_large x n_small), c2 likewise for (y2, s2), from the
// PACKED real (SS|SS) tensor -- computed in slabs, so neither the dense n_small^4 tensor nor its dense transformed
// intermediates ever exist. The projection factorizes leg by leg, so the work is done chunk by chunk of the second
// index t: (1) a chunk X[k][t][lu] of (SS|SS) is read from the packed tensor once; (2) for every (y1,s1) the
// first leg for ALL Large indices a at once is a GEMM, T1[a][t][lu] = sum_k conj(c1(a,k)) X[k][t][lu]; (3) the
// second leg is accumulated into T2[y1][a][c][lu] += c1(c,t) T1[a][t][lu]; finally (4) the electron-2 pair is
// projected for every (a,c): sum_{lu} conj(c2(b,l)) c2(d,u) T2[a][c][lu]. Memory: the packed tensor plus
// ~2 n_large^2 n_small^2 complex numbers of T2 and a chunk of X/T1; flops ~ n_large n_small^4.
void smallSmallBlocksSlab(const PackedTwoElectronTensor& ss, const Matrix<std::complex<double>>& rkb,
                          std::size_t nl, std::size_t ns, Tensor4<std::complex<double>> out[2][2]) {
  using C = std::complex<double>;
  const std::size_t nq = ns * ns;
  // Coefficient blocks A[y][s] (nl x ns) and the derived matrices used below.
  Matrix<C> a_blk[2][2], a_re[2][2], a_imneg[2][2], a_conj[2][2], a_trans[2][2];
  for (std::size_t y = 0; y < 2; ++y)
    for (std::size_t s = 0; s < 2; ++s) {
      const Matrix<C> blk = subBlock(rkb, y * nl, nl, s * ns, ns);
      a_blk[y][s] = blk;
      a_conj[y][s] = conjMatrix(blk);
      a_trans[y][s] = Matrix<C>(ns, nl);
      Matrix<double> re(nl, ns), im(nl, ns);  // conj(A) = re + i*im
      for (std::size_t p = 0; p < nl; ++p)
        for (std::size_t k = 0; k < ns; ++k) {
          a_trans[y][s](k, p) = blk(p, k);
          re(p, k) = blk(p, k).real();
          im(p, k) = -blk(p, k).imag();
        }
      a_re[y][s] = Matrix<C>(nl, ns);  // (real parts of conj(A), as complex for storage convenience)
      a_imneg[y][s] = Matrix<C>(nl, ns);
      for (std::size_t i = 0; i < nl * ns; ++i) {
        a_re[y][s].data()[i] = C(re.data()[i], 0.0);
        a_imneg[y][s].data()[i] = C(im.data()[i], 0.0);
      }
    }
  const std::size_t budget_doubles = 12u * 1024u * 1024u;  // ~100 MB for the chunk of (SS|SS)
  const std::size_t tb_max = std::max<std::size_t>(1, std::min<std::size_t>(ns, budget_doubles / (ns * nq)));
  std::vector<C> t2[2];
  for (std::size_t y = 0; y < 2; ++y) t2[y].assign(nl * nl * nq, C{});
  std::vector<double> x(ns * tb_max * nq), t_re(nl * tb_max * nq), t_im(nl * tb_max * nq);
  std::vector<C> t1(nl * tb_max * nq);
  for (std::size_t t0 = 0; t0 < ns; t0 += tb_max) {
    const std::size_t tb = std::min(tb_max, ns - t0);
    // (1) chunk of (SS|SS): X[k][tl][l*ns+u] = (k, t0+tl | l, u)
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t k = 0; k < ns; ++k)
      for (std::size_t tl = 0; tl < tb; ++tl) {
        double* row = &x[(k * tb + tl) * nq];
        for (std::size_t l = 0; l < ns; ++l)
          for (std::size_t u = 0; u < ns; ++u) row[l * ns + u] = ss(k, t0 + tl, l, u);
      }
    for (std::size_t y1 = 0; y1 < 2; ++y1)
      for (std::size_t s1 = 0; s1 < 2; ++s1) {
        // (2) first leg for all a: T = conj(A) X  (real GEMMs with the real and imaginary parts of conj(A))
        std::vector<double> a_r(nl * ns), a_i(nl * ns);
        for (std::size_t i = 0; i < nl * ns; ++i) {
          a_r[i] = a_re[y1][s1].data()[i].real();
          a_i[i] = a_imneg[y1][s1].data()[i].real();
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(tb * nq),
                    static_cast<int>(ns), 1.0, a_r.data(), static_cast<int>(ns), x.data(), static_cast<int>(tb * nq),
                    0.0, t_re.data(), static_cast<int>(tb * nq));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(tb * nq),
                    static_cast<int>(ns), 1.0, a_i.data(), static_cast<int>(ns), x.data(), static_cast<int>(tb * nq),
                    0.0, t_im.data(), static_cast<int>(tb * nq));
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < nl * tb * nq; ++i) t1[i] = C(t_re[i], t_im[i]);
        // (3) second leg, accumulated over the chunk: T2[y1][a][c][lu] += sum_tl A(c,t0+tl) T1[a][tl][lu]
        {
          const SerialBlasScope serial_blas;
          const C one(1.0, 0.0);
#pragma omp parallel for schedule(static)
          for (std::size_t a = 0; a < nl; ++a)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(nq),
                        static_cast<int>(tb), &one, a_blk[y1][s1].data() + t0, static_cast<int>(ns),
                        t1.data() + a * tb * nq, static_cast<int>(nq), &one, t2[y1].data() + a * nl * nq,
                        static_cast<int>(nq));
        }
      }
  }
  // (4) electron-2 pair, for every Large pair (a,c): out[y1][y2](a,c,b,d) += sum_{lu} conj(A2(b,l)) A2(d,u) T2(l,u)
  for (std::size_t y1 = 0; y1 < 2; ++y1)
    for (std::size_t y2 = 0; y2 < 2; ++y2) out[y1][y2] = Tensor4<C>(nl, nl, nl, nl, C{});
  {
    const SerialBlasScope serial_blas;
#pragma omp parallel
    {
      Matrix<C> u(nl, ns), v(nl, nl);
      const C one(1.0, 0.0), zero(0.0, 0.0);
#pragma omp for schedule(dynamic)
      for (std::size_t ac = 0; ac < nl * nl; ++ac)
        for (std::size_t y1 = 0; y1 < 2; ++y1)
          for (std::size_t y2 = 0; y2 < 2; ++y2)
            for (std::size_t s2 = 0; s2 < 2; ++s2) {
              cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(ns),
                          static_cast<int>(ns), &one, a_conj[y2][s2].data(), static_cast<int>(ns),
                          t2[y1].data() + ac * nq, static_cast<int>(ns), &zero, u.data(), static_cast<int>(ns));
              cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(nl),
                          static_cast<int>(ns), &one, u.data(), static_cast<int>(ns), a_trans[y2][s2].data(),
                          static_cast<int>(nl), &zero, v.data(), static_cast<int>(nl));
              C* dst = out[y1][y2].data() + ac * nl * nl;
              for (std::size_t i = 0; i < nl * nl; ++i) dst[i] += v.data()[i];
            }
    }
  }
}

Matrix<std::complex<double>> promoteToComplex(const Matrix<double>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) result(i, j) = std::complex<double>(m(i, j), 0.0);
  }
  return result;
}

// Projects a single (n_small x n_small) Cholesky vector V of the real
// (Small,Small|Small,Small) chemist-notation tensor ss_ss into the
// RKB-Small(y) "partner" flavor (row_offset = y*n_large), applying
// EXACTLY the same bra-conjugated/ket-unconjugated, 2-spin-block-summed
// projection transformPairToRkbSmall applies to a full tensor's leg pair
// -- see the derivation in rkbTwoElectronIntegrals's own comment for why
// this is valid to apply to a single Cholesky vector instead of the full
// tensor. Costs O(n_small^2 * n_large) per spin block (two ordinary
// matrix products), vs. transformPairToRkbSmall's O(n_small^2 * n_large *
// n_small^2) when applied directly to ss_ss's own two electron-pair legs
// (both legs still n_small before this transform touches them).
Matrix<std::complex<double>> projectCholeskyVectorToRkbSmall(
    const Matrix<double>& v, const Matrix<std::complex<double>>& rkb_coefficients,
    std::size_t row_offset, std::size_t n_large, std::size_t n_small) {
  const Matrix<std::complex<double>> v_complex = promoteToComplex(v);
  Matrix<std::complex<double>> total(n_large, n_large, std::complex<double>(0.0, 0.0));
  bool have_total = false;
  for (std::size_t spin_offset : {std::size_t{0}, n_small}) {
    const Matrix<std::complex<double>> c_spin =
        subBlock(rkb_coefficients, row_offset, n_large, spin_offset, n_small);
    const Matrix<std::complex<double>> c_spin_conj = conjMatrix(c_spin);
    // W(p,q) = sum_{a,b} conj(c_spin(p,a)) * v(a,b) * c_spin(q,b)
    //        = c_spin_conj * v_complex * transpose(c_spin), and
    // transpose(c_spin) == dagger(c_spin_conj) (dagger conjugates AND
    // transposes, and c_spin_conj is already conjugated once, so
    // conjugating it again cancels out, leaving a plain transpose).
    Matrix<std::complex<double>> contrib = c_spin_conj * (v_complex * dagger(c_spin_conj));
    if (!have_total) {
      total = std::move(contrib);
      have_total = true;
    } else {
      const std::size_t len = total.rows() * total.cols();
      std::complex<double>* t = total.data();
      const std::complex<double>* s = contrib.data();
      for (std::size_t i = 0; i < len; ++i) t[i] += s[i];
    }
  }
  return total;
}

// Builds a dense Tensor4 as a sum of outer products of two equal-length
// vector lists: result(p,q,r,s) = sum_L left[L](p,q) * right[L](r,s). Used
// to reconstruct ss_Y1Y2[y1][y2] from the projected Cholesky vectors
// W^{y1}, W^{y2} (see rkbTwoElectronIntegrals) -- NOT the same formula as
// Cholesky_Decomposition.h's own choleskyReconstructEri (which conjugates
// its second factor and uses the SAME vector list for both), since here
// `left` and `right` are generally DIFFERENT projections of the SAME
// underlying vector, and the electron-2 leg's own construction already
// leaves V_L unconjugated (see this function's caller for the derivation).
Tensor4<std::complex<double>> outerSumTensor(const std::vector<Matrix<std::complex<double>>>& left,
                                              const std::vector<Matrix<std::complex<double>>>& right) {
  const std::size_t n = left.front().rows();
  Tensor4<std::complex<double>> result(n, n, n, n, std::complex<double>(0.0, 0.0));
  for (std::size_t l = 0; l < left.size(); ++l) {
    const Matrix<std::complex<double>>& a = left[l];
    const Matrix<std::complex<double>>& b = right[l];
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        const std::complex<double> aval = a(p, q);
        for (std::size_t r = 0; r < n; ++r) {
          for (std::size_t s = 0; s < n; ++s) {
            result(p, q, r, s) += aval * b(r, s);
          }
        }
      }
    }
  }
  return result;
}

}  // namespace

Matrix<std::complex<double>> rkbProjectSmallVector(const Matrix<double>& v,
                                                    const Matrix<std::complex<double>>& rkb_coefficients,
                                                    std::size_t y, std::size_t n_large, std::size_t n_small) {
  return projectCholeskyVectorToRkbSmall(v, rkb_coefficients, y * n_large, n_large, n_small);
}

RkbTwoElectronTensor rkbTwoElectronIntegrals(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis,
                                              const Matrix<std::complex<double>>& rkb_coefficients,
                                              bool use_cholesky, double cholesky_threshold) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 4 * n_large;

  // Real spatial-AO chemist-notation tensors (pq|rs): ll_ll all-Large,
  // ll_ss Large electron-1 pair / Small electron-2 pair, ss_ss all-Small.
  const Tensor4<double> ll_ll = twoElectronIntegrals(large_basis);
  const Tensor4<double> ll_ss = twoElectronIntegralsCross(large_basis, small_basis);
  const PackedTwoElectronTensor ss_ss = twoElectronIntegralsPacked(small_basis);  // 8-fold packed, never dense

  // (Large,Large | RKB-Small(y2),RKB-Small(y2)): transform ll_ss's
  // electron-2 pair (legs 2,3). Index y in {0,1} means {alpha-partner,
  // beta-partner}, i.e. rkb_coefficients row_offset in {0, nLarge}.
  Tensor4<std::complex<double>> ll_sY[2];
  for (std::size_t y2 = 0; y2 < 2; ++y2) {
    ll_sY[y2] = transformPairToRkbSmall(ll_ss, 2, 3, rkb_coefficients, y2 * n_large, n_large,
                                         n_small);
  }

  // (RKB-Small(y1),RKB-Small(y1) | RKB-Small(y2),RKB-Small(y2)): the
  // dominant-cost piece (ss_ss has dimension n_small, typically >>
  // n_large, e.g. 57 vs. 7 for water/STO-3G).
  Tensor4<std::complex<double>> ss_Y1Y2[2][2];
  if (use_cholesky) {
    // ss_ss(a,b,c,d) is a REAL, chemist-notation (ab|cd) tensor: its
    // (0,1)/(2,3) leg-pair grouping IS the bra/ket grouping
    // Cholesky_Decomposition.h requires (see its own header comment),
    // so it decomposes directly: ss_ss(a,b,c,d) = sum_L V_L(a,b)*V_L(c,d).
    // Since transformPairToRkbSmall's projection of a same-electron leg
    // pair (bra index gets conj(c_spin), ket index gets un-conjugated
    // c_spin, summed over the Small basis's own 2 spin blocks) is LINEAR
    // in the tensor being transformed, it commutes with this sum: each
    // Cholesky vector can be projected independently --
    //   W_L^y(p,q) := sum_spin [conj(c_spin) * V_L * transpose(c_spin)](p,q)
    // (projectCholeskyVectorToRkbSmall, y*n_large row offset) -- and the
    // fully-transformed block follows as
    //   ss_Y1Y2[y1][y2](p,q,r,s) = sum_L W_L^{y1}(p,q) * W_L^{y2}(r,s)
    // (computing all four y1,y2
    // combinations this way is already O(Nchol*n_large^4), cheaper than
    // even one direct quarter-transform of the untouched ss_ss tensor).
    const auto vectors = choleskyDecomposeEriChecked(ss_ss, cholesky_threshold);
    std::vector<Matrix<std::complex<double>>> w[2];
    for (std::size_t y = 0; y < 2; ++y) {
      w[y].reserve(vectors.size());
      for (const auto& v : vectors) {
        w[y].push_back(
            projectCholeskyVectorToRkbSmall(v, rkb_coefficients, y * n_large, n_large, n_small));
      }
    }
    for (std::size_t y1 = 0; y1 < 2; ++y1) {
      for (std::size_t y2 = 0; y2 < 2; ++y2) {
        ss_Y1Y2[y1][y2] = outerSumTensor(w[y1], w[y2]);
      }
    }
  } else {
    // Exact projection of (SS|SS) into the four RKB-Small(y1),RKB-Small(y2) blocks, in slabs from the packed tensor.
    smallSmallBlocksSlab(ss_ss, rkb_coefficients, n_large, n_small, ss_Y1Y2);
  }

  // Assemble the full (4*nLarge)^4 physics-notation tensor <A B|C D> (only
  // the electron-exchange-unique half is actually stored -- see
  // RkbTwoElectronTensor). The spinor index ranges over [Large-alpha,
  // Large-beta, RKB-Small-alpha-partner, RKB-Small-beta-partner], each
  // nLarge wide, matching H_RKB's ordering (RkbHamiltonian.h). Chemist/
  // physics relation used throughout: <A B|C D> = (A C|B D) (electron-1
  // pair A,C; electron-2 pair B,D).
  RkbTwoElectronTensor result(n);

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 3 * n_large;
  const std::size_t off_small[2] = {off_small_alpha, off_small_beta};

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t c = 0; c < n_large; ++c) {
      for (std::size_t b = 0; b < n_large; ++b) {
        for (std::size_t d = 0; d < n_large; ++d) {
          // electron-1 = Large, electron-2 = Large: identical values for
          // ALL FOUR alpha/beta slot combinations (not just matching
          // alpha-alpha/beta-beta), since 1/r12 does not depend on spin --
          // electron-1's own bra,ket must share ONE spin slot and
          // electron-2's own bra,ket must share ONE spin slot, but those
          // two slots are otherwise independent of each other (e.g.
          // electron-1 = Large-alpha with electron-2 = Large-beta is a
          // perfectly ordinary, generally nonzero Coulomb integral).
          const std::complex<double> ll_ll_val(ll_ll(a, c, b, d), 0.0);
          result.set(off_large_alpha + a, off_large_alpha + b, off_large_alpha + c,
                     off_large_alpha + d, ll_ll_val);
          result.set(off_large_beta + a, off_large_beta + b, off_large_beta + c,
                     off_large_beta + d, ll_ll_val);
          result.set(off_large_alpha + a, off_large_beta + b, off_large_alpha + c,
                     off_large_beta + d, ll_ll_val);
          result.set(off_large_beta + a, off_large_alpha + b, off_large_beta + c,
                     off_large_alpha + d, ll_ll_val);

          for (std::size_t y2 = 0; y2 < 2; ++y2) {
            const std::size_t off_e2 = off_small[y2];
            // electron-1 = Large, electron-2 = RKB-Small(y2):
            //   <Large_a RKBSmall(y2)_b | Large_c RKBSmall(y2)_d>
            //     = (Large_a Large_c | RKBSmall(y2)_b RKBSmall(y2)_d) = ll_sY[y2](a,c,b,d).
            const std::complex<double> ls_val = ll_sY[y2](a, c, b, d);
            result.set(off_large_alpha + a, off_e2 + b, off_large_alpha + c, off_e2 + d, ls_val);
            result.set(off_large_beta + a, off_e2 + b, off_large_beta + c, off_e2 + d, ls_val);

            // (SS|LL): electron-1 = RKB-Small(y2), electron-2 = Large. By
            // the standard real-ERI symmetry (pq|rs)=(rs|pq), this is
            // ll_sY[y2] with the electron-1/electron-2 roles swapped, i.e.
            // its (Large_bra,Large_ket,Small_bra,Small_ket) arguments
            // reordered to (Large_b,Large_d,Small_a,Small_c) -- NOT the
            // same index tuple as ls_val above.
            const std::complex<double> sl_val = ll_sY[y2](b, d, a, c);
            result.set(off_e2 + a, off_large_alpha + b, off_e2 + c, off_large_alpha + d, sl_val);
            result.set(off_e2 + a, off_large_beta + b, off_e2 + c, off_large_beta + d, sl_val);
          }

          for (std::size_t y1 = 0; y1 < 2; ++y1) {
            const std::size_t off_e1 = off_small[y1];
            for (std::size_t y2 = 0; y2 < 2; ++y2) {
              const std::size_t off_e2 = off_small[y2];
              result.set(off_e1 + a, off_e2 + b, off_e1 + c, off_e2 + d,
                         ss_Y1Y2[y1][y2](a, c, b, d));
            }
          }
        }
      }
    }
  }

  return result;
}

}  // namespace rerdmft
