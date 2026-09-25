#include "RkbTwoElectron.h"

#include <cblas.h>

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

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

// Swaps the electron-1 pair (legs 0,1) with the electron-2 pair (legs 2,3):
// result(p,q,r,s) = src(r,s,p,q). Used to derive the RKB-Small(beta-
// partner),RKB-Small(alpha-partner) block from the RKB-Small(alpha-
// partner),RKB-Small(beta-partner) one via the electron-exchange symmetry,
// instead of running a second, equally expensive quarter-transform chain.
Tensor4<std::complex<double>> swapElectronPairs(const Tensor4<std::complex<double>>& src) {
  const std::size_t d0 = src.dim0(), d1 = src.dim1(), d2 = src.dim2(), d3 = src.dim3();
  Tensor4<std::complex<double>> result(d2, d3, d0, d1);
  for (std::size_t p = 0; p < d0; ++p) {
    for (std::size_t q = 0; q < d1; ++q) {
      for (std::size_t r = 0; r < d2; ++r) {
        for (std::size_t s = 0; s < d3; ++s) {
          result(r, s, p, q) = src(p, q, r, s);
        }
      }
    }
  }
  return result;
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
  const Tensor4<double> ss_ss = twoElectronIntegrals(small_basis);

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
    // (no swapElectronPairs shortcut needed: computing all four y1,y2
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
    // Transform ss_ss's electron-1 pair (legs 0,1) once per y1 -- reused
    // for both y2 choices -- then its electron-2 pair (legs 2,3) once
    // per y2.
    Tensor4<std::complex<double>> ss_sY1[2];
    for (std::size_t y1 = 0; y1 < 2; ++y1) {
      ss_sY1[y1] =
          transformPairToRkbSmall(ss_ss, 0, 1, rkb_coefficients, y1 * n_large, n_large, n_small);
    }
    // ss_Y1Y2[1][0] (electron-1=beta-partner, electron-2=alpha-partner) is
    // never computed directly: electron-exchange symmetry gives
    //   ss_Y1Y2[1][0](p,q,r,s) = ss_Y1Y2[0][1](r,s,p,q),
    // so it is recovered from ss_Y1Y2[0][1] by an index permutation
    // instead of a second, equally expensive quarter-transform chain.
    ss_Y1Y2[0][0] =
        transformPairToRkbSmall(ss_sY1[0], 2, 3, rkb_coefficients, 0, n_large, n_small);
    ss_Y1Y2[0][1] =
        transformPairToRkbSmall(ss_sY1[0], 2, 3, rkb_coefficients, n_large, n_large, n_small);
    ss_Y1Y2[1][1] =
        transformPairToRkbSmall(ss_sY1[1], 2, 3, rkb_coefficients, n_large, n_large, n_small);
    ss_Y1Y2[1][0] = swapElectronPairs(ss_Y1Y2[0][1]);
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
