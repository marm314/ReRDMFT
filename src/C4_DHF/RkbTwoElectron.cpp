#include "RkbTwoElectron.h"

#include <complex>
#include <cstddef>
#include <utility>

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
// product of dims before `leg` and inner the product of dims after it --
// so this single routine handles all four legs without needing four
// near-duplicate loop nests.
template <typename SrcT>
Tensor4<std::complex<double>> transformLeg(const Tensor4<SrcT>& src, int leg,
                                            const Matrix<std::complex<double>>& matrix) {
  const std::size_t dims[4] = {src.dim0(), src.dim1(), src.dim2(), src.dim3()};
  std::size_t outer = 1;
  std::size_t inner = 1;
  for (int i = 0; i < leg; ++i) outer *= dims[i];
  for (int i = leg + 1; i < 4; ++i) inner *= dims[i];
  const std::size_t old_dim = dims[leg];
  const std::size_t new_dim = matrix.rows();

  std::size_t new_dims[4] = {dims[0], dims[1], dims[2], dims[3]};
  new_dims[static_cast<std::size_t>(leg)] = new_dim;
  Tensor4<std::complex<double>> result(new_dims[0], new_dims[1], new_dims[2], new_dims[3],
                                        std::complex<double>(0.0, 0.0));

  const SrcT* src_data = src.data();
  std::complex<double>* out_data = result.data();
  // Every (o,j) pair owns a disjoint, non-overlapping `inner`-sized output
  // slice and only reads from src -- never from `result` -- so looping
  // over the combined (o,j) space in parallel is safe regardless of which
  // leg this is (leg 0/2 calls have outer==1, so collapsing onto j alone
  // still gives new_dim-way parallelism instead of none).
#pragma omp parallel for collapse(2)
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t j = 0; j < new_dim; ++j) {
      std::complex<double>* out = out_data + (o * new_dim + j) * inner;
      for (std::size_t k = 0; k < old_dim; ++k) {
        const std::complex<double> m_jk = matrix(j, k);
        const SrcT* in = src_data + (o * old_dim + k) * inner;
        for (std::size_t m = 0; m < inner; ++m) {
          out[m] += m_jk * static_cast<std::complex<double>>(in[m]);
        }
      }
    }
  }
  return result;
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

}  // namespace

RkbTwoElectronTensor rkbTwoElectronIntegrals(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis,
                                              const Matrix<std::complex<double>>& rkb_coefficients) {
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

  // (RKB-Small(y1),RKB-Small(y1) | RKB-Small(y2),RKB-Small(y2)): transform
  // ss_ss's electron-1 pair (legs 0,1) once per y1 -- reused for both y2
  // choices -- then its electron-2 pair (legs 2,3) once per y2.
  Tensor4<std::complex<double>> ss_sY1[2];
  for (std::size_t y1 = 0; y1 < 2; ++y1) {
    ss_sY1[y1] =
        transformPairToRkbSmall(ss_ss, 0, 1, rkb_coefficients, y1 * n_large, n_large, n_small);
  }
  // ss_Y1Y2[1][0] (electron-1=beta-partner, electron-2=alpha-partner) is
  // never computed directly: electron-exchange symmetry gives
  //   ss_Y1Y2[1][0](p,q,r,s) = ss_Y1Y2[0][1](r,s,p,q),
  // so it is recovered from ss_Y1Y2[0][1] by an index permutation instead
  // of a second, equally expensive quarter-transform chain.
  Tensor4<std::complex<double>> ss_Y1Y2[2][2];
  ss_Y1Y2[0][0] =
      transformPairToRkbSmall(ss_sY1[0], 2, 3, rkb_coefficients, 0, n_large, n_small);
  ss_Y1Y2[0][1] =
      transformPairToRkbSmall(ss_sY1[0], 2, 3, rkb_coefficients, n_large, n_large, n_small);
  ss_Y1Y2[1][1] =
      transformPairToRkbSmall(ss_sY1[1], 2, 3, rkb_coefficients, n_large, n_large, n_small);
  ss_Y1Y2[1][0] = swapElectronPairs(ss_Y1Y2[0][1]);

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
