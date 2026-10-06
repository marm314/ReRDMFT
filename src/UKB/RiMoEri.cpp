#include "RiMoEri.h"

#include <cblas.h>
#include <omp.h>

#include <array>
#include <stdexcept>

namespace rerdmft {

namespace {
using C = std::complex<double>;

// B_block(P,p,q) = sum_{mu',nu' in this block's physical AO range} conj(c_block(mu',p)) *
// c_block(nu',q) * eri3_block(mu',nu',P), for ONE of the 4 UKB spin/spatial blocks. `c_block` is
// n_phys x n_total (this block's own AO rows of c_ukb); `eri3_block` is eri3_LL or eri3_SS, shape
// (n_aux, n_phys*n_phys), row P holding the contiguous n_phys x n_phys (mu',nu'|P) block (same
// convention as AO_ints/ThreeCenterIntegrals.h). Returned shape (n_aux, n_total*n_total), added
// into the caller's running sum across all 4 blocks.
Matrix<C> transformBlock(const Matrix<double>& eri3_block, std::size_t n_phys, const Matrix<C>& c_block,
                          std::size_t n_total) {
  const std::size_t n_aux = eri3_block.rows();
  Matrix<C> b_block(n_aux, n_total * n_total, C{});
  if (n_phys == 0 || n_total == 0) return b_block;

  // conj(c_block)^T == dagger(c_block), split into real/imaginary parts so the first (real-AO)
  // half-transform can use plain DGEMM.
  const Matrix<C> c_dagger = dagger(c_block);  // n_total x n_phys
  Matrix<double> c_dagger_re(n_total, n_phys), c_dagger_im(n_total, n_phys);
  for (std::size_t i = 0; i < n_total; ++i)
    for (std::size_t j = 0; j < n_phys; ++j) {
      c_dagger_re(i, j) = c_dagger(i, j).real();
      c_dagger_im(i, j) = c_dagger(i, j).imag();
    }

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const double* slice = eri3_block.data() + p * n_phys * n_phys;

    // T_P = dagger(c_block) @ slice_P : (n_total x n_phys) @ (n_phys x n_phys) -> n_total x n_phys.
    Matrix<double> t_re(n_total, n_phys, 0.0), t_im(n_total, n_phys, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_total), static_cast<int>(n_phys),
                static_cast<int>(n_phys), 1.0, c_dagger_re.data(), static_cast<int>(n_phys), slice,
                static_cast<int>(n_phys), 0.0, t_re.data(), static_cast<int>(n_phys));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_total), static_cast<int>(n_phys),
                static_cast<int>(n_phys), 1.0, c_dagger_im.data(), static_cast<int>(n_phys), slice,
                static_cast<int>(n_phys), 0.0, t_im.data(), static_cast<int>(n_phys));
    Matrix<C> t_p(n_total, n_phys);
    for (std::size_t i = 0; i < n_total; ++i)
      for (std::size_t j = 0; j < n_phys; ++j) t_p(i, j) = C(t_re(i, j), t_im(i, j));

    // B_P = T_P @ c_block : (n_total x n_phys) @ (n_phys x n_total) -> n_total x n_total, written
    // directly into b_block's row p (contiguous n_total*n_total slice).
    const C alpha(1.0, 0.0), beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_total), static_cast<int>(n_total),
                static_cast<int>(n_phys), &alpha, t_p.data(), static_cast<int>(n_phys), c_block.data(),
                static_cast<int>(n_total), &beta, b_block.data() + p * n_total * n_total, static_cast<int>(n_total));
  }
  return b_block;
}

}  // namespace

RiMoEri buildRiMoEri(const RiDirectEriSource& eri, const Matrix<C>& c_ukb) {
  const std::size_t nl = eri.nl, ns = eri.ns;
  const std::size_t n_ukb = 2 * nl + 2 * ns;
  if (c_ukb.rows() != n_ukb) {
    throw std::runtime_error("buildRiMoEri: c_ukb row count does not match 2*nLarge+2*nSmall");
  }
  const std::size_t n_total = c_ukb.cols();
  const std::array<std::size_t, 4> offset = {0, nl, 2 * nl, 2 * nl + ns};
  const std::array<std::size_t, 4> fsize = {nl, nl, ns, ns};

  const std::size_t n_aux = eri.eri3_LL.rows();
  Matrix<C> b_full(n_aux, n_total * n_total, C{});
  for (int f = 0; f < 4; ++f) {
    const std::size_t fu = static_cast<std::size_t>(f);
    Matrix<C> c_block(fsize[fu], n_total);
    for (std::size_t a = 0; a < fsize[fu]; ++a)
      for (std::size_t p = 0; p < n_total; ++p) c_block(a, p) = c_ukb(offset[fu] + a, p);
    const Matrix<double>& eri3_block = (f < 2) ? eri.eri3_LL : eri.eri3_SS;
    const Matrix<C> b_block = transformBlock(eri3_block, fsize[fu], c_block, n_total);
    for (std::size_t i = 0; i < n_aux * n_total * n_total; ++i) b_full.data()[i] += b_block.data()[i];
  }
  return RiMoEri(n_total, std::move(b_full));
}

RiMoEri RiMoEri::rotated(const Matrix<C>& u) const {
  if (u.rows() != n_total_ || u.cols() != n_total_) {
    throw std::runtime_error("RiMoEri::rotated: u dimensions do not match n_total");
  }
  const std::size_t n_aux = b_.rows();
  Matrix<C> b_new(n_aux, n_total_ * n_total_, C{});
  const Matrix<C> u_dagger = dagger(u);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const C* slice = b_.data() + p * n_total_ * n_total_;
    Matrix<C> t_p(n_total_, n_total_);
    const C alpha(1.0, 0.0), beta(0.0, 0.0);
    // T_P = U^dagger @ B_P : (n_total x n_total) @ (n_total x n_total).
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_total_), static_cast<int>(n_total_),
                static_cast<int>(n_total_), &alpha, u_dagger.data(), static_cast<int>(n_total_), slice,
                static_cast<int>(n_total_), &beta, t_p.data(), static_cast<int>(n_total_));
    // B'_P = T_P @ U : written directly into b_new's row p.
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_total_), static_cast<int>(n_total_),
                static_cast<int>(n_total_), &alpha, t_p.data(), static_cast<int>(n_total_), u.data(),
                static_cast<int>(n_total_), &beta, b_new.data() + p * n_total_ * n_total_, static_cast<int>(n_total_));
  }
  return RiMoEri(n_total_, std::move(b_new));
}

RiMoEri RiMoEri::positiveBlock(std::size_t off) const {
  const std::size_t m = n_total_ - off;
  const std::size_t n_aux = b_.rows();
  Matrix<C> b_new(n_aux, m * m);
  for (std::size_t p = 0; p < n_aux; ++p)
    for (std::size_t a = 0; a < m; ++a)
      for (std::size_t c = 0; c < m; ++c) b_new(p, a * m + c) = b_(p, (a + off) * n_total_ + (c + off));
  return RiMoEri(m, std::move(b_new));
}

}  // namespace rerdmft
