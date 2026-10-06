#include "X2C_FockMatrixRi.h"

#include <cblas.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;

void addInto(Matrix<C>& dst, const Matrix<C>& src) {
  for (std::size_t i = 0; i < dst.rows(); ++i)
    for (std::size_t j = 0; j < dst.cols(); ++j) dst(i, j) += src(i, j);
}

Matrix<C> zgemmMultiply(const Matrix<C>& a, const Matrix<C>& b) {
  Matrix<C> c(a.rows(), b.cols(), C{});
  const C alpha(1.0, 0.0), beta(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows()), static_cast<int>(b.cols()),
              static_cast<int>(a.cols()), &alpha, a.data(), static_cast<int>(a.cols()), b.data(),
              static_cast<int>(b.cols()), &beta, c.data(), static_cast<int>(b.cols()));
  return c;
}

}  // namespace

Matrix<C> x2cFockMatrix(const Matrix<C>& h_x2c, const RiX2cEriSource& eri, const Matrix<C>& density_matrix) {
  const std::size_t nl = eri.nl;
  const std::size_t n = 2 * nl;
  if (density_matrix.rows() != n || density_matrix.cols() != n) {
    throw std::runtime_error("x2cFockMatrix (RI): density_matrix dimension does not match 2*nLarge");
  }
  const std::size_t n_aux = eri.eri3_L.rows();
  if (eri.eri3_L.cols() != nl * nl) {
    throw std::runtime_error("x2cFockMatrix (RI): eri3_L column count does not match nl^2");
  }

  const std::array<std::size_t, 2> offset = {0, nl};

  // --- Hartree/J: spin-summed total density, real part only (eri3_L is real, P is Hermitian). ---
  std::vector<double> p_flat(nl * nl);
  for (std::size_t mu = 0; mu < nl; ++mu)
    for (std::size_t nu = 0; nu < nl; ++nu)
      p_flat[mu * nl + nu] =
          (density_matrix(offset[0] + mu, offset[0] + nu) + density_matrix(offset[1] + mu, offset[1] + nu)).real();

  std::vector<double> x_p(n_aux, 0.0);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri.eri3_L.data(),
              static_cast<int>(nl * nl), p_flat.data(), 1, 0.0, x_p.data(), 1);
  std::vector<double> j_flat(nl * nl, 0.0);
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri.eri3_L.data(),
              static_cast<int>(nl * nl), x_p.data(), 1, 0.0, j_flat.data(), 1);

  // --- Exchange/K: recover the occupied AO-basis coefficients via eri.x_large/eri.overlap -- SAME
  // non-orthogonal-basis recovery as UkbFockMatrixRi.h's own rkbFockMatrix (reuse the SCF loop's
  // OWN X/S, never a fresh orthogonalization of the AO overlap). ---
  const Matrix<C> sps = zgemmMultiply(zgemmMultiply(eri.overlap, density_matrix), eri.overlap);
  const Matrix<C> p_eff = zgemmMultiply(dagger(eri.x_large), zgemmMultiply(sps, eri.x_large));
  const HermitianEigenResult eig = diagonalizeHermitian(p_eff);
  std::vector<std::size_t> occ_idx;
  for (std::size_t i = 0; i < eig.eigenvalues.size(); ++i) {
    if (eig.eigenvalues[i] > 0.5) occ_idx.push_back(i);
  }
  const std::size_t n_occ = occ_idx.size();
  Matrix<C> y_occ(eri.x_large.cols(), n_occ);
  std::vector<double> occ(n_occ);
  for (std::size_t i = 0; i < n_occ; ++i) {
    occ[i] = eig.eigenvalues[occ_idx[i]];
    for (std::size_t a = 0; a < eri.x_large.cols(); ++a) y_occ(a, i) = eig.eigenvectors(a, occ_idx[i]);
  }
  const Matrix<C> c_occ = zgemmMultiply(eri.x_large, y_occ);  // n x n_occ

  std::array<Matrix<double>, 2> c_t_re, c_t_im;
  for (int f = 0; f < 2; ++f) {
    c_t_re[static_cast<std::size_t>(f)] = Matrix<double>(n_occ, nl);
    c_t_im[static_cast<std::size_t>(f)] = Matrix<double>(n_occ, nl);
    for (std::size_t i = 0; i < n_occ; ++i) {
      const double sqrt_occ = std::sqrt(occ[i]);
      for (std::size_t a = 0; a < nl; ++a) {
        const C coeff = c_occ(offset[static_cast<std::size_t>(f)] + a, i);
        c_t_re[static_cast<std::size_t>(f)](i, a) = coeff.real() * sqrt_occ;
        c_t_im[static_cast<std::size_t>(f)](i, a) = -coeff.imag() * sqrt_occ;
      }
    }
  }

  const std::vector<std::pair<int, int>> unique_pairs = {{0, 0}, {1, 1}, {0, 1}};
  const int n_threads = omp_get_max_threads();
  std::vector<std::array<std::array<Matrix<C>, 2>, 2>> km_part(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t)
    for (const auto& fu : unique_pairs)
      km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(fu.first)][static_cast<std::size_t>(fu.second)] =
          Matrix<C>(nl, nl, C{});

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const int tid = omp_get_thread_num();
    auto& km_t = km_part[static_cast<std::size_t>(tid)];
    const double* slice = eri.eri3_L.data() + p * nl * nl;

    std::array<Matrix<C>, 2> tmp;
    for (int f = 0; f < 2; ++f) {
      Matrix<double> tmp_re(n_occ, nl, 0.0), tmp_im(n_occ, nl, 0.0);
      if (n_occ > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_occ), static_cast<int>(nl),
                    static_cast<int>(nl), 1.0, c_t_re[static_cast<std::size_t>(f)].data(), static_cast<int>(nl),
                    slice, static_cast<int>(nl), 0.0, tmp_re.data(), static_cast<int>(nl));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_occ), static_cast<int>(nl),
                    static_cast<int>(nl), 1.0, c_t_im[static_cast<std::size_t>(f)].data(), static_cast<int>(nl),
                    slice, static_cast<int>(nl), 0.0, tmp_im.data(), static_cast<int>(nl));
      }
      Matrix<C> tmp_f(n_occ, nl);
      for (std::size_t i = 0; i < n_occ; ++i)
        for (std::size_t a = 0; a < nl; ++a) tmp_f(i, a) = C(tmp_re(i, a), tmp_im(i, a));
      tmp[static_cast<std::size_t>(f)] = std::move(tmp_f);
    }

    const C alpha(1.0, 0.0), beta(1.0, 0.0);
    for (const auto& fu : unique_pairs) {
      if (n_occ == 0) continue;
      const std::size_t f = static_cast<std::size_t>(fu.first), u = static_cast<std::size_t>(fu.second);
      cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(nl),
                  static_cast<int>(n_occ), &alpha, tmp[f].data(), static_cast<int>(nl), tmp[u].data(),
                  static_cast<int>(nl), &beta, km_t[f][u].data(), static_cast<int>(nl));
    }
  }

  std::array<std::array<Matrix<C>, 2>, 2> km;
  for (const auto& fu : unique_pairs) {
    const std::size_t f = static_cast<std::size_t>(fu.first), u = static_cast<std::size_t>(fu.second);
    km[f][u] = Matrix<C>(nl, nl, C{});
    for (int t = 0; t < n_threads; ++t) addInto(km[f][u], km_part[static_cast<std::size_t>(t)][f][u]);
    if (f != u) km[u][f] = dagger(km[f][u]);
  }

  Matrix<C> fock(n, n, C{});
  for (std::size_t a = 0; a < nl; ++a)
    for (std::size_t b = 0; b < nl; ++b) {
      const C j_val(j_flat[a * nl + b], 0.0);
      fock(offset[0] + a, offset[0] + b) += j_val;
      fock(offset[1] + a, offset[1] + b) += j_val;
    }
  for (int f = 0; f < 2; ++f)
    for (int u = 0; u < 2; ++u) {
      const Matrix<C>& km_fu = km[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)];
      for (std::size_t a = 0; a < nl; ++a)
        for (std::size_t b = 0; b < nl; ++b)
          fock(offset[static_cast<std::size_t>(f)] + a, offset[static_cast<std::size_t>(u)] + b) -= km_fu(a, b);
    }
  return h_x2c + fock;
}

}  // namespace rerdmft
