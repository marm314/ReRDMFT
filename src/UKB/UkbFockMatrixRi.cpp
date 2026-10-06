#include "UkbFockMatrixRi.h"

#include <cblas.h>
#include <omp.h>

#include <algorithm>
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

Matrix<C> riFockTwoElectronDirect(std::size_t nl, std::size_t ns, const Matrix<double>& eri3_LL,
                                   const Matrix<double>& eri3_SS, const Matrix<C>& density_ukb,
                                   const Matrix<C>& c_occ_ukb, const std::vector<double>& occupations) {
  const std::size_t n_ukb = 2 * nl + 2 * ns;
  if (density_ukb.rows() != n_ukb || density_ukb.cols() != n_ukb) {
    throw std::runtime_error("riFockTwoElectronDirect: density_ukb dimension does not match 2*nLarge+2*nSmall");
  }
  if (c_occ_ukb.rows() != n_ukb || c_occ_ukb.cols() != occupations.size()) {
    throw std::runtime_error("riFockTwoElectronDirect: c_occ_ukb dimension does not match n_ukb x occupations.size()");
  }
  const std::size_t n_occ = occupations.size();
  const std::size_t n_aux = eri3_LL.rows();
  if (eri3_SS.rows() != n_aux) {
    throw std::runtime_error("riFockTwoElectronDirect: eri3_LL and eri3_SS do not share the same aux dimension");
  }
  if (eri3_LL.cols() != nl * nl || eri3_SS.cols() != ns * ns) {
    throw std::runtime_error("riFockTwoElectronDirect: eri3_LL/eri3_SS column count does not match nl^2/ns^2");
  }

  // Four flavors, SAME storage order/offsets as UkbFockMatrixDirect.h's ukbFockTwoElectronDirect:
  // 0=Large-alpha, 1=Large-beta, 2=Small-alpha, 3=Small-beta.
  const std::array<std::size_t, 4> offset = {0, nl, 2 * nl, 2 * nl + ns};
  const std::array<std::size_t, 4> fsize = {nl, nl, ns, ns};

  // --- Hartree/J: total (spin-summed) density per spatial type, REAL part only (eri3 is real, P
  // is Hermitian -- same simplification MOLGW's setup_hartree_ri uses). ---
  std::vector<double> p_flat_L(nl * nl), p_flat_S(ns * ns);
  for (std::size_t mu = 0; mu < nl; ++mu)
    for (std::size_t nu = 0; nu < nl; ++nu)
      p_flat_L[mu * nl + nu] =
          (density_ukb(offset[0] + mu, offset[0] + nu) + density_ukb(offset[1] + mu, offset[1] + nu)).real();
  for (std::size_t mu = 0; mu < ns; ++mu)
    for (std::size_t nu = 0; nu < ns; ++nu)
      p_flat_S[mu * ns + nu] =
          (density_ukb(offset[2] + mu, offset[2] + nu) + density_ukb(offset[3] + mu, offset[3] + nu)).real();

  std::vector<double> x_p(n_aux, 0.0);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri3_LL.data(),
              static_cast<int>(nl * nl), p_flat_L.data(), 1, 0.0, x_p.data(), 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n_aux), static_cast<int>(ns * ns), 1.0, eri3_SS.data(),
              static_cast<int>(ns * ns), p_flat_S.data(), 1, 1.0, x_p.data(), 1);

  std::vector<double> j_flat_L(nl * nl, 0.0), j_flat_S(ns * ns, 0.0);
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri3_LL.data(),
              static_cast<int>(nl * nl), x_p.data(), 1, 0.0, j_flat_L.data(), 1);
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(n_aux), static_cast<int>(ns * ns), 1.0, eri3_SS.data(),
              static_cast<int>(ns * ns), x_p.data(), 1, 0.0, j_flat_S.data(), 1);

  // --- Exchange/K: half-transform using the CALLER-supplied occupied UKB-basis coefficients
  // (recovered robustly in rkbFockMatrix below, via the SCF loop's own x_full/s_full -- see this
  // file's header comment for why that, and not a fresh orthogonalization of the UKB overlap
  // here, is required). c_t_f(i,a) = conj(c_occ_ukb(offset[f]+a, i)) * sqrt(occupations[i]), split
  // into real/imaginary parts so the half-transform against the REAL eri3 tensors can use plain
  // DGEMM.
  std::array<Matrix<double>, 4> c_t_re, c_t_im;
  for (int f = 0; f < 4; ++f) {
    c_t_re[static_cast<std::size_t>(f)] = Matrix<double>(n_occ, fsize[static_cast<std::size_t>(f)]);
    c_t_im[static_cast<std::size_t>(f)] = Matrix<double>(n_occ, fsize[static_cast<std::size_t>(f)]);
    for (std::size_t i = 0; i < n_occ; ++i) {
      const double sqrt_occ = std::sqrt(occupations[i]);
      for (std::size_t a = 0; a < fsize[static_cast<std::size_t>(f)]; ++a) {
        const C coeff = c_occ_ukb(offset[static_cast<std::size_t>(f)] + a, i);
        c_t_re[static_cast<std::size_t>(f)](i, a) = coeff.real() * sqrt_occ;
        c_t_im[static_cast<std::size_t>(f)](i, a) = -coeff.imag() * sqrt_occ;
      }
    }
  }

  // Unique (f,u) blocks with f<=u -- K_uf = K_fu^dagger, not independently built.
  const std::vector<std::pair<int, int>> unique_pairs = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {0, 1},
                                                           {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  const int n_threads = omp_get_max_threads();
  std::vector<std::array<std::array<Matrix<C>, 4>, 4>> km_part(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t)
    for (const auto& fu : unique_pairs) {
      km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(fu.first)][static_cast<std::size_t>(fu.second)] =
          Matrix<C>(fsize[static_cast<std::size_t>(fu.first)], fsize[static_cast<std::size_t>(fu.second)], C{});
    }

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const int tid = omp_get_thread_num();
    auto& km_t = km_part[static_cast<std::size_t>(tid)];

    const double* slice[4] = {eri3_LL.data() + p * nl * nl, eri3_LL.data() + p * nl * nl,
                               eri3_SS.data() + p * ns * ns, eri3_SS.data() + p * ns * ns};

    std::array<Matrix<C>, 4> tmp;
    for (int f = 0; f < 4; ++f) {
      const std::size_t fs = fsize[static_cast<std::size_t>(f)];
      Matrix<double> tmp_re(n_occ, fs, 0.0), tmp_im(n_occ, fs, 0.0);
      if (n_occ > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_occ), static_cast<int>(fs),
                    static_cast<int>(fs), 1.0, c_t_re[static_cast<std::size_t>(f)].data(), static_cast<int>(fs),
                    slice[f], static_cast<int>(fs), 0.0, tmp_re.data(), static_cast<int>(fs));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_occ), static_cast<int>(fs),
                    static_cast<int>(fs), 1.0, c_t_im[static_cast<std::size_t>(f)].data(), static_cast<int>(fs),
                    slice[f], static_cast<int>(fs), 0.0, tmp_im.data(), static_cast<int>(fs));
      }
      Matrix<C> tmp_f(n_occ, fs);
      for (std::size_t i = 0; i < n_occ; ++i)
        for (std::size_t a = 0; a < fs; ++a) tmp_f(i, a) = C(tmp_re(i, a), tmp_im(i, a));
      tmp[static_cast<std::size_t>(f)] = std::move(tmp_f);
    }

    const C alpha(1.0, 0.0), beta(1.0, 0.0);
    for (const auto& fu : unique_pairs) {
      const std::size_t f = static_cast<std::size_t>(fu.first), u = static_cast<std::size_t>(fu.second);
      if (n_occ == 0) continue;
      cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, static_cast<int>(fsize[f]), static_cast<int>(fsize[u]),
                  static_cast<int>(n_occ), &alpha, tmp[f].data(), static_cast<int>(fsize[f]), tmp[u].data(),
                  static_cast<int>(fsize[u]), &beta, km_t[f][u].data(), static_cast<int>(fsize[u]));
    }
  }

  std::array<std::array<Matrix<C>, 4>, 4> km;
  for (const auto& fu : unique_pairs) {
    const std::size_t f = static_cast<std::size_t>(fu.first), u = static_cast<std::size_t>(fu.second);
    km[f][u] = Matrix<C>(fsize[f], fsize[u], C{});
    for (int t = 0; t < n_threads; ++t) addInto(km[f][u], km_part[static_cast<std::size_t>(t)][f][u]);
    if (f != u) km[u][f] = dagger(km[f][u]);
  }

  Matrix<C> fock(n_ukb, n_ukb, C{});
  for (std::size_t a = 0; a < nl; ++a)
    for (std::size_t b = 0; b < nl; ++b) {
      const C j_val(j_flat_L[a * nl + b], 0.0);
      fock(offset[0] + a, offset[0] + b) += j_val;
      fock(offset[1] + a, offset[1] + b) += j_val;
    }
  for (std::size_t a = 0; a < ns; ++a)
    for (std::size_t b = 0; b < ns; ++b) {
      const C j_val(j_flat_S[a * ns + b], 0.0);
      fock(offset[2] + a, offset[2] + b) += j_val;
      fock(offset[3] + a, offset[3] + b) += j_val;
    }
  for (int f = 0; f < 4; ++f)
    for (int u = 0; u < 4; ++u) {
      const Matrix<C>& km_fu = km[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)];
      for (std::size_t a = 0; a < fsize[static_cast<std::size_t>(f)]; ++a)
        for (std::size_t b = 0; b < fsize[static_cast<std::size_t>(u)]; ++b)
          fock(offset[static_cast<std::size_t>(f)] + a, offset[static_cast<std::size_t>(u)] + b) -= km_fu(a, b);
    }
  return fock;
}

Matrix<C> rkbFockMatrix(const Matrix<C>& h_rkb, const RiDirectEriSource& eri, const Matrix<C>& density_matrix) {
  // Recover the occupied RKB-basis orbitals robustly, via the SAME x_full/s_full the SCF loop
  // itself already uses every iteration to diagonalize the Fock matrix -- NOT a fresh
  // orthogonalization of the (rank-deficient) UKB AO overlap (see riFockTwoElectronDirect's
  // header comment for why that is required): x_full^dagger s_full x_full = I already holds to
  // whatever precision the SCF loop already trusts, and density_matrix was built FROM x_full in
  // the first place (c_dhf = x_full * eigenvectors, C4_DHF.cpp), so it has zero support outside
  // x_full's span -- P_eff_rkb = X_full^dagger (S_full density_matrix S_full) X_full is therefore
  // a genuine plain-matrix projector (same derivation as the header comment's Y Y^dagger
  // identity, with S_full/x_full in place of a from-scratch S/X), safe to eigendecompose directly.
  const Matrix<C> sps_rkb = zgemmMultiply(zgemmMultiply(eri.s_full, density_matrix), eri.s_full);
  const Matrix<C> p_eff_rkb = zgemmMultiply(dagger(eri.x_full), zgemmMultiply(sps_rkb, eri.x_full));
  const HermitianEigenResult eig = diagonalizeHermitian(p_eff_rkb);
  std::vector<std::size_t> occ_idx;
  for (std::size_t i = 0; i < eig.eigenvalues.size(); ++i) {
    if (eig.eigenvalues[i] > 0.5) occ_idx.push_back(i);
  }
  const std::size_t n_occ = occ_idx.size();
  Matrix<C> y_occ(eri.x_full.cols(), n_occ);
  std::vector<double> occupations(n_occ);
  for (std::size_t i = 0; i < n_occ; ++i) {
    occupations[i] = eig.eigenvalues[occ_idx[i]];
    for (std::size_t a = 0; a < eri.x_full.cols(); ++a) y_occ(a, i) = eig.eigenvectors(a, occ_idx[i]);
  }
  const Matrix<C> c_occ_rkb = zgemmMultiply(eri.x_full, y_occ);
  const Matrix<C> c_occ_ukb = eri.v_total * c_occ_rkb;

  const Matrix<C> p_ukb = eri.v_total * (density_matrix * dagger(eri.v_total));
  const Matrix<C> two_electron =
      riFockTwoElectronDirect(eri.nl, eri.ns, eri.eri3_LL, eri.eri3_SS, p_ukb, c_occ_ukb, occupations);
  return h_rkb + dagger(eri.v_total) * (two_electron * eri.v_total);
}

}  // namespace rerdmft
