#include "NonRelFockMatrixRi.h"

#include <cblas.h>
#include <omp.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

Matrix<double> dgemmMultiply(const Matrix<double>& a, const Matrix<double>& b) {
  Matrix<double> c(a.rows(), b.cols(), 0.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(a.rows()), static_cast<int>(b.cols()),
              static_cast<int>(a.cols()), 1.0, a.data(), static_cast<int>(a.cols()), b.data(),
              static_cast<int>(b.cols()), 0.0, c.data(), static_cast<int>(b.cols()));
  return c;
}

}  // namespace

Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const RiNonRelEriSource& eri,
                                 const Matrix<double>& density_matrix) {
  const std::size_t nl = eri.nl;
  if (density_matrix.rows() != nl || density_matrix.cols() != nl) {
    throw std::runtime_error("nonRelFockMatrix (RI): density_matrix dimension does not match nl");
  }
  const std::size_t n_aux = eri.eri3_L.rows();
  if (eri.eri3_L.cols() != nl * nl) {
    throw std::runtime_error("nonRelFockMatrix (RI): eri3_L column count does not match nl^2");
  }

  // --- Hartree/J: a single real contraction, P already includes the factor of 2. ---
  std::vector<double> p_flat(nl * nl);
  for (std::size_t mu = 0; mu < nl; ++mu)
    for (std::size_t nu = 0; nu < nl; ++nu) p_flat[mu * nl + nu] = density_matrix(mu, nu);

  std::vector<double> x_p(n_aux, 0.0);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri.eri3_L.data(),
              static_cast<int>(nl * nl), p_flat.data(), 1, 0.0, x_p.data(), 1);
  std::vector<double> j_flat(nl * nl, 0.0);
  cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(n_aux), static_cast<int>(nl * nl), 1.0, eri.eri3_L.data(),
              static_cast<int>(nl * nl), x_p.data(), 1, 0.0, j_flat.data(), 1);

  // --- Exchange/K: recover the (doubly-occupied) orbitals via eri.x_large/eri.overlap -- D = P/2
  // is the genuine idempotent object; see this file's header comment for the derivation. ---
  Matrix<double> sps = dgemmMultiply(dgemmMultiply(eri.overlap, density_matrix), eri.overlap);
  for (std::size_t i = 0; i < sps.rows() * sps.cols(); ++i) sps.data()[i] *= 0.5;  // D = P/2
  const Matrix<double> p_eff = dgemmMultiply(transpose(eri.x_large), dgemmMultiply(sps, eri.x_large));
  const SymmetricEigenResult eig = diagonalizeSymmetric(p_eff);
  std::vector<std::size_t> occ_idx;
  for (std::size_t i = 0; i < eig.eigenvalues.size(); ++i) {
    if (eig.eigenvalues[i] > 0.5) occ_idx.push_back(i);
  }
  const std::size_t n_occ = occ_idx.size();
  Matrix<double> y_occ(eri.x_large.cols(), n_occ);
  for (std::size_t i = 0; i < n_occ; ++i)
    for (std::size_t a = 0; a < eri.x_large.cols(); ++a) y_occ(a, i) = eig.eigenvectors(a, occ_idx[i]);
  const Matrix<double> c_occ = dgemmMultiply(eri.x_large, y_occ);  // nl x n_occ

  // c_t(i,q) = sqrt(2) * C_occ(q,i) -- folds the factor-of-2 occupancy into the half-transform.
  Matrix<double> c_t(n_occ, nl);
  for (std::size_t i = 0; i < n_occ; ++i)
    for (std::size_t q = 0; q < nl; ++q) c_t(i, q) = std::sqrt(2.0) * c_occ(q, i);

  const int n_threads = omp_get_max_threads();
  std::vector<Matrix<double>> k_part(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t) k_part[static_cast<std::size_t>(t)] = Matrix<double>(nl, nl, 0.0);

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    if (n_occ == 0) continue;
    const int tid = omp_get_thread_num();
    const double* slice = eri.eri3_L.data() + p * nl * nl;
    Matrix<double> tmp(n_occ, nl, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_occ), static_cast<int>(nl),
                static_cast<int>(nl), 1.0, c_t.data(), static_cast<int>(nl), slice, static_cast<int>(nl), 0.0,
                tmp.data(), static_cast<int>(nl));
    // k_part += tmp^T @ tmp : (nl x n_occ) @ (n_occ x nl) -> nl x nl.
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(nl), static_cast<int>(nl),
                static_cast<int>(n_occ), 1.0, tmp.data(), static_cast<int>(nl), tmp.data(), static_cast<int>(nl), 1.0,
                k_part[static_cast<std::size_t>(tid)].data(), static_cast<int>(nl));
  }
  Matrix<double> k(nl, nl, 0.0);
  for (int t = 0; t < n_threads; ++t)
    for (std::size_t i = 0; i < nl * nl; ++i) k.data()[i] += k_part[static_cast<std::size_t>(t)].data()[i];

  Matrix<double> fock(nl, nl);
  for (std::size_t a = 0; a < nl; ++a)
    for (std::size_t b = 0; b < nl; ++b) fock(a, b) = h_core(a, b) + j_flat[a * nl + b] - 0.5 * k(a, b);
  return fock;
}

}  // namespace rerdmft
