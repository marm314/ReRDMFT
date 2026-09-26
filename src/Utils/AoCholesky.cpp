#include "AoCholesky.h"

#include <cblas.h>

#include <omp.h>

#include <stdexcept>

#include "BlasThreads.h"
#include "ElectronRepulsion.h"

namespace rerdmft {

namespace {

using C = std::complex<double>;

// C(m x n) (+)= op(A) * op(B), row-major, real. `beta` = 0 overwrites, 1 accumulates.
void dgemmRow(bool trans_a, bool trans_b, int m, int n, int k, const double* a, int lda, const double* b,
              int ldb, double beta, double* c, int ldc) {
  cblas_dgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans, m, n, k, 1.0,
              a, lda, b, ldb, beta, c, ldc);
}

void zgemmRow(bool conj_trans_a, int m, int n, int k, const C* a, int lda, const C* b, int ldb, double beta, C* c,
              int ldc) {
  const C alpha(1.0, 0.0), bt(beta, 0.0);
  cblas_zgemm(CblasRowMajor, conj_trans_a ? CblasConjTrans : CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a, lda, b,
              ldb, &bt, c, ldc);
}

void checkSquare(const AoCholesky& ao, std::size_t rows, const char* who) {
  if (ao.n != rows) throw std::runtime_error(std::string(who) + ": AO Cholesky dimension does not match the matrices");
}

}  // namespace

AoCholesky AoCholesky::fromPacked(const PackedTwoElectronTensor& eri, double threshold, CholeskyCheckReport* report) {
  AoCholesky ao;
  ao.n = eri.dim();
  ao.vectors = choleskyDecomposeEriChecked(eri, threshold, report);
  return ao;
}

Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const AoCholesky& ao, const Matrix<double>& p) {
  const std::size_t n = h_core.rows();
  checkSquare(ao, n, "nonRelFockMatrix");
  if (p.rows() != n || p.cols() != n) throw std::runtime_error("nonRelFockMatrix: density dimensions inconsistent");
  Matrix<double> j(n, n, 0.0), k(n, n, 0.0);
  // Per-thread partial sums (static schedule), merged in thread order below: the result must not depend on thread
  // timing (reproducible SCF/optimization runs).
  const int max_threads = omp_get_max_threads();
  std::vector<Matrix<double>> j_parts(static_cast<std::size_t>(max_threads)), k_parts(static_cast<std::size_t>(max_threads));
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
    j_parts[tid] = Matrix<double>(n, n, 0.0);
    k_parts[tid] = Matrix<double>(n, n, 0.0);
    Matrix<double>& j_loc = j_parts[tid];
    Matrix<double>& k_loc = k_parts[tid];
    Matrix<double> tmp(n, n, 0.0);
#pragma omp for schedule(static)
    for (std::size_t l = 0; l < ao.vectors.size(); ++l) {
      const Matrix<double>& b = ao.vectors[l];
      double t = 0.0;  // tr(B P) = sum_{qs} B(q,s) P(s,q)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t s = 0; s < n; ++s) t += b(q, s) * p(s, q);
      for (std::size_t i = 0; i < n * n; ++i) j_loc.data()[i] += t * b.data()[i];
      dgemmRow(false, false, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), p.data(), static_cast<int>(n),
               b.data(), static_cast<int>(n), 0.0, tmp.data(), static_cast<int>(n));                  // tmp = P B
      dgemmRow(false, false, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), b.data(), static_cast<int>(n),
               tmp.data(), static_cast<int>(n), 1.0, k_loc.data(), static_cast<int>(n));              // k += B P B
    }
  }
  for (std::size_t t = 0; t < j_parts.size(); ++t) {
    if (j_parts[t].rows() == 0) continue;
    for (std::size_t i = 0; i < n * n; ++i) {
      j.data()[i] += j_parts[t].data()[i];
      k.data()[i] += k_parts[t].data()[i];
    }
  }
  Matrix<double> fock(n, n, 0.0);
  for (std::size_t i = 0; i < n * n; ++i) fock.data()[i] = h_core.data()[i] + j.data()[i] - 0.5 * k.data()[i];
  return fock;
}

Matrix<C> x2cFockMatrix(const Matrix<C>& h_x2c, const AoCholesky& ao, const Matrix<C>& p) {
  const std::size_t n2 = h_x2c.rows();
  if (n2 % 2 != 0) throw std::runtime_error("x2cFockMatrix: basis dimension must be even (2*nLarge)");
  const std::size_t n = n2 / 2;
  checkSquare(ao, n, "x2cFockMatrix");
  if (p.rows() != n2 || p.cols() != n2) throw std::runtime_error("x2cFockMatrix: density dimensions inconsistent");
  // Spin blocks P_{st}(kappa,lambda) = P(s*n + kappa, t*n + lambda), contiguous n x n copies.
  Matrix<C> pb[2][2];
  for (int s = 0; s < 2; ++s)
    for (int t = 0; t < 2; ++t) {
      pb[s][t] = Matrix<C>(n, n);
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) pb[s][t](a, b) = p(s * n + a, t * n + b);
    }
  Matrix<C> jm(n, n, C{}), km[2][2];
  for (int s = 0; s < 2; ++s)
    for (int t = 0; t < 2; ++t) km[s][t] = Matrix<C>(n, n, C{});
  struct Part {
    Matrix<C> j;
    Matrix<C> k[2][2];
  };
  const int max_threads = omp_get_max_threads();
  std::vector<Part> parts(static_cast<std::size_t>(max_threads));  // per-thread partial sums, merged in thread order
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    Part& part = parts[static_cast<std::size_t>(omp_get_thread_num())];
    part.j = Matrix<C>(n, n, C{});
    for (int s = 0; s < 2; ++s)
      for (int t = 0; t < 2; ++t) part.k[s][t] = Matrix<C>(n, n, C{});
    Matrix<C>& j_loc = part.j;
    Matrix<C> bc(n, n), tmp(n, n);
#pragma omp for schedule(static)
    for (std::size_t l = 0; l < ao.vectors.size(); ++l) {
      const Matrix<double>& b = ao.vectors[l];
      for (std::size_t i = 0; i < n * n; ++i) bc.data()[i] = C(b.data()[i], 0.0);
      C t{};  // sum over spins of tr(B P_ss)
      for (int s = 0; s < 2; ++s)
        for (std::size_t q = 0; q < n; ++q)
          for (std::size_t r = 0; r < n; ++r) t += b(q, r) * pb[s][s](r, q);
      for (std::size_t i = 0; i < n * n; ++i) j_loc.data()[i] += t * b.data()[i];
      for (int s = 0; s < 2; ++s)
        for (int u = 0; u < 2; ++u) {
          zgemmRow(false, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), pb[s][u].data(), static_cast<int>(n),
                   bc.data(), static_cast<int>(n), 0.0, tmp.data(), static_cast<int>(n));                 // P_su B
          zgemmRow(false, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), bc.data(), static_cast<int>(n),
                   tmp.data(), static_cast<int>(n), 1.0, part.k[s][u].data(), static_cast<int>(n));        // += B P_su B
        }
    }
  }
  for (const Part& part : parts) {
    if (part.j.rows() == 0) continue;
    for (std::size_t i = 0; i < n * n; ++i) jm.data()[i] += part.j.data()[i];
    for (int s = 0; s < 2; ++s)
      for (int u = 0; u < 2; ++u)
        for (std::size_t i = 0; i < n * n; ++i) km[s][u].data()[i] += part.k[s][u].data()[i];
  }
  Matrix<C> fock = h_x2c;
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      for (int s = 0; s < 2; ++s) {
        fock(s * n + a, s * n + b) += jm(a, b);
        for (int u = 0; u < 2; ++u) fock(s * n + a, u * n + b) -= km[s][u](a, b);
      }
    }
  return fock;
}

CholeskyEri<double> aoCholeskyToMoSpinOrbital(const AoCholesky& ao, const Matrix<double>& c) {
  const std::size_t n = ao.n, nmo = c.cols();
  if (c.rows() != n) throw std::runtime_error("aoCholeskyToMoSpinOrbital: C row count does not match the AO dimension");
  std::vector<Matrix<double>> out(ao.vectors.size());
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    Matrix<double> tmp(n, nmo), bp(nmo, nmo);
#pragma omp for schedule(dynamic)
    for (std::size_t l = 0; l < ao.vectors.size(); ++l) {
      dgemmRow(false, false, static_cast<int>(n), static_cast<int>(nmo), static_cast<int>(n), ao.vectors[l].data(),
               static_cast<int>(n), c.data(), static_cast<int>(nmo), 0.0, tmp.data(), static_cast<int>(nmo));   // B C
      dgemmRow(true, false, static_cast<int>(nmo), static_cast<int>(nmo), static_cast<int>(n), c.data(),
               static_cast<int>(nmo), tmp.data(), static_cast<int>(nmo), 0.0, bp.data(), static_cast<int>(nmo)); // C^T B C
      Matrix<double> w(2 * nmo, 2 * nmo, 0.0);
      for (std::size_t x = 0; x < nmo; ++x)
        for (std::size_t y = 0; y < nmo; ++y) {
          w(x, y) = bp(x, y);
          w(nmo + x, nmo + y) = bp(x, y);
        }
      out[l] = std::move(w);
    }
  }
  return CholeskyEri<double>::fromVectors(out);
}

CholeskyEri<C> aoCholeskyToMoSpinor(const AoCholesky& ao, const Matrix<C>& c) {
  const std::size_t n = ao.n, nmo = c.cols();
  if (c.rows() != 2 * n) throw std::runtime_error("aoCholeskyToMoSpinor: C must have 2*n_AO rows ([alpha; beta] blocks)");
  std::vector<Matrix<C>> out(ao.vectors.size());
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    Matrix<C> bc(n, n), tmp(n, nmo), bp(nmo, nmo);
#pragma omp for schedule(dynamic)
    for (std::size_t l = 0; l < ao.vectors.size(); ++l) {
      for (std::size_t i = 0; i < n * n; ++i) bc.data()[i] = C(ao.vectors[l].data()[i], 0.0);
      for (int s = 0; s < 2; ++s) {
        const C* cs = c.data() + static_cast<std::size_t>(s) * n * nmo;  // alpha / beta AO block, n x nmo
        zgemmRow(false, static_cast<int>(n), static_cast<int>(nmo), static_cast<int>(n), bc.data(), static_cast<int>(n),
                 cs, static_cast<int>(nmo), 0.0, tmp.data(), static_cast<int>(nmo));                        // B C_s
        zgemmRow(true, static_cast<int>(nmo), static_cast<int>(nmo), static_cast<int>(n), cs, static_cast<int>(nmo),
                 tmp.data(), static_cast<int>(nmo), s == 0 ? 0.0 : 1.0, bp.data(), static_cast<int>(nmo));  // += C_s^+ B C_s
      }
      Matrix<C> w(nmo, nmo);
      for (std::size_t x = 0; x < nmo; ++x)
        for (std::size_t y = 0; y < nmo; ++y) w(x, y) = bp(y, x);  // W = B'^T
      out[l] = std::move(w);
    }
  }
  return CholeskyEri<C>::fromVectors(out);
}

}  // namespace rerdmft
