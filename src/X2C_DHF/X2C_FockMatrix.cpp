#include "X2C_FockMatrix.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

// Which of the two Large-component spin flavors (alpha/beta -- see
// RkbHamiltonian.h's own block ordering, restricted here to just the
// Large half) index i falls in, given each flavor is n_large wide.
int flavorOf(std::size_t i, std::size_t n_large) { return static_cast<int>(i / n_large); }

}  // namespace

Matrix<std::complex<double>> x2cFockMatrix(const Matrix<std::complex<double>>& h_x2c,
                                            const Tensor4<double>& eri,
                                            const Matrix<std::complex<double>>& density_matrix) {
  const std::size_t n = h_x2c.rows();
  if (h_x2c.cols() != n) {
    throw std::runtime_error("x2cFockMatrix: h_x2c is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("x2cFockMatrix: eri dimensions are inconsistent with h_x2c");
  }
  if (density_matrix.rows() != n || density_matrix.cols() != n) {
    throw std::runtime_error("x2cFockMatrix: density matrix dimensions are inconsistent with h_x2c");
  }
  if (n % 2 != 0) {
    throw std::runtime_error("x2cFockMatrix: basis dimension must be even (2*nLarge)");
  }
  const std::size_t n_large = n / 2;

  Matrix<std::complex<double>> fock(n, n, std::complex<double>(0.0, 0.0));

  // Each (p,r) owns its own disjoint output position and only reads the
  // shared, const `eri`/`density_matrix` -- safe to parallelize.
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t r = 0; r < n; ++r) {
      std::complex<double> hartree(0.0, 0.0);
      std::complex<double> exchange_same(0.0, 0.0);
      std::complex<double> exchange_opposite(0.0, 0.0);

      for (std::size_t q = 0; q < n; ++q) {
        const int flavor_q = flavorOf(q, n_large);
        for (std::size_t s = 0; s < n; ++s) {
          const std::complex<double> p_sq = density_matrix(s, q);
          if (p_sq == std::complex<double>(0.0, 0.0)) continue;

          // Hartree: <p q|r s> is already exactly zero unless q,s share
          // a spin, so no explicit split is needed here.
          hartree += p_sq * eri(p, q, r, s);

          // Exchange: <p q|s r> requires p,s to share a spin (electron-1
          // pair) and q,r to share a spin (electron-2 pair); whether q
          // and s themselves share a spin is what separates the
          // "same-spin" and "opposite-spin" contributions.
          const std::complex<double> exch = p_sq * eri(p, q, s, r);
          if (flavorOf(s, n_large) == flavor_q) {
            exchange_same += exch;
          } else {
            exchange_opposite += exch;
          }
        }
      }

      fock(p, r) = h_x2c(p, r) + hartree - exchange_same - exchange_opposite;
    }
  }

  return fock;
}

Matrix<std::complex<double>> x2cFockMatrix(const Matrix<std::complex<double>>& h_x2c,
                                            const PackedTwoElectronTensor& eri,
                                            const Matrix<std::complex<double>>& density_matrix) {
  const std::size_t n2 = h_x2c.rows();
  if (h_x2c.cols() != n2 || n2 % 2 != 0) {
    throw std::runtime_error("x2cFockMatrix: h_x2c must be square with an even dimension (2*nLarge)");
  }
  const std::size_t n = n2 / 2;
  if (eri.dim() != n) throw std::runtime_error("x2cFockMatrix: spatial eri dimension inconsistent with h_x2c");
  if (density_matrix.rows() != n2 || density_matrix.cols() != n2) {
    throw std::runtime_error("x2cFockMatrix: density matrix dimensions are inconsistent with h_x2c");
  }
  using C = std::complex<double>;
  Matrix<C> fock = h_x2c;
#pragma omp parallel for schedule(dynamic)
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      C hartree(0.0, 0.0);
      C exch[2][2] = {{C{}, C{}}, {C{}, C{}}};
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
          hartree += eri(m, nu, k, l) * (density_matrix(l, k) + density_matrix(n + l, n + k));
          const double x = eri(m, k, l, nu);
          for (int s = 0; s < 2; ++s)
            for (int t = 0; t < 2; ++t) exch[s][t] += x * density_matrix(s * n + k, t * n + l);
        }
      }
      for (int s = 0; s < 2; ++s) {
        fock(s * n + m, s * n + nu) += hartree;
        for (int t = 0; t < 2; ++t) fock(s * n + m, t * n + nu) -= exch[s][t];
      }
    }
  }
  return fock;
}

}  // namespace rerdmft
