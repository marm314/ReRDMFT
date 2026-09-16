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

}  // namespace rerdmft
