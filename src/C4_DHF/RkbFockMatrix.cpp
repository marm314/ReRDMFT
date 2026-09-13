#include "RkbFockMatrix.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

// Which of the four RKB spinor blocks (Large-alpha, Large-beta, RKB-Small-
// alpha-partner, RKB-Small-beta-partner -- see RkbHamiltonian.h) index i
// falls in, given each block is n_large wide.
int flavorOf(std::size_t i, std::size_t n_large) {
  return static_cast<int>(i / n_large);
}

}  // namespace

Matrix<std::complex<double>> rkbFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                            const RkbTwoElectronTensor& eri,
                                            const Matrix<std::complex<double>>& density_matrix) {
  const std::size_t n = h_rkb.rows();
  if (h_rkb.cols() != n) {
    throw std::runtime_error("rkbFockMatrix: H_RKB is not square");
  }
  if (eri.dim() != n) {
    throw std::runtime_error("rkbFockMatrix: ERI tensor dimension is inconsistent with H_RKB");
  }
  if (density_matrix.rows() != n || density_matrix.cols() != n) {
    throw std::runtime_error(
        "rkbFockMatrix: density matrix dimensions are inconsistent with H_RKB");
  }
  if (n % 4 != 0) {
    throw std::runtime_error("rkbFockMatrix: RKB basis dimension must be a multiple of 4");
  }
  const std::size_t n_large = n / 4;

  Matrix<std::complex<double>> fock(n, n, std::complex<double>(0.0, 0.0));

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

          // Hartree: <p q|r s> is already exactly zero unless q,s share a
          // flavor, so no explicit split is needed here.
          hartree += p_sq * eri(p, q, r, s);

          // Exchange: <p q|s r> requires p,s to share a flavor (electron-1
          // pair) and q,r to share a flavor (electron-2 pair); whether q
          // and s themselves share a flavor is what separates the
          // "same-spin" and "opposite-spin" contributions.
          const std::complex<double> exch = p_sq * eri(p, q, s, r);
          if (flavorOf(s, n_large) == flavor_q) {
            exchange_same += exch;
          } else {
            exchange_opposite += exch;
          }
        }
      }

      fock(p, r) = h_rkb(p, r) + hartree - exchange_same - exchange_opposite;
    }
  }

  return fock;
}

}  // namespace rerdmft
