#include "DiracKinetic.h"

#include <array>
#include <cstddef>

#include "Integrals.h"
#include "NablaIntegrals.h"

namespace rerdmft {

Matrix<std::complex<double>> diracKineticMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, double speed_of_light) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> T(n, n, std::complex<double>(0.0, 0.0));

  const std::complex<double> i_unit(0.0, 1.0);
  const std::complex<double> factor = -i_unit * speed_of_light;

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::array<double, 3> d = nablaIntegral(large_basis[a], small_basis[b]);
      const double dx = d[0];
      const double dy = d[1];
      const double dz = d[2];

      // sigma_x=[[0,1],[1,0]], sigma_y=[[0,-i],[i,0]], sigma_z=[[1,0],[0,-1]]
      const std::complex<double> t_alpha_alpha = factor * dz;
      const std::complex<double> t_alpha_beta = factor * (dx - i_unit * dy);
      const std::complex<double> t_beta_alpha = factor * (dx + i_unit * dy);
      const std::complex<double> t_beta_beta = factor * (-dz);

      T(off_large_alpha + a, off_small_alpha + b) = t_alpha_alpha;
      T(off_large_alpha + a, off_small_beta + b) = t_alpha_beta;
      T(off_large_beta + a, off_small_alpha + b) = t_beta_alpha;
      T(off_large_beta + a, off_small_beta + b) = t_beta_beta;

      // Hermitian conjugate (Small,Large) blocks.
      T(off_small_alpha + b, off_large_alpha + a) = std::conj(t_alpha_alpha);
      T(off_small_beta + b, off_large_alpha + a) = std::conj(t_alpha_beta);
      T(off_small_alpha + b, off_large_beta + a) = std::conj(t_beta_alpha);
      T(off_small_beta + b, off_large_beta + a) = std::conj(t_beta_beta);
    }
  }

  return T;
}

Matrix<std::complex<double>> diracRestEnergyMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, double speed_of_light) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> m(n, n, std::complex<double>(0.0, 0.0));

  const Matrix<double> s_large = overlapMatrix(large_basis);
  const Matrix<double> s_small = overlapMatrix(small_basis);
  const double minus_two_c2 = -2.0 * speed_of_light * speed_of_light;

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_large; ++b) {
      const std::complex<double> value(s_large(a, b), 0.0);
      m(off_large_alpha + a, off_large_alpha + b) = value;
      m(off_large_beta + a, off_large_beta + b) = value;
    }
  }
  for (std::size_t a = 0; a < n_small; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::complex<double> value(minus_two_c2 * s_small(a, b), 0.0);
      m(off_small_alpha + a, off_small_alpha + b) = value;
      m(off_small_beta + a, off_small_beta + b) = value;
    }
  }

  return m;
}

}  // namespace rerdmft
