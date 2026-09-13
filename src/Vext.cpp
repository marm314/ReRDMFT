#include "Vext.h"

#include <cstddef>

#include "NuclearAttraction.h"

namespace rerdmft {

Matrix<std::complex<double>> vextMatrix(const std::vector<BasisFunction>& large_basis,
                                         const std::vector<BasisFunction>& small_basis,
                                         const std::vector<Atom>& geometry) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> vext(n, n, std::complex<double>(0.0, 0.0));

  const Matrix<double> v_large = nuclearAttractionMatrix(large_basis, geometry);
  const Matrix<double> v_small = nuclearAttractionMatrix(small_basis, geometry);

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_large; ++b) {
      const std::complex<double> value(v_large(a, b), 0.0);
      vext(off_large_alpha + a, off_large_alpha + b) = value;
      vext(off_large_beta + a, off_large_beta + b) = value;
    }
  }
  for (std::size_t a = 0; a < n_small; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::complex<double> value(v_small(a, b), 0.0);
      vext(off_small_alpha + a, off_small_alpha + b) = value;
      vext(off_small_beta + a, off_small_beta + b) = value;
    }
  }

  return vext;
}

}  // namespace rerdmft
