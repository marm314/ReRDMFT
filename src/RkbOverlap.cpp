#include "RkbOverlap.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "Integrals.h"
#include "NuclearAttraction.h"

namespace rerdmft {

namespace {

// Shared by rkbSmallOverlapMatrix and rkbSmallVextMatrix: RKB-transforms a
// scalar Small-AO matrix `m_small` (real, already block-diagonally repeated
// here across the alpha/beta spin blocks) via conj(C) * M_uKB * C^T.
Matrix<std::complex<double>> rkbTransformSmallScalarMatrix(
    const Matrix<double>& m_small, const Matrix<std::complex<double>>& rkb_coefficients,
    const char* caller) {
  const std::size_t n_small = m_small.rows();
  const std::size_t n_large2 = rkb_coefficients.rows();  // 2*nLarge
  const std::size_t n_small2 = rkb_coefficients.cols();  // 2*nSmall

  if (n_small2 != 2 * n_small) {
    throw std::runtime_error(std::string(caller) +
                              ": rkb_coefficients column count is inconsistent with small_basis");
  }

  // M_uKB, block-diagonal over the alpha/beta spin blocks.
  Matrix<std::complex<double>> m_ukb(n_small2, n_small2, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_small; ++i) {
    for (std::size_t j = 0; j < n_small; ++j) {
      const std::complex<double> value(m_small(i, j), 0.0);
      m_ukb(i, j) = value;
      m_ukb(n_small + i, n_small + j) = value;
    }
  }

  // conj(C): same shape as C (2*nLarge x 2*nSmall).
  Matrix<std::complex<double>> conj_c(n_large2, n_small2);
  for (std::size_t p = 0; p < n_large2; ++p) {
    for (std::size_t t = 0; t < n_small2; ++t) {
      conj_c(p, t) = std::conj(rkb_coefficients(p, t));
    }
  }

  // C^T: (2*nSmall x 2*nLarge).
  Matrix<std::complex<double>> c_transpose(n_small2, n_large2);
  for (std::size_t p = 0; p < n_large2; ++p) {
    for (std::size_t t = 0; t < n_small2; ++t) {
      c_transpose(t, p) = rkb_coefficients(p, t);
    }
  }

  return conj_c * (m_ukb * c_transpose);
}

}  // namespace

Matrix<std::complex<double>> rkbSmallOverlapMatrix(
    const std::vector<BasisFunction>& small_basis,
    const Matrix<std::complex<double>>& rkb_coefficients) {
  return rkbTransformSmallScalarMatrix(overlapMatrix(small_basis), rkb_coefficients,
                                        "rkbSmallOverlapMatrix");
}

Matrix<std::complex<double>> rkbSmallVextMatrix(const std::vector<BasisFunction>& small_basis,
                                                 const Matrix<std::complex<double>>& rkb_coefficients,
                                                 const std::vector<Atom>& geometry) {
  return rkbTransformSmallScalarMatrix(nuclearAttractionMatrix(small_basis, geometry),
                                        rkb_coefficients, "rkbSmallVextMatrix");
}

}  // namespace rerdmft
