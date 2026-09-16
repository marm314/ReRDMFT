#include "X2C_decoupling.h"

#include <algorithm>
#include <cmath>

#include "RkbDensityMatrix.h"
#include "RkbOrthogonalization.h"

namespace rerdmft {

X2CDecouplingResult x2cDecoupling(const Matrix<std::complex<double>>& h_rkb,
                                   const Matrix<double>& /*s_large*/,
                                   const Matrix<double>& /*x_large*/,
                                   const Matrix<std::complex<double>>& /*s_small*/,
                                   const Matrix<std::complex<double>>& s_full,
                                   const Matrix<std::complex<double>>& x_full) {
  X2CDecouplingResult result;
  result.h_rkb_ortho = hRkbOrthoMatrix(h_rkb, x_full);
  result.eigen = diagonalizeHermitian(result.h_rkb_ortho);
  result.c_tmp = rkbCoefficientMatrix(x_full, result.eigen.eigenvectors);

  // Verify C_tmp solves the ORIGINAL generalized eigenvalue problem
  // H_RKB * C_tmp = S_full * C_tmp * E, not just the orthonormalized
  // one already solved above by construction.
  const auto hc = h_rkb * result.c_tmp;
  const auto sc = s_full * result.c_tmp;
  const auto& eigenvalues = result.eigen.eigenvalues;
  double max_residual = 0.0;
  for (std::size_t c = 0; c < eigenvalues.size(); ++c) {
    for (std::size_t i = 0; i < hc.rows(); ++i) {
      const auto residual = hc(i, c) - eigenvalues[c] * sc(i, c);
      max_residual = std::max(max_residual, std::abs(residual));
    }
  }
  result.max_generalized_eigenproblem_residual = max_residual;

  return result;
}

}  // namespace rerdmft
