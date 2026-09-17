#include "RkbTransformation.h"

#include <cstddef>

#include "Integrals.h"
#include "LinearAlgebra.h"
#include "NablaIntegrals.h"

namespace rerdmft {

namespace {

Matrix<std::complex<double>> toComplex(const Matrix<double>& m) {
  Matrix<std::complex<double>> result(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows(); ++i) {
    for (std::size_t j = 0; j < m.cols(); ++j) {
      result(i, j) = std::complex<double>(m(i, j), 0.0);
    }
  }
  return result;
}

}  // namespace

Matrix<std::complex<double>> rkbCoefficients(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();

  // Dk[p][t] = <Large_p|d/dx_k|Small_t>.
  Matrix<double> dx(n_large, n_small);
  Matrix<double> dy(n_large, n_small);
  Matrix<double> dz(n_large, n_small);
  for (std::size_t p = 0; p < n_large; ++p) {
    for (std::size_t t = 0; t < n_small; ++t) {
      const auto d = nablaIntegral(large_basis[p], small_basis[t]);
      dx(p, t) = d[0];
      dy(p, t) = d[1];
      dz(p, t) = d[2];
    }
  }

  // M_tp = <Small_t|sigma.p|Large_p>, p = -i grad_r, so
  // <Small_t|sigma.p|Large_p> = -i sum_k sigma_k <Small_t|d/dx_k|Large_p>
  //                            = -i sum_k sigma_k (-Dk[p][t])   (integration by parts)
  //                            =  i sum_k sigma_k Dk[p][t].
  // With sigma_x=[[0,1],[1,0]], sigma_y=[[0,-i],[i,0]], sigma_z=[[1,0],[0,-1]],
  // built here transposed (rows=Large p, cols=Small t) to match M's usage
  // as the left factor of M * S^-1.
  const std::complex<double> i_unit(0.0, 1.0);
  Matrix<std::complex<double>> m_alpha_alpha(n_large, n_small);
  Matrix<std::complex<double>> m_alpha_beta(n_large, n_small);
  Matrix<std::complex<double>> m_beta_alpha(n_large, n_small);
  Matrix<std::complex<double>> m_beta_beta(n_large, n_small);
  for (std::size_t p = 0; p < n_large; ++p) {
    for (std::size_t t = 0; t < n_small; ++t) {
      m_alpha_alpha(p, t) = i_unit * dz(p, t);
      m_alpha_beta(p, t) = i_unit * (dx(p, t) - i_unit * dy(p, t));
      m_beta_alpha(p, t) = i_unit * (dx(p, t) + i_unit * dy(p, t));
      m_beta_beta(p, t) = i_unit * (-dz(p, t));
    }
  }

  const Matrix<double> s_small = overlapMatrix(small_basis);
  const Matrix<double> s_small_inv = invert(s_small);
  const Matrix<std::complex<double>> s_small_inv_c = toComplex(s_small_inv);

  const Matrix<std::complex<double>> c_alpha_alpha = m_alpha_alpha * s_small_inv_c;
  const Matrix<std::complex<double>> c_alpha_beta = m_alpha_beta * s_small_inv_c;
  const Matrix<std::complex<double>> c_beta_alpha = m_beta_alpha * s_small_inv_c;
  const Matrix<std::complex<double>> c_beta_beta = m_beta_beta * s_small_inv_c;

  Matrix<std::complex<double>> c(2 * n_large, 2 * n_small);
  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 0;
  const std::size_t off_small_beta = n_small;
  // m_alpha_beta/m_beta_alpha are the (row=Small-spin, col=Large-spin)
  // off-diagonal entries of the 2x2 sigma.p matrix (m_alpha_beta's row is
  // Small-alpha, column Large-beta; m_beta_alpha's row is Small-beta,
  // column Large-alpha) -- so, in the assembled C matrix (rows=Large spin,
  // cols=Small spin), m_alpha_beta belongs at (Large-beta, Small-alpha)
  // and m_beta_alpha at (Large-alpha, Small-beta). Cross-checked against
  // an independent reference implementation's MpSqL_me (M. Rodriguez-
  // Mayorga's m_relativistic.f90, MOLGW).
  for (std::size_t p = 0; p < n_large; ++p) {
    for (std::size_t t = 0; t < n_small; ++t) {
      c(off_large_alpha + p, off_small_alpha + t) = c_alpha_alpha(p, t);
      c(off_large_alpha + p, off_small_beta + t) = c_beta_alpha(p, t);
      c(off_large_beta + p, off_small_alpha + t) = c_alpha_beta(p, t);
      c(off_large_beta + p, off_small_beta + t) = c_beta_beta(p, t);
    }
  }

  return c;
}

}  // namespace rerdmft
