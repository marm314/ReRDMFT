#include "SpinorRotation.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "LinearAlgebra.h"

namespace rerdmft {

template <typename T>
Matrix<T> spinorRotationMatrix(const Matrix<T>& kappa) {
  const std::size_t n = kappa.rows();
  if (kappa.cols() != n) {
    throw std::runtime_error("spinorRotationMatrix: kappa is not square");
  }

  // Build i*kappa (complex Hermitian) while checking, along the way,
  // that kappa itself is anti-Hermitian: kappa_ij == -conj(kappa_ji),
  // i.e. kappa_ij + conj(kappa_ji) == 0. std::complex<double>(kappa(i,j))
  // promotes a real T uniformly (real-to-complex constructor) or copies
  // an already-complex one, so this one loop handles both T = double
  // and T = std::complex<double> without branching.
  constexpr double kAntiHermitianTolerance = 1e-8;
  double max_residual = 0.0;
  Matrix<std::complex<double>> i_kappa(n, n);
  const std::complex<double> i_unit(0.0, 1.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const std::complex<double> kij(kappa(i, j));
      const std::complex<double> kji_conj = std::conj(std::complex<double>(kappa(j, i)));
      max_residual = std::max(max_residual, std::abs(kij + kji_conj));
      i_kappa(i, j) = i_unit * kij;
    }
  }
  if (max_residual > kAntiHermitianTolerance) {
    throw std::runtime_error(
        "spinorRotationMatrix: kappa is not anti-Hermitian to within tolerance (max "
        "|kappa_ij + conj(kappa_ji)| = " +
        std::to_string(max_residual) + ")");
  }

  const HermitianEigenResult eig = diagonalizeHermitian(i_kappa);
  const Matrix<std::complex<double>>& v = eig.eigenvectors;

  // exp(i*w_k), precomputed once per eigenvalue rather than per (i,j).
  std::vector<std::complex<double>> phase(n);
  for (std::size_t k = 0; k < n; ++k) {
    phase[k] = std::complex<double>(std::cos(eig.eigenvalues[k]), std::sin(eig.eigenvalues[k]));
  }

  // U_rot = V diag(exp(i*w)) V^dagger:
  //   U_rot(i,j) = sum_k V(i,k) * exp(i*w_k) * conj(V(j,k)).
  Matrix<std::complex<double>> u_rot(n, n, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < n; ++k) {
        sum += v(i, k) * phase[k] * std::conj(v(j, k));
      }
      u_rot(i, j) = sum;
    }
  }

  if constexpr (std::is_same_v<T, double>) {
    constexpr double kRealnessTolerance = 1e-8;
    double max_imag = 0.0;
    Matrix<double> result(n, n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        max_imag = std::max(max_imag, std::abs(u_rot(i, j).imag()));
        result(i, j) = u_rot(i, j).real();
      }
    }
    if (max_imag > kRealnessTolerance) {
      throw std::runtime_error(
          "spinorRotationMatrix: exp(-kappa) has an unexpectedly large imaginary part for a "
          "real kappa (max |Im| = " +
          std::to_string(max_imag) + ") -- kappa may not truly be antisymmetric");
    }
    return result;
  } else {
    return u_rot;
  }
}

template Matrix<double> spinorRotationMatrix(const Matrix<double>& kappa);
template Matrix<std::complex<double>> spinorRotationMatrix(
    const Matrix<std::complex<double>>& kappa);

}  // namespace rerdmft
