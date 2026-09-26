#include "ClosedShellSpinOrbitals.h"

#include <stdexcept>

namespace rerdmft {

namespace {

std::size_t spatialIndex(std::size_t p, std::size_t n_spatial) { return p % n_spatial; }
int spinOf(std::size_t p, std::size_t n_spatial) { return static_cast<int>(p / n_spatial); }

}  // namespace

Matrix<double> closedShellSpinOrbitalOneElectron(const Matrix<double>& h_mo,
                                                  std::size_t n_spatial) {
  if (h_mo.rows() != n_spatial || h_mo.cols() != n_spatial) {
    throw std::runtime_error(
        "closedShellSpinOrbitalOneElectron: h_mo dimensions do not match n_spatial");
  }
  const std::size_t n_spin = 2 * n_spatial;
  Matrix<double> h_spin(n_spin, n_spin, 0.0);
  for (std::size_t p = 0; p < n_spin; ++p) {
    for (std::size_t q = 0; q < n_spin; ++q) {
      if (spinOf(p, n_spatial) != spinOf(q, n_spatial)) continue;
      h_spin(p, q) = h_mo(spatialIndex(p, n_spatial), spatialIndex(q, n_spatial));
    }
  }
  return h_spin;
}

Tensor4<double> closedShellSpinOrbitalTwoElectron(const Tensor4<double>& eri_mo_physics,
                                                   std::size_t n_spatial) {
  if (eri_mo_physics.dim0() != n_spatial || eri_mo_physics.dim1() != n_spatial ||
      eri_mo_physics.dim2() != n_spatial || eri_mo_physics.dim3() != n_spatial) {
    throw std::runtime_error(
        "closedShellSpinOrbitalTwoElectron: eri_mo_physics dimensions do not match n_spatial");
  }
  const std::size_t n_spin = 2 * n_spatial;
  Tensor4<double> eri_spin(n_spin, n_spin, n_spin, n_spin, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t a = 0; a < n_spin; ++a) {
    for (std::size_t b = 0; b < n_spin; ++b) {
      for (std::size_t c = 0; c < n_spin; ++c) {
        if (spinOf(a, n_spatial) != spinOf(c, n_spatial)) continue;
        for (std::size_t d = 0; d < n_spin; ++d) {
          if (spinOf(b, n_spatial) != spinOf(d, n_spatial)) continue;
          eri_spin(a, b, c, d) = eri_mo_physics(spatialIndex(a, n_spatial),
                                                 spatialIndex(b, n_spatial),
                                                 spatialIndex(c, n_spatial),
                                                 spatialIndex(d, n_spatial));
        }
      }
    }
  }
  return eri_spin;
}

SymmetricEri<double> closedShellSpinOrbitalTwoElectron(const SymmetricEri<double>& eri_mo_physics,
                                                        std::size_t n_spatial) {
  if (eri_mo_physics.dim0() != n_spatial) {
    throw std::runtime_error(
        "closedShellSpinOrbitalTwoElectron: eri_mo_physics dimension does not match n_spatial");
  }
  const std::size_t n_spin = 2 * n_spatial;
  SymmetricEri<double> eri_spin(n_spin);
  for (std::size_t a = 0; a < n_spin; ++a) {
    for (std::size_t c = a; c < n_spin; ++c) {
      if (spinOf(a, n_spatial) != spinOf(c, n_spatial)) continue;
      for (std::size_t b = 0; b < n_spin; ++b) {
        for (std::size_t d = 0; d < n_spin; ++d) {
          if (spinOf(b, n_spatial) != spinOf(d, n_spatial)) continue;
          eri_spin.set(a, b, c, d, eri_mo_physics(spatialIndex(a, n_spatial), spatialIndex(b, n_spatial),
                                                   spatialIndex(c, n_spatial), spatialIndex(d, n_spatial)));
        }
      }
    }
  }
  return eri_spin;
}

Matrix<double> closedShellSpinOrbitalDensity(std::size_t n_spatial, int n_occupied_spatial) {
  if (n_occupied_spatial < 0 || static_cast<std::size_t>(n_occupied_spatial) > n_spatial) {
    throw std::runtime_error(
        "closedShellSpinOrbitalDensity: n_occupied_spatial out of range for n_spatial");
  }
  const std::size_t n_spin = 2 * n_spatial;
  Matrix<double> d(n_spin, n_spin, 0.0);
  for (std::size_t p = 0; p < n_spin; ++p) {
    if (spatialIndex(p, n_spatial) < static_cast<std::size_t>(n_occupied_spatial)) {
      d(p, p) = 1.0;
    }
  }
  return d;
}

}  // namespace rerdmft
