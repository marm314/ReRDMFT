#include "RkbMoTransform.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

#include "Matrix.h"

namespace rerdmft {

namespace {

// T1(p,B,C,D) = sum_A conj(c(A,p)) * ao(A,B,C,D) -- transforms the
// first (bra) leg from the packed, physics-notation AO tensor into a
// dense intermediate.
Tensor4<std::complex<double>> transformLeg1(const RkbTwoElectronTensor& ao,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_ao = ao.dim();
  const std::size_t n_mo = c.cols();
  Tensor4<std::complex<double>> result(n_mo, n_ao, n_ao, n_ao, std::complex<double>(0.0, 0.0));
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo; ++p) {
    for (std::size_t bb = 0; bb < n_ao; ++bb) {
      for (std::size_t cc = 0; cc < n_ao; ++cc) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          std::complex<double> sum(0.0, 0.0);
          for (std::size_t aa = 0; aa < n_ao; ++aa) {
            sum += std::conj(c(aa, p)) * ao(aa, bb, cc, dd);
          }
          result(p, bb, cc, dd) = sum;
        }
      }
    }
  }
  return result;
}

// T2(p,q,C,D) = sum_B conj(c(B,q)) * t1(p,B,C,D) -- second (bra) leg,
// also conjugated.
Tensor4<std::complex<double>> transformLeg2(const Tensor4<std::complex<double>>& t1,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t1.dim0();
  const std::size_t n_ao = t1.dim1();
  const std::size_t n_mo = c.cols();
  Tensor4<std::complex<double>> result(n_mo1, n_mo, n_ao, n_ao, std::complex<double>(0.0, 0.0));
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo; ++q) {
      for (std::size_t cc = 0; cc < n_ao; ++cc) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          std::complex<double> sum(0.0, 0.0);
          for (std::size_t bb = 0; bb < n_ao; ++bb) {
            sum += std::conj(c(bb, q)) * t1(p, bb, cc, dd);
          }
          result(p, q, cc, dd) = sum;
        }
      }
    }
  }
  return result;
}

// T3(p,q,r,D) = sum_C c(C,r) * t2(p,q,C,D) -- first ket leg, NOT
// conjugated.
Tensor4<std::complex<double>> transformLeg3(const Tensor4<std::complex<double>>& t2,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t2.dim0();
  const std::size_t n_mo2 = t2.dim1();
  const std::size_t n_ao = t2.dim2();
  const std::size_t n_mo = c.cols();
  Tensor4<std::complex<double>> result(n_mo1, n_mo2, n_mo, n_ao, std::complex<double>(0.0, 0.0));
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo2; ++q) {
      for (std::size_t r = 0; r < n_mo; ++r) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          std::complex<double> sum(0.0, 0.0);
          for (std::size_t cc = 0; cc < n_ao; ++cc) {
            sum += c(cc, r) * t2(p, q, cc, dd);
          }
          result(p, q, r, dd) = sum;
        }
      }
    }
  }
  return result;
}

// T4(p,q,r,s) = sum_D c(D,s) * t3(p,q,r,D) -- second ket leg, NOT
// conjugated. T4(p,q,r,s) == <p q|r s>_MO directly (physics notation).
Tensor4<std::complex<double>> transformLeg4(const Tensor4<std::complex<double>>& t3,
                                             const Matrix<std::complex<double>>& c) {
  const std::size_t n_mo1 = t3.dim0();
  const std::size_t n_mo2 = t3.dim1();
  const std::size_t n_mo3 = t3.dim2();
  const std::size_t n_ao = t3.dim3();
  const std::size_t n_mo = c.cols();
  Tensor4<std::complex<double>> result(n_mo1, n_mo2, n_mo3, n_mo, std::complex<double>(0.0, 0.0));
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo2; ++q) {
      for (std::size_t r = 0; r < n_mo3; ++r) {
        for (std::size_t s = 0; s < n_mo; ++s) {
          std::complex<double> sum(0.0, 0.0);
          for (std::size_t dd = 0; dd < n_ao; ++dd) {
            sum += c(dd, s) * t3(p, q, r, dd);
          }
          result(p, q, r, s) = sum;
        }
      }
    }
  }
  return result;
}

}  // namespace

Matrix<std::complex<double>> rkbMoOneElectronTransform(const Matrix<std::complex<double>>& h_rkb,
                                                        const Matrix<std::complex<double>>& c_dhf) {
  if (h_rkb.rows() != h_rkb.cols()) {
    throw std::runtime_error("rkbMoOneElectronTransform: h_rkb is not square");
  }
  if (c_dhf.rows() != h_rkb.rows()) {
    throw std::runtime_error("rkbMoOneElectronTransform: c_dhf row count does not match h_rkb");
  }
  return dagger(c_dhf) * (h_rkb * c_dhf);
}

Tensor4<std::complex<double>> rkbMoTwoElectronTransformPhysics(
    const RkbTwoElectronTensor& eri_ao_physics, const Matrix<std::complex<double>>& c_dhf) {
  if (c_dhf.rows() != eri_ao_physics.dim()) {
    throw std::runtime_error(
        "rkbMoTwoElectronTransformPhysics: c_dhf row count does not match eri_ao_physics's "
        "dimension");
  }
  const Tensor4<std::complex<double>> t1 = transformLeg1(eri_ao_physics, c_dhf);
  const Tensor4<std::complex<double>> t2 = transformLeg2(t1, c_dhf);
  const Tensor4<std::complex<double>> t3 = transformLeg3(t2, c_dhf);
  return transformLeg4(t3, c_dhf);
}

Matrix<std::complex<double>> occupiedPositiveEnergyDensity(std::size_t rkb_dim, int n_electrons) {
  if (rkb_dim % 2 != 0) {
    throw std::runtime_error("occupiedPositiveEnergyDensity: rkb_dim must be even");
  }
  const std::size_t n_negative = rkb_dim / 2;
  if (n_electrons < 0 || n_negative + static_cast<std::size_t>(n_electrons) > rkb_dim) {
    throw std::runtime_error(
        "occupiedPositiveEnergyDensity: n_electrons exceeds the number of positive-energy "
        "states");
  }
  Matrix<std::complex<double>> d(rkb_dim, rkb_dim, std::complex<double>(0.0, 0.0));
  for (std::size_t p = n_negative; p < n_negative + static_cast<std::size_t>(n_electrons); ++p) {
    d(p, p) = std::complex<double>(1.0, 0.0);
  }
  return d;
}

}  // namespace rerdmft
