#include "MoIntegralTransform.h"

#include <cstddef>
#include <stdexcept>

namespace rerdmft {

namespace {

// T1(p,B,C,D) = sum_A c(A,p) * ao(A,B,C,D)  -- transforms the first
// (chemist) leg from the packed AO tensor into a dense intermediate.
Tensor4<double> transformLeg1(const PackedTwoElectronTensor& ao, const Matrix<double>& c) {
  const std::size_t n_ao = ao.dim();
  const std::size_t n_mo = c.cols();
  Tensor4<double> result(n_mo, n_ao, n_ao, n_ao, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo; ++p) {
    for (std::size_t bb = 0; bb < n_ao; ++bb) {
      for (std::size_t cc = 0; cc < n_ao; ++cc) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          double sum = 0.0;
          for (std::size_t aa = 0; aa < n_ao; ++aa) {
            sum += c(aa, p) * ao(aa, bb, cc, dd);
          }
          result(p, bb, cc, dd) = sum;
        }
      }
    }
  }
  return result;
}

// T2(p,q,C,D) = sum_B c(B,q) * t1(p,B,C,D)
Tensor4<double> transformLeg2(const Tensor4<double>& t1, const Matrix<double>& c) {
  const std::size_t n_mo1 = t1.dim0();
  const std::size_t n_ao = t1.dim1();
  const std::size_t n_mo = c.cols();
  Tensor4<double> result(n_mo1, n_mo, n_ao, n_ao, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo; ++q) {
      for (std::size_t cc = 0; cc < n_ao; ++cc) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          double sum = 0.0;
          for (std::size_t bb = 0; bb < n_ao; ++bb) {
            sum += c(bb, q) * t1(p, bb, cc, dd);
          }
          result(p, q, cc, dd) = sum;
        }
      }
    }
  }
  return result;
}

// T3(p,q,r,D) = sum_C c(C,r) * t2(p,q,C,D)
Tensor4<double> transformLeg3(const Tensor4<double>& t2, const Matrix<double>& c) {
  const std::size_t n_mo1 = t2.dim0();
  const std::size_t n_mo2 = t2.dim1();
  const std::size_t n_ao = t2.dim2();
  const std::size_t n_mo = c.cols();
  Tensor4<double> result(n_mo1, n_mo2, n_mo, n_ao, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo2; ++q) {
      for (std::size_t r = 0; r < n_mo; ++r) {
        for (std::size_t dd = 0; dd < n_ao; ++dd) {
          double sum = 0.0;
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

// T4(p,q,r,s) = sum_D c(D,s) * t3(p,q,r,D)  -- the fully-transformed,
// still CHEMIST-notation MO tensor: T4(p,q,r,s) == (p q|r s)_MO.
Tensor4<double> transformLeg4(const Tensor4<double>& t3, const Matrix<double>& c) {
  const std::size_t n_mo1 = t3.dim0();
  const std::size_t n_mo2 = t3.dim1();
  const std::size_t n_mo3 = t3.dim2();
  const std::size_t n_ao = t3.dim3();
  const std::size_t n_mo = c.cols();
  Tensor4<double> result(n_mo1, n_mo2, n_mo3, n_mo, 0.0);
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n_mo1; ++p) {
    for (std::size_t q = 0; q < n_mo2; ++q) {
      for (std::size_t r = 0; r < n_mo3; ++r) {
        for (std::size_t s = 0; s < n_mo; ++s) {
          double sum = 0.0;
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

Matrix<double> moOneElectronTransform(const Matrix<double>& h_ao, const Matrix<double>& c) {
  const std::size_t n_ao = h_ao.rows();
  if (h_ao.cols() != n_ao) {
    throw std::runtime_error("moOneElectronTransform: h_ao is not square");
  }
  if (c.rows() != n_ao) {
    throw std::runtime_error("moOneElectronTransform: c row count does not match h_ao");
  }
  const std::size_t n_mo = c.cols();

  // temp(A,q) = sum_B h_ao(A,B) c(B,q)
  Matrix<double> temp(n_ao, n_mo, 0.0);
  for (std::size_t aa = 0; aa < n_ao; ++aa) {
    for (std::size_t bb = 0; bb < n_ao; ++bb) {
      const double h_ab = h_ao(aa, bb);
      if (h_ab == 0.0) continue;
      for (std::size_t q = 0; q < n_mo; ++q) {
        temp(aa, q) += h_ab * c(bb, q);
      }
    }
  }
  // h_mo(p,q) = sum_A c(A,p) temp(A,q)
  Matrix<double> h_mo(n_mo, n_mo, 0.0);
  for (std::size_t aa = 0; aa < n_ao; ++aa) {
    for (std::size_t p = 0; p < n_mo; ++p) {
      const double c_ap = c(aa, p);
      if (c_ap == 0.0) continue;
      for (std::size_t q = 0; q < n_mo; ++q) {
        h_mo(p, q) += c_ap * temp(aa, q);
      }
    }
  }
  return h_mo;
}

Tensor4<double> moTwoElectronTransformPhysics(const PackedTwoElectronTensor& eri_ao_chemist,
                                               const Matrix<double>& c) {
  if (c.rows() != eri_ao_chemist.dim()) {
    throw std::runtime_error(
        "moTwoElectronTransformPhysics: c row count does not match eri_ao_chemist's dimension");
  }
  const std::size_t n_mo = c.cols();

  const Tensor4<double> t1 = transformLeg1(eri_ao_chemist, c);
  const Tensor4<double> t2 = transformLeg2(t1, c);
  const Tensor4<double> t3 = transformLeg3(t2, c);
  const Tensor4<double> chemist_mo = transformLeg4(t3, c);  // chemist_mo(p,q,r,s) == (pq|rs)_MO

  // physics<p q|r s>_MO = chemist_mo(p,r,q,s) (physics<A B|C D> =
  // chemist(A,C,B,D)).
  Tensor4<double> physics_mo(n_mo, n_mo, n_mo, n_mo, 0.0);
  for (std::size_t p = 0; p < n_mo; ++p) {
    for (std::size_t q = 0; q < n_mo; ++q) {
      for (std::size_t r = 0; r < n_mo; ++r) {
        for (std::size_t s = 0; s < n_mo; ++s) {
          physics_mo(p, q, r, s) = chemist_mo(p, r, q, s);
        }
      }
    }
  }
  return physics_mo;
}

}  // namespace rerdmft
