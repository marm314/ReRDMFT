#include "GeneralizedFock.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

template <typename T>
Matrix<T> generalizedFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri, const Matrix<T>& d,
                                 const Tensor4<T>& two_rdm) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("generalizedFockMatrix: h is not square");
  }
  if (d.rows() != n || d.cols() != n) {
    throw std::runtime_error("generalizedFockMatrix: D dimensions are inconsistent with h");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("generalizedFockMatrix: eri dimensions are inconsistent with h");
  }
  if (two_rdm.dim0() != n || two_rdm.dim1() != n || two_rdm.dim2() != n || two_rdm.dim3() != n) {
    throw std::runtime_error("generalizedFockMatrix: two_rdm dimensions are inconsistent with h");
  }

  Matrix<T> fock(n, n, T{});
  // Each (p,q) owns its own disjoint output position and only reads the
  // shared, const h/eri/d/two_rdm -- safe to parallelize. one_electron
  // and two_electron are declared inside the loop body, so each thread
  // accumulates into its own local variable -- no reduction needed.
  //
  // F_pq = sum_r h_rp D_rq + 2 sum_rst <sr|tp> two_rdm_strq (Dyall &
  // Faegri Eq. 8.30's first sum, rewritten for physics-notation eri and
  // an N(N-1)/2-normalized two_rdm -- see GeneralizedFock.h for the
  // derivation of both the eri(s,r,t,p) index mapping and the factor of
  // 2): eri(s,r,t,p) is the physics-notation integral <sr|tp> (== Dyall's
  // chemist-notation (st|rp)), two_rdm(s,t,r,q) is Dyall's P_strq
  // (Eq. 8.28's index convention) divided by 2.
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T one_electron{};
      for (std::size_t r = 0; r < n; ++r) {
        one_electron += h(r, p) * d(r, q);
      }

      T two_electron{};
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          for (std::size_t t = 0; t < n; ++t) {
            two_electron += eri(s, r, t, p) * two_rdm(s, t, r, q);
          }
        }
      }

      fock(p, q) = one_electron + T(2) * two_electron;
    }
  }
  return fock;
}

template Matrix<double> generalizedFockMatrix(const Matrix<double>& h,
                                               const Tensor4<double>& eri, const Matrix<double>& d,
                                               const Tensor4<double>& two_rdm);
template Matrix<std::complex<double>> generalizedFockMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const Matrix<std::complex<double>>& d, const Tensor4<std::complex<double>>& two_rdm);

}  // namespace rerdmft
