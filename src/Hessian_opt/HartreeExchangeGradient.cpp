#include "HartreeExchangeGradient.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

// Derivation (see the header for the statement): starting from
//   F_pq = sum_r h_rp D_rq + 2 sum_rst eri(s,r,t,p) two_rdm(s,r,t,q),
// substitute D_rq = n_r delta_rq (diagonal 1-RDM) into the first sum:
//   sum_r h_rp D_rq = n_q h_qp.
// Substitute
//   two_rdm(A,B,C,D) = (1/2)[two_rdm_h(A,B) delta_AC delta_BD
//                             - two_rdm_x(A,B) delta_AD delta_BC]
// at (A,B,C,D) = (s,r,t,q) into the second sum:
//   2 sum_rst eri(s,r,t,p) two_rdm(s,r,t,q)
//     = sum_rst eri(s,r,t,p) [two_rdm_h(s,r) delta_st delta_rq
//                             - two_rdm_x(s,r) delta_sq delta_rt]
//     = sum_s two_rdm_h(s,q) eri(s,q,s,p)     [delta_st, delta_rq collapse t,r]
//       - sum_r two_rdm_x(q,r) eri(q,r,r,p)   [delta_sq, delta_rt collapse s,t]
// Using the electron-exchange symmetry <AB|CD> = <BA|DC> (true for any
// physical two-electron integral, real or complex orbitals):
// eri(s,q,s,p) = eri(q,s,p,s), and renaming the dummy index in the
// second sum from r to s:
//   = sum_s two_rdm_h(s,q) eri(q,s,p,s) - sum_s two_rdm_x(q,s) eri(q,s,s,p).
// Combining with the one-electron part:
//   F_pq = occupations[q]*h(q,p)
//          + sum_s [two_rdm_h(s,q)*eri(q,s,p,s) - two_rdm_x(q,s)*eri(q,s,s,p)].
// Verified against the unsimplified generalizedFockMatrix formula on
// random complex data, with two_rdm_h and two_rdm_x deliberately
// UNEQUAL and non-symmetric (not tied to any occupation-number
// formula), to floating-point precision.
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("hartreeExchangeFockMatrix: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("hartreeExchangeFockMatrix: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error("hartreeExchangeFockMatrix: occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error("hartreeExchangeFockMatrix: two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("hartreeExchangeFockMatrix: two_rdm_x dimensions inconsistent with h");
  }

  // F_pq = occupations[q]*h(q,p)
  //        + sum_s [two_rdm_h(s,q)*eri(q,s,p,s) - two_rdm_x(q,s)*eri(q,s,s,p)]
  // Each (p,q) owns its own disjoint output position and only reads the
  // shared, const h/eri/occupations/two_rdm_h/two_rdm_x -- safe to
  // parallelize.
  Matrix<T> f(n, n, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T sum{};
      for (std::size_t s = 0; s < n; ++s) {
        sum += T(two_rdm_h(s, q)) * eri(q, s, p, s) - T(two_rdm_x(q, s)) * eri(q, s, s, p);
      }
      f(p, q) = T(occupations[q]) * h(q, p) + sum;
    }
  }
  return f;
}

template Matrix<double> hartreeExchangeFockMatrix(const Matrix<double>& h,
                                                    const Tensor4<double>& eri,
                                                    const std::vector<double>& occupations,
                                                    const Matrix<double>& two_rdm_h,
                                                    const Matrix<double>& two_rdm_x);
template Matrix<std::complex<double>> hartreeExchangeFockMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x);

}  // namespace rerdmft
