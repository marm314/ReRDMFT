#include "HartreeExchangeGradient.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

// Derivation (see the header for the statement): starting from
//   F_pq = sum_r h_rp D_rq + 2 sum_rst eri(s,r,t,p) two_rdm(s,r,t,q),
// substitute D_rq = n_r delta_rq (diagonal 1-RDM) into the first sum:
//   sum_r h_rp D_rq = n_q h_qp.
// Substitute two_rdm(s,r,t,q) = (1/2)(D_st D_rq - D_sq D_rt)
//   = (1/2) n_s n_r (delta_st delta_rq - delta_sq delta_rt)
// into the second sum:
//   2 sum_rst eri(s,r,t,p) two_rdm(s,r,t,q)
//     = sum_rst eri(s,r,t,p) n_s n_r (delta_st delta_rq - delta_sq delta_rt)
//     = n_q sum_s n_s eri(s,q,s,p)      [delta_st, delta_rq collapse t,r]
//       - n_q sum_r n_r eri(q,r,r,p)    [delta_sq, delta_rt collapse s,t]
// Using the electron-exchange symmetry <AB|CD> = <BA|DC> (true for any
// physical two-electron integral, real or complex orbitals):
// eri(s,q,s,p) = eri(q,s,p,s), so, renaming the dummy index in the
// second sum from r to s:
//   = n_q [ sum_s n_s eri(q,s,p,s) - sum_s n_s eri(q,s,s,p) ].
// Combining with the one-electron part and relabeling q -> r
// (F_pq = n_q * FockLike(q,p), FockLike's own free indices are (r,p)):
//   FockLike(r,p) = h(r,p) + sum_s n_s (eri(r,s,p,s) - eri(r,s,s,p))
//   F_pq = n_q * FockLike(q,p).
// Verified against the unsimplified generalizedFockMatrix formula on
// random complex data (fractional AND idempotent occupations) to
// floating-point precision.
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations) {
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

  // FockLike(r,p) = h(r,p) + sum_s n_s * (eri(r,s,p,s) - eri(r,s,s,p)).
  // Each (r,p) owns its own disjoint output position and only reads the
  // shared, const h/eri/occupations -- safe to parallelize.
  Matrix<T> fock_like(n, n, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t p = 0; p < n; ++p) {
      T sum{};
      for (std::size_t s = 0; s < n; ++s) {
        if (occupations[s] == 0.0) continue;
        sum += T(occupations[s]) * (eri(r, s, p, s) - eri(r, s, s, p));
      }
      fock_like(r, p) = h(r, p) + sum;
    }
  }

  // F_pq = n_q * FockLike(q,p).
  Matrix<T> f(n, n, T{});
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      f(p, q) = T(occupations[q]) * fock_like(q, p);
    }
  }
  return f;
}

template Matrix<double> hartreeExchangeFockMatrix(const Matrix<double>& h,
                                                    const Tensor4<double>& eri,
                                                    const std::vector<double>& occupations);
template Matrix<std::complex<double>> hartreeExchangeFockMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations);

}  // namespace rerdmft
