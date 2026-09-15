#include "HartreeExchangeGradient.h"

#include <complex>
#include <cstddef>
#include <stdexcept>

namespace rerdmft {

// Derivation (see the header for the statement): starting from
//   F_pq = sum_r h_rp D_rq + 2 sum_rst eri(s,r,t,p) two_rdm(s,r,t,q),
// substitute D_rq = n_r delta_rq (diagonal 1-RDM) into the first sum:
//   sum_r h_rp D_rq = n_q h_qp.
//
// Hartree/exchange part: substitute
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
//
// L1/L2 part (optional, Piris/PNOF-style pair terms): substitute the
// additional 2-RDM contributions
//   two_rdm(p,pbar,q,qbar) += two_rdm_l1(p,q)
//   two_rdm(p,pbar,qbar,q) += two_rdm_l2(p,q)
// (pbar/qbar = pair_of(p)/pair_of(q)) into the same sum at
// (A,B,C,D)=(s,r,t,q_free) [q_free is F_pq's own free index, renamed
// here to avoid clashing with the L1/L2 formula's own "q"]:
//   2 sum_rst eri(s,r,t,q_free) two_rdm(s,r,t,q_free)
// only contributes when r = pair_of(s) (from the first two indices
// (p,pbar)=(s,r)) and either t=qbar with q=q_free (L1 term) or
// t=q_free with q=... -- working through both cases the same way as
// above collapses r to pair_of(s) and t to pair_of(q_free) in BOTH
// terms, giving (renaming q_free back to q, s the remaining free sum):
//   2 sum_s eri(s, pair_of(s), pair_of(q), p)
//           * [two_rdm_l1(s, pair_of(q)) + two_rdm_l2(s, q)]
//
// Combining everything:
//   F_pq = occupations[q]*h(q,p)
//          + sum_s [two_rdm_h(s,q)*eri(q,s,p,s) - two_rdm_x(q,s)*eri(q,s,s,p)]
//          + 2*sum_s eri(s,pair_of(s),pair_of(q),p)
//                    *[two_rdm_l1(s,pair_of(q)) + two_rdm_l2(s,q)]
// (last line only when `pair_of` is supplied). Verified against the
// unsimplified generalizedFockMatrix formula on random complex data,
// with two_rdm_h/two_rdm_x/two_rdm_l1/two_rdm_l2 all deliberately
// UNEQUAL, non-symmetric, and unrelated to any occupation-number
// formula, to floating-point precision.
template <typename T>
Matrix<T> hartreeExchangeFockMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occupations,
                                     const Matrix<double>& two_rdm_h,
                                     const Matrix<double>& two_rdm_x,
                                     const std::vector<std::size_t>& pair_of,
                                     const Matrix<double>& two_rdm_l1,
                                     const Matrix<double>& two_rdm_l2) {
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
  const bool have_l = !pair_of.empty();
  if (have_l) {
    if (pair_of.size() != n) {
      throw std::runtime_error("hartreeExchangeFockMatrix: pair_of size inconsistent with h");
    }
    if (two_rdm_l1.rows() != n || two_rdm_l1.cols() != n) {
      throw std::runtime_error(
          "hartreeExchangeFockMatrix: two_rdm_l1 dimensions inconsistent with h");
    }
    if (two_rdm_l2.rows() != n || two_rdm_l2.cols() != n) {
      throw std::runtime_error(
          "hartreeExchangeFockMatrix: two_rdm_l2 dimensions inconsistent with h");
    }
  }

  // Each (p,q) owns its own disjoint output position and only reads the
  // shared, const inputs -- safe to parallelize.
  Matrix<T> f(n, n, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      T sum{};
      for (std::size_t s = 0; s < n; ++s) {
        sum += T(two_rdm_h(s, q)) * eri(q, s, p, s) - T(two_rdm_x(q, s)) * eri(q, s, s, p);
      }
      if (have_l) {
        const std::size_t qbar = pair_of[q];
        for (std::size_t s = 0; s < n; ++s) {
          const std::size_t sbar = pair_of[s];
          sum += T(2.0) * eri(s, sbar, qbar, p) *
                 T(two_rdm_l1(s, qbar) + two_rdm_l2(s, q));
        }
      }
      f(p, q) = T(occupations[q]) * h(q, p) + sum;
    }
  }
  return f;
}

template <typename T>
double hartreeExchangeEnergy(const Matrix<T>& h, const Tensor4<T>& eri,
                              const std::vector<double>& occupations,
                              const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x) {
  const std::size_t n = h.rows();
  if (h.cols() != n) {
    throw std::runtime_error("hartreeExchangeEnergy: h is not square");
  }
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("hartreeExchangeEnergy: eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error("hartreeExchangeEnergy: occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n) {
    throw std::runtime_error("hartreeExchangeEnergy: two_rdm_h dimensions inconsistent with h");
  }
  if (two_rdm_x.rows() != n || two_rdm_x.cols() != n) {
    throw std::runtime_error("hartreeExchangeEnergy: two_rdm_x dimensions inconsistent with h");
  }

  double energy = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    energy += occupations[p] * std::real(h(p, p));
  }

  double two_electron = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : two_electron)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      two_electron +=
          std::real(eri(p, q, p, q)) * two_rdm_h(p, q) - std::real(eri(p, q, q, p)) * two_rdm_x(p, q);
    }
  }
  energy += 0.5 * two_electron;
  return energy;
}

template double hartreeExchangeEnergy(const Matrix<double>& h, const Tensor4<double>& eri,
                                       const std::vector<double>& occupations,
                                       const Matrix<double>& two_rdm_h,
                                       const Matrix<double>& two_rdm_x);
template double hartreeExchangeEnergy(const Matrix<std::complex<double>>& h,
                                       const Tensor4<std::complex<double>>& eri,
                                       const std::vector<double>& occupations,
                                       const Matrix<double>& two_rdm_h,
                                       const Matrix<double>& two_rdm_x);

template Matrix<double> hartreeExchangeFockMatrix(const Matrix<double>& h,
                                                    const Tensor4<double>& eri,
                                                    const std::vector<double>& occupations,
                                                    const Matrix<double>& two_rdm_h,
                                                    const Matrix<double>& two_rdm_x,
                                                    const std::vector<std::size_t>& pair_of,
                                                    const Matrix<double>& two_rdm_l1,
                                                    const Matrix<double>& two_rdm_l2);
template Matrix<std::complex<double>> hartreeExchangeFockMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2);

}  // namespace rerdmft
