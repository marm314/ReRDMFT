#ifndef RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTOR_H
#define RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTOR_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// The resummed/"sigma-vector" counterpart of `hartreeExchangeJointHessianVector`'s real-T branch
// (HartreeExchangeHessian.h), covering BOTH PNOF and pCCD at once -- they are both thin wrappers
// around this exact formula (see Hessian_opt/PccdHessian.h's own header comment, and
// [[project-ri-hessian-neo-design]]'s Stage-2 writeup), differing only in which `two_rdm_h/x/l1/l2`
// and `pair_of` they pass in. Mirrors Hessian_opt/JkOnlySigmaHessianVector.h's design exactly:
//
//   w_I = 0.5 * [ (Phi(p,q)-Phi(q,p)) + (Psi(p,q)-Psi(q,p)) ],   I=(p,q) in pair_indices,
//   Phi(a,b) := sum_{r,s} rawG(a,b,r,s) * K(v)[r,s],   Psi(c,d) := sum_{a,b} rawG(a,b,c,d) * K(v)[a,b],
//
// where `rawG = rawHartreeExchangeHessianTerm + rawL1L2HessianTerm` (HartreeExchangeHessian.cpp's
// own internal helpers) and K(v) is the antisymmetric kappa(v) matrix. `pair_of` empty means no
// L1/L2 term (matches `hartreeExchangeHessianElement`'s own convention).
//
// Real T=double only. Same honest cost caveat as JkOnlySigmaHessianVector.h: generic `Eri` costs
// O(n^4) eri() accesses (one power of n cheaper than the O(n_pairs^2*n) element-by-element
// formula) -- NOT yet the O(n_aux*n^3) RI-native target, see [[project-ri-hessian-neo-design]].
template <typename Eri>
std::vector<double> hartreeExchangeSigmaHessianVector(
    const Matrix<double>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

// The COMPLEX/joint [t;y] counterpart, matching `hartreeExchangeJointHessianVector`'s own result
// exactly (confirmed it uses the IDENTICAL TT/YY/TY combination as JK_only's own joint formula,
// see [[project-ri-hessian-neo-design]] Stage 3). `v`/the result have size `2*pair_indices.size()`
// (`[t_0..t_{n-1}, y_0..y_{n-1}]`). Needs NO new derivation: calls the SAME `Phi`/`Psi` builders
// twice (antisymmetric `K_t` from the t-half of `v`, symmetric `K_y` from the y-half) exactly like
// `jkOnlySigmaJointHessianVector`.
template <typename Eri>
std::vector<double> hartreeExchangeSigmaJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const Matrix<std::complex<double>>& fock, const std::vector<std::size_t>& pair_of,
    const Matrix<double>& two_rdm_l1, const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTOR_H
