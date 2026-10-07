#ifndef RERDMFT_JKONLYSIGMAHESSIANVECTOR_H
#define RERDMFT_JKONLYSIGMAHESSIANVECTOR_H

#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// A SECOND, independent way to compute the SAME real-step orbital-rotation Hessian-vector
// product `jkOnlyHessianVectorImpl`'s T=double branch already computes element-by-element
// (JkOnlyHessian.h's `jkOnlyHessianElement`, O(n_pairs^2) element evaluations per product) --
// here instead resummed ONCE over the pair index, per [[project-ri-hessian-neo-design]]'s
// "Stage 1" derivation (see that memory file for the full algebra). Mathematically IDENTICAL
// result, different computational path:
//
//   w_I = 0.5 * [ (Phi(p,q)-Phi(q,p)) + (Psi(p,q)-Psi(q,p)) ],   I = (p,q) in pair_indices,
//   Phi(a,b) := sum_{r,s} rawG(a,b,r,s) * K(v)[r,s],   (contract rawG's LAST two slots)
//   Psi(c,d) := sum_{a,b} rawG(a,b,c,d) * K(v)[a,b],   (contract rawG's FIRST two slots)
//
// where K(v) is the antisymmetric kappa matrix v encodes (K(v)[p,q]=+v_I, K(v)[q,p]=-v_I), and
// rawG is JkOnlyHessian.cpp's own (internal, unexported) `rawJkOnlyG` -- this file re-derives
// Phi/Psi as explicit tensor contractions against a GENERIC `Eri` (any type offering
// `eri(a,b,c,d)`/`dim0()..dim3()`, same contract as Tensor4<T>/CholeskyEri<T>/SymmetricEri<T>),
// so it compiles for the same Eri types `jkOnlyJointHessianVector` already does.
//
// Real T=double ONLY (matches `jkOnlyHessianVectorImpl`'s own `if constexpr
// (std::is_same_v<T, double>)` branch -- the complex/joint [t;y] case is NOT covered here).
//
// COST, honestly (see [[feedback-target-large-systems]]): for a GENERIC `Eri` this still costs
// O(n^4) `eri()` ACCESSES (each intermediate tensor below is built once via an explicit loop,
// not per-pair), which is already one full power of `n` cheaper than the O(n_pairs^2 * n) =
// O(n^5)-ish element-evaluation count `jkOnlyHessianVectorImpl` pays -- but for `Eri = RiMoEri`
// specifically, each `eri()` access itself costs O(n_aux), so the TOTAL cost here is
// O(n^4 * n_aux) -- NOT yet the O(n_aux*n^3) ("one more gradient build") target worked out by
// hand in [[project-ri-hessian-neo-design]]'s Stage 1b. This function is Stage 1c's
// CORRECTNESS checkpoint (validated against the existing trusted formula on Tensor4 data,
// tests/test_jkonly_sigma_hessian.cpp) -- a genuinely RI-NATIVE (O(n_aux*n^3)) specialization
// for `Eri = RiMoEri` specifically is the separate, not-yet-written, follow-up.
template <typename Eri>
std::vector<double> jkOnlySigmaHessianVector(
    const Matrix<double>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

// The COMPLEX/joint [t;y] counterpart, matching `jkOnlyJointHessianVector`'s own result exactly
// (same symmetrized TT/YY/TY block structure -- see JkOnlyHessian.cpp's `jkOnlyJointHessianVector`
// for the reference formula this reproduces). `v`/the result have size `2*pair_indices.size()`
// (`[t_0..t_{n-1}, y_0..y_{n-1}]`).
//
// Needs NO new derivation beyond the real case above: per
// [[project-ri-hessian-neo-design]]'s Stage 3, the SAME `Phi`/`Psi` functions (now genericized
// over `T`, see the .cpp) are called TWICE -- once with the usual antisymmetric kappa(v) matrix
// `K_t` (built from the `t` half of `v`, `K_t[p,q]=+t_I`, `K_t[q,p]=-t_I`), once with a SYMMETRIC
// matrix `K_y` (built from the `y` half, `K_y[p,q]=K_y[q,p]=+y_I` -- matching the imaginary-step
// perturbation convention `kappa_pq=kappa_qp=+iy` already established in
// [[project-hessian-opt]]) -- and combined via a fixed, derivation-verified formula. Verified in
// Python against a literal replica of `jkOnlyJointHessianVector`'s C++ formula to `2.1e-14`
// before this was written.
template <typename Eri>
std::vector<double> jkOnlySigmaJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYSIGMAHESSIANVECTOR_H
