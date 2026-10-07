#ifndef RERDMFT_JKONLYSIGMAHESSIANVECTORRI_H
#define RERDMFT_JKONLYSIGMAHESSIANVECTORRI_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// The RI-NATIVE counterpart of JkOnlySigmaHessianVector.h -- built ONLY from the real RI tensor
// `b` (n_aux x n^2, `b(P,p*n+q)==B(P,p,q)`) and the kappa(v) matrix, via
// RiSigmaHessianToolkit.h's shared `KB1`/`KB2`/`Bdiag` objects, NEVER forming a dense `eri`
// tensor. See [[project-ri-hessian-neo-design]]'s Stage 1d for the full per-term derivation
// (every one of `rawJkOnlyG`'s 5 bare-G pieces -- the commutator, two "sum_t" pieces, two
// unconditional pieces -- substituted with `eri(A,B,C,D)=sum_P B(P,A,C)*B(P,B,D)` and resummed).
//
// Real T=double only (NON_REL). O(n_aux*n^3) FLOPs total (dominated by the shared toolkit build,
// reused here, plus O(n_aux) per entry of each O(n^3)-sized per-term intermediate) -- see this
// project's [[feedback-target-large-systems]] for why this is the right cost class (same as one
// more RI gradient build), not O(n^4*n_aux) (the generic-Eri path) or O(n^4) (a dense tensor).
std::vector<double> jkOnlySigmaHessianVectorRi(const Matrix<double>& h, const Matrix<double>& b,
                                                std::size_t n, const std::vector<double>& occupations,
                                                const Matrix<double>& two_rdm_h,
                                                const Matrix<double>& two_rdm_x,
                                                const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
                                                const std::vector<double>& v);

// The COMPLEX/joint [t;y] RI-direct counterpart, for `Eri = RiMoEri`-style complex RI tensors
// (C4_SPINOR/X2C). `b` is the COMPLEX RI tensor (n_aux x n^2, same layout as the real case).
// Mirrors `jkOnlySigmaJointHessianVector`'s own K_t/K_y combination exactly -- builds the shared
// toolkit TWICE (once per real K matrix) and combines via the same fixed formula; see
// [[project-ri-hessian-neo-design]] Stage 4 for the complex-RI-specific validation (confirming no
// conjugate sneaks into the RI reconstruction itself, unlike the fock terms in the
// HartreeExchange case).
std::vector<double> jkOnlySigmaJointHessianVectorRi(
    const Matrix<std::complex<double>>& h, const Matrix<std::complex<double>>& b, std::size_t n,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYSIGMAHESSIANVECTORRI_H
