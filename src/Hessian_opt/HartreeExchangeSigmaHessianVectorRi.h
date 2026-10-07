#ifndef RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTORRI_H
#define RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTORRI_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// The RI-NATIVE counterpart of HartreeExchangeSigmaHessianVector.h -- covers PNOF AND pCCD at
// once (same reasoning as the generic version: both are thin wrappers around this one formula),
// built ONLY from the real RI tensor `b` (n_aux x n^2, `b(P,p*n+q)==B(P,p,q)`) and kappa(v), via
// RiSigmaHessianToolkit.h's shared `KB1`/`KB2`/`Bdiag`, NEVER forming a dense `eri`. See
// [[project-ri-hessian-neo-design]]'s Stage 2d for the full per-term derivation of both the H/X
// part (`rawHartreeExchangeHessianTerm`) and the L1/L2 pair term (`rawL1L2HessianTerm`) -- the
// `pair_of` Kramers permutations there turned out to need NO new RI machinery, just reading
// `B`/`KB1`/`KB2` at a permuted orbital index.
//
// Real T=double only. `pair_of` empty means no L1/L2 term (matches the generic version's
// convention). Same O(n_aux*n^3) FLOPs / O(n_aux*n^2)+O(n^3)-per-term storage accounting as
// JkOnlySigmaHessianVectorRi.h -- see that file and [[feedback-target-large-systems]].
std::vector<double> hartreeExchangeSigmaHessianVectorRi(
    const Matrix<double>& h, const Matrix<double>& b, std::size_t n, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const Matrix<double>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v);

// The COMPLEX/joint [t;y] RI-direct counterpart, for `Eri = RiMoEri`-style complex RI tensors
// (C4_SPINOR/X2C), covering PNOF+pCCD. `b`/`fock`/`h` are COMPLEX. Mirrors
// `hartreeExchangeSigmaJointHessianVector`'s own K_t/K_y combination exactly. Carries the SAME
// `conjugateT` fix the generic complex version needed (see
// [[project-ri-hessian-neo-design]] Stage 3's "real bug" entry) into the RI-direct `P1b`/`Q2`
// terms -- confirmed necessary again here, not assumed to carry over automatically.
std::vector<double> hartreeExchangeSigmaJointHessianVectorRi(
    const Matrix<std::complex<double>>& h, const Matrix<std::complex<double>>& b, std::size_t n,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, const Matrix<std::complex<double>>& fock,
    const std::vector<std::size_t>& pair_of, const Matrix<double>& two_rdm_l1,
    const Matrix<double>& two_rdm_l2,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices, const std::vector<double>& v);

}  // namespace rerdmft

#endif  // RERDMFT_HARTREEEXCHANGESIGMAHESSIANVECTORRI_H
