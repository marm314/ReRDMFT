#ifndef RERDMFT_JKONLYHESSIAN_H
#define RERDMFT_JKONLYHESSIAN_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Orbital-rotation Hessian element for JK_only functionals
// (Occ_opt/JK_only.h), Hess_pq,rs = d^2E/dkappa_pq dkappa_rs (real-step
// convention of OrbitalGradient.h: kappa_pq=+t, kappa_qp=-t), from the
// rederivation in doc/orbital_hessian_jk_only.pdf -- which does NOT use
// the operator-commutator machinery of GeneralizedHessian.h/Eq. 9 of
// doc/orbital_hessian.tex, because that machinery needs
// Gamma_pqrs = -Gamma_qprs (see project memory,
// project_jk_only_relativistic_hessian_gap.md), which a JK-only 2-RDM
// violates whenever two_rdm_x != two_rdm_h.
//
// The energy at FIXED occupations is
//   E = sum_a n_a h_aa + (1/2) sum_ab H_ab <ab|ab> - (1/2) sum_ab X_ab <ab|ba>
// with H = `two_rdm_h`, X = `two_rdm_x` two FIXED symmetric coupling
// matrices (they depend only on occupations/indices, never on the
// orbitals, so an orbital rotation leaves them untouched). Differentiating
// each integral leg under a rotation (bra legs +kappa, ket legs -kappa,
// i.e. h' = h + [kappa,h], and likewise per leg of <ab|cd>) gives the
// first-order coefficient of kappa_pq
//   C_pq = (n_p-n_q) h_qp + sum_t (H_pt-H_qt) <tq|tp>
//                         - sum_t (X_pt-X_qt) <qt|tp>
// and, differentiating C_pq itself along kappa_rs (sequential rotation,
// exactly what a finite difference of the analytic gradient at a rotated
// point measures), the "bare" second-derivative coefficient
//   G_pq,rs = (n_p-n_q) (delta_qr h_sp - delta_sp h_qr)
//     + delta_qr sum_t (H_pt-H_qt) <ts|tp> - delta_sp sum_t (H_pt-H_qt) <tq|tr>
//     + (H_pr-H_qr-H_ps+H_qs) <sq|rp>
//     - delta_qr sum_t (X_pt-X_qt) <st|tp> + delta_sp sum_t (X_pt-X_qt) <qt|tr>
//     + (X_ps-X_pr-X_qs+X_qr) <qs|rp>
// The real-step Hessian is then, exactly as for GeneralizedHessian.h,
//   Hess_pq,rs = G_pq,rs - G_pq,sr - G_qp,rs + G_qp,sr.
// For H_ab = n_a n_b this is term-by-term the PDF's Eq. (5) (its
// V_qp = h_qp + sum_t n_t <tq|tp>, (n_q-n_p)(n_s-n_r) = H_qs-H_qr-H_ps+H_pr,
// f -> X); the generalization to an INDEPENDENT Hartree coupling H is
// what `kMullerAs` needs (its Hartree coupling is g(n_p,n_q), not
// n_p n_q). Index-dependent functionals (BBC2's strong/weak split,
// MLSIC/GU's i==j branch) need nothing special: X is just the matrix
// jkExchangeCoupling built.
//
// O(n) per element (one sum over t per delta term). Returns T (complex
// for T=complex<double>); the imaginary part is roundoff for physical
// input, the real part is the Hessian. Validated against a finite
// difference of the gradient at rotated integrals for every JkFunctional
// on real (NON_REL) and complex (X2C, C4_DHF) data -- see main.cpp's
// jkOnlyRotationValidationReport.
//
// Deliberately a SEPARATE, self-contained implementation from
// HartreeExchangeHessian.h/PnofHessian.h (PNOF's correct Hessian is
// untouched).
template <typename T>
T jkOnlyHessianElement(const Matrix<T>& h, const Tensor4<T>& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYHESSIAN_H
