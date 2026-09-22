#ifndef RERDMFT_JKONLYHESSIAN_H
#define RERDMFT_JKONLYHESSIAN_H

#include <complex>
#include <cstddef>
#include <utility>
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
template <typename T, typename Eri>
T jkOnlyHessianElement(const Matrix<T>& h, const Eri& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s);

// The FULL dense real-step orbital-rotation Hessian over the independent
// pairs `pair_indices` (Hessian_opt/HartreeExchangeHessian.h's
// hessianPairIndices): element (I,J) = jkOnlyHessianElement at (p,q) =
// pair_indices[I], (r,s) = pair_indices[J]. NOT symmetrized (the bare G
// combination is asymmetric off orbital stationarity by an amount
// tracking the orbital gradient) -- the caller symmetrizes before
// diagonalizing. O(n^5) total.
template <typename T>
Matrix<T> jkOnlyHessianMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                               const std::vector<double>& occupations,
                               const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                               const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

// IMAGINARY-step (y = Im kappa_pq: kappa_pq=kappa_qp=iy) and MIXED
// real/imaginary second derivatives, complex spinors only, from the SAME
// bare G_pq,rs as jkOnlyHessianElement (each block is a fixed sign
// combination of it -- see HartreeExchangeHessian.h for the general
// statement):
//   jkOnlyHessianElementImag(pq;rs)  = SS_yy(pq;rs)
//                                    = -(G_pq,rs+G_pq,sr+G_qp,rs+G_qp,sr)
//   jkOnlyHessianElementMixed(pq,rs) = i(G_rs,pq+G_rs,qp-G_sr,pq-G_sr,qp)
//                                    = SS_ty(rs;pq) (hartreeExchange
//                                      HessianElementMixed's convention),
// where SS_ab(I;J) = d/da_J of the b-component gradient of pair I
// (sequential derivative, exactly what a finite difference of the
// gradient at a rotated point measures). Values are real for physical
// input (the returned complex has roundoff imaginary part).
template <typename T, typename Eri>
T jkOnlyHessianElementImag(const Matrix<T>& h, const Eri& eri,
                            const std::vector<double>& occupations,
                            const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                            std::size_t p, std::size_t q, std::size_t r, std::size_t s);
template <typename T, typename Eri>
T jkOnlyHessianElementMixed(const Matrix<T>& h, const Eri& eri,
                             const std::vector<double>& occupations,
                             const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                             std::size_t p, std::size_t q, std::size_t r, std::size_t s);

// The TRUE symmetric Hessian of E over the joint real orbital-rotation
// parameters [t_0..t_{n-1}, y_0..y_{n-1}] (t_I = Re kappa_pq:
// kappa_pq=+t, kappa_qp=-t; y_I = Im kappa_pq: kappa_pq=kappa_qp=iy; pairs
// p>q in `pair_indices`), i.e. the Hessian of E(exp(-kappa)) at kappa=0 --
// what a Newton-Raphson orbital step needs. Off orbital stationarity the
// bare G_pq,rs combinations are only SEQUENTIAL derivatives (not
// symmetric), so each block is symmetrized from the same bare G:
//   tt(I,J) = (1/2)[SS_tt(I;J)+SS_tt(J;I)]
//   yy(I,J) = (1/2)[SS_yy(I;J)+SS_yy(J;I)]
//   ty(I,J) = (1/2)[SS_ty(I;J)+SS_yt(J;I)],
//   SS_ty(pq;rs) = i(G_pq,rs+G_pq,sr-G_qp,rs-G_qp,sr),
//   SS_yt(pq;rs) = i(G_pq,rs-G_pq,sr+G_qp,rs-G_qp,sr).
// Real symmetric 2*n_pairs matrix (real parts). Complex orbitals only.
// O(n^5).
// Hessian-VECTOR product of jkOnlyJointHessianMatrix's own matrix, without ever forming it --
// same idea and cost accounting as HartreeExchangeHessian.h's hartreeExchangeJointHessianVector
// (O(n_pairs) memory instead of O(n_pairs^2), same O(n^5) total element cost). `v`/the returned
// vector have size 2*pair_indices.size(). Complex spinors only.
template <typename Eri>
std::vector<double> jkOnlyJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

Matrix<double> jkOnlyJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft

#endif  // RERDMFT_JKONLYHESSIAN_H
