#ifndef RERDMFT_X2C_DHF_X2C_HAMILTONIAN_H
#define RERDMFT_X2C_DHF_X2C_HAMILTONIAN_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// The EXACT two-component (X2C) Hamiltonian, built by eliminating the
// small component from the positive-energy block of the (already
// diagonalized, X2C_decoupling.h) 4-component Dirac problem. Given the
// RKB one-electron Hamiltonian `h_rkb`, its metric `s_full`
// (RkbOrthogonalization.h's sFullMatrix), and `c_tmp` = X_full*U
// (X2C_decoupling.h's own AO-basis eigenvector matrix, columns in
// ascending-energy order -- X2CDecouplingResult::c_tmp), this:
//
// 1. Splits h_rkb/s_full into Large/Small 2x2 blocks, each of size
//    n2 = h_rkb.rows()/2 (spanning BOTH alpha and beta together -- the
//    RKB ordering is [Large-alpha, Large-beta, Small-alpha, Small-beta]):
//      H_LL H_LS      S_LL  0
//      H_SL H_SS       0   S_SS
//    (H_SL = H_LS^dagger, S_full block-diagonal, so only H_LL/H_LS/H_SS
//    and S_LL/S_SS are actually needed below.)
// 2. Takes c_tmp's POSITIVE-ENERGY columns (the upper half in energy,
//    i.e. columns [n2, 2*n2)) and splits each into its Large
//    (C_L, rows [0,n2)) and Small (C_S, rows [n2,2*n2)) blocks -- both
//    n2 x n2 and square.
// 3. Builds the exact-decoupling matrix R = C_S * C_L^-1 (the linear
//    relation C_S = R*C_L that holds identically for every positive-
//    energy eigenvector -- LinearAlgebra.h's invertGeneral, since C_L
//    is an ordinary, non-Hermitian coefficient block).
// 4. Builds the Hermitian, positive-definite "renormalization" metric
//      Lambda = S_LL + R^dagger S_SS R
//    (C_L is exactly Lambda-orthonormal, C_L^dagger Lambda C_L = I --
//    an exact algebraic consequence of the original eigenvectors'
//    S_full-orthonormality, not an approximation).
// 5. Builds the Hermitian "raw" (ORIGINAL, non-orthogonal AO large-
//    component basis) effective Hamiltonian
//      h_x2c = H_LL + H_LS R + (H_LS R)^dagger + R^dagger H_SS R
//    which satisfies EXACTLY h_x2c C_L = Lambda C_L E (E = the SAME
//    positive-energy eigenvalues X2C_decoupling.h already found) -- a
//    Hermitian generalized eigenvalue problem in the Lambda metric,
//    NOT the plain large-component overlap S_LL (see
//    approximateX2COrtho below for what happens if one INCORRECTLY
//    uses S_LL here instead of Lambda).
// 6. Returns h_x2c_ortho = Lambda^-1/2 h_x2c Lambda^-1/2 (Hermitian,
//    metric-free/orthonormal-basis): diagonalizing h_x2c_ortho
//    reproduces the SAME positive-energy spectrum E EXACTLY (to
//    floating-point precision), which is the decisive end-to-end
//    validation of this whole construction (see main.cpp's X2C
//    report, which performs exactly this check).
//
// No approximation is introduced anywhere above -- this is an EXACT
// (not perturbative Foldy-Wouthuysen) decoupling, "one-electron X2C"
// in the standard nomenclature (Ilias & Saue, J. Chem. Phys. 126,
// 064102 (2007); Liu and Peng's exact-two-component reviews). The
// NO-PAIR approximation is applied by the caller simply by using
// h_x2c_ortho alone (discarding the negative-energy block entirely,
// which never appears here at all) for any subsequent many-electron
// treatment.
struct X2CHamiltonianResult {
  Matrix<std::complex<double>> r_matrix;
  Matrix<std::complex<double>> lambda;
  Matrix<std::complex<double>> h_x2c;        // raw, ORIGINAL AO large-component basis
  Matrix<std::complex<double>> h_x2c_ortho;  // EXACT: Lambda^-1/2 h_x2c Lambda^-1/2
};

X2CHamiltonianResult buildX2CHamiltonian(const Matrix<std::complex<double>>& h_rkb,
                                          const Matrix<std::complex<double>>& s_full,
                                          const Matrix<std::complex<double>>& c_tmp);

// An APPROXIMATE orthogonalization of the raw AO-basis X2C Hamiltonian
// `h_x2c` (X2CHamiltonianResult::h_x2c above), using ONLY the plain
// LARGE-component Loewdin orthogonalization -- `x_full`'s own upper-
// left n x n block (diag(X_Large, X_Large), RkbOrthogonalization.h's
// xFullMatrix), where n = h_x2c.rows() -- INSTEAD of the exact
// renormalization metric Lambda that h_x2c_ortho above uses. This is
// the simplification of skipping the exact "picture-change"/
// renormalization correction that some approximate two-component
// treatments make (using the untouched large-component metric S_LL as
// if it were the correct metric for the decoupled problem, which it is
// NOT in general -- Lambda = S_LL + R^dagger S_SS R differs from S_LL
// by the relativistic R-dependent correction term).
//
// Diagonalizing this result's eigenvalues will therefore be CLOSE to,
// but will NOT exactly reproduce, the true DHF positive-energy
// spectrum -- unlike h_x2c_ortho's exact agreement (see main.cpp's X2C
// report, which computes and prints both for direct comparison).
Matrix<std::complex<double>> approximateX2COrtho(const Matrix<std::complex<double>>& h_x2c,
                                                  const Matrix<std::complex<double>>& x_full);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_HAMILTONIAN_H
