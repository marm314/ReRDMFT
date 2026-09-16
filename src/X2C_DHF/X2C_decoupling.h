#ifndef RERDMFT_X2C_DHF_X2C_DECOUPLING_H
#define RERDMFT_X2C_DHF_X2C_DECOUPLING_H

#include <complex>

#include "LinearAlgebra.h"
#include "Matrix.h"

namespace rerdmft {

// One-electron X2C ("exact two-component") decoupling of the bare RKB
// Dirac Hamiltonian: diagonalizing the ORTHONORMALIZED one-electron
// Hamiltonian
//   H_RKB_ortho = X_full^dagger H_RKB X_full
// (RkbOrthogonalization.h's hRkbOrthoMatrix) yields eigenvectors that
// block-diagonalize the one-electron Dirac equation into decoupled
// positive-/negative-energy branches -- this diagonalization IS the
// (free-particle, "X2C-1e") one-electron X2C transformation itself,
// exactly the same H_RKB_ortho build-and-diagonalize step this
// project's own C4_DHF pipeline already performs in main.cpp to seed
// its initial guess. Pulled out into its own reusable function here so
// that a genuinely STANDALONE X2C report (Input.h's X2C keyword) can
// call the identical procedure without duplicating it.
//
// `h_rkb` is the bare RKB one-electron Hamiltonian (RkbHamiltonian.h).
// `x_full` is RkbOrthogonalization.h's own full Loewdin
// orthonormalization matrix (already combining the Large- and Small-
// component pieces) -- the only other input hRkbOrthoMatrix itself
// needs to build H_RKB_ortho. `s_full` is RkbOrthogonalization.h's own
// full RKB overlap (metric) matrix S_full = diag(S_Large, S_Large,
// S_small) -- used below (see C_tmp) to verify C_tmp actually solves
// the ORIGINAL generalized eigenvalue problem, not just the
// orthonormalized one. `s_large`/`x_large` (the Large-component AO
// overlap and its Loewdin inverse square root, Integrals.h/
// LinearAlgebra.h) and `s_small` (RkbOverlap.h's RKB-basis Small-
// component overlap, S_full's own Small-component block) are accepted
// for parity with how S_full/X_full are themselves built elsewhere in
// this project, but are NOT otherwise used by this function --
// `s_full` alone (already combining them) is what the check below
// needs.
//
// `c_tmp = X_full * U` (U = `eigen.eigenvectors`,
// C4_DHF/RkbDensityMatrix.h's own rkbCoefficientMatrix -- the SAME
// relation main.cpp already uses to build C_DHF from H_RKB_ortho's own
// eigenvectors) maps the orthonormal eigenbasis back to the original,
// non-orthonormal RKB spinor-AO basis. `max_generalized_eigenproblem_
// residual` confirms C_tmp is genuinely a solution of the ORIGINAL
// (non-orthonormal) generalized eigenvalue problem
// H_RKB * C_tmp = S_full * C_tmp * E (E = diag(eigen.eigenvalues)) --
// the defining property of X_full = S_full^-1/2 as the transformation
// that reduces this generalized problem to the ordinary Hermitian
// eigenproblem H_RKB_ortho U = U E already solved above -- computed as
// the maximum, over every matrix element of every eigenvector column
// c, of |H_RKB C_tmp - S_full C_tmp E|_{i,c} (0 = exact).
struct X2CDecouplingResult {
  Matrix<std::complex<double>> h_rkb_ortho;
  HermitianEigenResult eigen;
  Matrix<std::complex<double>> c_tmp;
  double max_generalized_eigenproblem_residual = 0.0;
};

X2CDecouplingResult x2cDecoupling(const Matrix<std::complex<double>>& h_rkb,
                                   const Matrix<double>& s_large, const Matrix<double>& x_large,
                                   const Matrix<std::complex<double>>& s_small,
                                   const Matrix<std::complex<double>>& s_full,
                                   const Matrix<std::complex<double>>& x_full);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_DECOUPLING_H
