#ifndef RERDMFT_KRAMERSSYMMETRY_H
#define RERDMFT_KRAMERSSYMMETRY_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// Checks that the eigenvectors of H_RKB_ortho respect Kramers' symmetry
// (time reversal), which is a stronger statement than the eigenvalues
// merely being degenerate in pairs: for a Hamiltonian with no external
// symmetry-breaking terms, every eigenvector psi must have a degenerate
// "Kramers partner" equal (up to an overall phase) to Theta*psi, where
// Theta = -i Sigma_y K is the antiunitary time-reversal operator
// (Sigma_y = diag(sigma_y, sigma_y), K = complex conjugation).
//
// Theta acts simply within the *original* (pre-orthonormalization) Large
// / unrestricted-kinetic-balance-Small spinor-AO basis, where every basis
// function is a real spatial AO times a fixed alpha or beta spin:
// Theta(AO*alpha) = +AO*beta, Theta(AO*beta) = -AO*alpha. So each
// eigenvector of H_RKB_ortho (given in its own orthonormal basis) is
// mapped back to that original representation via
// psi = rkbEmbeddingMatrix(rkb_coefficients) * x_full * eigenvector_column
// (RkbHamiltonian.h, RkbOrthogonalization.h), where Theta is applied, and
// the result compared against the eigenvector's partner using the
// physical inner product weighted by the original (non-orthonormal)
// overlap S_full = diag(S_Large, S_Large, S_smallUKB, S_smallUKB).
//
// `eigenvectors` (columns, in H_RKB_ortho's own basis) is assumed sorted
// so that consecutive pairs (2k, 2k+1) are the Kramers-degenerate pairs
// (as produced by diagonalizeHermitian and already relied on for the
// even/odd eigenvalue printing). `s_large` is the Large-component AO
// overlap (Integrals.h's overlapMatrix on the Large basis) and
// `s_small_ukb` the analogous overlap for the unrestricted-kinetic-
// balance Small basis (not RkbOverlap.h's RKB-basis S_small, which has a
// different dimension).
//
// Returns the maximum, over all pairs, of 1 - |<psi_odd|Theta psi_even>_S|
// (0 = perfect Kramers symmetry). Throws std::runtime_error on
// inconsistent input dimensions.
double maxKramersPartnerDeviation(const Matrix<std::complex<double>>& eigenvectors,
                                   const Matrix<std::complex<double>>& rkb_coefficients,
                                   const Matrix<std::complex<double>>& x_full,
                                   const Matrix<double>& s_large,
                                   const Matrix<double>& s_small_ukb);

// Same check as maxKramersPartnerDeviation, but for a strictly
// two-component (Large-component-only) spinor basis with no small
// component at all -- as used by X2C's own converged orbitals
// (X2C_DHF/X2C_HF.h's c_matrix = X_Large * U), which already live
// purely in the [Large-alpha, Large-beta] AO representation, so no
// RKB embedding step is needed to get there (unlike
// maxKramersPartnerDeviation, which must first map H_RKB_ortho's own
// eigenvectors back through the Large+Small RKB basis).
//
// `c_matrix` columns are assumed sorted so that consecutive pairs
// (2k, 2k+1) are the Kramers-degenerate pairs (as produced by
// diagonalizing Fock_ortho and already relied on for the even/odd
// orbital-energy printing). `s_large` is the Large-component AO
// overlap. Returns the maximum, over all pairs, of
// 1 - |<psi_odd|Theta psi_even>_S| (0 = perfect Kramers symmetry).
double maxKramersPartnerDeviationLarge(const Matrix<std::complex<double>>& c_matrix,
                                        const Matrix<double>& s_large);

}  // namespace rerdmft

#endif  // RERDMFT_KRAMERSSYMMETRY_H
