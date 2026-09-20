#ifndef RERDMFT_KRAMERSSYMMETRY_H
#define RERDMFT_KRAMERSSYMMETRY_H

#include <complex>

#include "KramersPairing.h"
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

// Rotates `eigenvectors`' ODD columns (index 2k+1) by whatever complex
// phase makes Theta*psi_even EXACTLY equal to the (rotated) psi_odd --
// the CANONICAL Kramers phase convention (Theta|even> = |odd> exactly,
// not merely "up to an overall phase" -- see this file's own
// maxKramersPartnerDeviation, which only ever checks the ABSOLUTE VALUE
// of that overlap and so is satisfied regardless of this phase).
// LAPACK's own diagonalization of a degenerate eigenvalue has no reason
// to respect this convention -- it returns SOME orthonormal basis of the
// degenerate 2D eigenspace, with an uncontrolled relative phase between
// the two columns. That is invisible to any downstream quantity built
// only from EACH orbital separately (occupations, orbital energies,
// densities, or a 2-RDM element that only ever touches ONE member of a
// Kramers pair, like Occ_opt/PNOFs.h's own K_ij/L_ij), but corrupts any
// quantity that MIXES a pair's own two members together in one integral
// (e.g. Hessian_opt/PnofFock.h's L1/L2 pattern, eri(i,ibar,j,jbar)) --
// confirmed directly: without this fix, that specific quantity picks up
// a large, spurious, run-dependent imaginary part for X2C/C4_DHF's
// genuinely complex spinors (never an issue for NON_REL's real
// orbitals, where no such phase freedom exists in the first place).
//
// EVEN columns are left untouched; each ODD column is multiplied by a
// single complex phase (magnitude 1, or 1+0i if the pair's own overlap
// is degenerately small). This is an EXACT correction, not an
// approximation: Theta(psi_even) is guaranteed, by Kramers' theorem
// together with the diagonalization's own orthogonality within the
// degenerate pair, to be a PURE PHASE multiple of psi_odd already (both
// span the same 2D degenerate eigenspace, and Theta(psi_even) is
// orthogonal to psi_even, exactly like psi_odd is) -- so this changes
// neither the eigenvalue, the orthonormality, nor the span of the
// eigenvector set, only the arbitrary relative phase within each pair.
//
// Same inputs, dimensions, and throw conditions as
// maxKramersPartnerDeviation -- call this BEFORE building any coupled-
// Kramers-pair quantity (Hessian_opt/PnofFock.h) from `eigenvectors`,
// immediately after diagonalizing H_RKB_ortho and before
// rkbCoefficientMatrix builds c_dhf from it.
Matrix<std::complex<double>> fixKramersPhase(const Matrix<std::complex<double>>& eigenvectors,
                                              const Matrix<std::complex<double>>& rkb_coefficients,
                                              const Matrix<std::complex<double>>& x_full,
                                              const Matrix<double>& s_large,
                                              const Matrix<double>& s_small_ukb);

// Same idea as fixKramersPhase, for the strictly two-component (X2C)
// case -- same inputs, dimensions, and throw conditions as
// maxKramersPartnerDeviationLarge. Call this on X2C_DHF/X2C_HF.h's own
// converged `c_matrix` before it is used to build any coupled-Kramers-
// pair quantity.
Matrix<std::complex<double>> fixKramersPhaseLarge(const Matrix<std::complex<double>>& c_matrix,
                                                   const Matrix<double>& s_large);

// Exact Kramers re-pairing of the 4-component (RKB) eigenvectors inside
// near-degenerate clusters, the C4_DHF counterpart of Utils/KramersPairing.h's
// fixKramersPairingLarge (same algorithm and motivation: consecutive columns are only
// guaranteed to be (psi, Theta psi) pairs while every level is well separated; a
// spin-orbit-split p shell of a stretched molecule breaks that). `eigenvectors` (columns,
// in H_RKB_ortho's orthonormal basis, ascending `energies`, consecutive pairs) is mapped to
// the original [Large-alpha, Large-beta, uKB-Small-alpha, uKB-Small-beta] representation
// only to build Theta there (Theta(AO alpha) = AO beta, Theta(AO beta) = -AO alpha, on the
// Large and the Small blocks), projected back to the orthonormal coefficient space, and the
// repair runs there (KramersPairing.h's fixKramersPairingOrthonormal). Returns the repaired
// eigenvectors (Theta|2k> = |2k+1> exact, same subspaces); call it after
// fixKramersPhase, before c_dhf = x_full * eigenvectors. Throws std::runtime_error if the
// mapped basis is not orthonormal (V^dagger S V != 1), on inconsistent dimensions, or on the
// conditions of fixKramersPairingOrthonormal.
Matrix<std::complex<double>> fixKramersPairing(const Matrix<std::complex<double>>& eigenvectors,
                                                const std::vector<double>& energies,
                                                const Matrix<std::complex<double>>& rkb_coefficients,
                                                const Matrix<std::complex<double>>& x_full,
                                                const Matrix<double>& s_large,
                                                const Matrix<double>& s_small_ukb,
                                                double cluster_tolerance = 1e-4,
                                                KramersPairingReport* report = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_KRAMERSSYMMETRY_H
