#ifndef RERDMFT_SPHERICALTRANSFORM_H
#define RERDMFT_SPHERICALTRANSFORM_H

#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Cartesian-to-real-solid-harmonic transformation, derived from first principles rather than a
// hand-transcribed coefficient table (this is foundational basis-set code; a transcription error
// here would be very hard to notice downstream). A degree-l Cartesian monomial space has dimension
// (l+1)(l+2)/2, of which only 2l+1 combinations are genuine angular-momentum-l (harmonic, i.e.
// Laplacian-annihilated) functions for l >= 2 -- the rest ("contaminant" combinations, e.g. the
// trace x^2+y^2+z^2 direction inside a Cartesian d shell) are redundant with lower-l shells. Per
// shell self-overlap (Integrals.h's shellSelfOverlap), evaluated once with a single reference
// exponent/coefficient, this is EXACTLY a scalar multiple of a universal, l-only-dependent matrix
// (the overlap of two same-center same-exponent Cartesian Gaussians factors into independent 1-D
// moment integrals depending only on the summed power per axis, so the same combined-exponent
// prefactor multiplies every entry); after normalizeCartesianBasis's own per-component rescaling
// that overall scalar cancels exactly, leaving a matrix that depends on l alone -- so a SINGLE
// reference calculation per l gives the transformation for every shell of that l in any basis set,
// regardless of its actual exponents or contraction. Finding the universal real-spherical-harmonic
// combinations is then exactly canonicalOrthogonalize (Linear_Algebra/LinearAlgebra.h) applied to
// that reference overlap: the contaminant directions are an EXACT (not approximate) degeneracy, so
// they diagonalize to eigenvalue ~0 (machine precision) and the genuine ones to O(1), with a huge
// gap between the two -- canonicalOrthogonalize's threshold only has to land anywhere in that gap.

// The (nf x (2l+1)) Cartesian-to-spherical transformation block for angular momentum l (nf =
// cartesianComponents(l).size()). Columns are an orthonormal basis of the l-degree harmonic
// subspace in some basis-independent (but fixed, deterministic) orientation -- nothing in this
// project's own math depends on matching any particular external "m" convention, since no step
// anywhere does anything m-label-specific (multipoles, point-group symmetry, etc.). Throws
// std::runtime_error if the resulting rank is ever not exactly 2l+1 (would indicate l is out of
// libcint's supported Cartesian range, or a genuine bug -- this is an exact mathematical identity,
// not a numerically-contingent one). Cached internally per l (cheap to recompute either way: nf is
// at most 28 for l=6).
Matrix<double> cartesianToSphericalBlock(int l);

// Metadata produced by buildSphericalTransform: how the ORIGINAL (cartesian) and TRANSFORMED
// (spherical) basis are laid out, shell by shell, in the same order -- needed by anything that must
// still reason per-shell after the transform (e.g. Phase B's analytic RKB construction, which needs
// to know each large shell's own (l, atomic center, exponents) even though the working AO dimension
// has changed).
struct SphericalShellLayout {
  int l = 0;
  std::size_t cartesian_offset = 0;  // offset into the original cartesian function list
  std::size_t spherical_offset = 0;  // offset into the transformed spherical function list
};

struct SphericalTransformResult {
  Matrix<double> transform;                     // (n_cart x n_sph) block-diagonal, one block per shell
  std::vector<SphericalShellLayout> shells;      // one entry per contracted shell, in order
  std::size_t n_spherical = 0;
};

// Builds the full block-diagonal Cartesian-to-spherical transform for an entire (already
// normalizeCartesianBasis'd) cartesian AO list, shell by shell -- relies on the same "shells are
// nf=cartesianComponents(l).size() consecutive entries" layout invariant normalizeCartesianBasis
// itself assumes. l=0,1 shells pass through unchanged (nf == 2l+1 already, transform block is the
// identity).
SphericalTransformResult buildSphericalTransform(const std::vector<BasisFunction>& cartesian_basis);

// Applies a basis change T^T M T to a real matrix M built from the cartesian basis `buildSphericalTransform`
// was given, producing the corresponding matrix in the spherical basis (dimension n_sph x n_sph).
Matrix<double> transformToSpherical(const Matrix<double>& m, const Matrix<double>& transform);

// Same, for a complex matrix (T is always real, so this is T^T M T with T promoted elementwise).
Matrix<std::complex<double>> transformToSpherical(const Matrix<std::complex<double>>& m,
                                                   const Matrix<double>& transform);

// Spin-duplicates a large-component Cartesian-to-spherical transform (buildSphericalTransform's own
// `transform`, n_large_cart x n_large_sph) block-diagonally across the alpha/beta spin channels,
// promoted to complex (T is always real; this exists purely to sandwich the complex RKB one-electron
// Hamiltonian/coefficient matrices below) -- T_spin = diag(T, T), shape
// (2*n_large_cart x 2*n_large_sph), matching the [Large-alpha, Large-beta, ...] ordering
// DiracKinetic.h/Vext.h/UkbHamiltonian.h/RkbTransformation.h all use.
Matrix<std::complex<double>> spinDuplicateComplex(const Matrix<double>& transform);

// Builds the (2*n_large_cart + n_small2) x (2*n_large_sph + n_small2) embedding matrix that applies a
// large-component spherical transform to h_ukb (UkbHamiltonian.h's one-electron Hamiltonian, basis
// order [Large-alpha, Large-beta, Small-alpha, Small-beta]) BEFORE the RKB small-component projection
// (RkbHamiltonian.h's rkbHamiltonianMatrix/rkbEmbeddingMatrix) happens: the top-left block is
// spinDuplicateComplex(transform) (shrinking the Large legs), the bottom-right is the identity on the
// untouched Small-component legs (n_small2 = 2 * the raw analytic RKB small basis's own size --
// unchanged by this step). Applying this BEFORE rkb_coefficients is fed into the RKB projection, in
// place of the original h_ukb/rkb_coefficients, is what makes the projection's own "exactly one RKB
// small partner per large function" count land at the new, smaller SPHERICAL large dimension instead
// of the original Cartesian one -- rkbEmbeddingMatrix's compressed-small output size is defined as
// rkb_coefficients.rows(), so shrinking rkb_coefficients' rows via this SAME transform (dagger(T_spin)
// * rkb_coefficients) keeps the Large and RKB-Small dimensions matched by construction, exactly as
// DIRAC's own (fully spherical, large AND RKB-small) construction does -- no n_large != n_small
// plumbing is needed anywhere downstream (xFullMatrix, sFullMatrix, X2C_hamiltonian.cpp, RkbCholesky,
// ...) as a result. Verified numerically (toy s+d-shell basis) before being wired in: the resulting
// RKB-small overlap comes out exactly (2*n_large_sph) x (2*n_large_sph) and well-conditioned.
Matrix<std::complex<double>> sphericalLargeEmbedding(const Matrix<double>& transform,
                                                      std::size_t n_small2);

}  // namespace rerdmft

#endif  // RERDMFT_SPHERICALTRANSFORM_H
