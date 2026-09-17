#ifndef RERDMFT_CHOLESKY_DECOMPOSITION_H
#define RERDMFT_CHOLESKY_DECOMPOSITION_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Pivoted (incomplete) Cholesky decomposition of a two-electron integral
// tensor `eri(A,B,C,D)`, used to REDUCE THE COST of transforming that
// tensor into a different orbital basis (see choleskyTransformVectors/
// choleskyReconstructEri below). Works IDENTICALLY for either of this
// project's two integral conventions -- PHYSICS notation
// (eri(A,B,C,D) == <AB|CD>, Hessian_opt/GeneralizedFock.h/C4_DHF/
// RkbTwoElectron.h/X2C_DHF's own convention) or CHEMIST notation
// (eri(A,B,C,D) == (AB|CD), AO_ints/ElectronRepulsion.h's own
// convention) -- as long as the SAME convention is used consistently by
// the caller on both sides (decomposing and later reconstructing); the
// two conventions differ only in how EXTERNAL code interprets the same
// 4-index array physically, not in the algebraic structure this file
// actually needs.
//
// Mathematical basis: the ONE property both conventions share is
//   eri(A,B,C,D) = conj(eri(C,D,A,B))
// (a direct consequence of <AB|CD> = <bra AB| ...|ket CD> becoming its
// own complex conjugate under bra<->ket exchange for physics notation;
// exactly (AB|CD)=(CD|AB), real, for chemist notation) -- i.e. treating
// the FIRST PAIR of indices (A,B) as a "bra" row-index and the SECOND
// PAIR (C,D) as a "ket" column-index, M[(A,B)][(C,D)] := eri(A,B,C,D)
// is an honest HERMITIAN matrix over the combined pair-index space
// (diagonal M[(A,B)][(A,B)] = eri(A,B,A,B) = <AB|AB> = the Coulomb
// self-energy of two ORDINARY, non-negative densities |A|^2/|B|^2,
// hence real and >= 0 -- confirming M is also POSITIVE SEMI-DEFINITE).
// IMPORTANT: this (A,B)/(C,D) "bra/ket" grouping is NOT the same as the
// "which electron" grouping used elsewhere in this project's own
// physics-notation docs (A,C on electron 1, B,D on electron 2, e.g.
// GeneralizedFock.h) -- that (A,C)/(B,D) grouping gives a matrix that
// is only COMPLEX SYMMETRIC, not Hermitian, for genuinely complex
// orbitals (its own "diagonal" eri(A,A,C,C) is generally complex, not a
// valid Cholesky pivot) -- do not conflate the two when extending this
// file. Any Hermitian PSD matrix admits a factorization M = sum_L v_L
// v_L^dagger, i.e.
//   eri(A,B,C,D) = sum_{L=1}^{Nchol} V_L(A,B) * conj(V_L(C,D))
// (conjugation is a no-op for T = double, so this collapses to the
// familiar real/chemist form eri(A,B,C,D) = sum_L V_L(A,B)*V_L(C,D)
// automatically), where each V_L is stored as an ordinary n x n
// Matrix<T> (the (A,B) pair reshaped back into two indices) -- the SAME
// shape as a one-electron matrix like h or S, deliberately, so the
// existing dagger()/operator* machinery already used for one-electron
// MO transforms (e.g. C4_DHF/RkbMoTransform.h's
// rkbMoOneElectronTransform) applies to EACH Cholesky vector unchanged
// (see choleskyTransformVectors).
//
// Algorithm (Beebe-Linderberg / the standard "pivoted Cholesky
// decomposition of the ERI matrix" used throughout quantum chemistry for
// exactly this cost-reduction purpose -- not a novel method): starting
// from the exact diagonal D[(A,B)] = eri(A,B,A,B) (always real and
// >= 0), repeatedly (1) pick the pivot (A*,B*) with the LARGEST current
// residual diagonal, (2) stop once that maximum drops below
// `threshold` (or `max_vectors` vectors have been found, if given and
// positive), (3) otherwise read the corresponding ROW of the ORIGINAL
// tensor, eri(A*, B*, C, D) for all (C,D) (an O(n^2) slice, since the
// full dense tensor is already available), subtract off the already-
// found vectors' own contribution to that row, take the COMPLEX
// CONJUGATE of the result (needed to solve V_k(C,D) out of the defining
// sum's own conj(V_k(C,D)) factor -- a no-op for T = double, but easy
// to get backwards for T = complex<double> since a real-only test would
// never catch its absence; verified by direct hand substitution and a
// dedicated complex numerical test before trusting this), and divide by
// sqrt(D[(A*,B*)]) to get the next vector V_k, then (4) update every
// remaining residual diagonal by subtracting |V_k(C,D)|^2. Each
// iteration costs O(n^2) given the tensor is already dense in memory,
// so the WHOLE decomposition costs O(Nchol * n^2) -- empirically
// Nchol scales roughly linearly with n for a fixed `threshold` (NOT
// quadratically, i.e. NOT with the O(n^2) dimension of M itself), so
// this is O(n^3) overall: strictly cheaper than the O(n^4) tensor it
// decomposes, let alone the O(n^4 * n_new) cost of a direct 4-leg basis
// transform of the untouched tensor (see the .cpp of each caller for
// the concrete savings).
//
// `threshold` is the actual accuracy control: below the true numerical
// double-precision noise floor (~1e-10 to 1e-12 in practice for a
// molecular ERI tensor of realistic magnitude), Nchol stops growing and
// the decomposition becomes, for all practical purposes, EXACT (the
// residual diagonal cannot be driven further down by floating-point
// error, not because more vectors wouldn't help mathematically) -- this
// is why CHOLESKY TRUE reproduces CHOLESKY FALSE's energies to full
// displayed precision at the default threshold, not merely
// approximately.
//
// Throws std::runtime_error if `eri` is not square in all four
// dimensions, or if a negative residual diagonal larger in magnitude
// than a small numerical-noise allowance is ever encountered (a
// genuine sign of a non-Hermitian-PSD input, e.g. a bug elsewhere, NOT
// swallowed silently).
template <typename T>
std::vector<Matrix<T>> choleskyDecomposeEri(const Tensor4<T>& eri, double threshold = 1e-10,
                                             std::size_t max_vectors = 0);

// Transforms every Cholesky vector from its old (n_old x n_old) basis
// into a new one:
//   V'_L = C^dagger * V_L * conj(C)
// (n_new x n_new), where `c` is the (n_old x n_new) AO/spinor -> MO/
// spinor coefficient matrix (for T = double, conj(C) = C, so this
// reduces to the familiar C^T * V_L * C form automatically). This
// specific combination -- NOT the plain C^dagger * V_L * C one-electron
// pattern used elsewhere in this project -- is what correctly
// reproduces a direct 4-leg transform of the FULL tensor by `c` on all
// four legs (bra legs conjugated, ket legs not, exactly like every
// other MO transform in this project, e.g. C4_DHF/RkbMoTransform.h's
// own rkbMoTwoElectronTransformPhysics) once expanded back out via
// choleskyReconstructEri -- verified algebraically and numerically
// (see this file's own validation, not committed) before trusting it
// here. Costs O(Nchol * (n_old^2*n_new + n_old*n_new^2)) total --
// compare the DIRECT 4-leg transform's O(n_old^4 * n_new) (see e.g.
// C4_DHF/RkbMoTransform.h's own docstring) -- the saving is exactly the
// standard Cholesky/RI one: replacing one O(n_old^4)-scaling tensor
// operation with O(Nchol) ~ O(n_old) O(n_old^3)-scaling ones.
template <typename T>
std::vector<Matrix<T>> choleskyTransformVectors(const std::vector<Matrix<T>>& vectors,
                                                 const Matrix<T>& c);

// Rebuilds the dense two-electron tensor from a set of (already-
// transformed, new-basis) Cholesky vectors:
//   eri(A,B,C,D) = sum_L V_L(A,B) * conj(V_L(C,D))
// Costs O(Nchol * n_new^4) -- together with choleskyTransformVectors
// above, the full "decompose once in the old basis, transform the
// (much smaller) vectors, reconstruct in the new basis" pipeline is
// what choleskyTransformEri/choleskyTransformEriMixed below wrap into
// a single drop-in replacement for a direct transform.
template <typename T>
Tensor4<T> choleskyReconstructEri(const std::vector<Matrix<T>>& vectors);

// Convenience wrapper: decompose + transform + reconstruct in one call,
// for the common case where the SOURCE tensor and the coefficient
// matrix `c` share the same scalar type T (NON_REL's real AO -> real MO
// transform; C4_DHF's complex RKB-spinor AO -> complex DHF-spinor MO
// transform; C4_DHF/RkbTwoElectron.cpp's own real UKB-Small AO ->
// complex RKB-Small-spinor "electron-pair" projections, chemist
// notation). A direct drop-in alternative to e.g. NON_REL/
// MoIntegralTransform.h's moTwoElectronTransformPhysics or C4_DHF/
// RkbMoTransform.h's rkbMoTwoElectronTransformPhysics whenever the
// CALLER already has (or is willing to build) a DENSE AO tensor in
// whichever convention it itself uses consistently -- unlike those two
// functions, this one does not itself understand any PACKED storage
// format, by design (keeping this file generic and independently
// testable; densifying a packed tensor once, if needed, is the
// caller's job, exactly as the existing MO-transform files already do
// internally for their own direct-transform path).
template <typename T>
Tensor4<T> choleskyTransformEri(const Tensor4<T>& eri, const Matrix<T>& c,
                                 double threshold = 1e-10);

// Same idea, for the MIXED-type case where the source tensor is REAL
// (no relativistic/picture-change correction applied to it, e.g.
// X2C_DHF's own eri_x2c_spin, or C4_DHF/RkbTwoElectron.cpp's
// unrestricted-kinetic-balance Small-basis integrals) but the
// coefficient matrix is COMPLEX (a genuinely relativistic/spin-orbit-
// coupled spinor basis). Decomposes in REAL arithmetic first (cheaper,
// and mathematically exact: a real PSD matrix's Cholesky vectors are
// themselves real), THEN promotes only the (much smaller) vectors to
// complex before transforming -- deliberately never upcasts the full
// O(n^4) source tensor to complex, unlike a direct 4-leg transform of a
// real tensor by a complex matrix (see X2C_DHF/X2C_MoTransform.cpp's
// own toComplex step, which this specifically avoids needing).
Tensor4<std::complex<double>> choleskyTransformEriMixed(const Tensor4<double>& eri,
                                                         const Matrix<std::complex<double>>& c,
                                                         double threshold = 1e-10);

}  // namespace rerdmft

#endif  // RERDMFT_CHOLESKY_DECOMPOSITION_H
