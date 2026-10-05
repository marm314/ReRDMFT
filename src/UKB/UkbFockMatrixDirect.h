#ifndef RERDMFT_UKBFOCKMATRIXDIRECT_H
#define RERDMFT_UKBFOCKMATRIXDIRECT_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds, DIRECTLY from AO integrals (no stored two-electron representation of any kind -- dense,
// packed, or Cholesky), the two-electron (J-K) contribution to the Fock matrix in the UKB spinor
// basis [Large-alpha, Large-beta, Small-alpha, Small-beta] (SpinorBasis's own storage order), from
// a general (not necessarily spin-block-diagonal) Hermitian UKB density matrix. `large_basis`/
// `small_basis` are the RAW Cartesian Large basis and the RKB small-component "elementary
// derivative term" basis (RkbDerivativeTerms.h's buildRkbSmallBasis -- same inputs
// ukbHamiltonianMatrix/rkbTwoElectronIntegrals already take).
//
// SCF_DIRECT_4C's Phase 1 core: this is the piece that makes an integral-direct 4-component SCF
// possible at all, by recomputing the needed AO contributions fresh every Fock build instead of
// persisting an AO tensor or a growing set of Cholesky vectors -- memory stays O(n^2) (the density/
// Fock matrices alone) regardless of basis size or angular degeneracy, matching DIRAC's own DIRAOF
// (dirtwo.F). Only the (Large,Large|Large,Large) sector is shell-quartet batched (via
// ElectronRepulsion.h's twoElectronShellQuartet, Phase 0): the RKB small basis's entries are each
// one large function's own individual elementary derivative term, not complete angular-momentum
// shells, so they cannot be grouped for libcint's per-shell batching at all -- every sector
// touching the Small basis ((LL|SS), (SS|LL), (SS|SS)) falls back to twoElectronQuadruplet, one AO
// quadruplet at a time (same per-call cost the existing dense path already pays for these same
// sectors via twoElectronIntegralsCross/twoElectronIntegralsPacked). Still O(1) persistent memory
// either way -- the shell-batched speedup for the Small-touching sectors, and Schwarz screening on
// top of the per-AO loops, are both deferred to a later performance pass.
//
// Derivation (verified algebraically, not guessed, before trusting this): 1/r12 does not see spin,
// so J only needs each spatial type's own TOTAL (summed-over-spin) density, and both spin blocks of
// a given spatial type receive the IDENTICAL J contribution (the same pattern X2C's own
// AoCholesky.cpp::x2cFockMatrix already uses for its single spatial basis, generalized here to two
// -- Large and Small):
//   J_f(a,b) = sum_g sum_{q,r in g's own basis} (a b | q r)_{f,g} * P_gg(r,q)
// K couples any pair of flavors (f,u) via whichever spatial integral type matches f's and u's own
// bases -- (LL|LL) if both Large, (SS|SS) if both Small, (LL|SS) [or its (AB|CD)=(CD|AB) symmetric
// equivalent] if mixed. Crucially this needs NO "mixed first-pair" integral type outside the
// existing (LL|LL)/(LL|SS)/(SS|SS) framework: in P_fu(q,r), q always lives in f's own basis and r
// in u's own basis, so (a,q) [a in f's basis] is always a same-type pair, and (b,r) [b in u's
// basis] is always a same-type pair too, independently:
//   K_fu(a,b) = sum_{q in f's basis, r in u's basis} (a q | b r)_{f,u} * P_fu(q,r)
// exactly the cross-flavor "K_opp" exchange RkbFockMatrix.h's own (already RKB-projected) formula
// documents for the analogous case. Spin only decides which density block is read and which output
// block is written; the spatial integral type depends only on which of {Large, Small} f and u
// belong to.
//
// Validated (not just derived) against the existing dense RkbTwoElectronTensor-based rkbFockMatrix
// on real molecules (LiH, CO) and a hand-built toy basis with d orbitals -- see
// tests/test_ukb_fock_direct.cpp.
Matrix<std::complex<double>> ukbFockTwoElectronDirect(const std::vector<BasisFunction>& large_basis,
                                                       const std::vector<BasisFunction>& small_basis,
                                                       const Matrix<std::complex<double>>& density_ukb);

// SCF_DIRECT_4C's "ERI source" for C4_DHF.h's generic dhfScfImpl -- the integral-direct counterpart
// of RkbTwoElectronTensor/RkbCholesky, except it holds no two-electron data at all (dense, packed,
// or Cholesky vectors): everything is recomputed fresh from large_basis/small_basis every Fock
// build via ukbFockTwoElectronDirect above. `v_total` embeds an RKB-basis (n_rkb x n_rkb) density
// into the raw Cartesian-Large/elementary-Small UKB basis ukbFockTwoElectronDirect operates in, and
// projects its raw two-electron output back -- v_total = v_sph * rkbEmbeddingMatrix(rkb_coefficients)
// (v_sph = SphericalTransform.h's sphericalLargeEmbedding, rkb_coefficients already row-projected to
// the final spherical large dimension), exactly as tests/test_ukb_fock_direct.cpp's own ctx.v_total
// is built and validated.
struct UkbDirectEriSource {
  std::vector<BasisFunction> large_basis, small_basis;
  Matrix<std::complex<double>> v_total;
};

// F = h_rkb + V_total^dagger * ukbFockTwoElectronDirect(..., V_total * P * V_total^dagger) * V_total.
// `density_matrix` is in the RKB spinor basis, same as h_rkb and every other rkbFockMatrix overload
// (RkbFockMatrix.h, RkbCholesky.h) -- this is simply a different way of computing the SAME quantity
// those overloads compute from a stored two-electron representation, so it can be dropped into
// C4_DHF.cpp's dhfScfImpl<Eri> (which only ever calls rkbFockMatrix(h_rkb, eri, density)) with no
// other changes to the SCF loop itself.
Matrix<std::complex<double>> rkbFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                            const UkbDirectEriSource& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_UKBFOCKMATRIXDIRECT_H
