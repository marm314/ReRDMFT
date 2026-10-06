#ifndef RERDMFT_THREECENTERINTEGRALS_H
#define RERDMFT_THREECENTERINTEGRALS_H

#include <cstddef>
#include <vector>

#include "LinearAlgebra.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Resolution-of-identity (RI/density-fitting) 3-center and 2-center Coulomb integrals, via
// libcint's cint3c2e_cart/cint2c2e_cart -- the same minimal-per-call libcint system style as
// ElectronRepulsion.h's twoElectronQuadruplet/twoElectronShellQuartet, just with 3 (or 2) shells
// instead of 4. See project memory project-ri-pauto-kr-validation for the derivation: the dense
// 4-component reference code (C4_DHF/RkbTwoElectron.cpp) never needs a Large-Small MIXED bra/ket
// pair, so RI only ever needs (LL|Q) and (SS|Q) -- this file builds either, for whichever single
// `basis` is passed; it never takes two different bases on the p,q side.

// (P|Q) for one pair of ENTIRE auxiliary shells (all of each shell's cartesian components, in one
// libcint call), with the same per-component relative-rescale correction as
// ElectronRepulsion.h's twoElectronShellQuartet (normalizeCartesianBasis has already rescaled each
// cartesian component of `aux_basis` by its own scalar by the time this is called). Returned flat,
// row-major (shell_p.count, shell_q.count), i.e. entry [i*shell_q.count+j].
std::vector<double> twoCenterAuxShellPair(const std::vector<BasisFunction>& aux_basis,
                                           const ShellInfo& shell_p, const ShellInfo& shell_q);

// (p q | P) for one specific pair of individually-normalized cartesian AOs p, q (each its own
// minimal "shell", exactly like twoElectronQuadruplet) and one ENTIRE auxiliary shell P (all of
// its cartesian components in one call). Returned as a vector of aux_shell.count values.
std::vector<double> threeCenterPairAuxShell(const BasisFunction& p, const BasisFunction& q,
                                             const std::vector<BasisFunction>& aux_basis,
                                             const ShellInfo& aux_shell);

// The full 2-center metric (P|Q), n_aux x n_aux, real symmetric -- loops over SHELL pairs of
// `aux_basis` (groupIntoShells is safe here: an auxiliary basis built by
// AuxiliaryBasis.cpp's buildAutoAuxiliaryBasis is, by construction, laid out as consecutive
// shells, unlike the Small-component UKB basis).
Matrix<double> auxMetric(const std::vector<BasisFunction>& aux_basis);

// The RAW (not metric-orthogonalized) 3-center integrals (p q | P) for p, q BOTH drawn from
// `basis`, P from `aux_basis`. AO-level loop on the p,q side (not shell-batched) -- deliberately
// uniform for either the Large basis (shell-consecutive) or the Small basis (NOT
// shell-consecutive -- see UkbFockMatrixDirect.h's own note on why its own integral-direct Fock
// build cannot shell-batch the Small-touching sectors either); the aux side IS shell-batched
// (`aux_basis` is always consecutive). Exploits p<->q symmetry (only p<=q evaluated). Returned
// aux-major: shape (aux_basis.size(), basis.size()^2), entry (P, p*basis.size()+q) -- this
// orientation is what UkbFockMatrixRi.h's per-aux-index RI-K half-transform needs (a contiguous
// basis.size() x basis.size() slice per aux row).
Matrix<double> threeCenterIntegralsRaw(const std::vector<BasisFunction>& basis,
                                        const std::vector<BasisFunction>& aux_basis);

// Builds X = canonicalOrthogonalize(auxMetric(aux_basis), threshold, report): the (n_aux_raw x
// n_aux_kept) matrix such that X^T (P|Q) X = I, near-singular aux directions genuinely dropped
// (same DIRAC-LOWGEN-style treatment this project already uses for AO overlaps elsewhere) rather
// than kept-and-inverted. Shared by every `basis` this aux_basis fits (e.g. both the Large-only
// and Small-only RI tensors below), so it is built ONCE and passed into riThreeCenterTensor.
Matrix<double> auxMetricOrthogonalization(const std::vector<BasisFunction>& aux_basis,
                                           double threshold = 1e-10, RankReductionReport* report = nullptr);

// The metric-orthogonalized RI 3-center tensor, (p q | P') with P' the orthogonalized aux index:
// transpose(metric_x) * threeCenterIntegralsRaw(basis, aux_basis) (a single GEMM -- MOLGW's own
// `eri_3center = eri_3center_tmp @ eri_2center_sqrtinv`, folding the metric's inverse square root
// into the stored tensor so no further (P|Q)^-1 is ever needed at Fock-build time). Shape
// (metric_x.cols(), basis.size()^2), same (P, p*n+q) row-major convention as
// threeCenterIntegralsRaw.
Matrix<double> riThreeCenterTensor(const std::vector<BasisFunction>& basis,
                                    const std::vector<BasisFunction>& aux_basis,
                                    const Matrix<double>& metric_x);

// Applies a REAL linear transform `t` (n_ao x n_out, e.g. a Cartesian->spherical/LOWGEN reduction
// -- main.cpp's own `nonrel_transform`/`large_transform_final`) to BOTH AO legs of an RI 3-center
// tensor already built in `t`'s own INPUT (n_ao-dimensional, raw Cartesian) basis:
//   eri3_out(P, p, q) = sum_{mu,nu} t(mu,p) * t(nu,q) * eri3_in(P, mu, nu)
// Needed whenever the caller's own SCF loop (NON_REL/X2C: no RKB/UKB-style coefficient embedding
// to absorb this the way C4_SPINOR's own `v_total` does) works in a DIFFERENT-dimensioned basis
// than the raw Cartesian one libcint natively evaluates integrals in -- i.e. whenever `t` is not
// simply the identity. Same per-aux-index-P small-GEMM-loop pattern as UKB/RiMoEri.h's own
// transformBlock, real-only (no complex split needed, `t` and `eri3_in` are both real). Shape
// (eri3_in.rows(), t.cols()^2), same (P, p*n_out+q) row-major convention as `riThreeCenterTensor`.
Matrix<double> transformRiThreeCenterAoLegs(const Matrix<double>& eri3_in, std::size_t n_ao,
                                             const Matrix<double>& t);

}  // namespace rerdmft

#endif  // RERDMFT_THREECENTERINTEGRALS_H
