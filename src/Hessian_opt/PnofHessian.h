#ifndef RERDMFT_PNOFHESSIAN_H
#define RERDMFT_PNOFHESSIAN_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "Matrix.h"
#include "Orb_subspaces.h"
#include "PNOFs.h"
#include "PnofFock.h"
#include "Tensor4.h"

namespace rerdmft {

// The "cheap" (no dense 2-RDM tensor, no O(n^2)-per-element general
// contraction) orbital-rotation Hessian element for a PNOF functional,
//   Hess_pq,rs = d^2E/dkappa_pq dkappa_rs,
// reusing Hessian_opt/HartreeExchangeHessian.h's own
// hartreeExchangeHessianElement, INCLUDING its L1/L2 pair-term
// extension: PnofFock.h's own buildPnofFullTwoRdm/buildPnofPairOf
// unfold a PNOF functional into EXACTLY HartreeExchangeHessian.h's own
// assumed FULL (H+X+L1+L2) ansatz -- so hartreeExchangeHessianElement's
// own already-validated formula (each of its raw terms builds directly
// on `fock`, `pnofFockMatrix`'s own generalized-Fock output, "the
// Hessian expression that uses the generalized Fock to be cheaper")
// applies here with NO new derivation at all, exactly how
// pnofFockMatrix itself already reuses hartreeExchangeFockMatrix
// directly for the gradient.
//
// The L1/L2 part is NOT optional here, unlike hartreeExchangeHessianElement's
// own default-empty parameters: an earlier version of this file passed
// PnofFock.h's Pi-pairing contribution folded into two_rdm_x instead
// (mathematically Fock/energy-equivalent, via an exact Kramers integral
// identity, but NOT Hessian-equivalent -- confirmed the hard way, via
// an e^kappa finite-difference check that failed with the folded
// version and matched to machine precision with this literal one; see
// PnofFock.h's own header comment for the full story).
//
// `fock` MUST be `pnofFockMatrix(functional, h, eri, geminals,
// occupations, relativistic)`'s own output (not recomputed here),
// exactly like hartreeExchangeHessianElement's own `fock` parameter --
// pass it in once and reuse it for every (p,q,r,s) element, rather than
// recomputing it per call. `buildPnofFullTwoRdm`/`buildPnofPairOf`
// themselves ARE rebuilt inside this function (cheap, O(n_geminals^2),
// negligible next to the Hessian cost they feed into) -- unlike `fock`,
// there is no benefit to hoisting them out for a single-element call;
// see `pnofHessianMatrix` below for the version that hoists them out
// once for a full matrix build instead.
template <typename T, typename Eri>
T pnofHessianElement(PnofFunctional functional, const Matrix<T>& h, const Eri& eri,
                      const std::vector<PnofGeminal>& geminals,
                      const std::vector<double>& occupations, bool relativistic,
                      const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                      std::size_t s);

// The FULL, dense orbital-rotation Hessian for a PNOF functional,
// indexed by the independent rotation pairs `pair_indices` (typically
// Hessian_opt/HartreeExchangeHessian.h's own `hessianPairIndices(h.rows())`)
// -- a thin wrapper building `buildPnofFullTwoRdm` ONCE and handing it,
// together with the caller-supplied `fock`, straight to
// Hessian_opt/HartreeExchangeHessian.h's own `hartreeExchangeHessianMatrix`
// (same O(n) per-element / O(n^5) total cost as that function, see its
// own header for the full complexity accounting).
template <typename T>
Matrix<T> pnofHessianMatrix(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                             const std::vector<PnofGeminal>& geminals,
                             const std::vector<double>& occupations, bool relativistic,
                             const Matrix<T>& fock,
                             const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

// CONVENTION NOTE (measured, main.cpp's pnofJointBlocksReport): off orbital
// stationarity the bare G_pq,rs of Eq. 9 (used by pnofHessianElement and
// the two functions below) is the TRANSPOSED sequential derivative
// relative to jkOnlyHessianElement's: pnofHessianElement(pq,rs) and
// pnofHessianElementImag(pq,rs) equal d/dkappa_pq of the gradient of pair
// (r,s) (agreement ~1e-9 vs finite differences of the gradient),
// pnofHessianElementMixed(pq,rs) equals d/dt_pq of the y-gradient of
// pair (r,s). They coincide with the true symmetric second derivative for
// pairs sharing no orbital index and at stationary points, but differ
// from it by half the (gradient-proportional) asymmetry for shared-index
// pairs at a non-stationary point (e.g. fixed HF orbitals with optimized
// occupations). pnofJointHessianMatrix below sums both orders and is
// convention independent -- USE IT (or symmetrize) for a Newton step.
//
// IMAGINARY-step (y = Im kappa_pq: kappa_pq=kappa_qp=iy) second derivative
//   -(G_pq,rs+G_pq,sr+G_qp,rs+G_qp,sr)
// and the MIXED real/imaginary one, hartreeExchangeHessianElementMixed's
// own formula i(G_rs,pq+G_rs,qp-G_sr,pq-G_sr,qp), for a PNOF functional,
// including the L1/L2 pair terms. Complex spinors only
// (T = std::complex<double>). Thin wrappers over HartreeExchangeHessian.h,
// exactly like pnofHessianElement (see the convention note above for
// what they measure off stationarity).
template <typename T, typename Eri>
T pnofHessianElementImag(PnofFunctional functional, const Matrix<T>& h, const Eri& eri,
                          const std::vector<PnofGeminal>& geminals,
                          const std::vector<double>& occupations, bool relativistic,
                          const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s);
template <typename T, typename Eri>
T pnofHessianElementMixed(PnofFunctional functional, const Matrix<T>& h, const Eri& eri,
                           const std::vector<PnofGeminal>& geminals,
                           const std::vector<double>& occupations, bool relativistic,
                           const Matrix<T>& fock, std::size_t p, std::size_t q, std::size_t r,
                           std::size_t s);

// The TRUE symmetric Hessian of E over the joint real parameters
// [t_I..., y_I...] (t_I=Re kappa_pq, y_I=Im kappa_pq, pairs p>q in
// `pair_indices`) at these occupations -- the matrix a Newton-Raphson
// orbital step needs -- for a PNOF functional; see
// hartreeExchangeSymmetricJointHessianMatrix for the exact definition
// (each block symmetrized from the same bare G_pq,rs, L1/L2 included).
// `fock` = pnofFockMatrix(...)'s own output. Complex spinors only.
// Hessian-VECTOR product of pnofJointHessianMatrix's own matrix, without ever forming it (see
// HartreeExchangeHessian.h's hartreeExchangeJointHessianVector, which this thin wrapper reuses
// exactly as pnofJointHessianMatrix reuses hartreeExchangeSymmetricJointHessianMatrix). `fock` =
// pnofFockMatrix(...)'s own output. `v`/the returned vector have size 2*pair_indices.size().
// Complex spinors only.
template <typename Eri>
std::vector<double> pnofJointHessianVector(
    PnofFunctional functional, const Matrix<std::complex<double>>& h,
    const Eri& eri, const std::vector<PnofGeminal>& geminals,
    const std::vector<double>& occupations, bool relativistic,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

// The diagonal of pnofJointHessianVector's matrix, joint ordering [t_0.., y_0..] (see
// HartreeExchangeHessian.h's hartreeExchangeJointHessianDiagonal). Complex spinors only.
template <typename Eri>
std::vector<double> pnofJointHessianDiagonal(
    PnofFunctional functional, const Matrix<std::complex<double>>& h,
    const Eri& eri, const std::vector<PnofGeminal>& geminals,
    const std::vector<double>& occupations, bool relativistic,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

Matrix<double> pnofJointHessianMatrix(
    PnofFunctional functional, const Matrix<std::complex<double>>& h,
    const Tensor4<std::complex<double>>& eri, const std::vector<PnofGeminal>& geminals,
    const std::vector<double>& occupations, bool relativistic,
    const Matrix<std::complex<double>>& fock,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft

#endif  // RERDMFT_PNOFHESSIAN_H
