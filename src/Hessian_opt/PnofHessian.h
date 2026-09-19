#ifndef RERDMFT_PNOFHESSIAN_H
#define RERDMFT_PNOFHESSIAN_H

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
template <typename T>
T pnofHessianElement(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
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

}  // namespace rerdmft

#endif  // RERDMFT_PNOFHESSIAN_H
