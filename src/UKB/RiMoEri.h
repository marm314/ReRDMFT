#ifndef RERDMFT_RIMOERI_H
#define RERDMFT_RIMOERI_H

#include <complex>
#include <cstddef>

#include "Matrix.h"
#include "Tensor4.h"
#include "UkbFockMatrixRi.h"

namespace rerdmft {

// A generic (physics-notation) two-electron integral accessor, eri(A,B,C,D) == <A B|C D> ==
// chemist(A,C,B,D) (this project's standard convention -- see e.g. NON_REL/NonRelHartreeFock.h,
// C4_DHF/RkbTwoElectron.h), backed by a resolution-of-identity (RI) 3-center tensor already
// transformed into the CURRENT orbital/natural-spinor basis -- same role as Tensor4<T>/
// SymmetricEri<T>/CholeskyEri<T> (the SAME `operator()`/`dim0()..dim3()` interface
// Hessian_opt/PnofFock.h's pnofFockMatrix<T,Eri>, Hessian_opt/JkOnlyFock.h's jkOnlyFockMatrix<T,
// Eri>, and Hessian_opt/PccdFock.h's pccdFockMatrix<T,Eri> are already templated over), so it
// drops into any of those with NO changes to their own, already-validated code.
//
// Built from `b_`, shape (n_aux_kept, n_total*n_total), b_(P, p*n_total+q) =
//   B(P,p,q) := sum_{mu,nu} conj(C(mu,p)) * C(nu,q) * (mu,nu|P)_AO
// (a chemist-convention MO-RI half-transform: conjugated on the FIRST leg, plain on the second,
// matching chemist(p,q|...) = integral conj(phi_p(1)) phi_q(1) (1/r12) ...). Then
//   chemist(p,q|r,s)_MO = sum_P B(P,p,q) * B(P,r,s)
//   eri(A,B,C,D) = <A B|C D> = chemist(A,C,B,D) = sum_P B(P,A,C) * B(P,B,D)
// -- a cheap O(n_aux) dot product per element, with the O(n_aux*n_total^2) transform itself paid
// ONCE per Fock build (buildRiMoEri below), not once per element or per density.
class RiMoEri {
 public:
  RiMoEri() = default;
  RiMoEri(std::size_t n_total, Matrix<std::complex<double>> b) : n_total_(n_total), b_(std::move(b)) {}

  std::size_t dim0() const { return n_total_; }
  std::size_t dim1() const { return n_total_; }
  std::size_t dim2() const { return n_total_; }
  std::size_t dim3() const { return n_total_; }
  std::size_t dim() const { return n_total_; }

  // Raw access to the RI tensor itself (shape n_aux x n_total^2, `b(P,p*n_total+q)==B(P,p,q)`) --
  // needed by Hessian_opt/JkOnlySigmaHessianVectorRi.h / HartreeExchangeSigmaHessianVectorRi.h's
  // RI-direct (B-only, never-dense-eri) Hessian-vector product, see
  // [[project-ri-hessian-neo-design]] Stage 5 (wiring). Everything else in this class only ever
  // reads `b_` through `operator()`'s O(n_aux)-per-element contraction; this exposes the tensor
  // itself for callers that contract against it directly (O(n_aux*n^3)-style, not O(n_aux) per
  // single element).
  const Matrix<std::complex<double>>& b() const { return b_; }

  std::complex<double> operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
    std::complex<double> sum{};
    const std::size_t n_aux = b_.rows();
    const std::size_t ac = a * n_total_ + c;
    const std::size_t bd = b * n_total_ + d;
    for (std::size_t p = 0; p < n_aux; ++p) sum += b_(p, ac) * b_(p, bd);
    return sum;
  }

  // USE_RI's own FULL_OPTIMIZATION occupation-optimizer extension (Occ_opt/pCCD.h's
  // buildPccdCoefficients, Occ_opt/PNOFs.h's pnofElectronicEnergy/pnofOccupationGradient/Hessian,
  // Occ_opt/OccupationEnergy.h's jkFunctional*, all already generic over Eri): the macro loop's
  // own occupation re-optimization step (Full_opt/FullOptimization.cpp's syncIntegrals) rotates
  // `eri` once per macro-iteration via `rotateEri`, exactly like CholeskyEri's own `.rotated()` --
  // B'(P,p,q) = sum_{a,b} conj(U(a,p)) * B(P,a,b) * U(b,q), the SAME half-transform
  // buildRiMoEri's own per-aux-index loop already does, just applied to the ALREADY-MO-basis B
  // tensor with a SQUARE n_total x n_total U instead of the AO-basis legs -- no AO-level integral
  // is ever touched again after construction.
  RiMoEri rotated(const Matrix<std::complex<double>>& u) const;

  // C4_DHF's own no-pair trim (Full_opt/FullOptimization.cpp's positiveBlock overloads, same role
  // as Tensor4<T>::positiveBlock/SymmetricEri<T>::positiveBlock): extracts the submatrix spanning
  // [off, n_total) on both MO legs -- the B tensor's AUX index is untouched (same n_aux), only its
  // two MO legs are restricted, O(n_aux*(n_total-off)^2) copy, no AO-level data involved.
  RiMoEri positiveBlock(std::size_t off) const;

 private:
  std::size_t n_total_ = 0;
  Matrix<std::complex<double>> b_;
};

// Builds the MO-RI tensor `b_` RiMoEri wraps, at the CURRENT orbital coefficients `c_ukb` (UKB AO
// basis, n_ukb x n_total -- e.g. `eri.v_total * c_rkb` for the current rotated natural spinors).
// Transforms `eri.eri3_LL`/`eri.eri3_SS` (fixed, built once for the whole calculation) into the
// n_total-dimensional MO basis fresh every call -- the per-ADAM-step cost RI_4C's functional-level
// integration pays instead of either a full AO-direct integral pass (UkbGenFockPrimitives.h) or an
// O(n^5) stored-MO-tensor rotation (IntegralRotation.h). No (LS|Q) term exists or is needed (see
// project memory project-ri-pauto-kr-validation): the 4 UKB spin/spatial blocks [Large-alpha,
// Large-beta, Small-alpha, Small-beta] are each separately half-transformed against their own
// eri3_LL/eri3_SS and the 4 results summed, mirroring UkbFockMatrixRi.h's own riFockTwoElectronDirect
// block decomposition.
RiMoEri buildRiMoEri(const RiDirectEriSource& eri, const Matrix<std::complex<double>>& c_ukb);

// A dense view (DEBUG-only validation code that needs a Tensor4, e.g. main.cpp's/Full_opt/
// FullOptimization.cpp's own generic buildFunctionalReport<T,Eri>/runFullOptimization<T,Eri>
// bodies): a plain O(n_total^4) element-by-element expansion, same role as CholeskyEri.h's own
// denseOf(CholeskyEri<T>)/denseOf(SymmetricEri<T>). Never actually REACHED at runtime (DEBUG is
// mutually exclusive with USE_RI, Input.cpp's own check), but the lambda bodies/branches that
// call it are compiled regardless of whether they are ever taken, so this must exist.
inline Tensor4<std::complex<double>> denseOf(const RiMoEri& eri) {
  const std::size_t n = eri.dim0();
  Tensor4<std::complex<double>> t(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) t(a, b, c, d) = eri(a, b, c, d);
  return t;
}

}  // namespace rerdmft

#endif  // RERDMFT_RIMOERI_H
