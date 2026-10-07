#ifndef RERDMFT_NONRELSPINRIMOERI_H
#define RERDMFT_NONRELSPINRIMOERI_H

#include <cstddef>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// USE_RI counterpart of ClosedShellSpinOrbitals.h's closedShellSpinOrbitalTwoElectron, for
// FULL_OPTIMIZATION's generic Eri-templated Fock builders (Hessian_opt/PnofFock.h's
// pnofFockMatrix<T,Eri>, JkOnlyFock.h's jkOnlyFockMatrix<T,Eri>, PccdFock.h's pccdFockMatrix<T,
// Eri>, all already instantiated for T=double). Builds ONE real, SPATIAL-only RI tensor
// B(P,p,q) (p,q in [0,n_spatial)) from eri3_L and the current spatial coefficient half --
// NON_REL's FULL_OPTIMIZATION always ties alpha and beta together (spin restriction enforced on
// the accumulated orbital rotation itself, see main.cpp's nonrelSpatialRotation comment), so the
// current spin-orbital coefficient matrix's alpha-block columns [0,n_spatial) and beta-block
// columns [n_spatial,2*n_spatial) stay numerically identical throughout, and only ONE spatial
// half-transform is ever needed per Fock build (not two, not four).
//
// operator()(A,B,C,D) reproduces ClosedShellSpinOrbitals.h's closedShellSpinOrbitalTwoElectron's
// OWN spin-selection rule EXACTLY: spin(A)==spin(C) AND spin(B)==spin(D), else 0 (a spin-free
// two-electron operator never connects different spins) -- this is the piece that is genuinely
// DIFFERENT from the relativistic UKB/RKB case (UKB/RiMoEri.h's RiMoEri has no such selection: all
// spinor components mix freely there). The underlying spatial value is read through the SAME
// physics-notation convention Tensor4<double>/the dense NON_REL path already use: eri(A,B,C,D) =
// chemist(A,C,B,D) = sum_P B(P,spatial(A),spatial(C)) * B(P,spatial(B),spatial(D)).
class RiNonRelSpinMoEri {
 public:
  RiNonRelSpinMoEri() = default;
  RiNonRelSpinMoEri(std::size_t n_spatial, Matrix<double> b) : n_spatial_(n_spatial), b_(std::move(b)) {}

  std::size_t dim0() const { return 2 * n_spatial_; }
  std::size_t dim1() const { return 2 * n_spatial_; }
  std::size_t dim2() const { return 2 * n_spatial_; }
  std::size_t dim3() const { return 2 * n_spatial_; }
  std::size_t dim() const { return 2 * n_spatial_; }

  double operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
    if (a / n_spatial_ != c / n_spatial_) return 0.0;
    if (b / n_spatial_ != d / n_spatial_) return 0.0;
    const std::size_t sa = a % n_spatial_, sb = b % n_spatial_, sc = c % n_spatial_, sd = d % n_spatial_;
    const std::size_t n_aux = b_.rows();
    const std::size_t ac = sa * n_spatial_ + sc;
    const std::size_t bd = sb * n_spatial_ + sd;
    double sum = 0.0;
    for (std::size_t p = 0; p < n_aux; ++p) sum += b_(p, ac) * b_(p, bd);
    return sum;
  }

  // USE_RI's own FULL_OPTIMIZATION occupation-optimizer extension (see UKB/RiMoEri.h's identical
  // `rotated` for the role: Full_opt/FullOptimization.cpp's syncIntegrals rotates `eri` once per
  // macro-iteration via `rotateEri`). `u` is the FULL n_total x n_total spin-orbital rotation --
  // NON_REL's FULL_OPTIMIZATION always ties alpha and beta together (see the class comment above),
  // so `u` is guaranteed exactly blockdiag(u_spatial, u_spatial); only the top-left n_spatial x
  // n_spatial block is ever read (checked, not just assumed -- see the .cpp).
  RiNonRelSpinMoEri rotated(const Matrix<double>& u) const;

  // Embeds the spatial-only `b_` into a BLOCK-DIAGONAL `(n_aux, dim0()^2)` tensor over the FULL
  // spin-orbital space (same spatial block repeated at [alpha,alpha] and [beta,beta], zero at the
  // cross [alpha,beta]/[beta,alpha] blocks) -- needed by
  // Hessian_opt/JkOnlySigmaHessianVectorRi.h / HartreeExchangeSigmaHessianVectorRi.h's RI-direct
  // (B-only) Hessian-vector product, which assumes a flat `B(P,p,q)` over the SAME orbital space
  // the Hessian acts on, with no spin-selection rule of its own (see
  // [[project-ri-hessian-neo-design]] Stage 6). Confirmed by direct substitution that this
  // reproduces `operator()`'s own spin-selection rule EXACTLY: `eri(A,B,C,D) = sum_P
  // B_embedded(P,A,C)*B_embedded(P,B,D)` is zero whenever `A,C` (or `B,D`) are in different spin
  // blocks, because `B_embedded(P,A,C)` itself is zero there by construction -- not an
  // approximation, an exact reformulation of the same operator. O(n_aux*n_spatial^2) to build,
  // cheap (same cost class as `rotated()`).
  Matrix<double> embeddedB() const {
    const std::size_t n_aux = b_.rows();
    const std::size_t n = 2 * n_spatial_;
    Matrix<double> out(n_aux, n * n, 0.0);
    for (std::size_t P = 0; P < n_aux; ++P)
      for (std::size_t p = 0; p < n_spatial_; ++p)
        for (std::size_t q = 0; q < n_spatial_; ++q) {
          const double val = b_(P, p * n_spatial_ + q);
          out(P, p * n + q) = val;                                            // [alpha,alpha] block
          out(P, (n_spatial_ + p) * n + (n_spatial_ + q)) = val;              // [beta,beta] block
        }
    return out;
  }

 private:
  std::size_t n_spatial_ = 0;
  Matrix<double> b_;
};

// Builds `b_` above: b_(P, p*n_spatial+q) = sum_{mu,nu} c_spatial(mu,p) * c_spatial(nu,q) *
// eri3_L(mu,nu,P) (REAL, no conjugation -- T=double throughout NON_REL). `eri3_L` is
// AO_ints/ThreeCenterIntegrals.h's convention, shape (n_aux, n_spatial*n_spatial). `c_spatial` is
// n_spatial x n_spatial: the FIRST HALF (alpha-block columns) of the current spin-orbital
// coefficient matrix -- identical to the beta-block half, see the class comment above.
RiNonRelSpinMoEri buildRiNonRelSpinMoEri(const Matrix<double>& eri3_L, std::size_t n_spatial,
                                          const Matrix<double>& c_spatial);

// A dense view (DEBUG-only validation code -- see UKB/RiMoEri.h's identical denseOf(RiMoEri) for
// the full rationale): a plain O(n_total^4) element-by-element expansion.
inline Tensor4<double> denseOf(const RiNonRelSpinMoEri& eri) {
  const std::size_t n = eri.dim0();
  Tensor4<double> t(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) t(a, b, c, d) = eri(a, b, c, d);
  return t;
}

}  // namespace rerdmft

#endif  // RERDMFT_NONRELSPINRIMOERI_H
