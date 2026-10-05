// Fast, focused cross-check of RkbTransformation.cpp's analytic rkb_coefficients (the sigma.p
// expansion) against the earlier numerical construction C = M * S^{-1}, where
// M(Large_p, elementary_r) = <elementary_r | sigma.p-component | Large_p> and S = overlap of the
// elementary (UKB) small basis -- on LiH/6-31G and a hand-built toy basis with a d shell. Does NOT
// touch two-electron integrals at all (that question -- whether rkbTwoElectronIntegrals' construction
// has a genuine gap for cross-RKB-flavor chemist pairs -- is tracked separately); this test is only
// about whether the expansion coefficients themselves are correct. Confirmed on H2/STO-3G already
// (tests/test_ukb_fock_direct.cpp): the two constructions agree in magnitude and sparsity pattern
// exactly, differing only by a benign overall global sign (irrelevant to physics, since C only ever
// enters bilinear forms C^dagger(.)C). Build/run: make test_rkb_coefficients_cross_check
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "BasisSet.h"
#include "Input.h"
#include "Integrals.h"
#include "LinearAlgebra.h"
#include "Matrix.h"
#include "MolecularBasis.h"
#include "NablaIntegrals.h"
#include "RkbTransformation.h"
#include "SmallComponentBasis.h"

using namespace rerdmft;

namespace {

int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

constexpr double kAngstromToBohr = 1.0 / 0.52917721067;
using C = std::complex<double>;

// M(Large_p-spin, elementary_r-spin) = <elementary_r | sigma.p-component | Large_p>, the (2*nl x
// 2*ns) target matrix the analytic rkb_coefficients C is SUPPOSED to satisfy via C*S_blockdiag = M
// (S_blockdiag = diag(S_elem, S_elem), S_elem = elementary-basis overlap). nablaIntegral(bra,ket)
// differentiates the BRA (libcint's own "<NABLA i|j>" convention), so Large_p must be passed as the
// bra to get <(d/dx_k)Large_p | elementary_r>.
Matrix<C> buildM(const std::vector<BasisFunction>& large_cart, const std::vector<BasisFunction>& small_cart) {
  const std::size_t nl = large_cart.size(), ns = small_cart.size();
  const C i_unit(0.0, 1.0);
  Matrix<C> m(2 * nl, 2 * ns, C{});
  for (std::size_t p = 0; p < nl; ++p) {
    for (std::size_t r = 0; r < ns; ++r) {
      const std::array<double, 3> nab = nablaIntegral(large_cart[p], small_cart[r]);
      // sigma.p|Large_p,alpha> = -i*D_z(Large_p)|alpha> + [-i*D_x(Large_p)+D_y(Large_p)]|beta>
      m(p, r) += -i_unit * nab[2];                // (La_p, Sa_r)
      m(p, ns + r) += -i_unit * nab[0] + nab[1];  // (La_p, Sb_r)
      // sigma.p|Large_p,beta> = [-i*D_x(Large_p)-D_y(Large_p)]|alpha> + i*D_z(Large_p)|beta>
      m(nl + p, r) += -i_unit * nab[0] - nab[1];  // (Lb_p, Sa_r)
      m(nl + p, ns + r) += i_unit * nab[2];       // (Lb_p, Sb_r)
    }
  }
  return m;
}

void runSystem(const std::string& name, std::vector<BasisFunction> large_cart) {
  normalizeCartesianBasis(large_cart);
  SmallComponentBasis small_basis;
  small_basis.build(large_cart);
  const std::vector<BasisFunction>& small_cart = small_basis.functions();
  const std::size_t ns = small_cart.size();
  const Matrix<C> rc_analytic = rkbCoefficients(large_cart, small_basis.termIndex());
  const Matrix<C> m_target = buildM(large_cart, small_cart);
  const Matrix<double> s_elem = overlapMatrix(small_cart);

  // Check C*S == M directly -- the DEFINING relationship, valid regardless of whether S is
  // invertible (the elementary basis is generally linearly dependent for a real multi-shell atom,
  // e.g. a p-shell's own lowering piece can coincide with a genuine s-shell's raising piece --
  // exactly why the project moved away from an M*S^-1 CONSTRUCTION of C in the first place; trying to
  // invert S to SOLVE for C would pick one of several equally-valid representations and could show a
  // spurious mismatch purely from that ambiguity, not from an actual error in C).
  Matrix<C> cs(rc_analytic.rows(), 2 * ns, C{});
  for (std::size_t row = 0; row < rc_analytic.rows(); ++row)
    for (std::size_t col = 0; col < ns; ++col) {
      C sum_a{}, sum_b{};
      for (std::size_t k = 0; k < ns; ++k) {
        sum_a += rc_analytic(row, k) * s_elem(k, col);
        sum_b += rc_analytic(row, ns + k) * s_elem(k, col);
      }
      cs(row, col) = sum_a;
      cs(row, ns + col) = sum_b;
    }
  // H2/STO-3G (tests/test_ukb_fock_direct.cpp) already established analytic C = -1 * (M*S^-1), a
  // benign global sign (nablaIntegral's bra-differentiation convention vs this file's own sign
  // choice -- irrelevant to physics, C only enters bilinear forms). Check both signs, report the
  // smaller.
  double worst_plus = 0.0, worst_minus = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < cs.rows(); ++i)
    for (std::size_t j = 0; j < cs.cols(); ++j) {
      worst_plus = std::max(worst_plus, std::abs(cs(i, j) - m_target(i, j)));
      worst_minus = std::max(worst_minus, std::abs(cs(i, j) + m_target(i, j)));
      scale = std::max(scale, std::abs(m_target(i, j)));
    }
  const double worst = std::min(worst_plus, worst_minus);
  const char* sign = worst_minus < worst_plus ? "-1" : "+1";
  std::printf("[%s] n_large_cart=%zu n_small=%zu  max|C*S - (%s)*M| = %.3e (max|M| = %.3e)\n", name.c_str(),
              large_cart.size(), ns, sign, worst, scale);
  check(worst < 1e-8 * std::max(scale, 1.0), name + ": rkb_coefficients satisfies C*S = +-M exactly");
}

}  // namespace

int main() {
  {
    BasisSet basis_set;
    basis_set.read("examples/lih-6-31g.gbs");
    const std::vector<Atom> geometry = {{"Li", 0.0, 0.0, 0.0}, {"H", 0.0, 0.0, 1.5949 * kAngstromToBohr}};
    MolecularBasis mb;
    mb.build(geometry, basis_set);
    runSystem("LiH/6-31G", mb.functions());
  }
  {
    // Hand-built toy basis: carbon with s and d shells, hydrogen with an s shell -- exercises the
    // d-shell's 6 Cartesian components (lx+ly+lz=2), which RkbDerivativeTerms.h's "lowering/raising
    // per axis" splitting must handle correctly (e.g. a d_xx component's x-direction lowering piece
    // is an s-type term, not simply absent the way it is for a p-type component).
    std::vector<BasisFunction> basis;
    auto addShell = [&](const std::string& elem, int l, double x, double y, double z, std::vector<double> exps,
                         std::vector<double> coefs) {
      for (const CartesianExponents& c : cartesianComponents(l)) {
        BasisFunction fn;
        fn.element = elem;
        fn.x = x;
        fn.y = y;
        fn.z = z;
        fn.l = l;
        fn.cartesian = c;
        fn.exponents = exps;
        fn.coefficients = coefs;
        basis.push_back(fn);
      }
    };
    addShell("C", 0, 0.0, 0.0, 0.0, {3.0475249, 0.6834831}, {0.1543290, 0.5353281});
    addShell("C", 2, 0.0, 0.0, 0.0, {0.8}, {1.0});
    addShell("H", 0, 0.0, 0.0, 1.4, {1.0}, {1.0});
    runSystem("hand-built C(s,d)+H(s)", basis);
  }

  std::printf("\n%d / %d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
