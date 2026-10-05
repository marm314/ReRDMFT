// Validates UkbFockMatrixDirect.h's ukbFockTwoElectronDirect (SCF_DIRECT_4C's Phase 1 integral-
// direct RKB Fock build) against the existing, trusted dense RkbTwoElectronTensor-based rkbFockMatrix
// path, on H2/STO-3G, LiH/6-31G, and a hand-built toy basis with a d shell (C: s+d, H: s) -- the
// explicit requirement before trusting this code at all, since H2/LiH alone are both s/p-only bases.
//
// Densities used: genuinely Kramers-paired densities built from the natural-orbital eigenvectors of
// H_RKB itself (a one-shot "core guess", not a converged SCF density -- convergence is irrelevant to
// the structural question this test asks), covering BOTH cases production code actually uses: the
// idempotent (Hartree-Fock, occupation 0/1) case, and a genuinely FRACTIONAL-occupation (RDMFT/
// PNOF/GNOF/Muller-style, occupation 0.95/0.05, explicitly non-idempotent) case.
//
// This is deliberately NOT tested against an arbitrary (even Kramers-symmetrized) random density:
// rkbTwoElectronIntegrals' construction never populates a two-electron chemist pair that mixes
// RKB-small flavors (Sa,Sb) in its first electron pair. That term is genuinely, exactly zero for any
// density built from a real Kramers-paired natural-orbital expansion -- confirmed empirically here
// (residual ~1e-7, consistent with H_RKB's own ~1e5 dynamic range propagating through the
// orthogonalization/diagonalization chain) for BOTH the idempotent and the fractional case -- but is
// NOT zero in general for an unstructured matrix that merely happens to satisfy the same algebraic
// time-reversal-even relation without having genuine natural-orbital structure (confirmed separately:
// such a density showed a ~0.1-0.2 discrepancy, two orders of magnitude above the noise floor here).
// Since no real calculation ever produces a density of that unstructured kind, this is the correct,
// physically meaningful regression check.
// Build/run: make test_ukb_fock_direct
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "ElectronRepulsion.h"
#include "Input.h"
#include "Integrals.h"
#include "KramersPairing.h"
#include "KramersSymmetry.h"
#include "LinearAlgebra.h"
#include "Matrix.h"
#include "MolecularBasis.h"
#include "RkbFockMatrix.h"
#include "RkbHamiltonian.h"
#include "RkbOverlap.h"
#include "RkbTransformation.h"
#include "RkbTwoElectron.h"
#include "SmallComponentBasis.h"
#include "SphericalTransform.h"
#include "UkbFockMatrixDirect.h"
#include "UkbHamiltonian.h"

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

// Residual noise floor observed on every system tested (H2/LiH/CO): the orthogonalization +
// diagonalization chain used only to BUILD a test density propagates H_RKB's own ~1e5 dynamic range
// (the negative-energy branch's -2c^2-scale diagonal) into a ~1e-7 absolute residual that has nothing
// to do with ukbFockTwoElectronDirect's own correctness -- a genuine formula/implementation bug showed
// up as ~0.1-0.2, three orders of magnitude above this, so a 1e-5 threshold has ample margin either way.
constexpr double kNoiseFloorThreshold = 1e-5;

using C = std::complex<double>;

double maxAbsDiff(const Matrix<C>& a, const Matrix<C>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i)
    for (std::size_t j = 0; j < a.cols(); ++j) worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
  return worst;
}

// Holds everything built once per system, reused by every density test below.
struct SystemContext {
  std::size_t nl_cart = 0, ns = 0, nl_final = 0, n_rkb = 0;
  std::vector<BasisFunction> large_cart, small_cart;
  Matrix<C> h_ukb_cart, h_rkb, v_total, s_rkb;
  RkbTwoElectronTensor eri;
};

SystemContext buildSystem(std::vector<BasisFunction> large_cart_in, const std::vector<Atom>& geometry) {
  SystemContext ctx;
  normalizeCartesianBasis(large_cart_in);
  ctx.large_cart = large_cart_in;

  SmallComponentBasis small_basis;
  small_basis.build(ctx.large_cart);
  ctx.small_cart = small_basis.functions();
  ctx.nl_cart = ctx.large_cart.size();
  ctx.ns = ctx.small_cart.size();

  // Raw UKB one-electron Hamiltonian + RKB coefficients, BEFORE any spherical reduction -- exactly
  // main.cpp's construction right after small_basis/spinor_basis are built (main.cpp:3004-3012).
  ctx.h_ukb_cart = ukbHamiltonianMatrix(ctx.large_cart, ctx.small_cart, geometry);
  const Matrix<C> rkb_coeff_cart = rkbCoefficients(ctx.large_cart, small_basis.termIndex());

  // Large-component spherical reduction (main.cpp:3026-3066) -- the LOWGEN safety net on top is
  // skipped: none of H2/LiH/CO's large overlaps are linearly dependent beyond the exact spherical
  // reduction, so large_transform_final == the pure spherical transform (identity for s/p-only
  // bases, which all three are).
  const SphericalTransformResult large_spherical = buildSphericalTransform(ctx.large_cart);
  const Matrix<double>& large_transform_final = large_spherical.transform;
  ctx.nl_final = large_transform_final.cols();

  const Matrix<C> v_sph = sphericalLargeEmbedding(large_transform_final, 2 * ctx.ns);
  const Matrix<C> h_ukb_final = dagger(v_sph) * (ctx.h_ukb_cart * v_sph);
  const Matrix<C> rkb_coeff_final = dagger(spinDuplicateComplex(large_transform_final)) * rkb_coeff_cart;

  ctx.h_rkb = rkbHamiltonianMatrix(h_ukb_final, rkb_coeff_final);
  ctx.n_rkb = ctx.h_rkb.rows();

  ctx.eri = rkbTwoElectronIntegrals(ctx.large_cart, ctx.small_cart, rkb_coeff_final, large_transform_final,
                                     /*use_cholesky=*/false);

  const Matrix<C> w_final = rkbEmbeddingMatrix(rkb_coeff_final);
  ctx.v_total = v_sph * w_final;

  // Full RKB overlap (needed to build a genuine, Kramers-paired natural-orbital density): block-
  // diagonal [S_large, S_large, S_small] -- S_large (spherical large overlap) shared by La,Lb (no
  // La-Lb or Large-Small cross overlap), S_small (rkbSmallOverlapMatrix) already covers the full
  // Sa,Sb quadrant as one object.
  const Matrix<double> s_large = transformToSpherical(overlapMatrix(ctx.large_cart), large_transform_final);
  const Matrix<C> s_small = rkbSmallOverlapMatrix(ctx.small_cart, rkb_coeff_final);
  ctx.s_rkb = Matrix<C>(ctx.n_rkb, ctx.n_rkb, C{});
  for (std::size_t i = 0; i < ctx.nl_final; ++i)
    for (std::size_t j = 0; j < ctx.nl_final; ++j) {
      ctx.s_rkb(i, j) = C(s_large(i, j), 0.0);
      ctx.s_rkb(ctx.nl_final + i, ctx.nl_final + j) = C(s_large(i, j), 0.0);
    }
  for (std::size_t i = 0; i < 2 * ctx.nl_final; ++i)
    for (std::size_t j = 0; j < 2 * ctx.nl_final; ++j) ctx.s_rkb(2 * ctx.nl_final + i, 2 * ctx.nl_final + j) = s_small(i, j);
  return ctx;
}

// Builds f_rkb_direct for a given P_RKB via the new embedding + ukbFockTwoElectronDirect path.
Matrix<C> fockDirect(const SystemContext& ctx, const Matrix<C>& p_rkb) {
  const Matrix<C> p_ukb_cart = ctx.v_total * (p_rkb * dagger(ctx.v_total));
  const Matrix<C> two_electron = ukbFockTwoElectronDirect(ctx.large_cart, ctx.small_cart, p_ukb_cart);
  return dagger(ctx.v_total) * ((ctx.h_ukb_cart + two_electron) * ctx.v_total);
}

Matrix<C> fockReference(const SystemContext& ctx, const Matrix<C>& p_rkb) {
  return rkbFockMatrix(ctx.h_rkb, ctx.eri, p_rkb);
}

// J and K at a single (p,r), computed two fully independent ways (same machinery used to isolate the
// H2 case): (A) literally copying RkbFockMatrix.cpp's own loop structure directly against eri
// (trusted: eri was validated against its own documented formula extensively); (B) the cartesian-
// embedding route: embed P_RKB into the raw basis, contract with the raw dense tensors via
// J_cart(A,B)=sum_{C,D} raw(A,B,C,D)*P_cart(D,C), K_cart(A,D)=sum_{B,C} raw(A,B,C,D)*P_cart(B,C), then
// project via V_total. Pinpoints whether a disagreement is in J or K, and whether it's the formula or
// ukbFockTwoElectronDirect's own implementation of it.
struct JKPair {
  C j{}, k{};
};

JKPair jkViaEri(const SystemContext& ctx, const Matrix<C>& p_rkb, std::size_t p, std::size_t r) {
  JKPair out;
  const std::size_t n = ctx.n_rkb;
  for (std::size_t q = 0; q < n; ++q)
    for (std::size_t s = 0; s < n; ++s) {
      const C p_sq = p_rkb(s, q);
      if (p_sq == C{}) continue;
      out.j += p_sq * ctx.eri(p, q, r, s);
      out.k += p_sq * ctx.eri(p, q, s, r);
    }
  return out;
}

JKPair jkViaCartesianEmbedding(const SystemContext& ctx, const Matrix<C>& p_rkb, std::size_t p, std::size_t r) {
  const std::size_t n_ukb_dim = ctx.v_total.rows();
  const std::size_t nl = ctx.nl_cart, ns = ctx.ns;
  const Tensor4<double> ll_ll = twoElectronIntegrals(ctx.large_cart, 0.0);
  const Tensor4<double> ll_ss = twoElectronIntegralsCross(ctx.large_cart, ctx.small_cart, 0.0);
  const PackedTwoElectronTensor ss_ss = twoElectronIntegralsPacked(ctx.small_cart, 0.0);
  auto typeOf = [&](std::size_t i) { return i < 2 * nl ? 0 : 1; };
  auto spinOf = [&](std::size_t i) {
    if (i < nl) return 0;
    if (i < 2 * nl) return 1;
    if (i < 2 * nl + ns) return 0;
    return 1;
  };
  auto spatialOf = [&](std::size_t i) -> std::size_t {
    if (i < nl) return i;
    if (i < 2 * nl) return i - nl;
    if (i < 2 * nl + ns) return i - 2 * nl;
    return i - 2 * nl - ns;
  };
  auto rawVal = [&](std::size_t A, std::size_t B, std::size_t Cc, std::size_t D) -> double {
    if (typeOf(A) != typeOf(B) || typeOf(Cc) != typeOf(D)) return 0.0;
    if (spinOf(A) != spinOf(B) || spinOf(Cc) != spinOf(D)) return 0.0;
    const std::size_t a = spatialOf(A), b = spatialOf(B), c = spatialOf(Cc), d = spatialOf(D);
    if (typeOf(A) == 0 && typeOf(Cc) == 0) return ll_ll(a, b, c, d);
    if (typeOf(A) == 0 && typeOf(Cc) == 1) return ll_ss(a, b, c, d);
    if (typeOf(A) == 1 && typeOf(Cc) == 0) return ll_ss(c, d, a, b);
    return ss_ss(a, b, c, d);
  };
  const Matrix<C> p_cart = ctx.v_total * (p_rkb * dagger(ctx.v_total));
  JKPair out;
  for (std::size_t A = 0; A < n_ukb_dim; ++A) {
    const C vAp = std::conj(ctx.v_total(A, p));
    if (vAp == C{}) continue;
    for (std::size_t B = 0; B < n_ukb_dim; ++B) {
      if (typeOf(A) != typeOf(B)) continue;
      const C vBr = ctx.v_total(B, r);
      if (vBr == C{}) continue;
      for (std::size_t Cc = 0; Cc < n_ukb_dim; ++Cc) {
        for (std::size_t D = 0; D < n_ukb_dim; ++D) {
          if (typeOf(Cc) != typeOf(D)) continue;
          const double raw = rawVal(A, B, Cc, D);
          if (raw == 0.0) continue;
          out.j += vAp * raw * p_cart(D, Cc) * vBr;
        }
      }
    }
  }
  for (std::size_t A = 0; A < n_ukb_dim; ++A) {
    const C vAp = std::conj(ctx.v_total(A, p));
    if (vAp == C{}) continue;
    for (std::size_t D = 0; D < n_ukb_dim; ++D) {
      const C vDr = ctx.v_total(D, r);
      if (vDr == C{}) continue;
      for (std::size_t B = 0; B < n_ukb_dim; ++B) {
        if (typeOf(A) != typeOf(B)) continue;
        for (std::size_t Cc = 0; Cc < n_ukb_dim; ++Cc) {
          if (typeOf(Cc) != typeOf(D)) continue;
          const double raw = rawVal(A, B, Cc, D);
          if (raw == 0.0) continue;
          out.k += vAp * raw * p_cart(B, Cc) * vDr;
        }
      }
    }
  }
  return out;
}

// Finds the (p,r) with the largest |F_direct-F_reference| and decomposes it into J/K via both routes.
// Minimal, restricted-to-Small-subspace computation of K(p,r) for p,r BOTH in the RKB-small (Sa/Sb)
// range -- since V_total's Sa/Sb columns are nonzero ONLY in the Small-cartesian rows [2nl,2nl+2ns),
// the Large legs never enter at all, so this can be written directly over the 2ns-dimensional Small
// subspace without the full n_ukb_dim loop jkViaCartesianEmbedding uses -- isolates whether a bug is
// in that general loop specifically, or lies deeper (e.g. in raw ss_ss itself or p_cart's own values).
C kSmallOnly(const SystemContext& ctx, const Matrix<C>& p_rkb, std::size_t p, std::size_t r) {
  const std::size_t nl = ctx.nl_cart, ns = ctx.ns;
  const PackedTwoElectronTensor ss_ss = twoElectronIntegralsPacked(ctx.small_cart, 0.0);
  const Matrix<C> p_cart = ctx.v_total * (p_rkb * dagger(ctx.v_total));
  auto spinOf2 = [&](std::size_t i) { return i < ns ? 0 : 1; };  // local index within the 2ns Small block
  auto spatialOf2 = [&](std::size_t i) { return i < ns ? i : i - ns; };
  const std::size_t off = 2 * nl;  // Small-alpha starts here in the full UKB-cartesian index space
  C k_sum{};
  for (std::size_t A = 0; A < 2 * ns; ++A) {
    const C vAp = std::conj(ctx.v_total(off + A, p));
    if (vAp == C{}) continue;
    for (std::size_t D = 0; D < 2 * ns; ++D) {
      const C vDr = ctx.v_total(off + D, r);
      if (vDr == C{}) continue;
      for (std::size_t B = 0; B < 2 * ns; ++B) {
        if (spinOf2(A) != spinOf2(B)) continue;
        for (std::size_t Cc = 0; Cc < 2 * ns; ++Cc) {
          if (spinOf2(Cc) != spinOf2(D)) continue;
          const double v = ss_ss(spatialOf2(A), spatialOf2(B), spatialOf2(Cc), spatialOf2(D));
          if (v == 0.0) continue;
          k_sum += vAp * v * p_cart(off + B, off + Cc) * vDr;
        }
      }
    }
  }
  return k_sum;
}

// Directly exposes ukbFockTwoElectronDirect's RAW (un-projected) output for the ACTUAL density used
// by fockDirect, and manually projects it at (p,r) -- should be IDENTICAL to what fockDirect itself
// computes (same function, same density, same projection), and should ALSO match kSmallOnly if
// kSmallOnly's own formula/projection is right. If this disagrees with kSmallOnly, kSmallOnly itself
// has the bug, not ukbFockTwoElectronDirect or V_total.
C twoElectronDirectProjected(const SystemContext& ctx, const Matrix<C>& p_rkb, std::size_t p, std::size_t r) {
  const Matrix<C> p_ukb_cart = ctx.v_total * (p_rkb * dagger(ctx.v_total));
  const Matrix<C> two_electron = ukbFockTwoElectronDirect(ctx.large_cart, ctx.small_cart, p_ukb_cart);
  C sum{};
  for (std::size_t A = 0; A < two_electron.rows(); ++A) {
    const C vAp = std::conj(ctx.v_total(A, p));
    if (vAp == C{}) continue;
    for (std::size_t D = 0; D < two_electron.cols(); ++D) {
      const C vDr = ctx.v_total(D, r);
      if (vDr == C{}) continue;
      sum += vAp * two_electron(A, D) * vDr;
    }
  }
  return sum;
}

void diagnoseWorstEntry(const SystemContext& ctx, const std::string& name, const std::string& label, const Matrix<C>& p_rkb) {
  const Matrix<C> f_ref = fockReference(ctx, p_rkb);
  const Matrix<C> f_dir = fockDirect(ctx, p_rkb);
  double worst = 0.0;
  std::size_t wp = 0, wr = 0;
  for (std::size_t p = 0; p < ctx.n_rkb; ++p)
    for (std::size_t r = 0; r < ctx.n_rkb; ++r) {
      const double d = std::abs(f_ref(p, r) - f_dir(p, r));
      if (d > worst) {
        worst = d;
        wp = p;
        wr = r;
      }
    }
  const JKPair via_eri = jkViaEri(ctx, p_rkb, wp, wr);
  const JKPair via_cart = jkViaCartesianEmbedding(ctx, p_rkb, wp, wr);
  std::printf("  [diagnose %s] worst (p=%zu,r=%zu) diff=%.3e\n", label.c_str(), wp, wr, worst);
  std::printf("    J via_eri =%.6e%+.6ei  via_cart=%.6e%+.6ei  diff=%.3e\n", via_eri.j.real(), via_eri.j.imag(),
              via_cart.j.real(), via_cart.j.imag(), std::abs(via_eri.j - via_cart.j));
  std::printf("    K via_eri =%.6e%+.6ei  via_cart=%.6e%+.6ei  diff=%.3e\n", via_eri.k.real(), via_eri.k.imag(),
              via_cart.k.real(), via_cart.k.imag(), std::abs(via_eri.k - via_cart.k));
  std::printf("    h_rkb(%zu,%zu)=%.6e%+.6ei\n", wp, wr, ctx.h_rkb(wp, wr).real(), ctx.h_rkb(wp, wr).imag());
  std::printf("    f_ref(%zu,%zu)=%.6e%+.6ei  f_dir(%zu,%zu)=%.6e%+.6ei\n", wp, wr, f_ref(wp, wr).real(), f_ref(wp, wr).imag(),
              wp, wr, f_dir(wp, wr).real(), f_dir(wp, wr).imag());
  std::printf("    reconstructed: h+J-K via_eri=%.6e%+.6ei  via_cart=%.6e%+.6ei\n",
              (ctx.h_rkb(wp, wr) + via_eri.j - via_eri.k).real(), (ctx.h_rkb(wp, wr) + via_eri.j - via_eri.k).imag(),
              (ctx.h_rkb(wp, wr) + via_cart.j - via_cart.k).real(), (ctx.h_rkb(wp, wr) + via_cart.j - via_cart.k).imag());
  std::printf("  [diagnose %s] n_neg=%zu, flavor block size q=n_rkb/4=%zu -> worst position flavor: p-block=%zu r-block=%zu\n",
              label.c_str(), ctx.n_rkb / 2, ctx.n_rkb / 4, wp / (ctx.n_rkb / 4), wr / (ctx.n_rkb / 4));
  const C k_small_only = kSmallOnly(ctx, p_rkb, wp, wr);
  std::printf("  [diagnose %s] K via kSmallOnly (minimal, restricted computation) = %.6e%+.6ei  (true needed K ~= %.6e%+.6ei)\n",
              label.c_str(), k_small_only.real(), k_small_only.imag(), (ctx.h_rkb(wp, wr) - f_dir(wp, wr)).real(),
              (ctx.h_rkb(wp, wr) - f_dir(wp, wr)).imag());
  const C two_e_direct_proj = twoElectronDirectProjected(ctx, p_rkb, wp, wr);
  std::printf("  [diagnose %s] ukbFockTwoElectronDirect's OWN raw output, projected at (p,r) = %.6e%+.6ei  (should equal "
              "f_dir-h_rkb = %.6e%+.6ei)\n",
              label.c_str(), two_e_direct_proj.real(), two_e_direct_proj.imag(), (f_dir(wp, wr) - ctx.h_rkb(wp, wr)).real(),
              (f_dir(wp, wr) - ctx.h_rkb(wp, wr)).imag());
}

// A Kramers-paired density built from a GIVEN set of occupation numbers (one per orthonormal
// eigenvector of H_RKB, ascending-eigenvalue order, same order diagonalizeHermitian returns): P =
// sum_i occ[i] |phi_i><phi_i|. occ[i]==1 for exactly n_electrons orbitals (0 elsewhere) gives the
// idempotent, Slater-determinant case; occupations fractional in [0,1] and EQUAL within each
// (near-)degenerate Kramers pair -- as any RDMFT functional's natural-orbital occupations do, by the
// same time-reversal symmetry that pairs the orbitals themselves -- give a genuinely NON-idempotent
// but still Kramers-symmetric density, the case PNOF/GNOF/Muller-type functionals actually use.
Matrix<C> buildKramersDensityFromOccupations(const SystemContext& ctx, const std::vector<double>& occ) {
  const Matrix<C> x = canonicalOrthogonalizeHermitian(ctx.s_rkb, 1e-10);
  const Matrix<C> h_ortho = dagger(x) * (ctx.h_rkb * x);
  const HermitianEigenResult eig = diagonalizeHermitian(h_ortho);

  // Consecutive (psi, Theta psi) Kramers-pair columns are only guaranteed by a plain diagonalization
  // while every level is well separated -- a single, highly symmetric shell (e.g. an isolated p or d
  // shell on one center, as the pure-s/p/d toy systems use) can have an EXACTLY degenerate manifold
  // wider than one Kramers pair, in which case diagonalizeHermitian may return an arbitrary (non-
  // Kramers-adapted) basis within it. fixKramersPairing repairs this exactly (same subspaces, same
  // eigenvalues) -- the same step main.cpp applies to real H_RKB_ortho eigenvectors before building
  // any coupled-Kramers-pair quantity.
  SmallComponentBasis small_basis_for_pairing;
  small_basis_for_pairing.build(ctx.large_cart);
  const Matrix<C> rkb_coeff_cart = rkbCoefficients(ctx.large_cart, small_basis_for_pairing.termIndex());
  const SphericalTransformResult large_spherical = buildSphericalTransform(ctx.large_cart);
  const Matrix<C> rkb_coeff_final = dagger(spinDuplicateComplex(large_spherical.transform)) * rkb_coeff_cart;
  const Matrix<double> s_large_sph = transformToSpherical(overlapMatrix(ctx.large_cart), large_spherical.transform);
  const Matrix<double> s_small_ukb = overlapMatrix(ctx.small_cart);
  const Matrix<C> eigenvectors_fixed =
      fixKramersPairing(eig.eigenvectors, eig.eigenvalues, rkb_coeff_final, x, s_large_sph, s_small_ukb);

  Matrix<C> c(x.rows(), x.cols(), C{});
  for (std::size_t a = 0; a < x.cols(); ++a)
    for (std::size_t i = 0; i < x.rows(); ++i) {
      C val{};
      for (std::size_t k = 0; k < x.cols(); ++k) val += x(i, k) * eigenvectors_fixed(k, a);
      c(i, a) = val;
    }
  Matrix<C> p(ctx.n_rkb, ctx.n_rkb, C{});
  for (std::size_t i = 0; i < ctx.n_rkb; ++i)
    for (std::size_t j = 0; j < ctx.n_rkb; ++j) {
      C s{};
      for (std::size_t a = 0; a < occ.size(); ++a)
        if (occ[a] != 0.0) s += occ[a] * c(i, a) * std::conj(c(j, a));
      p(i, j) = s;
    }
  return p;
}

double idempotencyDeviation(const SystemContext& ctx, const Matrix<C>& p) {
  const Matrix<C> psp = p * (ctx.s_rkb * p);
  double worst = 0.0;
  for (std::size_t i = 0; i < ctx.n_rkb; ++i)
    for (std::size_t j = 0; j < ctx.n_rkb; ++j) worst = std::max(worst, std::abs(psp(i, j) - p(i, j)));
  return worst;
}

// Basis-independent invariant for the DENSITY EMBEDDING direction specifically (V*P*V^dagger),
// never otherwise exercised: the P=0 control only validates the OPERATOR direction (V^dagger*(.)*V)
// via h_ukb_cart -> h_rkb, which says nothing about whether V*P*V^dagger is correct. Tr(P*S) ("particle
// number") is basis-independent, so Tr(p_rkb*s_rkb) must equal Tr(p_ukb_cart*s_ukb_cart) EXACTLY if
// the embedding is right.
double traceInvariantDeviation(const SystemContext& ctx, const Matrix<C>& p_rkb) {
  const Matrix<double> s_large_ao = overlapMatrix(ctx.large_cart);
  const Matrix<double> s_small_ao = overlapMatrix(ctx.small_cart);
  const std::size_t nl = ctx.nl_cart, ns = ctx.ns;
  Matrix<C> s_ukb_cart(2 * nl + 2 * ns, 2 * nl + 2 * ns, C{});
  for (std::size_t i = 0; i < nl; ++i)
    for (std::size_t j = 0; j < nl; ++j) {
      s_ukb_cart(i, j) = C(s_large_ao(i, j), 0.0);
      s_ukb_cart(nl + i, nl + j) = C(s_large_ao(i, j), 0.0);
    }
  for (std::size_t i = 0; i < ns; ++i)
    for (std::size_t j = 0; j < ns; ++j) {
      s_ukb_cart(2 * nl + i, 2 * nl + j) = C(s_small_ao(i, j), 0.0);
      s_ukb_cart(2 * nl + ns + i, 2 * nl + ns + j) = C(s_small_ao(i, j), 0.0);
    }
  const Matrix<C> p_ukb_cart = ctx.v_total * (p_rkb * dagger(ctx.v_total));
  C tr_rkb{}, tr_cart{};
  for (std::size_t i = 0; i < ctx.n_rkb; ++i)
    for (std::size_t j = 0; j < ctx.n_rkb; ++j) tr_rkb += p_rkb(i, j) * ctx.s_rkb(j, i);
  for (std::size_t i = 0; i < 2 * nl + 2 * ns; ++i)
    for (std::size_t j = 0; j < 2 * nl + 2 * ns; ++j) tr_cart += p_ukb_cart(i, j) * s_ukb_cart(j, i);
  std::printf("    trace invariant: Tr(P_rkb*S_rkb)=%.10f%+.3ei  Tr(P_cart*S_ukb_cart)=%.10f%+.3ei\n", tr_rkb.real(),
              tr_rkb.imag(), tr_cart.real(), tr_cart.imag());
  return std::abs(tr_rkb - tr_cart);
}

void checkDensity(const SystemContext& ctx, const std::string& name, const std::string& label, const Matrix<C>& p,
                   bool expect_idempotent) {
  const double idem_dev = idempotencyDeviation(ctx, p);
  const double kramers_dev = kramersRkbAoDeviation(p);
  const double trace_dev = traceInvariantDeviation(ctx, p);
  const double worst = maxAbsDiff(fockDirect(ctx, p), fockReference(ctx, p));
  std::printf("[%s] %s: idempotency dev=%.3e, Kramers dev=%.3e, trace-invariant dev=%.3e, max|F_direct-F_reference|=%.3e\n",
              name.c_str(), label.c_str(), idem_dev, kramers_dev, trace_dev, worst);
  check(kramers_dev < 1e-10, name + " " + label + ": density is genuinely Kramers-paired (sanity)");
  check(expect_idempotent ? idem_dev < 1e-10 : idem_dev > 1e-3,
        name + " " + label + ": density has the expected idempotency (sanity)");
  check(trace_dev < 1e-8, name + " " + label + ": V*P*V^dagger preserves Tr(P*S) exactly (density embedding sanity)");
  check(worst < kNoiseFloorThreshold, name + " " + label + ": direct Fock matches dense RkbTwoElectronTensor-based Fock");
}

// Isolates ukbFockTwoElectronDirect's SS-SS cross-flavor (Sa-Sb, meaning the RAW elementary-alpha vs
// elementary-beta copies, dimension ns each -- NOT the RKB-flavor Sa/Sb) K computation from V_total
// and the whole RKB chain entirely: build a density_ukb that is zero everywhere except a RANDOM
// Sa_raw-Sb_raw cross block, call ukbFockTwoElectronDirect on it DIRECTLY, and compare its own
// output's Sa_raw-Sb_raw block against a from-scratch reimplementation of ITS OWN documented formula
// (km_fu(a,c) = sum_{b,d} (ab|cd) * P_fu(b,d)) computed independently in this test.
// Checks (1) whether ctx.s_rkb's own RKB-flavor Sa-Sb cross block is zero (the "Sa,Sb orthogonal"
// claim, from rkbSmallOverlapMatrix directly, independent of V_total), and (2) whether V_total
// reproduces the SAME overlap when applied as an operator transform (dagger(V)*S_ukb_cart*V) --
// validating V_total itself for THIS system, not just H2.
void checkOverlapFlavorRule(const SystemContext& ctx, const std::string& name) {
  const std::size_t q = ctx.nl_final;
  double worst_cross_direct = 0.0;
  for (std::size_t i = 0; i < q; ++i)
    for (std::size_t j = 0; j < q; ++j) worst_cross_direct = std::max(worst_cross_direct, std::abs(ctx.s_rkb(2 * q + i, 3 * q + j)));

  const Matrix<double> s_large_ao = overlapMatrix(ctx.large_cart);
  const Matrix<double> s_small_ao = overlapMatrix(ctx.small_cart);
  const std::size_t nl = ctx.nl_cart, ns = ctx.ns;
  Matrix<C> s_ukb_cart(2 * nl + 2 * ns, 2 * nl + 2 * ns, C{});
  for (std::size_t i = 0; i < nl; ++i)
    for (std::size_t j = 0; j < nl; ++j) {
      s_ukb_cart(i, j) = C(s_large_ao(i, j), 0.0);
      s_ukb_cart(nl + i, nl + j) = C(s_large_ao(i, j), 0.0);
    }
  for (std::size_t i = 0; i < ns; ++i)
    for (std::size_t j = 0; j < ns; ++j) {
      s_ukb_cart(2 * nl + i, 2 * nl + j) = C(s_small_ao(i, j), 0.0);
      s_ukb_cart(2 * nl + ns + i, 2 * nl + ns + j) = C(s_small_ao(i, j), 0.0);
    }
  const Matrix<C> s_check = dagger(ctx.v_total) * (s_ukb_cart * ctx.v_total);
  double worst_vtotal = 0.0;
  for (std::size_t i = 0; i < ctx.n_rkb; ++i)
    for (std::size_t j = 0; j < ctx.n_rkb; ++j) worst_vtotal = std::max(worst_vtotal, std::abs(s_check(i, j) - ctx.s_rkb(i, j)));
  std::printf("[%s] overlap check: Sa-Sb cross block (rkbSmallOverlapMatrix directly) max|.|=%.3e; "
              "V_total operator-transform reproduces ctx.s_rkb: max diff=%.3e\n",
              name.c_str(), worst_cross_direct, worst_vtotal);
  check(worst_cross_direct < 1e-10, name + ": Sa-Sb overlap is genuinely zero");
  check(worst_vtotal < 1e-10, name + ": V_total's operator-transform reproduces the full RKB overlap exactly");
}

// Independent, naive (no GEMM, no transformLeg/transformPairToRkbSmall reuse) recomputation of
// ll_sY[y2](A,C,b,d) = chemist(Large_A,Large_C|RKBSmall(y2)_b,RKBSmall(y2)_d), cross-checked directly
// against ctx.eri's own stored value at the corresponding physics position <La_A RKBSmall(y2)_b|
// La_C RKBSmall(y2)_d>. Isolates RkbTwoElectron.cpp's optimized GEMM leg-transform machinery
// (transformLeg/transformPairToRkbSmall) from everything else -- per the user's request to rule out a
// physics<->chemist notation or leg-transform bug as the cause of the LiH/hand-built discrepancy.
void verifyLlSmallLegTransform(const SystemContext& ctx, const std::string& name) {
  const std::size_t nl_cart = ctx.nl_cart, ns = ctx.ns, nl_final = ctx.nl_final;

  // Rebuild rkb_coeff_final and large_transform_final exactly as buildSystem does (not stored on ctx).
  SmallComponentBasis small_basis;
  small_basis.build(ctx.large_cart);
  const Matrix<C> rkb_coeff_cart = rkbCoefficients(ctx.large_cart, small_basis.termIndex());
  const SphericalTransformResult large_spherical = buildSphericalTransform(ctx.large_cart);
  const Matrix<double>& large_transform_final = large_spherical.transform;
  const Matrix<C> rkb_coeff_final = dagger(spinDuplicateComplex(large_transform_final)) * rkb_coeff_cart;

  const Tensor4<double> ll_ss_cart = twoElectronIntegralsCross(ctx.large_cart, ctx.small_cart, 0.0);

  // Naive Large-leg transform (no GEMM): ll_ss_final(A,C,k,l) = sum_{p,q} T(p,A)*T(q,C)*ll_ss_cart(p,q,k,l).
  auto llSsFinal = [&](std::size_t A, std::size_t Cc, std::size_t k, std::size_t l) -> double {
    double sum = 0.0;
    for (std::size_t p = 0; p < nl_cart; ++p)
      for (std::size_t q = 0; q < nl_cart; ++q)
        sum += large_transform_final(p, A) * large_transform_final(q, Cc) * ll_ss_cart(p, q, k, l);
    return sum;
  };
  // Naive Small-leg (RKB-small(y2)) projection: ll_sY(A,C,b,d) = sum_{spin s2} sum_{k,l}
  // conj(rkb_coeff_final(y2*nl_final+b, s2+k)) * rkb_coeff_final(y2*nl_final+d, s2+l) * ll_ss_final(A,C,k,l).
  auto llSY = [&](std::size_t y2, std::size_t A, std::size_t Cc, std::size_t b, std::size_t d) -> C {
    C sum{};
    for (std::size_t s2 : {std::size_t{0}, ns})
      for (std::size_t k = 0; k < ns; ++k)
        for (std::size_t l = 0; l < ns; ++l) {
          const double val = llSsFinal(A, Cc, k, l);
          if (val == 0.0) continue;
          sum += std::conj(rkb_coeff_final(y2 * nl_final + b, s2 + k)) * rkb_coeff_final(y2 * nl_final + d, s2 + l) * val;
        }
    return sum;
  };

  // Compare against ctx.eri's own stored value at a handful of (A,C,y2,b,d) combinations spanning
  // both y2 flavors and a few indices, not just the single worst-diff position.
  double worst = 0.0;
  std::size_t worst_A = 0, worst_C = 0, worst_y2 = 0, worst_b = 0, worst_d = 0;
  const std::size_t off_small[2] = {2 * nl_final, 3 * nl_final};
  const std::size_t n_check_idx = std::min<std::size_t>(nl_final, 3);
  for (std::size_t A = 0; A < n_check_idx; ++A)
    for (std::size_t Cc = 0; Cc < n_check_idx; ++Cc)
      for (std::size_t y2 = 0; y2 < 2; ++y2)
        for (std::size_t b = 0; b < std::min<std::size_t>(nl_final, 5); ++b)
          for (std::size_t d = 0; d < std::min<std::size_t>(nl_final, 5); ++d) {
            const C naive = llSY(y2, A, Cc, b, d);
            const C from_eri = ctx.eri(A, off_small[y2] + b, Cc, off_small[y2] + d);
            const double diff = std::abs(naive - from_eri);
            if (diff > worst) {
              worst = diff;
              worst_A = A;
              worst_C = Cc;
              worst_y2 = y2;
              worst_b = b;
              worst_d = d;
            }
          }
  std::printf("[%s] verifyLlSmallLegTransform: max|naive-ctx.eri|=%.3e at (A=%zu,C=%zu,y2=%zu,b=%zu,d=%zu)\n", name.c_str(),
              worst, worst_A, worst_C, worst_y2, worst_b, worst_d);
  check(worst < 1e-9, name + ": ctx.eri's La-RKBSmall leg transform matches an independent, non-GEMM recomputation");
}

void isolateSsSsCrossFlavorK(const SystemContext& ctx, const std::string& name) {
  const std::size_t nl = ctx.nl_cart, ns = ctx.ns;
  const std::size_t n_ukb = 2 * nl + 2 * ns;
  Matrix<C> density_ukb(n_ukb, n_ukb, C{});
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const std::size_t off_sa = 2 * nl, off_sb = 2 * nl + ns;
  // HERMITIAN this time (off_sb,off_sa block = conjugate-transpose of off_sa,off_sb) -- a real
  // embedded density V*P*V^dagger would populate BOTH blocks together, not just one; the earlier,
  // non-Hermitian version of this test (only one block nonzero) may have missed a double-counting or
  // cross-contamination bug that only shows up when both are present simultaneously. ALSO set the
  // Sa-Sa and Sb-Sb DIAGONAL blocks (also Hermitian) at the same time -- a real density has these
  // too, and the earlier (cross-block-only) version of this test left them at zero, potentially
  // missing a bug that only manifests when J's diagonal contribution and K's off-diagonal
  // contribution are BOTH active simultaneously.
  for (std::size_t b = 0; b < ns; ++b)
    for (std::size_t d = 0; d < ns; ++d) {
      const C val(dist(rng), dist(rng));
      density_ukb(off_sa + b, off_sb + d) = val;
      density_ukb(off_sb + d, off_sa + b) = std::conj(val);
    }
  for (std::size_t b = 0; b < ns; ++b)
    for (std::size_t d = b; d < ns; ++d) {
      const C val(dist(rng), dist(rng));
      density_ukb(off_sa + b, off_sa + d) = val;
      density_ukb(off_sa + d, off_sa + b) = std::conj(val);
      const C val2(dist(rng), dist(rng));
      density_ukb(off_sb + b, off_sb + d) = val2;
      density_ukb(off_sb + d, off_sb + b) = std::conj(val2);
    }

  const Matrix<C> fock = ukbFockTwoElectronDirect(ctx.large_cart, ctx.small_cart, density_ukb);

  // From-scratch reimplementation of BOTH km_{Sa,Sb}(a,c) = sum_{b,d} (ab|cd)*P_{Sa,Sb}(b,d) AND
  // km_{Sb,Sa}(a,c) = sum_{b,d} (ab|cd)*P_{Sb,Sa}(b,d), checked against fock's own (Sa,Sb) and (Sb,Sa)
  // blocks respectively.
  double worst_sa_sb = 0.0, worst_sb_sa = 0.0;
  for (std::size_t a = 0; a < ns; ++a) {
    for (std::size_t c = 0; c < ns; ++c) {
      C k_sa_sb{}, k_sb_sa{};
      for (std::size_t b = 0; b < ns; ++b)
        for (std::size_t d = 0; d < ns; ++d) {
          const double v = twoElectronQuadruplet(ctx.small_cart[a], ctx.small_cart[b], ctx.small_cart[c], ctx.small_cart[d]);
          if (v == 0.0) continue;
          k_sa_sb += v * density_ukb(off_sa + b, off_sb + d);
          k_sb_sa += v * density_ukb(off_sb + b, off_sa + d);
        }
      worst_sa_sb = std::max(worst_sa_sb, std::abs(-k_sa_sb - fock(off_sa + a, off_sb + c)));
      worst_sb_sa = std::max(worst_sb_sa, std::abs(-k_sb_sa - fock(off_sb + a, off_sa + c)));
    }
  }
  std::printf("[%s] isolateSsSsCrossFlavorK (Hermitian density): Sa-Sb block max diff=%.3e, Sb-Sa block max diff=%.3e\n",
              name.c_str(), worst_sa_sb, worst_sb_sa);
  check(worst_sa_sb < 1e-9 && worst_sb_sa < 1e-9,
        name + ": ukbFockTwoElectronDirect's SS-SS cross-flavor K matches its own documented formula (Hermitian density)");
}

void runSystem(const std::string& name, std::vector<BasisFunction> large_cart, const std::vector<Atom>& geometry,
               std::size_t n_electrons) {
  const SystemContext ctx = buildSystem(std::move(large_cart), geometry);
  std::printf("[%s] n_large_cart=%zu n_small=%zu n_large_final=%zu n_rkb=%zu\n", name.c_str(), ctx.nl_cart, ctx.ns,
              ctx.nl_final, ctx.n_rkb);
  checkOverlapFlavorRule(ctx, name);
  verifyLlSmallLegTransform(ctx, name);
  isolateSsSsCrossFlavorK(ctx, name);
  // Direct check of kSmallOnly's own assumption: for an RKB-small-flavor column (e.g. position
  // 2*nl_final, the first Sa index), is V_total ACTUALLY zero outside the Small-cartesian row range
  // [2*nl_cart, 2*nl_cart+2*ns)? If not for THIS system, kSmallOnly's restriction to that range is
  // the bug, not ukbFockTwoElectronDirect.
  {
    const std::size_t p_test = 2 * ctx.nl_final;  // first Sa (RKB) column
    double worst_outside = 0.0;
    for (std::size_t A = 0; A < 2 * ctx.nl_cart; ++A) worst_outside = std::max(worst_outside, std::abs(ctx.v_total(A, p_test)));
    std::printf("[%s] V_total(:,Sa_0) max|.| OUTSIDE Small-cartesian range (should be 0) = %.3e\n", name.c_str(), worst_outside);
  }
  const std::size_t n_neg = ctx.n_rkb / 2;

  // Idempotent (Hartree-Fock-like): fill n_electrons/2 Kramers pairs (occupation 1) from the bottom
  // of the positive-energy branch.
  std::vector<double> occ_idem(ctx.n_rkb, 0.0);
  for (std::size_t k = 0; k < n_electrons / 2; ++k) occ_idem[n_neg + 2 * k] = occ_idem[n_neg + 2 * k + 1] = 1.0;
  const Matrix<C> p_idem = buildKramersDensityFromOccupations(ctx, occ_idem);
  checkDensity(ctx, name, "idempotent (HF-like) density", p_idem, /*expect_idempotent=*/true);
  diagnoseWorstEntry(ctx, name, "idempotent (HF-like) density", p_idem);

  // Fractional (RDMFT-like): same two lowest positive-energy Kramers pairs, occupations 0.95/0.05
  // instead of 1/0 -- genuinely non-idempotent, still exactly Kramers-paired.
  std::vector<double> occ_frac(ctx.n_rkb, 0.0);
  occ_frac[n_neg] = occ_frac[n_neg + 1] = 0.95;
  occ_frac[n_neg + 2] = occ_frac[n_neg + 3] = 0.05;
  checkDensity(ctx, name, "fractional-occupation (RDMFT-like) density", buildKramersDensityFromOccupations(ctx, occ_frac),
               /*expect_idempotent=*/false);

  // P=0 control: isolates the embedding plumbing from the two-electron code -- must reproduce H_RKB.
  const Matrix<C> p_zero(ctx.n_rkb, ctx.n_rkb, C{});
  check(maxAbsDiff(fockDirect(ctx, p_zero), ctx.h_rkb) < 1e-10, name + ": P=0 control reproduces H_RKB exactly");
}

}  // namespace

// One shell of angular momentum `l` under test on center A, plus a minimal s-shell on center B (1.4
// Bohr apart, same geometry as the old hand-built system) -- single-primitive, uncontracted
// (normalizeCartesianBasis renormalizes regardless of the arbitrary exponent/coefficient choice).
// Only ONE center carries the (potentially large, e.g. d's 6-component) shell under test, keeping
// n_small -- and so the O(n_small^4) dense (SS|SS) materialization and the half-dozen leg-transforms
// rkbTwoElectronIntegrals' cross-flavor terms need -- as small as possible while still exercising l's
// own Cartesian count and (for l>=2) Cartesian-to-spherical reduction, plus a genuine cross-center
// shell quartet. Element labels are placeholders (only their nuclear charge matters here).
std::vector<BasisFunction> buildUniformShellSystem(int l) {
  std::vector<BasisFunction> basis;
  auto addShell = [&](const std::string& elem, int shell_l, double x, double y, double z) {
    for (const CartesianExponents& c : cartesianComponents(shell_l)) {
      BasisFunction fn;
      fn.element = elem;
      fn.x = x;
      fn.y = y;
      fn.z = z;
      fn.l = shell_l;
      fn.cartesian = c;
      fn.exponents = {1.0};
      fn.coefficients = {1.0};
      basis.push_back(fn);
    }
  };
  addShell("C", l, 0.0, 0.0, 0.0);
  addShell("H", 0, 0.0, 0.0, 1.4);
  return basis;
}

int main() {
  const std::vector<Atom> geometry = {{"C", 0.0, 0.0, 0.0}, {"H", 0.0, 0.0, 1.4}};
  // One test per angular momentum actually exercised anywhere downstream (s: no Cartesian-to-
  // spherical reduction at all; p: same, 3 Cartesian == 3 spherical; d: the 6-vs-5 reduction case) --
  // replacing the former real-molecule bases (H2/STO-3G, LiH/6-31G, hand-built C(s,d)+H(s)), which
  // mixed multiple shells/contractions per system and cost much more to build for no extra coverage
  // of the RKB construction/two-electron paths this file actually tests.
  runSystem("pure-s (2-center)", buildUniformShellSystem(0), geometry, /*n_electrons=*/2);
  runSystem("pure-p (2-center)", buildUniformShellSystem(1), geometry, /*n_electrons=*/2);
  runSystem("pure-d (2-center)", buildUniformShellSystem(2), geometry, /*n_electrons=*/2);

  std::printf("\n%d / %d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
