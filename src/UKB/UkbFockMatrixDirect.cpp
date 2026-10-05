#include "UkbFockMatrixDirect.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <vector>

#include "ElectronRepulsion.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;

void addInto(Matrix<C>& dst, const Matrix<C>& src) {
  for (std::size_t i = 0; i < dst.rows(); ++i)
    for (std::size_t j = 0; j < dst.cols(); ++j) dst(i, j) += src(i, j);
}

// Schwarz bound index into sqrtPairDiagonal's own triangular layout (hi*(hi+1)/2+lo, hi=max,lo=min).
std::size_t triIndex(std::size_t p, std::size_t q) {
  const std::size_t hi = std::max(p, q), lo = std::min(p, q);
  return hi * (hi + 1) / 2 + lo;
}

}  // namespace

Matrix<C> ukbFockTwoElectronDirect(const std::vector<BasisFunction>& large_basis,
                                    const std::vector<BasisFunction>& small_basis,
                                    const Matrix<C>& density_ukb) {
  const std::size_t nl = large_basis.size(), ns = small_basis.size();
  const std::size_t n_ukb = 2 * nl + 2 * ns;
  if (density_ukb.rows() != n_ukb || density_ukb.cols() != n_ukb) {
    throw std::runtime_error("ukbFockTwoElectronDirect: density_ukb dimension does not match 2*nLarge+2*nSmall");
  }

  // Shell-quartet batching (twoElectronShellQuartet) needs every shell to be a COMPLETE,
  // consecutive run of cartesianComponents(l).size() basis functions sharing one center/l/
  // exponents (groupIntoShells's own invariant, which large_basis -- built by MolecularBasis::build
  // -- satisfies). The RKB small-component basis (buildRkbSmallBasis, RkbDerivativeTerms.h) does
  // NOT: each entry is one large function's own individual elementary derivative term (one specific
  // Cartesian monomial, e.g. only the x-lowering piece of one particular d component), not a
  // complete, shell-paired set of all 2l+1-worth of angular combinations -- so it cannot be grouped
  // into libcint shells at all. Only the (Large,Large|Large,Large) sector is therefore shell-quartet
  // batched here; every sector touching the Small basis falls back to twoElectronQuadruplet, one AO
  // quadruplet at a time (same per-call cost as the existing dense twoElectronIntegralsCross/
  // twoElectronIntegralsPacked use for these same sectors), Schwarz-prescreened (same exact bound,
  // same 1e-10 default threshold as rkbTwoElectronIntegrals's own screening) and parallelized over
  // the outer AO index with per-thread local accumulators merged at the end (not atomics on a
  // reinterpreted complex<double> -- simpler and exactly how e.g. RkbTwoElectron.cpp's own
  // smallSmallBlocksSlab parallelizes). Still O(1) persistent memory either way (nothing this
  // function builds is cached across Fock builds) -- genuine shell-batching for the Small-touching
  // sectors is deferred to a later performance pass.
  constexpr double kScreeningThreshold = 1e-10;
  const std::vector<ShellInfo> large_shells = groupIntoShells(large_basis);
  const std::vector<double> sqrt_diag_large = sqrtPairDiagonal(large_basis);
  const std::vector<double> sqrt_diag_small = sqrtPairDiagonal(small_basis);

  // Four flavors, in SpinorBasis's own storage order: 0=Large-alpha, 1=Large-beta, 2=Small-alpha,
  // 3=Small-beta. `offset[f]`/`fsize[f]` locate flavor f's own block in density_ukb.
  const std::size_t offset[4] = {0, nl, 2 * nl, 2 * nl + ns};
  const std::size_t fsize[4] = {nl, nl, ns, ns};

  // Total (summed-over-spin) density for each spatial type -- J only ever needs this, never the
  // individual flavor blocks (1/r12 does not see spin): J_f(a,b) is the SAME for both spin values
  // of a given spatial type, since it is linear in each g's own diagonal block and the spatial
  // integral itself does not distinguish spin -- see this file's header comment.
  auto totalDensity = [&](std::size_t f0, std::size_t f1, std::size_t n) {
    Matrix<C> out(n, n, C{});
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r) out(q, r) = density_ukb(offset[f0] + q, offset[f0] + r) + density_ukb(offset[f1] + q, offset[f1] + r);
    return out;
  };
  const Matrix<C> p_total_large = totalDensity(0, 1, nl);
  const Matrix<C> p_total_small = totalDensity(2, 3, ns);

  // The full 4x4 flavor exchange density blocks P_fu(q,r) = density_ukb(offset[f]+q, offset[u]+r),
  // q in f's own basis, r in u's own basis -- see this file's header comment for why (a,q) and
  // (b,r) are therefore always same-spatial-type pairs independently of whether f and u match.
  // UNLIKE J, K genuinely needs all 16 individual flavor pairs: the spatial integral for a given
  // (spatial(f),spatial(u)) combination is shared across its 4 spin sub-combinations, but the
  // DENSITY is not (P_{Lalpha,Salpha} != P_{Lbeta,Sbeta} in general), so summing them the way J's
  // densities are summed would be wrong -- confirmed by direct derivation, not assumed.
  Matrix<C> p_fu[4][4];
  for (int f = 0; f < 4; ++f)
    for (int u = 0; u < 4; ++u) {
      Matrix<C> blk(fsize[f], fsize[u]);
      for (std::size_t q = 0; q < fsize[f]; ++q)
        for (std::size_t r = 0; r < fsize[u]; ++r) blk(q, r) = density_ukb(offset[f] + q, offset[u] + r);
      p_fu[f][u] = std::move(blk);
    }

  Matrix<C> jm_large(nl, nl, C{}), jm_small(ns, ns, C{});
  Matrix<C> km[4][4];
  for (int f = 0; f < 4; ++f)
    for (int u = 0; u < 4; ++u) km[f][u] = Matrix<C>(fsize[static_cast<std::size_t>(f)], fsize[static_cast<std::size_t>(u)], C{});

  // --- (Large,Large | Large,Large): shell-quartet batched (Phase 0, ElectronRepulsion.h). ---
  for (std::size_t sa = 0; sa < large_shells.size(); ++sa) {
    for (std::size_t sb = 0; sb < large_shells.size(); ++sb) {
      for (std::size_t sc = 0; sc < large_shells.size(); ++sc) {
        for (std::size_t sd = 0; sd < large_shells.size(); ++sd) {
          const ShellQuartetBlock blk =
              twoElectronShellQuartet(large_basis, large_shells[sa], large_shells[sb], large_shells[sc], large_shells[sd]);
          const std::size_t a0 = large_shells[sa].first;
          const std::size_t b0 = large_shells[sb].first;
          const std::size_t c0 = large_shells[sc].first;
          const std::size_t d0 = large_shells[sd].first;

          // J(a,b) = sum_{q,s} (ab|qs) P(s,q) (RkbFockMatrix.h's own convention, physics <pq|rs> =
          // chemist (pr|qs) applied to J(p,r)=sum P(s,q)<pq|rs>) -- q is the integral's 3rd index,
          // s its 4th, so the density read is P(s,q) = p_total_large(d0+id, c0+ic), NOT (c0+ic,
          // d0+id): for a genuinely complex Hermitian density these differ by complex conjugation,
          // not just a relabeling, so getting this backwards is a real (not cosmetic) bug -- caught
          // by tests/test_ukb_fock_direct.cpp using a non-real P_RKB specifically to expose it.
          for (int ia = 0; ia < blk.ni; ++ia)
            for (int ib = 0; ib < blk.nj; ++ib) {
              C j_sum{};
              for (int ic = 0; ic < blk.nk; ++ic)
                for (int id = 0; id < blk.nl; ++id)
                  j_sum += blk(ia, ib, ic, id) * p_total_large(d0 + static_cast<std::size_t>(id), c0 + static_cast<std::size_t>(ic));
              jm_large(a0 + static_cast<std::size_t>(ia), b0 + static_cast<std::size_t>(ib)) += j_sum;
            }

          // K_fu(a,c) += sum_{b,d} (ab|cd) P_fu(b,d), for the 4 (f,u) in {0,1}x{0,1} (both Large).
          for (int f : {0, 1}) {
            for (int u : {0, 1}) {
              Matrix<C>& km_fu = km[f][u];
              const Matrix<C>& p = p_fu[f][u];
              for (int ia = 0; ia < blk.ni; ++ia)
                for (int ic = 0; ic < blk.nk; ++ic) {
                  C k_sum{};
                  for (int ib = 0; ib < blk.nj; ++ib)
                    for (int id = 0; id < blk.nl; ++id)
                      k_sum += blk(ia, ib, ic, id) * p(b0 + static_cast<std::size_t>(ib), d0 + static_cast<std::size_t>(id));
                  km_fu(a0 + static_cast<std::size_t>(ia), c0 + static_cast<std::size_t>(ic)) += k_sum;
                }
            }
          }
        }
      }
    }
  }

  // --- (Large,Large | Small,Small) and its (Small,Small | Large,Large) mirror: per-AO, both
  // directions from the SAME computed value v = (ab|cd) via the real-integral symmetry
  // (ab|cd) = (cd|ab). Parallelized over `a` with per-thread local accumulators (jm_small/km[u][f]
  // would otherwise race across different (a,b) iterations landing on the same (c,d) output).
  {
    const int n_threads = omp_get_max_threads();
    std::vector<Matrix<C>> jm_large_part(static_cast<std::size_t>(n_threads), Matrix<C>(nl, nl, C{}));
    std::vector<Matrix<C>> jm_small_part(static_cast<std::size_t>(n_threads), Matrix<C>(ns, ns, C{}));
    std::vector<std::array<std::array<Matrix<C>, 4>, 4>> km_part(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t)
      for (int f = 0; f < 4; ++f)
        for (int u = 0; u < 4; ++u)
          km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] =
              Matrix<C>(fsize[static_cast<std::size_t>(f)], fsize[static_cast<std::size_t>(u)], C{});

#pragma omp parallel for schedule(dynamic)
    for (std::size_t a = 0; a < nl; ++a) {
      const int tid = omp_get_thread_num();
      Matrix<C>& jm_large_t = jm_large_part[static_cast<std::size_t>(tid)];
      Matrix<C>& jm_small_t = jm_small_part[static_cast<std::size_t>(tid)];
      auto& km_t = km_part[static_cast<std::size_t>(tid)];
      for (std::size_t b = 0; b < nl; ++b) {
        const double sqrt_lab = sqrt_diag_large[triIndex(a, b)];
        for (std::size_t c = 0; c < ns; ++c) {
          for (std::size_t d = 0; d < ns; ++d) {
            if (sqrt_lab * sqrt_diag_small[triIndex(c, d)] < kScreeningThreshold) continue;
            const double v = twoElectronQuadruplet(large_basis[a], large_basis[b], small_basis[c], small_basis[d]);
            if (v == 0.0) continue;

            // J: jm_large(a,b) += v * P_total_small(d,c); jm_small(c,d) += v * P_total_large(b,a).
            jm_large_t(a, b) += v * p_total_small(d, c);
            jm_small_t(c, d) += v * p_total_large(b, a);

            // K: K_{Lf,Su}(a,c) += v * P_{Lf,Su}(b,d), f in {0,1}, u in {2,3} (4 combos); and its
            // mirror K_{Su,Lf}(c,a) += v * P_{Su,Lf}(d,b) (generally a DIFFERENT density block, not
            // simply the conjugate/transpose of the first -- see this file's header derivation).
            for (int f : {0, 1}) {
              for (int u : {2, 3}) {
                km_t[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](a, c) += v * p_fu[f][u](b, d);
                km_t[static_cast<std::size_t>(u)][static_cast<std::size_t>(f)](c, a) += v * p_fu[u][f](d, b);
              }
            }
          }
        }
      }
    }

    for (int t = 0; t < n_threads; ++t) {
      addInto(jm_large, jm_large_part[static_cast<std::size_t>(t)]);
      addInto(jm_small, jm_small_part[static_cast<std::size_t>(t)]);
      for (int f = 0; f < 4; ++f)
        for (int u = 0; u < 4; ++u)
          addInto(km[f][u], km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)]);
    }
  }

  // --- (Small,Small | Small,Small): per-AO, same screening + per-thread-local-accumulator pattern. ---
  {
    const int n_threads = omp_get_max_threads();
    std::vector<Matrix<C>> jm_small_part(static_cast<std::size_t>(n_threads), Matrix<C>(ns, ns, C{}));
    std::vector<std::array<std::array<Matrix<C>, 2>, 2>> km_part(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t)
      for (int f = 0; f < 2; ++f)
        for (int u = 0; u < 2; ++u) km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] = Matrix<C>(ns, ns, C{});

#pragma omp parallel for schedule(dynamic)
    for (std::size_t a = 0; a < ns; ++a) {
      const int tid = omp_get_thread_num();
      Matrix<C>& jm_small_t = jm_small_part[static_cast<std::size_t>(tid)];
      auto& km_t = km_part[static_cast<std::size_t>(tid)];
      for (std::size_t b = 0; b < ns; ++b) {
        const double sqrt_ab = sqrt_diag_small[triIndex(a, b)];
        for (std::size_t c = 0; c < ns; ++c) {
          for (std::size_t d = 0; d < ns; ++d) {
            if (sqrt_ab * sqrt_diag_small[triIndex(c, d)] < kScreeningThreshold) continue;
            const double v = twoElectronQuadruplet(small_basis[a], small_basis[b], small_basis[c], small_basis[d]);
            if (v == 0.0) continue;
            // Same J(a,b) = sum (ab|qs) P(s,q) convention as the LL|LL sector above: q=c (3rd
            // integral index), s=d (4th), so the density read is P(d,c), not P(c,d).
            jm_small_t(a, b) += v * p_total_small(d, c);
            for (int f : {0, 1}) {
              for (int u : {0, 1}) {
                km_t[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](a, c) += v * p_fu[f + 2][u + 2](b, d);
              }
            }
          }
        }
      }
    }

    for (int t = 0; t < n_threads; ++t) {
      addInto(jm_small, jm_small_part[static_cast<std::size_t>(t)]);
      for (int f = 0; f < 2; ++f)
        for (int u = 0; u < 2; ++u)
          addInto(km[f + 2][u + 2], km_part[static_cast<std::size_t>(t)][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)]);
    }
  }

  // Assemble: J added only to each spatial type's own two diagonal (spin-matching) blocks; K
  // written to every one of the 16 flavor-pair blocks individually.
  Matrix<C> fock(n_ukb, n_ukb, C{});
  for (std::size_t a = 0; a < nl; ++a)
    for (std::size_t b = 0; b < nl; ++b) {
      fock(offset[0] + a, offset[0] + b) += jm_large(a, b);
      fock(offset[1] + a, offset[1] + b) += jm_large(a, b);
    }
  for (std::size_t a = 0; a < ns; ++a)
    for (std::size_t b = 0; b < ns; ++b) {
      fock(offset[2] + a, offset[2] + b) += jm_small(a, b);
      fock(offset[3] + a, offset[3] + b) += jm_small(a, b);
    }
  for (int f = 0; f < 4; ++f)
    for (int u = 0; u < 4; ++u) {
      const Matrix<C>& km_fu = km[f][u];
      for (std::size_t a = 0; a < fsize[static_cast<std::size_t>(f)]; ++a)
        for (std::size_t b = 0; b < fsize[static_cast<std::size_t>(u)]; ++b)
          fock(offset[static_cast<std::size_t>(f)] + a, offset[static_cast<std::size_t>(u)] + b) -= km_fu(a, b);
    }
  return fock;
}

Matrix<C> rkbFockMatrix(const Matrix<C>& h_rkb, const UkbDirectEriSource& eri, const Matrix<C>& density_matrix) {
  const Matrix<C> p_ukb_cart = eri.v_total * (density_matrix * dagger(eri.v_total));
  const Matrix<C> two_electron = ukbFockTwoElectronDirect(eri.large_basis, eri.small_basis, p_ukb_cart);
  return h_rkb + dagger(eri.v_total) * (two_electron * eri.v_total);
}

}  // namespace rerdmft
