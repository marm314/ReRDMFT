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
  //
  // All three sectors below DO exploit the full real-orbital permutational symmetry of (pq|rs)
  // within a single build, though: each one only evaluates one representative integral per
  // symmetry-equivalence class (shell-level for the batched LL|LL sector, AO-level for the other
  // two) and scatters that single computed value into every Fock contribution its orbit covers,
  // instead of calling libcint again for a quartet that is numerically identical to one already
  // computed earlier in the same loop. See each sector's own comment for the exact scatter.
  // Nothing here reduces the PER-ITERATION-to-PER-ITERATION redundancy (every SCF iteration still
  // recomputes the full symmetry-reduced integral set from scratch, by design -- that is the
  // O(1)-memory/recompute tradeoff this kernel exists for); it only removes the within-one-build
  // redundancy of evaluating the same (pq|rs) value multiple times under different index orders.
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
  //
  // Exploits the full real-orbital 8-fold permutational symmetry (p<->q, r<->s, (pq)<->(rs)): the
  // outer shell loop below only visits one canonical representative per symmetry-equivalence
  // class of shell quartets (sa<=sb, sc<=sd, and shell-pair-compound(sa,sb)<=shell-pair-compound
  // (sc,sd), same triangular-restriction idea as twoElectronIntegralsPacked's own AO-level loop,
  // just lifted to shell granularity so it actually saves libcint CALLS, not merely storage) --
  // up to 8x fewer twoElectronShellQuartet/libcint calls for a generic, non-degenerate quartet.
  //
  // The computed block is then read under every DISTINCT shell-level role assignment in its orbit
  // {(sa,sb,sc,sd),(sb,sa,sc,sd),(sa,sb,sd,sc),(sb,sa,sd,sc),(sc,sd,sa,sb),(sd,sc,sa,sb),
  // (sc,sd,sb,sa),(sd,sc,sb,sa)}, deduped AT THE SHELL LEVEL (collapsing whenever two shells in the
  // quartet coincide, e.g. sa==sb, so a role assignment that is shell-identical to one already
  // counted is not read again) -- NOT deduped per AO quadruplet inside the block: a per-entry dedup
  // would independently rediscover and rescatter the SAME orbit from several different (ia,ib,ic,
  // id) entries of one block (e.g. entries (ia,ib,ic,id) and (ib,ia,id,ic) both mapping onto the
  // SAME physical AO quadruplet's orbit), overcounting by up to 8x -- caught by a standalone
  // cross-check against the unrestricted form on p/d toy bases before this landed. For each
  // surviving role assignment, every (ia,ib,ic,id) in the FULL block contributes J(p1,p2) +=
  // v*P(p4,p3) and, for both Large flavors f,u in {0,1}, K_fu(p1,p3) += v*P_fu(p2,p4), with
  // (p1,p2,p3,p4) read off that assignment's shell offsets -- exactly reproducing the former fully
  // unrestricted sa,sb,sc,sd loop (every unrestricted AO quadruplet belongs to exactly one such
  // orbit, visited by exactly one (role assignment, block entry) pair), without ever recomputing a
  // symmetry-equivalent integral via a second libcint call. Validated bit-for-bit against the
  // prior unrestricted form in tests/test_ukb_fock_direct.cpp.
  for (std::size_t sa = 0; sa < large_shells.size(); ++sa) {
    for (std::size_t sb = sa; sb < large_shells.size(); ++sb) {
      const std::size_t pq_shell = sa * large_shells.size() + sb;
      for (std::size_t sc = 0; sc < large_shells.size(); ++sc) {
        for (std::size_t sd = sc; sd < large_shells.size(); ++sd) {
          const std::size_t rs_shell = sc * large_shells.size() + sd;
          if (rs_shell < pq_shell) continue;  // (ab|cd) == (cd|ab); only need pq_shell<=rs_shell.

          const ShellQuartetBlock blk =
              twoElectronShellQuartet(large_basis, large_shells[sa], large_shells[sb], large_shells[sc], large_shells[sd]);
          // off[k]/dim index slot k (0=a/sa, 1=b/sb, 2=c/sc, 3=d/sd) -- slot, not role: a role
          // assignment below picks, for each of its 4 ROLES, which SLOT's offset and block
          // dimension to read.
          const std::size_t off[4] = {large_shells[sa].first, large_shells[sb].first, large_shells[sc].first,
                                       large_shells[sd].first};
          const std::size_t shell_of_slot[4] = {sa, sb, sc, sd};

          // J(p1,p2) = sum (p1 p2|p3 p4) P(p4,p3) (RkbFockMatrix.h's own convention, physics
          // <pq|rs> = chemist (pr|qs) applied to J(p,r)=sum P(s,q)<pq|rs>) -- the density read is
          // P(p4,p3), NOT (p3,p4): for a genuinely complex Hermitian density these differ by
          // complex conjugation, not just a relabeling, so getting this backwards is a real (not
          // cosmetic) bug -- caught by tests/test_ukb_fock_direct.cpp using a non-real P_RKB
          // specifically to expose it.
          const std::array<std::array<int, 4>, 8> slot_cand = {
              {{0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
               {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}}};
          std::array<std::array<int, 4>, 8> patterns{};
          std::size_t n_patterns = 0;
          for (const auto& sp : slot_cand) {
            const std::array<std::size_t, 4> shell_tuple = {shell_of_slot[sp[0]], shell_of_slot[sp[1]],
                                                              shell_of_slot[sp[2]], shell_of_slot[sp[3]]};
            bool dup = false;
            for (std::size_t j = 0; j < n_patterns; ++j) {
              const auto& ep = patterns[j];
              const std::array<std::size_t, 4> existing_tuple = {shell_of_slot[ep[0]], shell_of_slot[ep[1]],
                                                                   shell_of_slot[ep[2]], shell_of_slot[ep[3]]};
              if (existing_tuple == shell_tuple) { dup = true; break; }
            }
            if (!dup) patterns[n_patterns++] = sp;
          }

          for (std::size_t pi = 0; pi < n_patterns; ++pi) {
            const std::array<int, 4>& pat = patterns[pi];
            for (int ia = 0; ia < blk.ni; ++ia) {
              for (int ib = 0; ib < blk.nj; ++ib) {
                for (int ic = 0; ic < blk.nk; ++ic) {
                  for (int id = 0; id < blk.nl; ++id) {
                    const double v = blk(ia, ib, ic, id);
                    if (v == 0.0) continue;
                    const int idx[4] = {ia, ib, ic, id};
                    const std::size_t p1 = off[pat[0]] + static_cast<std::size_t>(idx[pat[0]]);
                    const std::size_t p2 = off[pat[1]] + static_cast<std::size_t>(idx[pat[1]]);
                    const std::size_t p3 = off[pat[2]] + static_cast<std::size_t>(idx[pat[2]]);
                    const std::size_t p4 = off[pat[3]] + static_cast<std::size_t>(idx[pat[3]]);
                    jm_large(p1, p2) += v * p_total_large(p4, p3);
                    for (int f : {0, 1}) {
                      for (int u : {0, 1}) {
                        km[f][u](p1, p3) += v * p_fu[f][u](p2, p4);
                      }
                    }
                  }
                }
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
      for (std::size_t b = a; b < nl; ++b) {
        const double sqrt_lab = sqrt_diag_large[triIndex(a, b)];
        for (std::size_t c = 0; c < ns; ++c) {
          for (std::size_t d = c; d < ns; ++d) {
            if (sqrt_lab * sqrt_diag_small[triIndex(c, d)] < kScreeningThreshold) continue;
            const double v = twoElectronQuadruplet(large_basis[a], large_basis[b], small_basis[c], small_basis[d]);
            if (v == 0.0) continue;

            // a<->b (within Large) and c<->d (within Small) are both exact real-integral
            // symmetries not yet exploited by the a<=b, c<=d restriction above (that restriction
            // only avoids recomputing them via a second twoElectronQuadruplet call -- it does not
            // by itself apply their Fock contributions). Apply each distinct (p1,p2,p3,p4) role
            // assignment from {(a,b,c,d),(b,a,c,d),(a,b,d,c),(b,a,d,c)} exactly once (deduped so
            // a==b or c==d is not double counted); each one, via the SAME v, gets both the direct
            // J/K contribution and the already-known (ab|cd)=(cd|ab) mirror -- together
            // reproducing the full unrestricted a,b,c,d quadruple sum exactly.
            const std::array<std::array<std::size_t, 4>, 4> cand = {
                {{a, b, c, d}, {b, a, c, d}, {a, b, d, c}, {b, a, d, c}}};
            for (std::size_t i = 0; i < cand.size(); ++i) {
              bool dup = false;
              for (std::size_t j = 0; j < i; ++j) {
                if (cand[j] == cand[i]) { dup = true; break; }
              }
              if (dup) continue;
              const std::size_t p1 = cand[i][0], p2 = cand[i][1], p3 = cand[i][2], p4 = cand[i][3];

              // J: jm_large(p1,p2) += v * P_total_small(p4,p3); mirror jm_small(p3,p4) += v *
              // P_total_large(p2,p1).
              jm_large_t(p1, p2) += v * p_total_small(p4, p3);
              jm_small_t(p3, p4) += v * p_total_large(p2, p1);

              // K: K_{Lf,Su}(p1,p3) += v * P_{Lf,Su}(p2,p4), f in {0,1}, u in {2,3} (4 combos); and
              // its mirror K_{Su,Lf}(p3,p1) += v * P_{Su,Lf}(p4,p2) (generally a DIFFERENT density
              // block, not simply the conjugate/transpose of the first -- see this file's header
              // derivation).
              for (int f : {0, 1}) {
                for (int u : {2, 3}) {
                  km_t[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p1, p3) += v * p_fu[f][u](p2, p4);
                  km_t[static_cast<std::size_t>(u)][static_cast<std::size_t>(f)](p3, p1) += v * p_fu[u][f](p4, p2);
                }
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
      for (std::size_t b = a; b < ns; ++b) {
        const double sqrt_ab = sqrt_diag_small[triIndex(a, b)];
        const std::size_t pq = a * ns + b;
        for (std::size_t c = 0; c < ns; ++c) {
          for (std::size_t d = c; d < ns; ++d) {
            const std::size_t rs = c * ns + d;
            if (rs < pq) continue;  // (ab|cd) == (cd|ab); only need the pq<=rs half.
            if (sqrt_ab * sqrt_diag_small[triIndex(c, d)] < kScreeningThreshold) continue;
            const double v = twoElectronQuadruplet(small_basis[a], small_basis[b], small_basis[c], small_basis[d]);
            if (v == 0.0) continue;

            // Full real-orbital 8-fold symmetry (p<->q, r<->s, (pq)<->(rs)) -- same orbit-scatter
            // pattern as the (Large,Large|Large,Large) sector above (see its own comment): deduped
            // so degenerate a==b, c==d or (ab)==(cd) cases are not double counted.
            const std::array<std::array<std::size_t, 4>, 8> cand = {
                {{a, b, c, d}, {b, a, c, d}, {a, b, d, c}, {b, a, d, c},
                 {c, d, a, b}, {d, c, a, b}, {c, d, b, a}, {d, c, b, a}}};
            for (std::size_t i = 0; i < cand.size(); ++i) {
              bool dup = false;
              for (std::size_t j = 0; j < i; ++j) {
                if (cand[j] == cand[i]) { dup = true; break; }
              }
              if (dup) continue;
              const std::size_t p1 = cand[i][0], p2 = cand[i][1], p3 = cand[i][2], p4 = cand[i][3];
              // Same J(p1,p2) = sum (p1 p2|p3 p4) P(p4,p3) convention as the LL|LL sector above.
              jm_small_t(p1, p2) += v * p_total_small(p4, p3);
              for (int f : {0, 1}) {
                for (int u : {0, 1}) {
                  km_t[static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p1, p3) += v * p_fu[f + 2][u + 2](p2, p4);
                }
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
