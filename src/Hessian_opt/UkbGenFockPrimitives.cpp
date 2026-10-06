#include "UkbGenFockPrimitives.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "ElectronRepulsion.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;

void addInto(Matrix<C>& dst, const Matrix<C>& src) {
  for (std::size_t i = 0; i < dst.rows(); ++i)
    for (std::size_t j = 0; j < dst.cols(); ++j) dst(i, j) += src(i, j);
}

// Schwarz bound index into sqrtPairDiagonal's own triangular layout (hi*(hi+1)/2+lo, hi=max,lo=min)
// -- identical to UKB/UkbFockMatrixDirect.cpp's own helper.
std::size_t triIndex(std::size_t p, std::size_t q) {
  const std::size_t hi = std::max(p, q), lo = std::min(p, q);
  return hi * (hi + 1) / 2 + lo;
}

}  // namespace

UkbGenFockBuild ukbGenFockBuild(const std::vector<BasisFunction>& large_basis,
                                 const std::vector<BasisFunction>& small_basis,
                                 const std::vector<Matrix<C>>& densities) {
  const std::size_t nl = large_basis.size(), ns = small_basis.size();
  const std::size_t n_ukb = 2 * nl + 2 * ns;
  const std::size_t nd = densities.size();
  if (nd == 0) {
    throw std::runtime_error("ukbGenFockBuild: no densities given");
  }
  for (std::size_t k = 0; k < nd; ++k) {
    if (densities[k].rows() != n_ukb || densities[k].cols() != n_ukb) {
      throw std::runtime_error("ukbGenFockBuild: density dimension does not match 2*nLarge+2*nSmall");
    }
  }

  // Same shell-batching/screening infrastructure as UKB/UkbFockMatrixDirect.cpp's
  // ukbFockTwoElectronDirect (not reused directly -- see this file's own header comment on why).
  constexpr double kScreeningThreshold = 1e-10;
  const std::vector<ShellInfo> large_shells = groupIntoShells(large_basis);
  const std::vector<double> sqrt_diag_large = sqrtPairDiagonal(large_basis);
  const std::vector<double> sqrt_diag_small = sqrtPairDiagonal(small_basis);

  const std::size_t offset[4] = {0, nl, 2 * nl, 2 * nl + ns};
  const std::size_t fsize[4] = {nl, nl, ns, ns};

  // Per-density reductions J/K need -- same formulas as UkbFockMatrixDirect.cpp's own
  // p_total_large/p_total_small/p_fu (verified this session, project memory
  // project-ukb-direct-genfock-scheme: the spin-summed shortcut for J and the direct per-flavor-
  // pair block read for K are both exact regardless of the density's own cross-flavor structure --
  // a property of the operator, not of the density), just built once per density here.
  std::vector<Matrix<C>> p_total_large(nd), p_total_small(nd);
  std::vector<std::array<std::array<Matrix<C>, 4>, 4>> p_fu(nd);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t k = 0; k < nd; ++k) {
    const Matrix<C>& d = densities[k];
    auto totalDensity = [&](std::size_t f0, std::size_t f1, std::size_t n) {
      Matrix<C> out(n, n, C{});
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r) out(q, r) = d(offset[f0] + q, offset[f0] + r) + d(offset[f1] + q, offset[f1] + r);
      return out;
    };
    p_total_large[k] = totalDensity(0, 1, nl);
    p_total_small[k] = totalDensity(2, 3, ns);
    for (int f = 0; f < 4; ++f) {
      for (int u = 0; u < 4; ++u) {
        Matrix<C> blk(fsize[static_cast<std::size_t>(f)], fsize[static_cast<std::size_t>(u)]);
        for (std::size_t q = 0; q < fsize[static_cast<std::size_t>(f)]; ++q)
          for (std::size_t r = 0; r < fsize[static_cast<std::size_t>(u)]; ++r)
            blk(q, r) = d(offset[static_cast<std::size_t>(f)] + q, offset[static_cast<std::size_t>(u)] + r);
        p_fu[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] = std::move(blk);
      }
    }
  }

  std::vector<Matrix<C>> jm_large(nd), jm_small(nd);
  std::vector<std::array<std::array<Matrix<C>, 4>, 4>> km(nd);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t k = 0; k < nd; ++k) {
    jm_large[k] = Matrix<C>(nl, nl, C{});
    jm_small[k] = Matrix<C>(ns, ns, C{});
    for (int f = 0; f < 4; ++f)
      for (int u = 0; u < 4; ++u)
        km[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] =
            Matrix<C>(fsize[static_cast<std::size_t>(f)], fsize[static_cast<std::size_t>(u)], C{});
  }

  // --- (Large,Large | Large,Large): shell-quartet batched, IDENTICAL symmetry-orbit scatter to
  // UkbFockMatrixDirect.cpp's own LL|LL sector (shell-level pattern dedup, not per-AO-entry -- see
  // that file's own comment for why per-entry dedup overcounts). UkbFockMatrixDirect.cpp's own
  // sector stays serial because it only ever has ONE density (nd=1) to scatter into -- here `nd`
  // can be O(n_mo) (e.g. PNOF's 3*n_total), turning the innermost `for k in 0..nd` scatter into
  // the dominant cost, so this sector (unlike that one) IS parallelized, over the outer shell-pair
  // index `sa`, with the same per-thread-accumulator pattern the LL/SS and SS/SS sectors below
  // already use.
  {
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<Matrix<C>>> jm_large_part(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<std::array<std::array<Matrix<C>, 2>, 2>>> km_part(static_cast<std::size_t>(n_threads));
    // Allocation is O(n_threads * nd) Matrix objects -- parallelized over `t` (each thread
    // allocates/first-touches only its own accumulators) rather than left serial, since `nd` can
    // be O(n_mo) and this would otherwise be a real, purely-serial cost paid on every call.
#pragma omp parallel for
    for (int t = 0; t < n_threads; ++t) {
      jm_large_part[static_cast<std::size_t>(t)].assign(nd, Matrix<C>(nl, nl, C{}));
      km_part[static_cast<std::size_t>(t)].resize(nd);
      for (std::size_t k = 0; k < nd; ++k)
        for (int f = 0; f < 2; ++f)
          for (int u = 0; u < 2; ++u) km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] = Matrix<C>(nl, nl, C{});
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t sa = 0; sa < large_shells.size(); ++sa) {
      const int tid = omp_get_thread_num();
      std::vector<Matrix<C>>& jm_large_t = jm_large_part[static_cast<std::size_t>(tid)];
      auto& km_t = km_part[static_cast<std::size_t>(tid)];
      for (std::size_t sb = sa; sb < large_shells.size(); ++sb) {
        const std::size_t pq_shell = sa * large_shells.size() + sb;
        for (std::size_t sc = 0; sc < large_shells.size(); ++sc) {
          for (std::size_t sd = sc; sd < large_shells.size(); ++sd) {
            const std::size_t rs_shell = sc * large_shells.size() + sd;
            if (rs_shell < pq_shell) continue;  // (ab|cd) == (cd|ab); only need pq_shell<=rs_shell.

            const ShellQuartetBlock blk =
                twoElectronShellQuartet(large_basis, large_shells[sa], large_shells[sb], large_shells[sc], large_shells[sd]);
            const std::size_t off[4] = {large_shells[sa].first, large_shells[sb].first, large_shells[sc].first,
                                         large_shells[sd].first};
            const std::size_t shell_of_slot[4] = {sa, sb, sc, sd};

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
                      for (std::size_t k = 0; k < nd; ++k) {
                        jm_large_t[k](p1, p2) += v * p_total_large[k](p4, p3);
                        for (int f : {0, 1}) {
                          for (int u : {0, 1}) {
                            km_t[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p1, p3) +=
                                v * p_fu[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p2, p4);
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
    }

    // Merge: parallelized over `k` (O(n_mo), far more parallelism than over `t` -- O(n_threads) --
    // and no two `k` ever touch the same output, unlike two `t` for the SAME `k`, which is exactly
    // why `t` stays the serial (innermost) loop here).
#pragma omp parallel for schedule(dynamic)
    for (std::size_t k = 0; k < nd; ++k) {
      for (int t = 0; t < n_threads; ++t) {
        addInto(jm_large[k], jm_large_part[static_cast<std::size_t>(t)][k]);
        for (int f = 0; f < 2; ++f)
          for (int u = 0; u < 2; ++u)
            addInto(km[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)],
                    km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)]);
      }
    }
  }

  // --- (Large,Large | Small,Small) and its (Small,Small | Large,Large) mirror: per-AO, IDENTICAL
  // symmetry scatter to UkbFockMatrixDirect.cpp's own sector, parallelized over `a` with per-thread
  // local accumulators -- now one vector of `nd` accumulators per thread instead of one.
  {
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<Matrix<C>>> jm_large_part(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<Matrix<C>>> jm_small_part(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<std::array<std::array<Matrix<C>, 4>, 4>>> km_part(static_cast<std::size_t>(n_threads));
#pragma omp parallel for
    for (int t = 0; t < n_threads; ++t) {
      jm_large_part[static_cast<std::size_t>(t)].assign(nd, Matrix<C>(nl, nl, C{}));
      jm_small_part[static_cast<std::size_t>(t)].assign(nd, Matrix<C>(ns, ns, C{}));
      km_part[static_cast<std::size_t>(t)].resize(nd);
      for (std::size_t k = 0; k < nd; ++k)
        for (int f = 0; f < 4; ++f)
          for (int u = 0; u < 4; ++u)
            km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] =
                Matrix<C>(fsize[static_cast<std::size_t>(f)], fsize[static_cast<std::size_t>(u)], C{});
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t a = 0; a < nl; ++a) {
      const int tid = omp_get_thread_num();
      std::vector<Matrix<C>>& jm_large_t = jm_large_part[static_cast<std::size_t>(tid)];
      std::vector<Matrix<C>>& jm_small_t = jm_small_part[static_cast<std::size_t>(tid)];
      auto& km_t = km_part[static_cast<std::size_t>(tid)];
      for (std::size_t b = a; b < nl; ++b) {
        const double sqrt_lab = sqrt_diag_large[triIndex(a, b)];
        for (std::size_t c = 0; c < ns; ++c) {
          for (std::size_t d = c; d < ns; ++d) {
            if (sqrt_lab * sqrt_diag_small[triIndex(c, d)] < kScreeningThreshold) continue;
            const double v = twoElectronQuadruplet(large_basis[a], large_basis[b], small_basis[c], small_basis[d]);
            if (v == 0.0) continue;

            const std::array<std::array<std::size_t, 4>, 4> cand = {
                {{a, b, c, d}, {b, a, c, d}, {a, b, d, c}, {b, a, d, c}}};
            for (std::size_t i = 0; i < cand.size(); ++i) {
              bool dup = false;
              for (std::size_t j = 0; j < i; ++j) {
                if (cand[j] == cand[i]) { dup = true; break; }
              }
              if (dup) continue;
              const std::size_t p1 = cand[i][0], p2 = cand[i][1], p3 = cand[i][2], p4 = cand[i][3];

              for (std::size_t k = 0; k < nd; ++k) {
                jm_large_t[k](p1, p2) += v * p_total_small[k](p4, p3);
                jm_small_t[k](p3, p4) += v * p_total_large[k](p2, p1);
                for (int f : {0, 1}) {
                  for (int u : {2, 3}) {
                    km_t[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p1, p3) +=
                        v * p_fu[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p2, p4);
                    km_t[k][static_cast<std::size_t>(u)][static_cast<std::size_t>(f)](p3, p1) +=
                        v * p_fu[k][static_cast<std::size_t>(u)][static_cast<std::size_t>(f)](p4, p2);
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t k = 0; k < nd; ++k) {
      for (int t = 0; t < n_threads; ++t) {
        addInto(jm_large[k], jm_large_part[static_cast<std::size_t>(t)][k]);
        addInto(jm_small[k], jm_small_part[static_cast<std::size_t>(t)][k]);
        for (int f = 0; f < 4; ++f)
          for (int u = 0; u < 4; ++u)
            addInto(km[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)],
                    km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)]);
      }
    }
  }

  // --- (Small,Small | Small,Small): per-AO, IDENTICAL symmetry scatter + per-thread-local-
  // accumulator pattern, now `nd`-wide.
  {
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<Matrix<C>>> jm_small_part(static_cast<std::size_t>(n_threads));
    std::vector<std::vector<std::array<std::array<Matrix<C>, 2>, 2>>> km_part(static_cast<std::size_t>(n_threads));
#pragma omp parallel for
    for (int t = 0; t < n_threads; ++t) {
      jm_small_part[static_cast<std::size_t>(t)].assign(nd, Matrix<C>(ns, ns, C{}));
      km_part[static_cast<std::size_t>(t)].resize(nd);
      for (std::size_t k = 0; k < nd; ++k)
        for (int f = 0; f < 2; ++f)
          for (int u = 0; u < 2; ++u)
            km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)] =
                Matrix<C>(ns, ns, C{});
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t a = 0; a < ns; ++a) {
      const int tid = omp_get_thread_num();
      std::vector<Matrix<C>>& jm_small_t = jm_small_part[static_cast<std::size_t>(tid)];
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
              for (std::size_t k = 0; k < nd; ++k) {
                jm_small_t[k](p1, p2) += v * p_total_small[k](p4, p3);
                for (int f : {0, 1}) {
                  for (int u : {0, 1}) {
                    km_t[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)](p1, p3) +=
                        v * p_fu[k][static_cast<std::size_t>(f) + 2][static_cast<std::size_t>(u) + 2](p2, p4);
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t k = 0; k < nd; ++k) {
      for (int t = 0; t < n_threads; ++t) {
        addInto(jm_small[k], jm_small_part[static_cast<std::size_t>(t)][k]);
        for (int f = 0; f < 2; ++f)
          for (int u = 0; u < 2; ++u)
            addInto(km[k][static_cast<std::size_t>(f) + 2][static_cast<std::size_t>(u) + 2],
                    km_part[static_cast<std::size_t>(t)][k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)]);
      }
    }
  }

  // --- Assemble the n_ukb x n_ukb J[D_k]/K[D_k] outputs for every density k. ---
  UkbGenFockBuild result;
  result.j.resize(nd);
  result.k.resize(nd);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t k = 0; k < nd; ++k) {
    Matrix<C> j_full(n_ukb, n_ukb, C{});
    for (std::size_t a = 0; a < nl; ++a)
      for (std::size_t b = 0; b < nl; ++b) {
        j_full(offset[0] + a, offset[0] + b) = jm_large[k](a, b);
        j_full(offset[1] + a, offset[1] + b) = jm_large[k](a, b);
      }
    for (std::size_t a = 0; a < ns; ++a)
      for (std::size_t b = 0; b < ns; ++b) {
        j_full(offset[2] + a, offset[2] + b) = jm_small[k](a, b);
        j_full(offset[3] + a, offset[3] + b) = jm_small[k](a, b);
      }
    result.j[k] = std::move(j_full);

    Matrix<C> k_full(n_ukb, n_ukb, C{});
    for (int f = 0; f < 4; ++f)
      for (int u = 0; u < 4; ++u) {
        const Matrix<C>& km_fu = km[k][static_cast<std::size_t>(f)][static_cast<std::size_t>(u)];
        for (std::size_t a = 0; a < fsize[static_cast<std::size_t>(f)]; ++a)
          for (std::size_t b = 0; b < fsize[static_cast<std::size_t>(u)]; ++b)
            k_full(offset[static_cast<std::size_t>(f)] + a, offset[static_cast<std::size_t>(u)] + b) = km_fu(a, b);
      }
    result.k[k] = std::move(k_full);
  }
  return result;
}

}  // namespace rerdmft
