#include "FullOptimization.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "ADAM.h"
#include "CholeskyEri.h"
#include "HartreeExchangeGradient.h"
#include "HartreeExchangeHessian.h"
#include "JkOnlyFock.h"
#include "JkOnlyHessian.h"
#include "KramersPairing.h"
#include "KramersRestriction.h"
#include "LBFGS.h"
#include "OccupationEnergy.h"
#include "OrbitalGradient.h"
#include "PnofFock.h"
#include "PnofHessian.h"
#include "SQP.h"
#include "SpinorRotation.h"

namespace rerdmft {

namespace {

constexpr double kOccupationEpsilon = 1e-6;

inline double conjugate(double x) { return x; }
inline std::complex<double> conjugate(const std::complex<double>& x) { return std::conj(x); }

using Pair = std::pair<std::size_t, std::size_t>;

// `n_negative` (default 0): excludes every pair touching an index < n_negative -- C4_DHF's
// negative-energy (Dirac sea) branch, which orbital rotations must never mix into an occupied
// positive-energy orbital (Talman 1986's min-max characterization of relativistic SCF; Saue,
// ChemPhysChem 12, 3077 (2011)) to stay within the no-pair approximation. Checking q < n_negative
// alone suffices since q < p always, so q is the smaller of the two indices.
std::vector<Pair> lowerPairs(std::size_t n, std::size_t n_negative = 0) {
  std::vector<Pair> pairs;
  pairs.reserve(n * (n - 1) / 2);
  for (std::size_t p = 1; p < n; ++p) {
    for (std::size_t q = 0; q < p; ++q) {
      if (q < n_negative) continue;
      pairs.emplace_back(p, q);
    }
  }
  return pairs;
}

// Drops every pair touching an index < n_negative from an arbitrary (e.g. adamPairIndices')
// pair list, preserving order -- same exclusion as lowerPairs' own n_negative, for a caller
// that cannot build its list with lowerPairs directly (RotationProblem, below).
std::vector<Pair> excludeNegative(std::vector<Pair> pairs, std::size_t n_negative) {
  if (n_negative == 0) return pairs;
  std::vector<Pair> out;
  out.reserve(pairs.size());
  for (const auto& pr : pairs) {
    if (pr.first < n_negative || pr.second < n_negative) continue;
    out.push_back(pr);
  }
  return out;
}

}  // namespace

// =====================================================================
// exact integral rotation
// =====================================================================

namespace {

// h' = U^dagger h U.
template <typename T>
Matrix<T> oneElectronRotated(const Matrix<T>& h, const Matrix<T>& u) {
  const std::size_t n = h.rows();
  if (h.cols() != n || u.rows() != n || u.cols() != n) {
    throw std::runtime_error("oneElectronRotated: inconsistent input dimensions");
  }
  Matrix<T> uh(n, n, T{});
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      const T hab = h(a, b);
      for (std::size_t q = 0; q < n; ++q) uh(a, q) += hab * u(b, q);
    }
  }
  Matrix<T> out(n, n, T{});
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t p = 0; p < n; ++p) {
      const T uap = conjugate(u(a, p));
      for (std::size_t q = 0; q < n; ++q) out(p, q) += uap * uh(a, q);
    }
  }
  return out;
}

// eri: transform the LAST index of an (n^3 x n) view with W, then cyclically
// permute (a,b,c,s) -> (s,a,b,c) so the next leg is last; legs d, c, b, a
// carry U, U, conj(U), conj(U) (physics notation <ab|cd>: the BRA legs, the
// first two indices, are conjugated -- the pattern rotateIntegrals'
// Cholesky path actually implements: V' = U^dagger V conj(U) per vector,
// eri' = sum_L V'(a,b) conj(V'(c,d))) and after four cycles the order is
// (p,q,r,s).
template <typename T>
Tensor4<T> twoElectronRotated(const Tensor4<T>& eri, const Matrix<T>& u) {
  const std::size_t n = u.rows();
  if (u.cols() != n || eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("twoElectronRotated: inconsistent input dimensions");
  }
  Matrix<T> uc(n, n, T{});
  for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) uc(i, j) = conjugate(u(i, j));
  const std::size_t n3 = n * n * n;
  std::vector<T> cur(eri.data(), eri.data() + n3 * n);
  std::vector<T> tmp(n3 * n);
  const Matrix<T>* legs[4] = {&u, &u, &uc, &uc};
  for (int leg = 0; leg < 4; ++leg) {
    const Matrix<T>& w = *legs[leg];
#pragma omp parallel for
    for (std::size_t x = 0; x < n3; ++x) {
      T* row = &tmp[x * n];
      for (std::size_t s = 0; s < n; ++s) row[s] = T{};
      for (std::size_t d = 0; d < n; ++d) {
        const T c = cur[x * n + d];
        for (std::size_t s = 0; s < n; ++s) row[s] += c * w(d, s);
      }
    }
#pragma omp parallel for
    for (std::size_t s = 0; s < n; ++s) {
      for (std::size_t x = 0; x < n3; ++x) cur[s * n3 + x] = tmp[x * n + s];
    }
  }
  Tensor4<T> out(n, n, n, n);
  std::copy(cur.begin(), cur.end(), out.data());
  return out;
}

// The two-electron part of a rotation for either ERI representation: a dense tensor is
// transformed leg by leg (O(n^5)), Cholesky vectors are transformed one by one (O(Nchol n^3)).
template <typename T>
Tensor4<T> rotateEri(const Tensor4<T>& eri, const Matrix<T>& u) { return twoElectronRotated(eri, u); }
template <typename T>
CholeskyEri<T> rotateEri(const CholeskyEri<T>& eri, const Matrix<T>& u) { return eri.rotated(u); }

// The dense tensor: a reference for a dense ERI, a freshly assembled O(n^4) tensor for Cholesky
// vectors (used only by start-up/final checks).
template <typename T>
const Tensor4<T>& denseOf(const Tensor4<T>& eri) { return eri; }
template <typename T>
Tensor4<T> denseOf(const CholeskyEri<T>& eri) { return eri.toDense(); }

}  // namespace

template <typename T>
RotatedIntegrals<T> rotateIntegralsExact(const Matrix<T>& h, const Tensor4<T>& eri,
                                         const Matrix<T>& u) {
  const std::size_t n = h.rows();
  if (h.cols() != n || u.rows() != n || u.cols() != n || eri.dim0() != n || eri.dim1() != n ||
      eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("rotateIntegralsExact: inconsistent input dimensions");
  }
  RotatedIntegrals<T> out;
  out.h = oneElectronRotated(h, u);
  out.eri = twoElectronRotated(eri, u);
  return out;
}

template RotatedIntegrals<double> rotateIntegralsExact(const Matrix<double>&, const Tensor4<double>&,
                                                       const Matrix<double>&);
template RotatedIntegrals<std::complex<double>> rotateIntegralsExact(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const Matrix<std::complex<double>>&);

namespace {

// Shared body of RdmftModel::hessian_vector/hessian_vector_dense (JK_only and PNOF, below):
// factored into a free function template, parameterized on the ERI type SEPARATELY from the
// model's own `Eri`, so the SAME logic can be instantiated once for the model's native
// representation (Tensor4 or CholeskyEri) and once more, always, for `Tensor4<T>` -- the second
// instantiation is what NeoOrbitalProblem's dense-tensor CACHE (see its own comment) calls for a
// Cholesky-vector model, avoiding CholeskyEri::operator()'s O(Nchol)-per-element cost on every one
// of the many Hessian-vector products a single Newton step's Davidson solve needs.
// Diagonal of the matrix jkOnlyHessianVectorImpl applies (NEO's Davidson preconditioner): real T
// -> jkOnlyHessianElement(p,q,p,q) (symmetrization 0.5(e_ij+e_ji) is the element itself at I = J);
// complex T -> the joint [t;y] diagonal, size 2*n_pairs.
template <typename T, typename EriT>
std::vector<double> jkOnlyHessianDiagonalImpl(JkFunctional functional, std::size_t f_l,
                                              const std::vector<Pair>& pair_indices,
                                              const Matrix<T>& h, const EriT& eri,
                                              const std::vector<double>& occ) {
  const auto hc = jkHartreeCoupling(functional, occ, f_l);
  const auto xc = jkExchangeCoupling(functional, occ, f_l);
  if constexpr (std::is_same_v<T, double>) {
    const std::size_t n_pairs = pair_indices.size();
    std::vector<double> d(n_pairs, 0.0);
#pragma omp parallel for
    for (std::size_t i = 0; i < n_pairs; ++i) {
      const auto& [p, q] = pair_indices[i];
      d[i] = jkOnlyHessianElement(h, eri, occ, hc, xc, p, q, p, q);
    }
    return d;
  } else {
    return jkOnlyJointHessianDiagonal(h, eri, occ, hc, xc, pair_indices);
  }
}

template <typename T, typename EriT>
std::vector<double> jkOnlyHessianVectorImpl(JkFunctional functional, std::size_t f_l,
                                            const std::vector<Pair>& pair_indices,
                                            const Matrix<T>& h, const EriT& eri,
                                            const std::vector<double>& occ,
                                            const std::vector<double>& v) {
  const auto hc = jkHartreeCoupling(functional, occ, f_l);
  const auto xc = jkExchangeCoupling(functional, occ, f_l);
  if constexpr (std::is_same_v<T, double>) {
    const std::size_t n_pairs = pair_indices.size();
    if (v.size() != n_pairs) throw std::runtime_error("JK_only hessian_vector: v has the wrong size");
    std::vector<double> w(n_pairs, 0.0);
    // jkOnlyHessianElement(pq,rs) is the SEQUENTIAL derivative (d/dkappa_pq of pair rs's
    // gradient), asymmetric off orbital stationarity, exactly like the complex-T joint
    // functions below -- symmetrize the same way main.cpp's symmetrizeReal(jkOnlyHessianMatrix
    // (...)) does before using it as a genuine (Newton-usable) Hessian.
#pragma omp parallel for
    for (std::size_t i = 0; i < n_pairs; ++i) {
      const auto& [p, q] = pair_indices[i];
      double acc = 0.0;
      for (std::size_t j = 0; j < n_pairs; ++j) {
        const auto& [r, s] = pair_indices[j];
        const double e_ij = jkOnlyHessianElement(h, eri, occ, hc, xc, p, q, r, s);
        const double e_ji = jkOnlyHessianElement(h, eri, occ, hc, xc, r, s, p, q);
        acc += 0.5 * (e_ij + e_ji) * v[j];
      }
      w[i] = acc;
    }
    return w;
  } else {
    return jkOnlyJointHessianVector(h, eri, occ, hc, xc, pair_indices, v);
  }
}

// Diagonal of the matrix pnofHessianVectorImpl applies -- see jkOnlyHessianDiagonalImpl. `full`/
// `pair_of` are the caller's cached 2-RDM matrices (same cache the Hessian-vector product uses).
template <typename T, typename EriT>
std::vector<double> pnofHessianDiagonalImpl(PnofFunctional functional,
                                            const std::vector<PnofGeminal>& geminals,
                                            const PnofFullTwoRdm& full,
                                            const std::vector<std::size_t>& pair_of, bool relativistic,
                                            const std::vector<Pair>& pair_indices, const Matrix<T>& h,
                                            const EriT& eri, const std::vector<double>& occ) {
  const auto fock = pnofFockMatrix(functional, h, eri, geminals, occ, relativistic);
  if constexpr (std::is_same_v<T, double>) {
    const std::size_t n_pairs = pair_indices.size();
    std::vector<double> d(n_pairs, 0.0);
#pragma omp parallel for
    for (std::size_t i = 0; i < n_pairs; ++i) {
      const auto& [p, q] = pair_indices[i];
      d[i] = hartreeExchangeHessianElement(h, eri, occ, full.two_rdm_h, full.two_rdm_x, fock, p, q, p, q,
                                           pair_of, full.two_rdm_l1, full.two_rdm_l2);
    }
    return d;
  } else {
    return hartreeExchangeJointHessianDiagonal(h, eri, occ, full.two_rdm_h, full.two_rdm_x, fock, pair_indices,
                                               pair_of, full.two_rdm_l1, full.two_rdm_l2);
  }
}

template <typename T, typename EriT>
std::vector<double> pnofHessianVectorImpl(PnofFunctional functional,
                                          const std::vector<PnofGeminal>& geminals,
                                          const PnofFullTwoRdm& full,
                                          const std::vector<std::size_t>& pair_of,
                                          bool relativistic, const std::vector<Pair>& pair_indices,
                                          const Matrix<T>& h, const EriT& eri,
                                          const std::vector<double>& occ,
                                          const std::vector<double>& v) {
  const auto fock = pnofFockMatrix(functional, h, eri, geminals, occ, relativistic);
  if constexpr (std::is_same_v<T, double>) {
    const std::size_t n_pairs = pair_indices.size();
    if (v.size() != n_pairs) throw std::runtime_error("PNOF hessian_vector: v has the wrong size");
    std::vector<double> w(n_pairs, 0.0);
    // Same symmetrization as the JK_only branch above (see its comment): pnofHessianElement
    // is the sequential derivative, asymmetric off stationarity.
#pragma omp parallel for
    for (std::size_t i = 0; i < n_pairs; ++i) {
      const auto& [p, q] = pair_indices[i];
      double acc = 0.0;
      for (std::size_t j = 0; j < n_pairs; ++j) {
        const auto& [r, s] = pair_indices[j];
        // `full`/`pair_of` are built ONCE per occupation vector (the caller's cache) -- rebuilding
        // them inside pnofHessianElement for each of the 2*n_pairs^2 elements cost about half of
        // every Hessian-vector product.
        const double e_ij = hartreeExchangeHessianElement(h, eri, occ, full.two_rdm_h, full.two_rdm_x, fock,
                                                          p, q, r, s, pair_of, full.two_rdm_l1, full.two_rdm_l2);
        const double e_ji = hartreeExchangeHessianElement(h, eri, occ, full.two_rdm_h, full.two_rdm_x, fock,
                                                          r, s, p, q, pair_of, full.two_rdm_l1, full.two_rdm_l2);
        acc += 0.5 * (e_ij + e_ji) * v[j];
      }
      w[i] = acc;
    }
    return w;
  } else {
    return pnofJointHessianVector(functional, h, eri, geminals, occ, relativistic, fock, pair_indices, v);
  }
}

}  // namespace

// =====================================================================
// models
// =====================================================================

template <typename T, typename Eri>
RdmftModel<T, Eri> makeJkOnlyModel(JkFunctional functional, std::size_t f_l, double n_electrons,
                              std::size_t n_total, std::size_t n_frozen,
                              std::size_t n_inactive_below,
                              std::size_t n_active, bool two_columns, std::size_t n_negative) {
  RdmftModel<T, Eri> model;
  const std::size_t frozen_base = n_inactive_below - n_frozen;
  // Actual full-array indices of the frozen/active windows -- MUST use the exact same algorithm
  // as main.cpp's buildFunctionalReport (its own frozen_indices/active_indices, same comment
  // there for the full story): a plain contiguous range is wrong for NON_REL's BLOCK-layout
  // spin-orbitals (alpha_k and beta_k are n_total/2 apart in the array, not adjacent), so NON_REL
  // gathers by INTERLEAVED spatial rank instead (rank r -> alpha at r, beta at r+n_spatial),
  // making LOCAL indices (2k, 2k+1) an adjacent pair uniformly for every T from here on.
  std::vector<std::size_t> frozen_indices(n_frozen), active_indices(n_active);
  if constexpr (std::is_same_v<T, double>) {
    const std::size_t n_spatial = n_total / 2;
    for (std::size_t i = 0; i < n_frozen; ++i) {
      const std::size_t rank = i / 2;
      frozen_indices[i] = (i % 2 == 0) ? rank : rank + n_spatial;
    }
    for (std::size_t i = 0; i < n_active; ++i) {
      const std::size_t rank = n_frozen / 2 + i / 2;
      active_indices[i] = (i % 2 == 0) ? rank : rank + n_spatial;
    }
  } else {
    for (std::size_t i = 0; i < n_frozen; ++i) frozen_indices[i] = frozen_base + i;
    for (std::size_t i = 0; i < n_active; ++i) active_indices[i] = n_inactive_below + i;
  }
  // Same layout as main.cpp's "Optimized occupation numbers" table.
  model.print_occupations = [=](const std::vector<double>& occ, std::ostream& out) {
    const auto round5 = [](double x) { return std::round(x * 1e5) / 1e5; };
    out << "    Optimized occupation numbers (index: n_p, global index into the full "
           "orbital/spinor space, fixed 5 decimals):\n";
    out << std::fixed << std::setprecision(5);
    double displayed_sum = 0.0;
    // JK_FROZEN_PAIRS: the frozen block, pinned at exactly 1 (never an SQP variable) --
    // PNOF's own "core geminal ... (frozen)" style.
    for (std::size_t i = 0; i + 1 < n_frozen; i += 2) {
      const std::size_t g0 = frozen_indices[i], g1 = frozen_indices[i + 1];
      out << "      " << std::setw(6) << g0 << std::setw(12) << occ[g0] << std::setw(10) << g1
          << std::setw(12) << occ[g1] << "  (frozen)\n";
      displayed_sum += round5(occ[g0]) + round5(occ[g1]);
    }
    if (two_columns) {
      for (std::size_t i = 0; i + 1 < n_active; i += 2) {
        const std::size_t g0 = active_indices[i], g1 = active_indices[i + 1];
        out << "      " << std::setw(6) << g0 << std::setw(12) << occ[g0] << std::setw(10) << g1
            << std::setw(12) << occ[g1] << "\n";
        displayed_sum += round5(occ[g0]) + round5(occ[g1]);
      }
      if (n_active % 2 == 1) {
        const std::size_t g = active_indices[n_active - 1];
        out << "      " << std::setw(6) << g << std::setw(12) << occ[g] << "\n";
        displayed_sum += round5(occ[g]);
      }
    } else {
      for (std::size_t i = 0; i < n_active; ++i) {
        const std::size_t g = active_indices[i];
        out << "      " << std::setw(6) << g << std::setw(12) << occ[g] << "\n";
        displayed_sum += round5(occ[g]);
      }
    }
    out << std::defaultfloat << std::setprecision(6);
    out << "    Sum of the occupation numbers shown above, at that same 5-decimal precision "
           "(expect close to "
        << (n_electrons + static_cast<double>(n_frozen)) << "): " << std::setprecision(10)
        << displayed_sum << std::setprecision(6) << "\n";
  };
  model.energy = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
    return jkFunctionalEnergy(h, eri, occ, functional, f_l);
  };
  model.gradient = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
    const auto hc = jkHartreeCoupling(functional, occ, f_l);
    const auto xc = jkExchangeCoupling(functional, occ, f_l);
    return jkOnlyOrbitalGradient(h, eri, occ, hc, xc);
  };
  {
    // MUST be the SAME pair list RotationProblem/NeoOrbitalProblem actually rotate over (dropped
    // via excludeNegative there, C4_DHF only) -- otherwise the joint [t;y] vector NEO passes in
    // has a different size than this row-based Hv expects (hessianPairIndices(n_total) alone,
    // unfiltered, would silently mismatch whenever n_negative > 0).
    const auto pair_indices = lowerPairs(n_total, n_negative);
    model.hessian_vector = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
                               const std::vector<double>& v) {
      return jkOnlyHessianVectorImpl(functional, f_l, pair_indices, h, eri, occ, v);
    };
    // ALWAYS Tensor4-based (even when Eri = CholeskyEri) -- see NeoOrbitalProblem's own comment
    // on why (its dense-tensor cache for the Cholesky case calls this, not `hessian_vector`).
    model.hessian_vector_dense = [=](const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occ, const std::vector<double>& v) {
      return jkOnlyHessianVectorImpl(functional, f_l, pair_indices, h, eri, occ, v);
    };
    model.hessian_diagonal = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
      return jkOnlyHessianDiagonalImpl(functional, f_l, pair_indices, h, eri, occ);
    };
    model.hessian_diagonal_dense = [=](const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occ) {
      return jkOnlyHessianDiagonalImpl(functional, f_l, pair_indices, h, eri, occ);
    };
  }
  // Kramers/spin pairs within the active window -- SAME convention and SAME reasoning as
  // main.cpp's buildFunctionalReport (its own `state` is exactly this function's reduced
  // per-pair vector, carried across the FULL_OPTIMIZATION macro loop's own occupation
  // re-optimizations, so the two MUST agree): adjacent LOCAL indices (2k, 2k+1), which
  // active_indices[] above already resolves to each T's own actual-array layout.
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(n_active / 2);
  for (std::size_t i = 0; i + 1 < n_active; i += 2) pairs.emplace_back(i, i + 1);
  const std::size_t n_pairs = pairs.size();
  model.optimize_occupations = [=](const Matrix<T>& h, const Eri& eri,
                                   std::vector<double>& state) {
    auto embed = [&](const std::vector<double>& reduced) {
      std::vector<double> full(n_total, 0.0);
      for (std::size_t i = 0; i < n_frozen; ++i) full[frozen_indices[i]] = 1.0;
      for (std::size_t i = 0; i < n_pairs; ++i) {
        full[active_indices[pairs[i].first]] = reduced[i];
        full[active_indices[pairs[i].second]] = reduced[i];
      }
      return full;
    };
    const SqpValueFn value_fn = [&](const std::vector<double>& x) {
      return jkFunctionalEnergy(h, eri, embed(x), functional, f_l);
    };
    const SqpGradientFn gradient_fn = [&](const std::vector<double>& x) {
      const auto g = jkFunctionalGradient(h, eri, embed(x), functional, f_l);
      std::vector<double> reduced_grad(n_pairs);
      for (std::size_t i = 0; i < n_pairs; ++i) {
        reduced_grad[i] = g[active_indices[pairs[i].first]] + g[active_indices[pairs[i].second]];
      }
      return reduced_grad;
    };
    const SqpHessianFn hessian_fn = [&](const std::vector<double>& x) {
      const auto full = jkFunctionalHessian(h, eri, embed(x), functional, f_l);
      Matrix<double> reduced_hess(n_pairs, n_pairs);
      for (std::size_t r = 0; r < n_pairs; ++r) {
        const std::size_t pr = active_indices[pairs[r].first], qr = active_indices[pairs[r].second];
        for (std::size_t s = 0; s < n_pairs; ++s) {
          const std::size_t ps = active_indices[pairs[s].first], qs = active_indices[pairs[s].second];
          reduced_hess(r, s) = full(pr, ps) + full(pr, qs) + full(qr, ps) + full(qr, qs);
        }
      }
      return reduced_hess;
    };
    const Matrix<double> a_eq(1, n_pairs, 2.0);  // 2 electrons per pair (both members tied)
    const std::vector<double> b_eq = {n_electrons};
    const std::vector<double> lb(n_pairs, kOccupationEpsilon);
    const std::vector<double> ub(n_pairs, 1.0 - kOccupationEpsilon);
    const auto res = solveSqp(value_fn, gradient_fn, hessian_fn, a_eq, b_eq, lb, ub, state);
    state = res.x;
    return RdmftOccupationResult{embed(res.x), res.objective_value, res.converged, res.iterations};
  };
  return model;
}

template <typename T, typename Eri>
RdmftModel<T, Eri> makePnofModel(PnofFunctional functional, std::vector<PnofGeminal> geminals,
                            std::size_t n_core, int pnof_subspaces, int pnof_coupling,
                            bool relativistic, bool sqp_occupations, std::size_t n_total,
                            std::size_t n_negative) {
  RdmftModel<T, Eri> model;
  const std::size_t n_frontier = geminals.size() - n_core;
  // Same layout as main.cpp's "Optimized geminal occupation numbers" listing.
  model.print_occupations = [=](const std::vector<double>& occ, std::ostream& out) {
    out << "    Optimized geminal occupation numbers (n_p, both members of each Kramers/spin "
           "pair share this value; fixed 5 decimals):\n";
    out << std::fixed << std::setprecision(5);
    for (std::size_t a = 0; a < n_core; ++a) {
      out << "      core        geminal (" << geminals[a].i << "," << geminals[a].ibar
          << "): n = 1.00000 (frozen)\n";
    }
    for (std::size_t a = n_core; a < geminals.size(); ++a) {
      out << "      subspace " << std::setw(2) << geminals[a].subspace_id << " "
          << (geminals[a].is_principal ? "principal" : "virtual  ") << " geminal (" << geminals[a].i
          << "," << geminals[a].ibar << "): n = " << occ[geminals[a].i] << "\n";
    }
    out << std::defaultfloat << std::setprecision(6);
  };
  auto embed = [=](const std::vector<double>& frontier) {
    std::vector<double> full(n_total, 0.0);
    for (std::size_t a = 0; a < n_core; ++a) full[geminals[a].i] = full[geminals[a].ibar] = 1.0;
    for (std::size_t a = n_core; a < geminals.size(); ++a) {
      full[geminals[a].i] = full[geminals[a].ibar] = frontier[a - n_core];
    }
    return full;
  };
  // The ORBITAL-rotation energy must be valid for ANY orbitals: pnofElectronicEnergy
  // reads one bar-combination of K_ij/L_ij and doubles it, i.e. assumes eri keeps
  // its full Kramers/spin-pair symmetry, which a general rotation breaks. The
  // full two-RDM path (PnofFock.h's buildPnofFullTwoRdm + hartreeExchangeEnergy,
  // the same one the gradient/Hessian were validated against) sums over every
  // actual orbital pair independently and is exact either way (and equals
  // pnofElectronicEnergy when the symmetry does hold).
  const std::vector<std::size_t> pair_of = buildPnofPairOf(geminals, n_total);
  model.energy = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
    const auto full = buildPnofFullTwoRdm(functional, geminals, occ, n_total, relativistic);
    return hartreeExchangeEnergy(h, eri, occ, full.two_rdm_h, full.two_rdm_x, pair_of,
                                 full.two_rdm_l1, full.two_rdm_l2);
  };
  model.gradient = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
    return orbitalGradient(pnofFockMatrix(functional, h, eri, geminals, occ, relativistic));
  };
  {
    // Same requirement as JK_only's own comment: must match RotationProblem/NeoOrbitalProblem's
    // actual (possibly n_negative-restricted, C4_DHF only) pair list exactly.
    const auto pair_indices = lowerPairs(n_total, n_negative);
    // The 2-RDM matrices (two_rdm_h/x/l1/l2) depend only on the occupations, which are FIXED during
    // an orbital-optimization step: build them once per occupation vector and reuse them for every
    // Hessian-vector product (each Davidson iteration of every Newton step) instead of per element.
    struct TwoRdmCache {
      std::vector<double> occ;
      PnofFullTwoRdm full;
      bool valid = false;
    };
    const auto cache = std::make_shared<TwoRdmCache>();
    const auto cachedTwoRdm = [=](const std::vector<double>& occ) -> const PnofFullTwoRdm& {
      if (!cache->valid || cache->occ != occ) {
        cache->full = buildPnofFullTwoRdm(functional, geminals, occ, n_total, relativistic);
        cache->occ = occ;
        cache->valid = true;
      }
      return cache->full;
    };
    model.hessian_vector = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
                               const std::vector<double>& v) {
      return pnofHessianVectorImpl(functional, geminals, cachedTwoRdm(occ), pair_of, relativistic, pair_indices,
                                   h, eri, occ, v);
    };
    // ALWAYS Tensor4-based -- see JK_only's own comment on this field / NeoOrbitalProblem's cache.
    model.hessian_vector_dense = [=](const Matrix<T>& h, const Tensor4<T>& eri,
                                     const std::vector<double>& occ, const std::vector<double>& v) {
      return pnofHessianVectorImpl(functional, geminals, cachedTwoRdm(occ), pair_of, relativistic, pair_indices,
                                   h, eri, occ, v);
    };
    model.hessian_diagonal = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ) {
      return pnofHessianDiagonalImpl(functional, geminals, cachedTwoRdm(occ), pair_of, relativistic, pair_indices,
                                     h, eri, occ);
    };
    model.hessian_diagonal_dense = [=](const Matrix<T>& h, const Tensor4<T>& eri, const std::vector<double>& occ) {
      return pnofHessianDiagonalImpl(functional, geminals, cachedTwoRdm(occ), pair_of, relativistic, pair_indices,
                                     h, eri, occ);
    };
  }
  model.symmetric_shortcut_energy = [=](const Matrix<T>& h, const Eri& eri,
                                        const std::vector<double>& occ) {
    const auto two_rdm = buildPnofTwoRdm(functional, geminals, occ, relativistic);
    return pnofElectronicEnergy(functional, h, eri, occ, geminals, two_rdm, relativistic);
  };

  if (sqp_occupations) {
    model.optimize_occupations = [=](const Matrix<T>& h, const Eri& eri,
                                     std::vector<double>& state) {
      const SqpValueFn value_fn = [&](const std::vector<double>& x) {
        const auto occ = embed(x);
        const auto two_rdm = buildPnofTwoRdm(functional, geminals, occ, relativistic);
        return pnofElectronicEnergy(functional, h, eri, occ, geminals, two_rdm, relativistic);
      };
      const SqpGradientFn gradient_fn = [&](const std::vector<double>& x) {
        const auto g = pnofOccupationGradient(functional, h, eri, embed(x), geminals, relativistic);
        return std::vector<double>(g.begin() + static_cast<std::ptrdiff_t>(n_core), g.end());
      };
      const SqpHessianFn hessian_fn = [&](const std::vector<double>& x) {
        const auto full = (functional == PnofFunctional::kGnof)
                              ? pnofOccupationHessianFD(functional, h, eri, embed(x), geminals,
                                                         relativistic, 0.1 * kOccupationEpsilon)
                              : pnofOccupationHessian(functional, h, eri, embed(x), geminals, relativistic);
        Matrix<double> active(n_frontier, n_frontier);
        for (std::size_t r = 0; r < n_frontier; ++r)
          for (std::size_t s = 0; s < n_frontier; ++s) active(r, s) = full(n_core + r, n_core + s);
        return active;
      };
      Matrix<double> a_eq(static_cast<std::size_t>(pnof_subspaces), n_frontier, 0.0);
      for (int s = 0; s < pnof_subspaces; ++s) {
        const std::size_t base = static_cast<std::size_t>(s) * static_cast<std::size_t>(pnof_coupling);
        for (int v = 0; v < pnof_coupling; ++v) a_eq(static_cast<std::size_t>(s), base + static_cast<std::size_t>(v)) = 1.0;
      }
      const std::vector<double> b_eq(static_cast<std::size_t>(pnof_subspaces), 1.0);
      const std::vector<double> lb(n_frontier, kOccupationEpsilon);
      const std::vector<double> ub(n_frontier, 1.0 - kOccupationEpsilon);
      const auto res = solveSqp(value_fn, gradient_fn, hessian_fn, a_eq, b_eq, lb, ub, state);
      state = res.x;
      return RdmftOccupationResult{embed(res.x), res.objective_value, res.converged, res.iterations};
    };
    return model;
  }

  model.optimize_occupations = [=](const Matrix<T>& h, const Eri& eri,
                                   std::vector<double>& state) {
    const std::size_t per = static_cast<std::size_t>(pnof_coupling - 1);
    auto embedGammas = [&](const std::vector<double>& gammas) {
      std::vector<double> full(n_total, 0.0);
      for (std::size_t a = 0; a < n_core; ++a) full[geminals[a].i] = full[geminals[a].ibar] = 1.0;
      for (int s = 0; s < pnof_subspaces; ++s) {
        const std::size_t gb = static_cast<std::size_t>(s) * per;
        const std::vector<double> sg(gammas.begin() + static_cast<std::ptrdiff_t>(gb),
                                     gammas.begin() + static_cast<std::ptrdiff_t>(gb + per));
        const auto occ = pnofSubspaceOccupationsFromGammas(pnof_coupling, sg);
        const std::size_t fb = n_core + static_cast<std::size_t>(s) * static_cast<std::size_t>(pnof_coupling);
        for (int v = 0; v < pnof_coupling; ++v) {
          const auto& g = geminals[fb + static_cast<std::size_t>(v)];
          full[g.i] = full[g.ibar] = occ[static_cast<std::size_t>(v)];
        }
      }
      return full;
    };
    const LbfgsValueFn value_fn = [&](const std::vector<double>& gammas) {
      const auto occ = embedGammas(gammas);
      const auto two_rdm = buildPnofTwoRdm(functional, geminals, occ, relativistic);
      return pnofElectronicEnergy(functional, h, eri, occ, geminals, two_rdm, relativistic);
    };
    const LbfgsGradientFn gradient_fn = [&](const std::vector<double>& gammas) {
      const auto occ = embedGammas(gammas);
      const auto full_grad = pnofOccupationGradient(functional, h, eri, occ, geminals, relativistic);
      std::vector<double> grad(gammas.size(), 0.0);
      for (int s = 0; s < pnof_subspaces; ++s) {
        const std::size_t gb = static_cast<std::size_t>(s) * per;
        const std::vector<double> sg(gammas.begin() + static_cast<std::ptrdiff_t>(gb),
                                     gammas.begin() + static_cast<std::ptrdiff_t>(gb + per));
        const auto jac = pnofSubspaceOccupationsFromGammasWithGradient(pnof_coupling, sg);
        const std::size_t fb = n_core + static_cast<std::size_t>(s) * static_cast<std::size_t>(pnof_coupling);
        for (std::size_t k = 0; k < per; ++k) {
          double sum = 0.0;
          for (int v = 0; v < pnof_coupling; ++v) {
            sum += full_grad[fb + static_cast<std::size_t>(v)] * jac.docc_dgamma(static_cast<std::size_t>(v), k);
          }
          grad[gb + k] = sum;
        }
      }
      return grad;
    };
    const auto res = solveLbfgs(value_fn, gradient_fn, state);
    state = res.x;
    return RdmftOccupationResult{embedGammas(res.x), res.objective_value, res.converged, res.iterations};
  };
  return model;
}

// =====================================================================
// ADAM problem over the orbital rotations
// =====================================================================

namespace {

// Trial and best integrals in the ROTATED basis, plus the cumulative
// rotation matrices (C_new = C_start * U_total) -- occupations fixed.
template <typename T, typename Eri>
class RotationProblem : public AdamProblem<T> {
 public:
  // `n_negative` (default 0, C4_DHF only): excludes every pair touching the negative-energy
  // branch's first n_negative indices from the rotation entirely -- see lowerPairs' own comment.
  RotationProblem(const RdmftModel<T, Eri>& model, const Matrix<T>& h, const Eri& eri,
                  const std::vector<std::size_t>& spin_partner = {}, std::size_t n_negative = 0)
      : model_(model), n_(h.rows()),
        pairs_(excludeNegative(adamPairIndices(h.rows(), false), n_negative)), h_t_(h), eri_t_(eri),
        h_b_(h), eri_b_(eri), u_t_(h.rows(), h.rows(), T{}), u_b_(h.rows(), h.rows(), T{}) {
    for (std::size_t i = 0; i < n_; ++i) u_t_(i, i) = u_b_(i, i) = T(1.0);
    if (!spin_partner.empty()) {
      // twin[I] = index of the pair (P p, P q) when it is again a p > q pair (a same-spin pair's
      // twin); spin-mixing pairs map to a swapped pair and are left alone (their gradient is
      // exactly zero by Sz symmetry).
      std::vector<long> index(n_ * n_, -1);
      for (std::size_t i = 0; i < pairs_.size(); ++i) index[pairs_[i].first * n_ + pairs_[i].second] = static_cast<long>(i);
      twin_.assign(pairs_.size(), -1);
      for (std::size_t i = 0; i < pairs_.size(); ++i) {
        const std::size_t pp = spin_partner[pairs_[i].first], qq = spin_partner[pairs_[i].second];
        if (pp > qq) twin_[i] = index[pp * n_ + qq];
      }
    }
  }
  void setOccupations(const std::vector<double>& occ) { occ_ = occ; }
  std::size_t dimension() const override { return pairs_.size(); }
  double energy() override { return model_.energy(h_t_, eri_t_, occ_); }
  std::vector<T> gradient() override {
    std::vector<T> g = adamGradientVector(model_.gradient(h_t_, eri_t_, occ_), pairs_);
    if (!twin_.empty()) {
      const std::vector<T> raw = g;
      for (std::size_t i = 0; i < g.size(); ++i) {
        if (twin_[i] >= 0) g[i] = T(0.5) * (raw[i] + raw[static_cast<std::size_t>(twin_[i])]);
      }
    }
    return g;
  }
  void rotate(const std::vector<T>& step) override {
    const Matrix<T> u = spinorRotationMatrix(adamKappaMatrix(n_, pairs_, step));
    h_t_ = oneElectronRotated(h_t_, u);
    eri_t_ = rotateEri(eri_t_, u);
    u_t_ = u_t_ * u;
  }
  void saveBest() override { h_b_ = h_t_; eri_b_ = eri_t_; u_b_ = u_t_; }
  void restoreBest() override { h_t_ = h_b_; eri_t_ = eri_b_; u_t_ = u_b_; }
  const Matrix<T>& h() const { return h_t_; }
  const Eri& eri() const { return eri_t_; }
  const Matrix<T>& totalRotation() const { return u_t_; }
  const std::vector<Pair>& pairs() const { return pairs_; }

 private:
  const RdmftModel<T, Eri>& model_;
  std::size_t n_;
  std::vector<Pair> pairs_;
  std::vector<double> occ_;
  Matrix<T> h_t_;
  Eri eri_t_;
  Matrix<T> h_b_;
  Eri eri_b_;
  Matrix<T> u_t_, u_b_;
  std::vector<long> twin_;
};

// =====================================================================
// NEO problem over the orbital rotations
// =====================================================================

// Trial integrals and the cumulative rotation, in the JOINT real parametrization NEO.h expects
// (Utils/NEO.h: T = double, real orbitals directly for NON_REL; the real [t;y] joint vector,
// Hessian_opt/OrbitalGradient.h's own convention, for X2C's complex spinors -- see NEO.h's own
// note that this IS its intended use of T = double). Works for `Eri = Tensor4<Scalar>` AND
// `Eri = CholeskyEri<Scalar>` (RdmftModel::hessian_vector is set for both -- see
// FullOptimization.h). `Scalar` is the integrals' own scalar type (double for NON_REL,
// complex<double> for X2C); `dimension()` is pairs_.size() (NON_REL) or 2*pairs_.size() (X2C).
//
// NOTE (NON_REL, spin_partner): unlike RotationProblem, this does NOT average the alpha/beta twin
// gradient entries the way ADAM does. It relies instead on the underlying physics: a spin-
// restricted (Sz-conserving) starting point has IDENTICAL gradient/Hessian-action on a pure-alpha
// pair and its pure-beta twin, and exactly zero on a spin-mixing pair, so a trajectory that starts
// there stays there analytically. The end-of-run final test (spin symmetry of h/eri) still checks
// this rather than assuming it -- unlike ADAM's per-parameter normalization, NEO's Newton step has
// no mechanism that would spontaneously BREAK an exact tie, but a degenerate Davidson eigenvector
// choice in principle could; a genuine spin-restricted NEO problem (a "SpinRestriction" analogous
// to Utils/KramersRestriction.h) would remove even that risk and is a natural follow-up if the
// final test ever shows drift.
// A THIN wrapper, not a second state holder: it reads/rotates the SAME RotationProblem the ADAM
// branch would use (h()/eri()/pairs() for reading, rotate() -- via the identical adamKappaMatrix
// per-pair convention adamPairIndices/hessianPairIndices already share -- for the one-shot commit
// accept() needs), so `problem.totalRotation()`/`h()`/`eri()` stay the SINGLE source of truth the
// rest of runFullOptimization (occupation re-optimization, the final tests) already reads,
// whichever optimizer actually advanced them. Only trialEnergy() needs its own (non-mutating)
// rotation -- built with the SAME oneElectronRotated/rotateEri pair RotationProblem::rotate()
// itself uses (Tensor4: exact O(n^5); CholeskyEri: rotates the vectors), rather than the
// Tensor4-only rotateIntegralsExact -- since NEO must be able to try a step without committing it.
template <typename Scalar, typename Eri>
class NeoOrbitalProblem : public NeoProblem<double> {
 public:
  NeoOrbitalProblem(const RdmftModel<Scalar, Eri>& model, RotationProblem<Scalar, Eri>& problem)
      : model_(model), problem_(problem) {}
  void setOccupations(const std::vector<double>& occ) { occ_ = occ; }
  std::size_t dimension() const override {
    return std::is_same_v<Scalar, double> ? problem_.pairs().size() : 2 * problem_.pairs().size();
  }
  double energy() override { return model_.energy(problem_.h(), problem_.eri(), occ_); }
  std::vector<double> gradient() override {
    return jointOrbitalGradient(model_.gradient(problem_.h(), problem_.eri(), occ_), problem_.pairs());
  }
  std::vector<double> hessianVector(const std::vector<double>& v) override {
    if constexpr (std::is_same_v<Eri, Tensor4<Scalar>>) {
      if (!model_.hessian_vector) {
        throw std::runtime_error("NeoOrbitalProblem: the model has no hessian_vector");
      }
      return model_.hessian_vector(problem_.h(), problem_.eri(), occ_, v);
    } else {
      // CholeskyEri: model_.hessian_vector would call CholeskyEri::operator()'s O(n_chol) element
      // access O(n_pairs^2) times PER CALL, and a single Newton step's Davidson solve makes many
      // such calls -- materialize a dense Tensor4 ONCE per accepted step instead (invalidated in
      // accept() below) and reuse it for every hessianVector() call until the next one. Measured
      // on lih_gnof_full_optimization_cholesky.inp (X2C): without this, a single macro-iteration
      // did not finish in 5 minutes; with it, the whole run matches the dense/ADAM result and
      // takes about as long as the dense NEO case.
      if (!model_.hessian_vector_dense) {
        throw std::runtime_error("NeoOrbitalProblem: the model has no hessian_vector_dense");
      }
      if (!dense_eri_valid_) {
        dense_eri_ = problem_.eri().toDense();
        dense_eri_valid_ = true;
      }
      return model_.hessian_vector_dense(problem_.h(), dense_eri_, occ_, v);
    }
  }
  // Diagonal of the Hessian the Hessian-vector product applies: NEO's Davidson preconditioner
  // (without it the unpreconditioned solve needs 150+ Hessian products per step on CO/cc-pVDZ).
  // O(n_pairs) elements, so cheap next to ONE hessianVector() call. Empty if the model has none.
  std::vector<double> hessianDiagonal() override {
    if constexpr (std::is_same_v<Eri, Tensor4<Scalar>>) {
      if (!model_.hessian_diagonal) return {};
      return model_.hessian_diagonal(problem_.h(), problem_.eri(), occ_);
    } else {
      if (!model_.hessian_diagonal_dense) return {};
      if (!dense_eri_valid_) {
        dense_eri_ = problem_.eri().toDense();
        dense_eri_valid_ = true;
      }
      return model_.hessian_diagonal_dense(problem_.h(), dense_eri_, occ_);
    }
  }
  double trialEnergy(const std::vector<double>& d) override {
    const auto u = spinorRotationMatrix(buildKappa(d));
    const auto h_trial = oneElectronRotated(problem_.h(), u);
    const auto eri_trial = rotateEri(problem_.eri(), u);
    return model_.energy(h_trial, eri_trial, occ_);
  }
  void accept(const std::vector<double>& d) override {
    problem_.rotate(toStep(d));
    dense_eri_valid_ = false;  // problem_.eri() just changed -- the cache (if any) is stale.
  }

 private:
  Matrix<Scalar> buildKappa(const std::vector<double>& x) const {
    const auto& pairs = problem_.pairs();
    Matrix<Scalar> kappa(problem_.h().rows(), problem_.h().rows(), Scalar{});
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      const auto& [p, q] = pairs[i];
      if constexpr (std::is_same_v<Scalar, double>) {
        kappa(p, q) = x[i];
        kappa(q, p) = -x[i];
      } else {
        const Scalar z(x[i], x[pairs.size() + i]);
        kappa(p, q) = z;
        kappa(q, p) = -std::conj(z);
      }
    }
    return kappa;
  }
  std::vector<Scalar> toStep(const std::vector<double>& d) const {
    const auto& pairs = problem_.pairs();
    std::vector<Scalar> step(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      if constexpr (std::is_same_v<Scalar, double>) {
        step[i] = d[i];
      } else {
        step[i] = Scalar(d[i], d[pairs.size() + i]);
      }
    }
    return step;
  }

  const RdmftModel<Scalar, Eri>& model_;
  RotationProblem<Scalar, Eri>& problem_;
  std::vector<double> occ_;
  // Only used when Eri != Tensor4<Scalar> -- see hessianVector()'s own comment.
  Tensor4<Scalar> dense_eri_;
  bool dense_eri_valid_ = false;
};

// The (pure-alpha-index, pure-beta-twin-index) pairs of pairs, one entry per orbit, canonically
// ordered (alpha < beta): SAME twin relation as RotationProblem's own `twin_` (spin_partner maps
// each half rigidly onto the other, preserving p>q order within a pure pair), just returned as a
// list instead of consumed inline. A spin-mixing pair (one index alpha, one beta) has no orbit --
// left out entirely, matching that its gradient/Hessian-action is exactly zero by Sz symmetry.
std::vector<std::pair<std::size_t, std::size_t>> spinOrbits(
    const std::vector<Pair>& pairs, const std::vector<std::size_t>& spin_partner) {
  const std::size_t n = spin_partner.size();
  std::vector<long> index(n * n, -1);
  for (std::size_t i = 0; i < pairs.size(); ++i) index[pairs[i].first * n + pairs[i].second] = static_cast<long>(i);
  std::vector<std::pair<std::size_t, std::size_t>> orbits;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const std::size_t pp = spin_partner[pairs[i].first], qq = spin_partner[pairs[i].second];
    if (pp > qq) {
      const long twin = index[pp * n + qq];
      if (twin >= 0 && i < static_cast<std::size_t>(twin)) orbits.emplace_back(i, static_cast<std::size_t>(twin));
    }
  }
  return orbits;
}

// NEO in the spin-restricted (Sz-conserving) sector for NON_REL, the T=double analogue of
// Utils/KramersRestriction.h's KramersNeoProblem: an ISOMETRIC embedding Q (Q^T Q = 1) tying each
// pure-alpha pair to its pure-beta twin with EQUAL weight 1/sqrt(2) (no signs -- unlike Kramers
// orbits, real orbitals have no y and the two twins are related by nothing more than a spin swap),
// so every accepted step is exactly alpha/beta-symmetric by construction, closing the gap ADAM's
// own per-parameter gradient AVERAGING only approximates (see NeoOrbitalProblem's own header
// comment: without this wrapper the end-of-run spin-symmetry test can fail by ~1e-6, confirmed on
// water_muller_as_full_optimization.inp with ORBITAL_OPTIMIZER NEO).
class SpinRestrictedNeoProblem : public NeoProblem<double> {
 public:
  SpinRestrictedNeoProblem(NeoProblem<double>& full, std::vector<std::pair<std::size_t, std::size_t>> orbits)
      : full_(full), orbits_(std::move(orbits)) {}
  std::size_t dimension() const override { return orbits_.size(); }
  double energy() override { return full_.energy(); }
  std::vector<double> gradient() override { return contract(full_.gradient()); }
  std::vector<double> hessianVector(const std::vector<double>& v) override {
    return contract(full_.hessianVector(expand(v)));
  }
  // Preconditioner only: the reduced diagonal is (H_aa + H_bb)/2 + H_ab; the alpha-beta cross term
  // H_ab is dropped (same approximation as KramersRestriction::contractDiagonal), which affects
  // only the convergence rate of the Davidson solve, never its result.
  std::vector<double> hessianDiagonal() override {
    const std::vector<double> d = full_.hessianDiagonal();
    if (d.empty()) return {};
    std::vector<double> reduced(orbits_.size());
    for (std::size_t k = 0; k < orbits_.size(); ++k) reduced[k] = 0.5 * (d[orbits_[k].first] + d[orbits_[k].second]);
    return reduced;
  }
  double trialEnergy(const std::vector<double>& d) override { return full_.trialEnergy(expand(d)); }
  void accept(const std::vector<double>& d) override { full_.accept(expand(d)); }

 private:
  std::vector<double> expand(const std::vector<double>& reduced) const {
    std::vector<double> full(full_.dimension(), 0.0);
    constexpr double kInvSqrt2 = 0.70710678118654752440;
    for (std::size_t k = 0; k < orbits_.size(); ++k) {
      full[orbits_[k].first] = kInvSqrt2 * reduced[k];
      full[orbits_[k].second] = kInvSqrt2 * reduced[k];
    }
    return full;
  }
  std::vector<double> contract(const std::vector<double>& full) const {
    std::vector<double> reduced(orbits_.size());
    constexpr double kInvSqrt2 = 0.70710678118654752440;
    for (std::size_t k = 0; k < orbits_.size(); ++k) {
      reduced[k] = kInvSqrt2 * (full[orbits_[k].first] + full[orbits_[k].second]);
    }
    return reduced;
  }

  NeoProblem<double>& full_;
  std::vector<std::pair<std::size_t, std::size_t>> orbits_;
};

// Kramers-structure deviations (Utils/KramersPairing.h) for either scalar type (complex only used).
template <typename T>
double timeReversalDeviation(const Matrix<T>& m) {
  Matrix<std::complex<double>> c(m.rows(), m.cols());
  for (std::size_t p = 0; p < m.rows(); ++p)
    for (std::size_t q = 0; q < m.cols(); ++q) c(p, q) = std::complex<double>(m(p, q));
  return kramersOneBodyDeviation(c);
}

template <typename T>
std::pair<double, double> twoBodyTimeReversalDeviation(const Tensor4<T>& eri) {
  Tensor4<std::complex<double>> c(eri.dim0(), eri.dim1(), eri.dim2(), eri.dim3());
  for (std::size_t a = 0; a < eri.dim0(); ++a)
    for (std::size_t b = 0; b < eri.dim1(); ++b)
      for (std::size_t x = 0; x < eri.dim2(); ++x)
        for (std::size_t d = 0; d < eri.dim3(); ++d) c(a, b, x, d) = std::complex<double>(eri(a, b, x, d));
  double scale = 0.0;
  const double dev = kramersTwoBodyDeviation(c, &scale);
  return {dev, scale};
}

template <typename T>
Matrix<T> generatorMatrix(std::size_t n, std::size_t p, std::size_t q, double t, double y) {
  Matrix<T> k(n, n, T{});
  k(p, q) = T(t);
  k(q, p) = T(-t);
  if constexpr (!std::is_same_v<T, double>) {
    k(p, q) += std::complex<double>(0.0, y);
    k(q, p) += std::complex<double>(0.0, y);
  }
  return k;
}

// Part (a): validation of the pieces the macro loop relies on.
template <typename T, typename Eri>
bool runChecks(const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
               const RdmftModel<T, Eri>& model, bool kramers, const std::vector<std::size_t>& spin_partner,
               std::ostream& log, std::size_t n_negative = 0) {
  const std::size_t n = h.rows();
  const auto pairs = lowerPairs(n, n_negative);
  bool ok = true;
  auto verdict = [&](bool pass, const std::string& what) {
    log << "    [" << (pass ? "PASS" : "FAIL") << "] " << what << "\n";
    ok = ok && pass;
  };
  const Matrix<T> g0 = model.gradient(h, eri, occ);
  const auto joint_gradient = jointOrbitalGradient(g0, pairs);
  double g_max = 0.0;
  for (double v : joint_gradient) g_max = std::max(g_max, std::abs(v));
  log << std::scientific << std::setprecision(2);
  log << "    max |orbital gradient entry| at the start: " << g_max << "\n";
  if (model.symmetric_shortcut_energy) {
    // The occupation optimizer minimizes the pair-symmetric shortcut energy, the orbital
    // stage the full two-RDM energy whose derivative is the gradient: they must be the same
    // functional, else the two alternating stages would optimize different things.
    const double e_full = model.energy(h, eri, occ);
    const double e_sym = model.symmetric_shortcut_energy(h, eri, occ);
    log << std::setprecision(10) << std::defaultfloat << "    energy used by the orbital stage (full two-RDM) " << e_full
        << ", by the occupation optimizer (pair-symmetric) " << e_sym << std::scientific << std::setprecision(2)
        << " (|diff| = " << std::abs(e_full - e_sym) << ")\n";
    verdict(std::abs(e_full - e_sym) < 1e-8,
            "orbital-stage and occupation-stage energies are the same functional");
  }
  // The two pairs with the largest |g| carry the FD test.
  std::vector<std::size_t> order(pairs.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  const std::size_t m = pairs.size();
  auto mag = [&](std::size_t i) {
    return std::abs(std::complex<double>(g0(pairs[i].first, pairs[i].second)));
  };
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return mag(a) > mag(b); });
  const std::size_t n_test = std::min<std::size_t>(2, m);

  // (1) the rotation of the two-electron integrals, at a probe rotation.
  {
    const auto [p, q] = pairs[order[0]];
    const Matrix<T> u = spinorRotationMatrix(generatorMatrix<T>(n, p, q, 0.3, 0.15));
    if constexpr (std::is_same_v<Eri, Tensor4<T>>) {
      // Dense: the exact O(n^5) leg transform against the Cholesky-based reference rotation.
      const auto exact = rotateIntegralsExact(h, eri, u);
      const auto ref = rotateIntegrals(h, eri, u);
      double dh = 0.0, de = 0.0;
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) dh = std::max(dh, std::abs(std::complex<double>(exact.h(a, b) - ref.h(a, b))));
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b)
          for (std::size_t c = 0; c < n; ++c)
            for (std::size_t d = 0; d < n; ++d)
              de = std::max(de, std::abs(std::complex<double>(exact.eri(a, b, c, d) - ref.eri(a, b, c, d))));
      log << "    exact integral rotation vs Cholesky rotation: max |dh| = " << dh << ", max |deri| = " << de << "\n";
      verdict(dh < 1e-8 && de < 1e-7, "exact O(n^5) integral rotation reproduces the reference rotation");
    } else {
      // Cholesky vectors: rotating the vectors must equal the exact dense rotation of the
      // integrals they represent.
      const Tensor4<T> dense_start = denseOf(eri);
      const Tensor4<T> exact = twoElectronRotated(dense_start, u);
      const Eri rotated = rotateEri(eri, u);
      double de = 0.0;
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b)
          for (std::size_t c = 0; c < n; ++c)
            for (std::size_t d = 0; d < n; ++d)
              de = std::max(de, std::abs(std::complex<double>(exact(a, b, c, d) - rotated(a, b, c, d))));
      log << "    Cholesky-vector rotation (" << eri.nVectors() << " vectors, O(Nchol n^3)) vs exact dense rotation: max |deri| = " << de << "\n";
      verdict(de < 1e-9, "rotating the Cholesky vectors reproduces the exact rotation of the integrals");
    }
  }

  // (2) orbital gradient vs finite differences of the energy (t, and y for complex).
  if (g_max < 1e-7) {
    log << "    (gradient ~ 0 at the start: finite-difference test not informative, skipped)\n";
  } else {
    constexpr double kStep = 1e-4;
    double worst = 0.0;
    for (std::size_t k = 0; k < n_test; ++k) {
      const auto [p, q] = pairs[order[k]];
      auto energy_at = [&](double t, double y) {
        const Matrix<T> u = spinorRotationMatrix(generatorMatrix<T>(n, p, q, t, y));
        return model.energy(oneElectronRotated(h, u), rotateEri(eri, u), occ);
      };
      const double dt = (energy_at(kStep, 0.0) - energy_at(-kStep, 0.0)) / (2.0 * kStep);
      const std::complex<double> g(g0(p, q));
      worst = std::max(worst, std::abs(g.real() - dt));
      if constexpr (!std::is_same_v<T, double>) {
        const double dy = (energy_at(0.0, kStep) - energy_at(0.0, -kStep)) / (2.0 * kStep);
        worst = std::max(worst, std::abs(g.imag() - dy));
      }
    }
    log << "    gradient (dE/dt" << (std::is_same_v<T, double> ? "" : " + i dE/dy")
        << ") vs central finite difference of the energy at the " << n_test
        << " largest-gradient pairs: max |diff| = " << worst << "\n";
    verdict(worst < 1e-6, "orbital gradient matches the energy finite difference");
  }

  // (2a) NEO's Hessian-diagonal preconditioner against finite differences of the (local) gradient at
  // the same pairs: H_II = d g_I / d t_I (and, for complex spinors, H_(y_I) = d g_(y_I) / d y_I) via
  // a central difference of the gradient at the rotated integrals -- the definition NeoProblem's
  // hessianVector documents -- and against the Hessian-vector product's own I-th entry for the unit
  // vector e_I (must agree to roundoff: the diagonal is the diagonal of that very operator).
  if (model.hessian_diagonal_dense && model.hessian_vector_dense) {
    // Dense-tensor variants (identical computation; a Cholesky model's own would pay O(n_chol) per
    // element access). The Hessian-vector comparison costs one product per tested direction, so it
    // is skipped for very large rotation spaces.
    const auto& eri_dense = denseOf(eri);
    const bool compare_hv = pairs.size() <= 4000;
    const std::vector<double> diag = model.hessian_diagonal_dense(h, eri_dense, occ);
    constexpr double kStepD = 1e-4;
    double worst_fd = 0.0, worst_hv = 0.0, scale = 0.0;
    const std::size_t n_pairs_all = pairs.size();
    for (std::size_t k = 0; k < n_test; ++k) {
      const std::size_t idx = order[k];
      const auto [p, q] = pairs[idx];
      auto gradient_at = [&](double t, double y) {
        const Matrix<T> u = spinorRotationMatrix(generatorMatrix<T>(n, p, q, t, y));
        return jointOrbitalGradient(model.gradient(oneElectronRotated(h, u), rotateEri(eri, u), occ), pairs);
      };
      const auto gt_plus = gradient_at(kStepD, 0.0), gt_minus = gradient_at(-kStepD, 0.0);
      const double fd_t = (gt_plus[idx] - gt_minus[idx]) / (2.0 * kStepD);
      std::vector<double> e(diag.size(), 0.0);
      e[idx] = 1.0;
      worst_fd = std::max(worst_fd, std::abs(diag[idx] - fd_t));
      if (compare_hv) worst_hv = std::max(worst_hv, std::abs(diag[idx] - model.hessian_vector_dense(h, eri_dense, occ, e)[idx]));
      scale = std::max(scale, std::abs(diag[idx]));
      if constexpr (!std::is_same_v<T, double>) {
        const auto gy_plus = gradient_at(0.0, kStepD), gy_minus = gradient_at(0.0, -kStepD);
        const double fd_y = (gy_plus[n_pairs_all + idx] - gy_minus[n_pairs_all + idx]) / (2.0 * kStepD);
        std::vector<double> ey(diag.size(), 0.0);
        ey[n_pairs_all + idx] = 1.0;
        worst_fd = std::max(worst_fd, std::abs(diag[n_pairs_all + idx] - fd_y));
        if (compare_hv) {
          worst_hv = std::max(worst_hv, std::abs(diag[n_pairs_all + idx] -
                                                 model.hessian_vector_dense(h, eri_dense, occ, ey)[n_pairs_all + idx]));
        }
        scale = std::max(scale, std::abs(diag[n_pairs_all + idx]));
      }
    }
    log << "    Hessian diagonal (NEO preconditioner) at the " << n_test << " largest-gradient pairs: max |diag - central"
        << " finite difference of the gradient| = " << worst_fd;
    if (compare_hv) log << ", max |diag - (H e_I)_I| = " << worst_hv;
    log << " (largest |diag| tested " << scale << ")\n";
    verdict(worst_hv < 1e-8 * std::max(1.0, scale) && worst_fd < 1e-5 * std::max(1.0, scale),
            "Hessian diagonal matches the Hessian-vector product and the gradient finite difference");
  }

  // (2b) PNOF only: the orbital gradient against a finite difference of the INDEPENDENT
  // occupation-side energy (pnofElectronicEnergy, the pair-symmetric shortcut that the occupation
  // optimizer minimizes and that equals the exact pair-CI energy for two electrons), along a
  // direction that keeps the Kramers / spin-pair symmetry that shortcut assumes (a rotation of a
  // single pair would break it). The check (2) above differentiates the full two-RDM energy the
  // gradient itself is built from, so it cannot see an error common to both.
  if (model.symmetric_shortcut_energy && g_max >= 1e-7) {
    std::vector<double> dir(joint_gradient.size(), 0.0);
    bool have_dir = false;
    if constexpr (!std::is_same_v<T, double>) {
      if (kramers && n % 2 == 0) {
        const KramersRestriction kr(n, pairs);
        const std::vector<double> reduced_grad = kr.contractRepresentative(joint_gradient);
        std::size_t best = 0;
        for (std::size_t j = 0; j < reduced_grad.size(); ++j)
          if (std::abs(reduced_grad[j]) > std::abs(reduced_grad[best])) best = j;
        std::vector<double> e(kr.reducedSize(), 0.0);
        e[best] = 1.0;
        dir = kr.expandRepresentative(e);
        have_dir = true;
      }
    } else {
      if (!spin_partner.empty()) {
        std::vector<long> index(n * n, -1);
        for (std::size_t i = 0; i < pairs.size(); ++i) index[pairs[i].first * n + pairs[i].second] = static_cast<long>(i);
        std::size_t best = pairs.size();
        double best_mag = 0.0;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
          const std::size_t pp = spin_partner[pairs[i].first], qq = spin_partner[pairs[i].second];
          if (pp > qq && std::abs(joint_gradient[i]) > best_mag) { best = i; best_mag = std::abs(joint_gradient[i]); }
        }
        if (best < pairs.size()) {
          const long twin = index[spin_partner[pairs[best].first] * n + spin_partner[pairs[best].second]];
          dir[best] = 1.0;
          if (twin >= 0) dir[static_cast<std::size_t>(twin)] = 1.0;
          have_dir = true;
        }
      }
    }
    if (have_dir) {
      double analytic = 0.0;
      for (std::size_t i = 0; i < dir.size(); ++i) analytic += joint_gradient[i] * dir[i];
      auto shortcut_at = [&](double scale) {
        Matrix<T> kappa(n, n, T{});
        for (std::size_t i = 0; i < pairs.size(); ++i) {
          const auto [p, q] = pairs[i];
          if constexpr (std::is_same_v<T, double>) {
            kappa(p, q) = scale * dir[i];
            kappa(q, p) = -scale * dir[i];
          } else {
            const std::complex<double> z(scale * dir[i], scale * dir[pairs.size() + i]);
            kappa(p, q) = z;
            kappa(q, p) = -std::conj(z);
          }
        }
        const Matrix<T> u = spinorRotationMatrix(kappa);
        return model.symmetric_shortcut_energy(oneElectronRotated(h, u), rotateEri(eri, u), occ);
      };
      constexpr double kStep2 = 1e-4;
      const double fd = (shortcut_at(kStep2) - shortcut_at(-kStep2)) / (2.0 * kStep2);
      log << "    directional derivative along a symmetry-preserving rotation: analytic gradient " << analytic
          << " vs finite difference of the independent pair-symmetric energy " << fd << " (|diff| = "
          << std::abs(analytic - fd) << ")\n";
      verdict(std::abs(analytic - fd) < 1e-6 * std::max(1.0, std::abs(analytic)),
              "orbital gradient matches the finite difference of the independent occupation-side energy");
    }
  }

  // (3) Kramers-restriction machinery (complex spinors only).
  if constexpr (!std::is_same_v<T, double>) {
    if (kramers) {
      if (n % 2 != 0) {
        verdict(false, "even number of spinors for the Kramers pairing (2k, 2k+1)");
      } else {
        const KramersRestriction kr(n, pairs);
        // The STARTING integrals must already have the Kramers structure (Theta|2k> = |2k+1>
        // for every spinor): a Kramers-restricted rotation preserves it but cannot create it.
        // (Near-degenerate Kramers pairs, e.g. a spin-orbit-split p shell of a stretched
        // molecule, can leave the SCF eigenvectors slightly mixed across pairs.)
        {
          const double h_dev = timeReversalDeviation(h);
          double h_scale = 0.0;
          for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q) h_scale = std::max(h_scale, std::abs(std::complex<double>(h(p, q))));
          const auto [eri_dev, eri_scale] = twoBodyTimeReversalDeviation(denseOf(eri));
          log << "    starting integrals: max |h(P p,P q) - s_p s_q conj h(p,q)| = " << h_dev << " (max |h| = " << h_scale
              << "), max |<Pa Pb|Pc Pd> - s_a s_b s_c s_d conj <ab|cd>| = " << eri_dev << " (max |eri| = " << eri_scale << ")\n";
          verdict(h_dev <= 1e-8 * std::max(1.0, h_scale), "starting one-electron integrals have the Kramers-pair structure");
          verdict(eri_dev <= 1e-8 * std::max(1.0, eri_scale), "starting two-electron integrals have the Kramers-pair structure");
        }
        const double occ_dev = kramersOccupationDeviation(occ);
        log << "    Kramers restriction: " << kr.nOrbits() << " orbit(s) (" << kr.reducedSize()
            << " of " << kr.fullSize() << " real parameters), max |n_a - n_abar| = " << occ_dev << "\n";
        verdict(occ_dev < 1e-6, "occupation numbers equal within every Kramers pair");
        const double asym = kr.asymmetry(joint_gradient);
        log << "    gradient time-reversal asymmetry max |nu - R nu| = " << asym << " (max |nu| = " << g_max << ")\n";
        verdict(asym <= 1e-6 * std::max(g_max, 1e-10) + 1e-12,
                "orbital gradient is time-reversal symmetric (R nu = nu)");
        // Probe rotation from a random-looking reduced step: exp(-kappa) must commute with time reversal.
        std::vector<double> probe(kr.reducedSize());
        for (std::size_t i = 0; i < probe.size(); ++i) probe[i] = 0.05 * std::sin(1.7 * static_cast<double>(i) + 0.3);
        const auto x = kr.expandRepresentative(probe);
        Matrix<T> kappa(n, n, T{});
        for (std::size_t i = 0; i < pairs.size(); ++i) {
          const std::complex<double> z(x[i], x[pairs.size() + i]);
          kappa(pairs[i].first, pairs[i].second) = z;
          kappa(pairs[i].second, pairs[i].first) = -std::conj(z);
        }
        const double dev = timeReversalDeviation(spinorRotationMatrix(kappa));
        log << "    probe Kramers-restricted rotation: time-reversal deviation of exp(-kappa) = " << dev << "\n";
        verdict(dev < 1e-10, "Kramers-restricted rotations preserve the Kramers pairing");
      }
    }
  }
  if constexpr (std::is_same_v<T, double>) {
    if (!spin_partner.empty()) {
      double h_dev = 0.0, e_dev = 0.0, h_scale = 0.0, e_scale = 0.0;
      const auto& eri_view = denseOf(eri);
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          h_scale = std::max(h_scale, std::abs(h(p, q)));
          h_dev = std::max(h_dev, std::abs(h(spin_partner[p], spin_partner[q]) - h(p, q)));
        }
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b)
          for (std::size_t c = 0; c < n; ++c)
            for (std::size_t d = 0; d < n; ++d) {
              e_scale = std::max(e_scale, std::abs(eri_view(a, b, c, d)));
              e_dev = std::max(e_dev, std::abs(eri_view(spin_partner[a], spin_partner[b], spin_partner[c], spin_partner[d]) - eri_view(a, b, c, d)));
            }
      log << "    starting integrals, alpha/beta symmetry: max |h(alpha)-h(beta)| = " << h_dev << ", max |eri(alpha..)-eri(beta..)| = " << e_dev << "\n";
      verdict(h_dev <= 1e-8 * std::max(1.0, h_scale), "starting one-electron integrals are spin-restricted (alpha = beta)");
      verdict(e_dev <= 1e-8 * std::max(1.0, e_scale), "starting two-electron integrals are spin-restricted (alpha = beta)");
    }
  }
  log << std::defaultfloat << std::setprecision(6);
  return ok;
}

}  // namespace

// =====================================================================
// the macro loop
// =====================================================================

template <typename T, typename Eri>
FullOptResult runFullOptimization(const Matrix<T>& h, const Eri& eri,
                                  const std::vector<double>& occupations,
                                  const std::vector<double>& state, const RdmftModel<T, Eri>& model,
                                  const FullOptSettings& settings, bool kramers_restricted,
                                  double nuclear_repulsion_energy, std::ostream& log,
                                  const std::vector<std::size_t>& spin_partner,
                                  std::size_t n_negative) {
  FullOptResult result;
  result.occupations = occupations;
  log << "\n  FULL orbital + occupation optimization ("
      << (settings.orbital_optimizer == OrbitalOptimizer::kNeo ? "NEO" : "ADAM")
      << " orbital rotations at fixed occupations, then\n"
         "  occupation re-optimization at the new orbitals, macro-iterated to convergence; energy tolerance "
      << settings.energy_tolerance << ", orbital-gradient tolerance " << settings.gradient_tolerance << ", at most "
      << settings.max_macro_iterations << " macro-iterations";
  if (kramers_restricted) log << "; orbital rotations Kramers-restricted (Utils/KramersRestriction.h)";
  if (!spin_partner.empty()) log << "; spin-restricted orbital rotations (alpha and beta rotate identically)";
  if (n_negative > 0) {
    log << "; rotations excluded from the " << n_negative
        << " negative-energy (Dirac sea) spinors (no-pair approximation, Talman 1986/Saue"
           " ChemPhysChem 2011's min-max characterization -- see project notes)";
  }
  log << "):\n";

  if constexpr (!std::is_same_v<Eri, Tensor4<T>>) {
    log << "  Two-electron integrals held as " << eri.nVectors()
        << " Cholesky vectors (threshold " << settings.cholesky_threshold
        << "): orbital rotations act on the vectors (O(Nchol n^3) instead of O(n^5)), elements are\n"
           "  evaluated on demand, and no dense n^4 tensor is kept in the loop (start-up and final\n"
           "  Kramers/spin-structure tests assemble one transiently).\n";
  }
  log << "  a) Validation of the ADAM / Kramers-restriction machinery on this system:\n";
  result.checks_passed =
      runChecks<T, Eri>(h, eri, occupations, model, kramers_restricted, spin_partner, log, n_negative);
  if (!result.checks_passed) {
    log << "  A validation check FAILED -- the macro-iteration loop is NOT run.\n";
    return result;
  }
  log << "  All validation checks passed.\n";

  log << "  b) Macro-iteration loop:\n";
  RotationProblem<T, Eri> problem(model, h, eri, spin_partner, n_negative);
  AdamOptions adam_options;
  adam_options.gradient_tolerance = settings.gradient_tolerance;
  adam_options.energy_tolerance = settings.energy_tolerance;
  AdamOptimizer<T> adam(adam_options);
  std::unique_ptr<KramersRestriction> kr;
  if constexpr (!std::is_same_v<T, double>) {
    if (kramers_restricted) kr = std::make_unique<KramersRestriction>(h.rows(), problem.pairs());
  }
  // NON_REL's NEO analogue of `kr`: ties every pure-alpha pair to its pure-beta twin (see
  // SpinRestrictedNeoProblem's own comment on why this is needed, unlike ADAM's averaging).
  std::vector<std::pair<std::size_t, std::size_t>> spin_orbits;
  if constexpr (std::is_same_v<T, double>) {
    if (!spin_partner.empty()) spin_orbits = spinOrbits(problem.pairs(), spin_partner);
  }

  // NEO branch: needs the model's hessian_vector (set for both Tensor4 and CholeskyEri models --
  // see RdmftModel's own note) AND ORBITAL_OPTIMIZER NEO; falls back to ADAM (with a printed
  // note) if hessian_vector is somehow unset (defensive -- every model built by makeJkOnlyModel/
  // makePnofModel sets it).
  bool use_neo = false;
  std::unique_ptr<NeoOrbitalProblem<T, Eri>> neo_problem;
  if (settings.orbital_optimizer == OrbitalOptimizer::kNeo && model.hessian_vector) {
    neo_problem = std::make_unique<NeoOrbitalProblem<T, Eri>>(model, problem);
    use_neo = true;
  }
  if (settings.orbital_optimizer == OrbitalOptimizer::kNeo && !use_neo) {
    log << "    ORBITAL_OPTIMIZER NEO requested but the model has no Hessian-vector callback: using ADAM.\n";
  }
  NeoOptions neo_options;
  // Inexact Newton: the Davidson solve for each step only needs a residual of ~0.1*|g| (the
  // library default 1e-6 is an absolute cap, which over-solves every step while |g| >> 1e-5 -- on
  // CO/cc-pVDZ (GNOF, NON_REL) the residual stagnated near 2e-5 for hundreds of Hessian products).
  // Tolerance = min(residual_tolerance, 0.1*|g|), so it still tightens as the gradient falls.
  neo_options.step.residual_tolerance = 1e-3;
  neo_options.step.target_order = 0;  // ground state (a minimum) only -- no saddle-point search here.
  neo_options.gradient_tolerance = settings.gradient_tolerance;
  neo_options.max_iterations = settings.neo_max_iterations;  // fixed budget -- see its own comment.

  std::vector<double> occ = occupations;
  std::vector<double> occ_state = state;
  double e_elec = model.energy(h, eri, occ);
  double e_old = e_elec;
  const double e_start = e_elec;
  log << "    starting point (occupation-optimized HF orbitals): total energy " << std::setprecision(10)
      << e_elec + nuclear_repulsion_energy << std::setprecision(6) << " Hartree\n";
  log << (use_neo ? "    iter      total energy (Ha)          dE        max|grad|  NEO steps    trust radius\n"
                  : "    iter      total energy (Ha)          dE        max|grad|  ADAM steps  learning rate\n");

  // NEO only, when target_order = 0 (ground state) was requested: is the point the macro loop
  // stopped at actually a MINIMUM of the orbital-rotation energy, within the same symmetry-
  // restricted subspace NEO searched (Kramers for X2C, spin-restricted for NON_REL, unrestricted
  // otherwise)? A stationary point with zero gradient can still be a saddle -- NEO's own stopping
  // rule is gradient-only, and a shallow negative-curvature direction can become numerically
  // decoupled (doc/NEO.tex Sec. 3.2: |z0| < decoupled_z0*||g||) before it has been descended, so
  // NEO can report "converged" at a genuine saddle (confirmed on lih_pnof5/7/7s NON_REL). When
  // that happens here, escape along the offending eigenvector -- whichever sign lowers the trial
  // energy -- and re-enter the macro loop from the perturbed point, up to a small retry cap.
  auto withReduced = [&](auto&& fn) {
    if constexpr (!std::is_same_v<T, double>) {
      if (kr) {
        KramersNeoProblem reduced(*neo_problem, *kr);
        fn(static_cast<NeoProblem<double>&>(reduced));
        return;
      }
    } else {
      if (!spin_orbits.empty()) {
        SpinRestrictedNeoProblem reduced(*neo_problem, spin_orbits);
        fn(static_cast<NeoProblem<double>&>(reduced));
        return;
      }
    }
    fn(static_cast<NeoProblem<double>&>(*neo_problem));
  };
  constexpr int kMaxSaddleEscapeAttempts = 3;

  int iter = 0;
  int n_occ_unconverged = 0;
  int total_macro_iterations = 0;
  Matrix<T> g_final(h.rows(), h.rows(), T{});
  double g_max = 0.0;
  for (int escape_attempt = 0;; ++escape_attempt) {
    result.converged = false;
    for (iter = 1; iter <= settings.max_macro_iterations; ++iter) {
      problem.setOccupations(occ);
      double max_gradient = 0.0;
      int orbital_iterations = 0;
      bool gradient_converged = false, orbital_restart_requested = false;
      double log_extra = 0.0;  // ADAM's learning rate, or NEO's final trust radius.
      if (use_neo) {
        neo_problem->setOccupations(occ);
        NeoResult nr;
        if constexpr (!std::is_same_v<T, double>) {
          if (kr) {
            KramersNeoProblem reduced(*neo_problem, *kr);
            nr = neoOptimize(reduced, neo_options);
          } else {
            nr = neoOptimize(*neo_problem, neo_options);
          }
        } else {
          if (!spin_orbits.empty()) {
            SpinRestrictedNeoProblem reduced(*neo_problem, spin_orbits);
            nr = neoOptimize(reduced, neo_options);
          } else {
            nr = neoOptimize(*neo_problem, neo_options);
          }
        }
        max_gradient = nr.gradient_max;
        orbital_iterations = nr.iterations;
        gradient_converged = nr.converged;
        log_extra = nr.history.empty() ? 0.0 : nr.history.back().radius;
      } else {
        AdamResult ar;
        if constexpr (!std::is_same_v<T, double>) {
          if (kr) {
            KramersAdamProblem reduced(problem, *kr);
            ar = adam.run(reduced);
          } else {
            ar = adam.run(problem);
          }
        } else {
          ar = adam.run(problem);
        }
        max_gradient = ar.max_gradient;
        orbital_iterations = ar.iterations;
        gradient_converged = ar.gradient_converged;
        orbital_restart_requested = ar.restart_requested;
        log_extra = ar.learning_rate;
      }

      RdmftOccupationResult occ_result;
      try {
        occ_result = model.optimize_occupations(problem.h(), problem.eri(), occ_state);
      } catch (const std::exception& e) {
        log << "    occupation re-optimization FAILED (" << e.what() << ") -- stopping.\n";
        break;
      }
      if (!occ_result.converged) ++n_occ_unconverged;
      occ = occ_result.occupations;
      // Same energy definition as the orbital stage (see makePnofModel: the
      // optimizer's own value can differ for a non-symmetric eri).
      e_elec = model.energy(problem.h(), problem.eri(), occ);
      const double d_e = e_elec - e_old;
      log << "    " << std::setw(4) << iter << "  " << std::fixed << std::setprecision(10) << std::setw(20)
          << e_elec + nuclear_repulsion_energy << "  " << std::scientific << std::setprecision(2) << std::setw(10)
          << d_e << "  " << std::setw(10) << max_gradient << "  " << std::defaultfloat << std::setw(6)
          << orbital_iterations << "      " << std::scientific << std::setprecision(2) << log_extra
          << (orbital_restart_requested ? "  restart" : "") << (gradient_converged ? "  gradient-converged" : "")
          << (occ_result.converged ? "" : "  occupations-not-converged")
          << std::defaultfloat << std::setprecision(6) << "\n";
      if (std::abs(d_e) < settings.energy_tolerance && !(use_neo ? false : adam.restartRequested())) {
        result.converged = true;
        break;
      }
      e_old = e_elec;
    }
    total_macro_iterations += std::min(iter, settings.max_macro_iterations);
    result.occupations = occ;
    result.occupation_state = occ_state;
    {
      const Matrix<T>& u_final = problem.totalRotation();
      result.total_rotation = Matrix<std::complex<double>>(u_final.rows(), u_final.cols());
      for (std::size_t i = 0; i < u_final.rows() * u_final.cols(); ++i) {
        result.total_rotation.data()[i] = std::complex<double>(u_final.data()[i]);
      }
    }
    result.electronic_energy = e_elec;

    g_final = model.gradient(problem.h(), problem.eri(), occ);
    g_max = 0.0;
    for (const auto& [p, q] : problem.pairs()) g_max = std::max(g_max, std::abs(std::complex<double>(g_final(p, q))));
    result.gradient_max = g_max;

    if (!use_neo) break;  // ADAM has no Hessian to check and nothing to escape.

    neo_problem->setOccupations(occ);
    NeoEigenOptions eig_options;
    constexpr std::size_t kRoots = 3;
    bool escaping = false;
    withReduced([&](NeoProblem<double>& reduced) {
      const std::size_t n_roots = std::min<std::size_t>(kRoots, reduced.dimension());
      const NeoEigenResult<double> eig = neoLowestHessianEigenpairs<double>(
          reduced.dimension(), [&](const std::vector<double>& v) { return reduced.hessianVector(v); }, n_roots, {},
          eig_options);
      const int n_negative = static_cast<int>(
          std::count_if(eig.eigenvalues.begin(), eig.eigenvalues.end(), [](double e) { return e < -1e-6; }));
      log << "    Hessian check (lowest " << eig.eigenvalues.size() << " eigenvalue(s) of the "
          << (kr ? "Kramers-restricted" : (!spin_orbits.empty() ? "spin-restricted" : "unrestricted"))
          << " orbital-rotation Hessian at this point):";
      for (double e : eig.eigenvalues) log << " " << std::scientific << std::setprecision(3) << e;
      log << std::defaultfloat << std::setprecision(6) << (eig.converged ? "" : "  (Davidson NOT converged)") << "\n";
      if (n_negative > 0 && !eig.eigenvectors.empty() && escape_attempt < kMaxSaddleEscapeAttempts) {
        const auto& v = eig.eigenvectors.front();  // most negative eigenvalue: eigenvalues is ascending
        constexpr double kEscapeStep = 0.3;
        std::vector<double> step_plus(v.size()), step_minus(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
          step_plus[i] = kEscapeStep * v[i];
          step_minus[i] = -kEscapeStep * v[i];
        }
        const double e0 = reduced.energy();
        const double e_plus = reduced.trialEnergy(step_plus);
        const double e_minus = reduced.trialEnergy(step_minus);
        log << "    Escaping the saddle (eigenvalue " << std::scientific << std::setprecision(3)
            << eig.eigenvalues.front() << std::defaultfloat << std::setprecision(6)
            << "): perturbing along its eigenvector, sign chosen by trial energy (" << std::setprecision(10)
            << e_plus << " vs " << e_minus << ", from " << e0 << std::setprecision(6) << "), attempt "
            << (escape_attempt + 1) << "/" << kMaxSaddleEscapeAttempts << "; re-optimizing.\n";
        reduced.accept(e_plus <= e_minus ? step_plus : step_minus);
        escaping = true;
      } else {
        log << "    [" << (n_negative == 0 ? "PASS" : "FAIL")
            << "] the point is a genuine minimum (0 negative eigenvalues)\n";
      }
    });
    if (!escaping) break;
    e_elec = model.energy(problem.h(), problem.eri(), occ);
    e_old = e_elec;
  }
  result.iterations = total_macro_iterations;

  log << "  " << (result.converged ? "CONVERGED" : "NOT converged") << " after " << result.iterations
      << " macro-iteration(s).\n";
  log << "    Final total energy: " << std::setprecision(10) << e_elec + nuclear_repulsion_energy << std::setprecision(6)
      << " Hartree (" << std::scientific << std::setprecision(3) << e_elec - e_start << std::defaultfloat
      << std::setprecision(6) << " Hartree relative to the occupation-only optimum)\n";
  if (model.print_occupations) {
    log << "  Occupation numbers after the macro-iteration loop:\n";
    model.print_occupations(occ, log);
  }
  if (n_occ_unconverged > 0) {
    log << "    Note: the occupation optimizer reported no convergence in " << n_occ_unconverged
        << " of the macro-iterations (its own iteration limit).\n";
  }
  log << "    Final max |orbital gradient entry|: " << std::scientific << std::setprecision(3) << g_max
      << std::defaultfloat << std::setprecision(6) << "\n";
  // Final test: are the optimized orbitals still Kramers pairs (and is the PNOF
  // pair-symmetric energy shortcut still valid)?
  bool final_ok = true;
  auto final_verdict = [&](bool pass, const std::string& what) {
    log << "    [" << (pass ? "PASS" : "FAIL") << "] " << what << "\n";
    final_ok = final_ok && pass;
  };
  log << "  c) Final test of the optimized orbitals:\n";
  log << std::scientific << std::setprecision(2);
  if constexpr (!std::is_same_v<T, double>) {
    if (kr) {
      const double u_dev = timeReversalDeviation(problem.totalRotation());
      const double h_dev = timeReversalDeviation(problem.h());
      double h_scale = 0.0;
      for (std::size_t p = 0; p < problem.h().rows(); ++p)
        for (std::size_t q = 0; q < problem.h().cols(); ++q)
          h_scale = std::max(h_scale, std::abs(std::complex<double>(problem.h()(p, q))));
      const auto [eri_dev, eri_scale] = twoBodyTimeReversalDeviation(denseOf(problem.eri()));
      const double occ_dev = kramersOccupationDeviation(occ);
      const auto joint_final = jointOrbitalGradient(g_final, problem.pairs());
      double nu_scale = 0.0;
      for (double v : joint_final) nu_scale = std::max(nu_scale, std::abs(v));
      const double nu_dev = kr->asymmetry(joint_final);
      log << "    total rotation U: max |U(P p,P q) - s_p s_q conj U(p,q)| = " << u_dev << "\n";
      log << "    h in the final basis: max |h(P p,P q) - s_p s_q conj h(p,q)| = " << h_dev << " (max |h| = " << h_scale << ")\n";
      log << "    eri in the final basis: max |<Pa Pb|Pc Pd> - s_a s_b s_c s_d conj <ab|cd>| = " << eri_dev << " (max |eri| = " << eri_scale << ")\n";
      log << "    occupations: max |n_a - n_abar| = " << occ_dev << "\n";
      log << "    final orbital gradient: max |nu - R nu| = " << nu_dev << " (max |nu| = " << nu_scale << ")\n";
      final_verdict(u_dev < 1e-8, "total orbital rotation commutes with time reversal (Theta|2k> = |2k+1> preserved)");
      final_verdict(h_dev <= 1e-8 * std::max(1.0, h_scale), "one-electron integrals keep the Kramers-pair structure");
      final_verdict(eri_dev <= 1e-8 * std::max(1.0, eri_scale), "two-electron integrals keep the Kramers-pair structure");
      final_verdict(occ_dev < 1e-6, "occupation numbers equal within every Kramers pair");
      // The 1e-12 floor alone assumes roundoff scales with the GRADIENT's own magnitude
      // (nu_scale); for C4_DHF the gradient/occupation-optimization sums also touch h/eri
      // entries up to ~h_scale/eri_scale in magnitude (the negative-energy branch, order
      // -2mc^2 ~ 1e4 Hartree, always multiplied by an exactly-zero occupation coefficient but
      // still summed over), so roundoff can reach ~machine-epsilon*h_scale even though the
      // PHYSICAL asymmetry is zero -- confirmed directly (JK_only/MULLER, water/STO-3G C4_DHF:
      // nu_dev 7.35e-12 with h_scale 3.76e4, comfortably explained by ~100*eps*h_scale but far
      // above the un-scaled 1e-12 floor). X2C's own h_scale is chemical-size, so this term stays
      // well below 1e-12 there and does not loosen that check.
      final_verdict(nu_dev <= 1e-6 * std::max(nu_scale, 1e-10) + 1e-12 +
                                  100.0 * std::numeric_limits<double>::epsilon() *
                                      std::max(h_scale, eri_scale),
                    "final orbital gradient is time-reversal symmetric");
    } else {
      log << "    (Kramers pairing test not applicable: no Kramers restriction requested)\n";
    }
  } else if (!spin_partner.empty()) {
    const std::size_t n_so = problem.h().rows();
    const auto& eri_final = denseOf(problem.eri());
    double h_dev = 0.0, e_dev = 0.0, h_scale = 0.0, e_scale = 0.0;
    for (std::size_t p = 0; p < n_so; ++p)
      for (std::size_t q = 0; q < n_so; ++q) {
        h_scale = std::max(h_scale, std::abs(problem.h()(p, q)));
        h_dev = std::max(h_dev, std::abs(problem.h()(spin_partner[p], spin_partner[q]) - problem.h()(p, q)));
      }
    for (std::size_t a = 0; a < n_so; ++a)
      for (std::size_t b = 0; b < n_so; ++b)
        for (std::size_t c = 0; c < n_so; ++c)
          for (std::size_t d = 0; d < n_so; ++d) {
            e_scale = std::max(e_scale, std::abs(eri_final(a, b, c, d)));
            e_dev = std::max(e_dev, std::abs(eri_final(spin_partner[a], spin_partner[b], spin_partner[c], spin_partner[d]) -
                                              eri_final(a, b, c, d)));
          }
    log << "    spin symmetry of the final basis: max |h(alpha)-h(beta)| = " << h_dev << " (max |h| = " << h_scale
        << "), max |eri(alpha..)-eri(beta..)| = " << e_dev << " (max |eri| = " << e_scale << ")\n";
    final_verdict(h_dev <= 1e-8 * std::max(1.0, h_scale), "one-electron integrals keep alpha/beta (spin-restricted) symmetry");
    final_verdict(e_dev <= 1e-8 * std::max(1.0, e_scale), "two-electron integrals keep alpha/beta (spin-restricted) symmetry");
  } else {
    log << "    (Kramers pairing test not applicable to real NON_REL spin-orbitals)\n";
  }
  if (model.symmetric_shortcut_energy) {
    const double e_full = model.energy(problem.h(), problem.eri(), occ);
    const double e_sym = model.symmetric_shortcut_energy(problem.h(), problem.eri(), occ);
    log << "    energy, full two-RDM vs pair-symmetric shortcut (pnofElectronicEnergy): " << std::setprecision(10)
        << std::defaultfloat << e_full << " vs " << e_sym << std::scientific << std::setprecision(2)
        << "  (|diff| = " << std::abs(e_full - e_sym) << ")\n";
    final_verdict(std::abs(e_full - e_sym) < 1e-8, "integrals still have the Kramers/spin-pair symmetry the PNOF shortcut assumes");
  }
  log << "  Final test: " << (final_ok ? "all checks PASSED" : "a check FAILED -- inspect the values above") << "\n";
  log << std::defaultfloat << std::setprecision(6);
  return result;
}

// ---------------------------------------------------------------------
// Public entry points: dense tensor or Cholesky vectors (settings.cholesky)
// ---------------------------------------------------------------------

namespace {

// Decomposes the dense MO integrals into Cholesky vectors and verifies the reconstruction
// (all elements for small n, a strided sample otherwise). `n_negative` (C4_DHF only): occupations
// are EXACTLY 0 on the negative-energy branch (buildPnofFullTwoRdm/jkHartree|ExchangeCoupling both
// vanish whenever either index has occ 0), and n_negative>0 additionally excludes those indices
// from every orbital rotation (excludeNegative) -- so no energy/gradient/Hessian sum ever needs
// the TRUE VALUE of an eri element touching that branch, only that IT NOT POISON the decomposition
// of the (chemically relevant) positive-energy sub-block it's fetched alongside. Confirmed
// directly it does: on LiH/6-31G C4_DHF the negative branch's sheer magnitude (~-2mc^2, order
// 1e4 Hartree) made the pivoted decomposition of the FULL tensor reconstruct even the
// positive-only sub-block only to ~3.5e-05 at the default 1e-10 threshold (vs. ~1e-15 for
// NON_REL/X2C's own, single-scale tensors) -- so those entries are zeroed out (not merely
// excluded from the check) before decomposition when n_negative>0, eliminating the mixed-scale
// pivot competition entirely; the check below then confirms the positive-block reconstruction
// this actually achieves.
template <typename T>
CholeskyEri<T> makeCholeskyEri(const Tensor4<T>& eri, double threshold, std::ostream& log,
                               std::size_t n_negative = 0) {
  Tensor4<T> sanitized;
  const Tensor4<T>* decompose_from = &eri;
  if (n_negative > 0) {
    sanitized = eri;
    const std::size_t n_full = eri.dim0();
    for (std::size_t a = 0; a < n_full; ++a) {
      for (std::size_t b = 0; b < n_full; ++b) {
        for (std::size_t c = 0; c < n_full; ++c) {
          for (std::size_t d = 0; d < n_full; ++d) {
            if (a < n_negative || b < n_negative || c < n_negative || d < n_negative) {
              sanitized(a, b, c, d) = T{};
            }
          }
        }
      }
    }
    decompose_from = &sanitized;
  }
  const CholeskyEri<T> ch = CholeskyEri<T>::fromDense(*decompose_from, threshold);
  const std::size_t n = eri.dim0();
  const std::size_t n_check = n - n_negative;
  const std::size_t total = n_check * n_check * n_check * n_check;
  const std::size_t stride = total > 20000000 ? total / 20000000 + 1 : 1;
  double worst = 0.0;
  for (std::size_t flat = 0; flat < total; flat += stride) {
    const std::size_t d = n_negative + flat % n_check, c = n_negative + (flat / n_check) % n_check,
                       b = n_negative + (flat / (n_check * n_check)) % n_check,
                       a = n_negative + flat / (n_check * n_check * n_check);
    worst = std::max(worst, std::abs(std::complex<double>(ch(a, b, c, d) - eri(a, b, c, d))));
  }
  log << "  Cholesky decomposition of the MO integrals (Coulomb grouping, threshold " << threshold << "): "
      << ch.nVectors() << " vectors for n = " << n << " spinors (n^2 = " << n * n << ")";
  if (n_negative > 0) {
    log << "; reconstruction checked over the " << n_check
        << " positive-energy spinors only (the rest never enters an energy/gradient/Hessian sum"
           " with a nonzero coefficient)";
  }
  log << "; max |reconstruction - dense| = " << std::scientific << std::setprecision(2) << worst
      << std::defaultfloat << std::setprecision(6) << (stride > 1 ? " (sampled)" : "") << "\n";
  if (worst > 100.0 * threshold + 1e-9) {
    throw std::runtime_error("the Cholesky vectors do not reproduce the two-electron integrals");
  }
  return ch;
}

}  // namespace

template <typename T>
FullOptResult runFullOptimizationJk(const Matrix<T>& h, const Tensor4<T>& eri,
                                    const std::vector<double>& occupations,
                                    const std::vector<double>& state, JkFunctional functional,
                                    std::size_t f_l, double n_electrons, std::size_t n_total,
                                    std::size_t n_frozen, std::size_t n_inactive_below,
                                    std::size_t n_active,
                                    bool two_columns, const FullOptSettings& settings,
                                    bool kramers_restricted, double nuclear_repulsion_energy,
                                    std::ostream& log, const std::vector<std::size_t>& spin_partner,
                                    std::size_t n_negative) {
  if (settings.cholesky) {
    const CholeskyEri<T> ch = makeCholeskyEri(eri, settings.cholesky_threshold, log, n_negative);
    const auto model = makeJkOnlyModel<T, CholeskyEri<T>>(functional, f_l, n_electrons, n_total,
                                                          n_frozen, n_inactive_below, n_active,
                                                          two_columns, n_negative);
    return runFullOptimization<T, CholeskyEri<T>>(h, ch, occupations, state, model, settings,
                                                   kramers_restricted, nuclear_repulsion_energy, log,
                                                   spin_partner, n_negative);
  }
  const auto model = makeJkOnlyModel<T, Tensor4<T>>(functional, f_l, n_electrons, n_total,
                                                    n_frozen, n_inactive_below, n_active,
                                                    two_columns, n_negative);
  return runFullOptimization<T, Tensor4<T>>(h, eri, occupations, state, model, settings,
                                              kramers_restricted, nuclear_repulsion_energy, log,
                                              spin_partner, n_negative);
}

template <typename T>
FullOptResult runFullOptimizationPnof(const Matrix<T>& h, const Tensor4<T>& eri,
                                      const std::vector<double>& occupations,
                                      const std::vector<double>& state, PnofFunctional functional,
                                      const std::vector<PnofGeminal>& geminals, std::size_t n_core,
                                      int pnof_subspaces, int pnof_coupling, bool relativistic,
                                      bool sqp_occupations, std::size_t n_total,
                                      const FullOptSettings& settings, bool kramers_restricted,
                                      double nuclear_repulsion_energy, std::ostream& log,
                                      const std::vector<std::size_t>& spin_partner,
                                      std::size_t n_negative) {
  if (settings.cholesky) {
    const CholeskyEri<T> ch = makeCholeskyEri(eri, settings.cholesky_threshold, log, n_negative);
    const auto model = makePnofModel<T, CholeskyEri<T>>(functional, geminals, n_core, pnof_subspaces,
                                                        pnof_coupling, relativistic, sqp_occupations,
                                                        n_total, n_negative);
    return runFullOptimization<T, CholeskyEri<T>>(h, ch, occupations, state, model, settings,
                                                   kramers_restricted, nuclear_repulsion_energy, log,
                                                   spin_partner, n_negative);
  }
  const auto model = makePnofModel<T, Tensor4<T>>(functional, geminals, n_core, pnof_subspaces,
                                                  pnof_coupling, relativistic, sqp_occupations, n_total,
                                                  n_negative);
  return runFullOptimization<T, Tensor4<T>>(h, eri, occupations, state, model, settings,
                                             kramers_restricted, nuclear_repulsion_energy, log,
                                             spin_partner, n_negative);
}

#define RERDMFT_INSTANTIATE_FULLOPT(T, ERI)                                                        \
  template RdmftModel<T, ERI> makeJkOnlyModel<T, ERI>(JkFunctional, std::size_t, double,           \
                                                      std::size_t, std::size_t, std::size_t,        \
                                                      std::size_t, bool, std::size_t);              \
  template RdmftModel<T, ERI> makePnofModel<T, ERI>(PnofFunctional, std::vector<PnofGeminal>,      \
                                                    std::size_t, int, int, bool, bool, std::size_t, \
                                                    std::size_t);                                   \
  template FullOptResult runFullOptimization<T, ERI>(                                              \
      const Matrix<T>&, const ERI&, const std::vector<double>&, const std::vector<double>&,        \
      const RdmftModel<T, ERI>&, const FullOptSettings&, bool, double, std::ostream&,              \
      const std::vector<std::size_t>&, std::size_t);

RERDMFT_INSTANTIATE_FULLOPT(double, Tensor4<double>)
RERDMFT_INSTANTIATE_FULLOPT(double, CholeskyEri<double>)
RERDMFT_INSTANTIATE_FULLOPT(std::complex<double>, Tensor4<std::complex<double>>)
RERDMFT_INSTANTIATE_FULLOPT(std::complex<double>, CholeskyEri<std::complex<double>>)
#undef RERDMFT_INSTANTIATE_FULLOPT

template FullOptResult runFullOptimizationJk<double>(
    const Matrix<double>&, const Tensor4<double>&, const std::vector<double>&,
    const std::vector<double>&, JkFunctional, std::size_t, double, std::size_t, std::size_t,
    std::size_t, std::size_t, bool, const FullOptSettings&, bool, double, std::ostream&,
    const std::vector<std::size_t>&, std::size_t);
template FullOptResult runFullOptimizationJk<std::complex<double>>(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const std::vector<double>&, JkFunctional, std::size_t, double,
    std::size_t, std::size_t, std::size_t, std::size_t, bool, const FullOptSettings&, bool, double,
    std::ostream&, const std::vector<std::size_t>&, std::size_t);
template FullOptResult runFullOptimizationPnof<double>(
    const Matrix<double>&, const Tensor4<double>&, const std::vector<double>&,
    const std::vector<double>&, PnofFunctional, const std::vector<PnofGeminal>&, std::size_t, int,
    int, bool, bool, std::size_t, const FullOptSettings&, bool, double, std::ostream&,
    const std::vector<std::size_t>&, std::size_t);
template FullOptResult runFullOptimizationPnof<std::complex<double>>(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const std::vector<double>&, PnofFunctional,
    const std::vector<PnofGeminal>&, std::size_t, int, int, bool, bool, std::size_t,
    const FullOptSettings&, bool, double, std::ostream&, const std::vector<std::size_t>&,
    std::size_t);

}  // namespace rerdmft
