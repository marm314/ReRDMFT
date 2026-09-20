#include "NEO.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "LinearAlgebra.h"
#include "Matrix.h"

namespace rerdmft {

namespace {

// ---- scalar helpers valid for both T=double and T=std::complex<double> ----

inline double conjugate(double x) { return x; }
inline std::complex<double> conjugate(const std::complex<double>& x) { return std::conj(x); }
inline double realPart(double x) { return x; }
inline double realPart(const std::complex<double>& x) { return x.real(); }

// sum_i conj(a_i) b_i over `len` elements.
template <typename T>
T dotc(const T* a, const T* b, std::size_t len) {
  T sum{};
  for (std::size_t i = 0; i < len; ++i) sum += conjugate(a[i]) * b[i];
  return sum;
}

template <typename T>
double norm2(const std::vector<T>& v) {
  double sum = 0.0;
  for (const T& x : v) sum += std::norm(std::complex<double>(x));
  return std::sqrt(sum);
}

template <typename T>
double maxAbs(const std::vector<T>& v) {
  double m = 0.0;
  for (const T& x : v) m = std::max(m, std::abs(x));
  return m;
}

// Modified Gram-Schmidt (two passes) of `v` against the orthonormal set
// `basis`, then normalization. Returns false (v unusable) if v is
// (numerically) inside span(basis).
template <typename T>
bool orthonormalize(const std::vector<std::vector<T>>& basis, std::vector<T>& v,
                    double linear_dependence) {
  const double before = norm2(v);
  if (before < 1e-14) return false;
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& b : basis) {
      const T c = dotc(b.data(), v.data(), v.size());
      for (std::size_t i = 0; i < v.size(); ++i) v[i] -= c * b[i];
    }
  }
  const double after = norm2(v);
  if (after < linear_dependence * before || after < 1e-14) return false;
  for (T& x : v) x /= after;
  return true;
}

// Eigen-decomposition of the small reduced matrix (real symmetric or
// complex Hermitian), eigenvectors returned as rows y[r] (length m).
template <typename T>
struct ReducedEigen {
  std::vector<double> w;
  std::vector<std::vector<T>> y;
};

ReducedEigen<double> reducedEigen(const Matrix<double>& r) {
  const SymmetricEigenResult e = diagonalizeSymmetric(r);
  ReducedEigen<double> out;
  out.w = e.eigenvalues;
  const std::size_t m = r.rows();
  out.y.assign(m, std::vector<double>(m));
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t j = 0; j < m; ++j) out.y[j][k] = e.eigenvectors(k, j);
  }
  return out;
}

ReducedEigen<std::complex<double>> reducedEigen(const Matrix<std::complex<double>>& r) {
  const HermitianEigenResult e = diagonalizeHermitian(r);
  ReducedEigen<std::complex<double>> out;
  out.w = e.eigenvalues;
  const std::size_t m = r.rows();
  out.y.assign(m, std::vector<std::complex<double>>(m));
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t j = 0; j < m; ++j) out.y[j][k] = e.eigenvectors(k, j);
  }
  return out;
}

// Diagonal preconditioner denominator (theta - H_pp), sign-preserving floor.
inline double safeDenominator(double den, double floor_value) {
  return std::abs(den) < floor_value ? std::copysign(floor_value, den) : den;
}

}  // namespace

// =====================================================================
// NeoStepSolver
// =====================================================================

template <typename T>
struct NeoStepSolver<T>::RitzSet {
  std::vector<double> w;                  // ascending eigenvalues of L_red(alpha)
  std::vector<std::vector<T>> y;          // y[r]: coefficients of root r in the basis
  std::vector<T> z0;                      // z0 component of each root
  std::vector<std::size_t> coupled;       // roots with |z0| > decoupled_z0, ascending
};

template <typename T>
NeoStepSolver<T>::NeoStepSolver(std::vector<T> gradient, NeoHessianVectorFn<T> hessian_vector,
                                std::vector<double> hessian_diagonal, NeoStepOptions options,
                                std::vector<std::vector<T>> guess_vectors)
    : n_(gradient.size()),
      g_(std::move(gradient)),
      hvec_(std::move(hessian_vector)),
      diag_(std::move(hessian_diagonal)),
      opt_(options),
      guesses_(std::move(guess_vectors)) {
  if (n_ == 0) throw std::runtime_error("NeoStepSolver: empty gradient");
  if (!hvec_) throw std::runtime_error("NeoStepSolver: no Hessian-vector callback");
  if (!diag_.empty() && diag_.size() != n_) {
    throw std::runtime_error("NeoStepSolver: hessian_diagonal size differs from the gradient");
  }
  if (opt_.target_order > n_) {
    throw std::runtime_error("NeoStepSolver: target_order exceeds the number of parameters");
  }
  for (const auto& v : guesses_) {
    if (v.size() != n_ + 1) {
      throw std::runtime_error("NeoStepSolver: guess vectors must have size n+1");
    }
  }
  g_norm_ = norm2(g_);
  opt_.max_subspace = std::max(opt_.max_subspace, 4 * (opt_.target_order + 1));
}

template <typename T>
double NeoStepSolver<T>::effectiveTolerance() const {
  return std::max(1e-13, std::min(opt_.residual_tolerance, opt_.residual_relative * g_norm_));
}

// Adds an EXTENDED vector to the trial space: orthonormalize, one
// Hessian product on its lower part (skipped when that part is zero, e.g.
// e_0), and update the reduced pieces A and C.
template <typename T>
bool NeoStepSolver<T>::addVector(std::vector<T> ext) {
  if (ext.size() != n_ + 1) throw std::runtime_error("NeoStepSolver: bad trial-vector size");
  if (!orthonormalize(basis_, ext, opt_.linear_dependence)) return false;

  std::vector<T> bx(ext.begin() + 1, ext.end());
  std::vector<T> hb(n_, T{});
  if (norm2(bx) > 0.0) {
    hb = hvec_(bx);
    ++products_;
    if (hb.size() != n_) throw std::runtime_error("NeoStepSolver: Hessian product has wrong size");
  }
  const T gx = dotc(g_.data(), bx.data(), n_);

  const std::size_t m = basis_.size();
  for (std::size_t j = 0; j < m; ++j) {
    const T ajk = T(0.5) * (dotc(basis_[j].data() + 1, hb.data(), n_) +
                             conjugate(dotc(bx.data(), hb_[j].data(), n_)));
    a_[j].push_back(ajk);
    const T cjk = conjugate(basis_[j][0]) * gx + conjugate(gx_[j]) * ext[0];
    c_[j].push_back(cjk);
  }
  std::vector<T> a_row(m + 1), c_row(m + 1);
  for (std::size_t j = 0; j < m; ++j) {
    a_row[j] = conjugate(a_[j][m]);
    c_row[j] = conjugate(c_[j][m]);
  }
  a_row[m] = T(realPart(dotc(bx.data(), hb.data(), n_)));
  c_row[m] = T(2.0 * realPart(conjugate(ext[0]) * gx));
  a_.push_back(std::move(a_row));
  c_.push_back(std::move(c_row));

  basis_.push_back(std::move(ext));
  hb_.push_back(std::move(hb));
  gx_.push_back(gx);
  return true;
}

// Adds the next not-yet-tried coordinate unit vector (coordinates ordered
// by ascending diag(H) when a diagonal is known, else naturally).
template <typename T>
bool NeoStepSolver<T>::addUnitVector() {
  if (unit_order_.empty()) {
    unit_order_.resize(n_);
    std::iota(unit_order_.begin(), unit_order_.end(), std::size_t{0});
    if (!diag_.empty()) {
      std::stable_sort(unit_order_.begin(), unit_order_.end(),
                       [&](std::size_t a, std::size_t b) { return diag_[a] < diag_[b]; });
    }
  }
  while (unit_cursor_ < n_) {
    std::vector<T> v(n_ + 1, T{});
    v[1 + unit_order_[unit_cursor_++]] = T(1.0);
    if (addVector(std::move(v))) return true;
  }
  return false;
}

// Sec. 6.3 step 1: e_0 = (1;0) and (0;g), then (saddle targets) unit vectors
// on the lowest diagonal elements, then any warm-start vectors.
template <typename T>
void NeoStepSolver<T>::initialize() {
  initialized_ = true;
  std::vector<T> e0(n_ + 1, T{});
  e0[0] = T(1.0);
  addVector(std::move(e0));
  if (g_norm_ > 0.0) {
    std::vector<T> eg(n_ + 1, T{});
    for (std::size_t p = 0; p < n_; ++p) eg[1 + p] = g_[p] / g_norm_;
    addVector(std::move(eg));
  }
  if (opt_.target_order > 0 && opt_.guess_from_diagonal && !diag_.empty()) {
    for (std::size_t s = 0; s < opt_.target_order; ++s) addUnitVector();
  }
  for (auto& v : guesses_) addVector(v);
  guesses_.clear();
}

// Eigenpairs of L_red(alpha) = alpha C + A.
template <typename T>
typename NeoStepSolver<T>::RitzSet NeoStepSolver<T>::ritz(double alpha) const {
  const std::size_t m = basis_.size();
  Matrix<T> r(m, m, T{});
  for (std::size_t j = 0; j < m; ++j) {
    for (std::size_t k = 0; k < m; ++k) r(j, k) = T(alpha) * c_[j][k] + a_[j][k];
  }
  ReducedEigen<T> e = reducedEigen(r);
  RitzSet set;
  set.w = std::move(e.w);
  set.y = std::move(e.y);
  set.z0.assign(m, T{});
  // A coupled root above the lowest has z0 ~ |gamma_j|/gap = O(||g||), so
  // the "decoupled" cut is relative to ||g|| (capped at 1); only exact
  // decoupling leaves z0 at roundoff level.
  const double z0_threshold = std::max(1e-14, opt_.decoupled_z0 * std::min(1.0, g_norm_));
  for (std::size_t root = 0; root < m; ++root) {
    T z0{};
    for (std::size_t k = 0; k < m; ++k) z0 += set.y[root][k] * basis_[k][0];
    set.z0[root] = z0;
    if (std::abs(z0) > z0_threshold) set.coupled.push_back(root);
  }
  return set;
}

// Replaces the trial space by the Ritz vectors of the roots `keep`; the
// stored H*b products and reduced pieces are rotated, not recomputed.
template <typename T>
void NeoStepSolver<T>::collapse(const RitzSet& set, const std::vector<std::size_t>& keep) {
  const std::size_t m = basis_.size();
  const std::size_t k_new = keep.size();
  std::vector<std::vector<T>> basis(k_new, std::vector<T>(n_ + 1, T{}));
  std::vector<std::vector<T>> hb(k_new, std::vector<T>(n_, T{}));
  std::vector<T> gx(k_new, T{});
  for (std::size_t j = 0; j < k_new; ++j) {
    const auto& y = set.y[keep[j]];
    for (std::size_t k = 0; k < m; ++k) {
      for (std::size_t p = 0; p <= n_; ++p) basis[j][p] += y[k] * basis_[k][p];
      for (std::size_t p = 0; p < n_; ++p) hb[j][p] += y[k] * hb_[k][p];
      gx[j] += y[k] * gx_[k];
    }
  }
  auto rotate = [&](const std::vector<std::vector<T>>& mat) {
    std::vector<std::vector<T>> out(k_new, std::vector<T>(k_new, T{}));
    for (std::size_t j = 0; j < k_new; ++j) {
      const auto& yj = set.y[keep[j]];
      for (std::size_t l = 0; l < k_new; ++l) {
        const auto& yl = set.y[keep[l]];
        T sum{};
        for (std::size_t a = 0; a < m; ++a) {
          T inner{};
          for (std::size_t b = 0; b < m; ++b) inner += mat[a][b] * yl[b];
          sum += conjugate(yj[a]) * inner;
        }
        out[j][l] = sum;
      }
    }
    return out;
  };
  a_ = rotate(a_);
  c_ = rotate(c_);
  basis_ = std::move(basis);
  hb_ = std::move(hb);
  gx_ = std::move(gx);
}

// Davidson on L(alpha) until the residuals of the target root and all
// coupled roots below it are within tolerance (block expansion over those
// roots, Sec. 6.3 step 2). Leaves lowestRoots() set.
template <typename T>
bool NeoStepSolver<T>::davidson(double alpha, int& micro, double& residual) {
  const std::size_t need = opt_.target_order + 1;
  const double tol = effectiveTolerance();
  const double floor_value = opt_.preconditioner_floor;
  for (int it = 0; it < opt_.max_micro_iterations; ++it) {
    ++micro;
    const RitzSet set = ritz(alpha);
    if (set.coupled.size() < need) {
      if (!addUnitVector()) return false;
      continue;
    }
    const std::size_t m = basis_.size();
    std::vector<std::size_t> keep(set.coupled.begin(), set.coupled.begin() + need);
    std::vector<std::vector<T>> corrections;
    std::vector<std::vector<T>> roots;
    double max_res = 0.0;
    for (std::size_t c = 0; c < need; ++c) {
      const std::size_t r = keep[c];
      const double theta = set.w[r];
      std::vector<T> z(n_ + 1, T{});
      std::vector<T> hz(n_, T{});
      T gz{};
      for (std::size_t k = 0; k < m; ++k) {
        const T yk = set.y[r][k];
        for (std::size_t p = 0; p <= n_; ++p) z[p] += yk * basis_[k][p];
        for (std::size_t p = 0; p < n_; ++p) hz[p] += yk * hb_[k][p];
        gz += yk * gx_[k];
      }
      std::vector<T> res(n_ + 1);
      res[0] = T(alpha) * gz - T(theta) * z[0];
      for (std::size_t p = 0; p < n_; ++p) {
        res[1 + p] = T(alpha) * z[0] * g_[p] + hz[p] - T(theta) * z[1 + p];
      }
      const double rn = norm2(res);
      max_res = std::max(max_res, rn);
      if (rn > tol) {
        std::vector<T> corr(n_ + 1);
        corr[0] = res[0] / T(safeDenominator(theta, floor_value));
        for (std::size_t p = 0; p < n_; ++p) {
          const double hpp = diag_.empty() ? 0.0 : diag_[p];
          corr[1 + p] = res[1 + p] / T(safeDenominator(theta - hpp, floor_value));
        }
        corrections.push_back(std::move(corr));
      }
      roots.push_back(std::move(z));
    }
    residual = max_res;
    roots_ = std::move(roots);
    if (max_res <= tol) return true;

    if (m + corrections.size() > opt_.max_subspace) collapse(set, keep);
    std::size_t added = 0;
    for (auto& corr : corrections) {
      if (addVector(std::move(corr))) ++added;
    }
    if (added == 0 && !addUnitVector()) {
      // Nothing left to add: exact if the space is complete, else stuck.
      return basis_.size() >= n_ + 1;
    }
  }
  return false;
}

// ||d_alpha|| for the target root of the CURRENT reduced matrix (no new
// Hessian products): with orthonormal basis and unit-norm Ritz vector,
// ||x||^2 = (1-|z0|^2)/|z0|^2 and d = x/alpha.
template <typename T>
bool NeoStepSolver<T>::reducedStepNorm(double alpha, double& norm) const {
  const RitzSet set = ritz(alpha);
  if (set.coupled.size() <= opt_.target_order) return false;
  const double z0 = std::abs(set.z0[set.coupled[opt_.target_order]]);
  norm = std::sqrt(std::max(0.0, 1.0 - z0 * z0)) / (z0 * alpha);
  return true;
}

// Sec. 4: alpha >= 1 with ||d_alpha|| = radius on the current reduced
// space: geometric scan for the first crossing below `radius`, then
// bisection in log(alpha). Returns a negative value if none exists (the
// saddle-point case of Sec. 4.1).
template <typename T>
double NeoStepSolver<T>::findAlpha(double radius, double /*alpha_start*/) const {
  double lo = 1.0;
  double hi = -1.0;
  for (double a = 1.25; a <= opt_.max_alpha; a *= 1.25) {
    double nrm = 0.0;
    if (!reducedStepNorm(a, nrm)) return -1.0;
    if (nrm <= radius) {
      hi = a;
      break;
    }
    lo = a;
  }
  if (hi < 0.0) return -1.0;
  for (int it = 0; it < 80; ++it) {
    const double mid = std::sqrt(lo * hi);
    double nrm = 0.0;
    if (!reducedStepNorm(mid, nrm)) return -1.0;
    if (nrm > radius) lo = mid; else hi = mid;
    if (hi / lo < 1.0 + 1e-6) break;
  }
  return hi;
}

template <typename T>
NeoStep<T> NeoStepSolver<T>::solve(double radius) {
  NeoStep<T> out;
  out.step.assign(n_, T{});
  if (g_norm_ < 1e-14) {
    out.trivial = true;
    out.converged = true;
    out.hessian_products = products_;
    return out;
  }
  if (!initialized_) initialize();

  double alpha = 1.0;
  int micro = 0;
  double residual = 0.0;
  bool conv = davidson(alpha, micro, residual);

  // Step, shift and quadratic-model pieces of the target root at `alpha`.
  std::vector<T> d(n_), hd(n_);
  double theta = 0.0;
  double norm_d = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  bool have_root = false;
  auto extract = [&](double a) {
    const RitzSet set = ritz(a);
    have_root = set.coupled.size() > opt_.target_order;
    if (!have_root) return;
    const std::size_t r = set.coupled[opt_.target_order];
    theta = set.w[r];
    const T z0 = set.z0[r];
    std::fill(d.begin(), d.end(), T{});
    std::fill(hd.begin(), hd.end(), T{});
    for (std::size_t k = 0; k < basis_.size(); ++k) {
      const T yk = set.y[r][k];
      for (std::size_t p = 0; p < n_; ++p) {
        d[p] += yk * basis_[k][1 + p];
        hd[p] += yk * hb_[k][p];
      }
    }
    const T inv = T(1.0) / (z0 * T(a));
    for (std::size_t p = 0; p < n_; ++p) {
      d[p] *= inv;
      hd[p] *= inv;
    }
    norm_d = norm2(d);
    a1 = realPart(dotc(g_.data(), d.data(), n_));
    a2 = realPart(dotc(d.data(), hd.data(), n_));
  };
  extract(alpha);
  if (!have_root) {
    out.hessian_products = products_;
    out.micro_iterations = micro;
    return out;  // zero step, converged = false
  }

  bool restricted = false;
  bool scaled = false;
  double scale = 1.0;
  if (radius > 0.0 && norm_d > radius) {
    for (int it = 0; it < opt_.max_alpha_iterations; ++it) {
      const double a_new = findAlpha(radius, alpha);
      if (a_new < 0.0) break;
      alpha = a_new;
      restricted = true;
      conv = davidson(alpha, micro, residual);
      extract(alpha);
      if (!have_root) break;
      if (std::abs(norm_d - radius) <= opt_.step_length_tolerance * radius) break;
    }
    if (!have_root) {
      out.hessian_products = products_;
      out.micro_iterations = micro;
      return out;
    }
    if (norm_d > radius * (1.0 + opt_.step_length_tolerance)) {
      scale = radius / norm_d;  // no admissible alpha: rescale to the boundary
      scaled = true;
      restricted = false;
    }
  }

  for (std::size_t p = 0; p < n_; ++p) out.step[p] = d[p] * T(scale);
  out.shift = theta;
  out.alpha = alpha;
  out.step_norm = norm_d * scale;
  out.predicted_change = scale * a1 + 0.5 * scale * scale * a2;
  out.residual_norm = residual;
  out.converged = conv;
  out.restricted = restricted;
  out.scaled = scaled;
  out.micro_iterations = micro;
  out.hessian_products = products_;
  return out;
}

// =====================================================================
// Trust-radius rule (Sec. 5)
// =====================================================================

NeoTrustDecision neoTrustDecision(std::size_t target_order, double ratio, double radius,
                                  const NeoTrustOptions& o) {
  NeoTrustDecision decision;
  if (target_order == 0) {
    if (ratio < 0.0) {
      decision.accept = false;
      decision.radius = o.shrink_factor * radius;
    } else if (ratio < o.ratio_low) {
      decision.accept = true;
      decision.radius = o.shrink_factor * radius;
    } else if (ratio <= o.ratio_high) {
      decision.accept = true;
      decision.radius = radius;
    } else {
      decision.accept = true;
      decision.radius = std::min(o.grow_factor * radius, o.max_radius);
    }
  } else {
    const double lo = o.saddle_ratio_min;
    const double good = o.saddle_ratio_good;
    if (ratio < lo || ratio > 2.0 - lo) {
      decision.accept = false;
      decision.radius = radius / o.saddle_grow;
    } else if (ratio < good || ratio > 2.0 - good) {
      decision.accept = true;
      decision.radius = radius;
    } else {
      decision.accept = true;
      decision.radius = std::min(o.saddle_grow * radius, o.max_radius);
    }
  }
  return decision;
}

// =====================================================================
// Lowest Hessian eigenpairs (block Davidson on H)
// =====================================================================

template <typename T>
NeoEigenResult<T> neoLowestHessianEigenpairs(std::size_t n,
                                             const NeoHessianVectorFn<T>& hessian_vector,
                                             std::size_t n_roots,
                                             const std::vector<double>& hessian_diagonal,
                                             const NeoEigenOptions& o) {
  if (n == 0 || n_roots == 0 || n_roots > n) {
    throw std::runtime_error("neoLowestHessianEigenpairs: invalid n / n_roots");
  }
  if (!hessian_vector) throw std::runtime_error("neoLowestHessianEigenpairs: no callback");
  if (!hessian_diagonal.empty() && hessian_diagonal.size() != n) {
    throw std::runtime_error("neoLowestHessianEigenpairs: diagonal size differs from n");
  }
  const std::size_t max_subspace = std::max(o.max_subspace, 4 * n_roots);

  NeoEigenResult<T> result;
  std::vector<std::vector<T>> basis, hb, a;

  auto addVec = [&](std::vector<T> v) {
    if (!orthonormalize(basis, v, o.linear_dependence)) return false;
    std::vector<T> h = hessian_vector(v);
    ++result.hessian_products;
    if (h.size() != n) throw std::runtime_error("neoLowestHessianEigenpairs: bad product size");
    const std::size_t m = basis.size();
    for (std::size_t j = 0; j < m; ++j) {
      a[j].push_back(T(0.5) * (dotc(basis[j].data(), h.data(), n) +
                                conjugate(dotc(v.data(), hb[j].data(), n))));
    }
    std::vector<T> row(m + 1);
    for (std::size_t j = 0; j < m; ++j) row[j] = conjugate(a[j][m]);
    row[m] = T(realPart(dotc(v.data(), h.data(), n)));
    a.push_back(std::move(row));
    basis.push_back(std::move(v));
    hb.push_back(std::move(h));
    return true;
  };

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (!hessian_diagonal.empty()) {
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
      return hessian_diagonal[x] < hessian_diagonal[y];
    });
  }
  std::size_t cursor = 0;
  auto addUnit = [&]() {
    while (cursor < n) {
      std::vector<T> v(n, T{});
      v[order[cursor++]] = T(1.0);
      if (addVec(std::move(v))) return true;
    }
    return false;
  };
  for (std::size_t s = 0; s < n_roots; ++s) addUnit();
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto uniform = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0 - 0.5;
  };
  for (std::size_t s = 0; s < o.n_random_guesses; ++s) {
    std::vector<T> v(n, T{});
    for (T& x : v) {
      if constexpr (std::is_same_v<T, double>) {
        x = uniform();
      } else {
        const double re = uniform();
        const double im = uniform();
        x = T(re, im);
      }
    }
    addVec(std::move(v));
  }

  for (int it = 0; it < o.max_iterations; ++it) {
    result.iterations = it + 1;
    const std::size_t m = basis.size();
    Matrix<T> r(m, m, T{});
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t k = 0; k < m; ++k) r(j, k) = a[j][k];
    }
    const ReducedEigen<T> e = reducedEigen(r);
    const std::size_t take = std::min(n_roots, m);

    std::vector<std::vector<T>> vecs, corrections;
    std::vector<double> res_norms;
    double max_res = 0.0;
    for (std::size_t root = 0; root < take; ++root) {
      const double theta = e.w[root];
      std::vector<T> z(n, T{}), res(n, T{});
      for (std::size_t k = 0; k < m; ++k) {
        const T yk = e.y[root][k];
        for (std::size_t p = 0; p < n; ++p) {
          z[p] += yk * basis[k][p];
          res[p] += yk * hb[k][p];
        }
      }
      for (std::size_t p = 0; p < n; ++p) res[p] -= T(theta) * z[p];
      const double rn = norm2(res);
      max_res = std::max(max_res, rn);
      res_norms.push_back(rn);
      if (rn > o.residual_tolerance) {
        for (std::size_t p = 0; p < n; ++p) {
          const double hpp = hessian_diagonal.empty() ? 0.0 : hessian_diagonal[p];
          res[p] /= T(safeDenominator(theta - hpp, o.preconditioner_floor));
        }
        corrections.push_back(std::move(res));
      }
      vecs.push_back(std::move(z));
    }
    result.eigenvalues.assign(e.w.begin(), e.w.begin() + take);
    result.eigenvectors = std::move(vecs);
    result.residual_norms = std::move(res_norms);
    if (take == n_roots && max_res <= o.residual_tolerance) {
      result.converged = true;
      return result;
    }
    if (take < n_roots && m >= n) {
      result.converged = m >= n;
      return result;
    }

    if (m + corrections.size() > max_subspace) {
      // Collapse onto the current Ritz vectors of the wanted roots.
      std::vector<std::vector<T>> nb(take, std::vector<T>(n, T{}));
      std::vector<std::vector<T>> nh(take, std::vector<T>(n, T{}));
      for (std::size_t j = 0; j < take; ++j) {
        for (std::size_t k = 0; k < m; ++k) {
          const T yk = e.y[j][k];
          for (std::size_t p = 0; p < n; ++p) {
            nb[j][p] += yk * basis[k][p];
            nh[j][p] += yk * hb[k][p];
          }
        }
      }
      std::vector<std::vector<T>> na(take, std::vector<T>(take, T{}));
      for (std::size_t j = 0; j < take; ++j) {
        for (std::size_t l = 0; l < take; ++l) na[j][l] = T(j == l ? e.w[j] : 0.0);
      }
      basis = std::move(nb);
      hb = std::move(nh);
      a = std::move(na);
    }
    std::size_t added = 0;
    for (auto& c : corrections) {
      if (addVec(std::move(c))) ++added;
    }
    if (added == 0 && !addUnit()) {
      result.converged = basis.size() >= n;
      return result;
    }
  }
  return result;
}

// =====================================================================
// Macro-iteration driver (Sec. 8)
// =====================================================================

template <typename T>
NeoResult neoOptimize(NeoProblem<T>& problem, const NeoOptions& o) {
  const std::size_t n = problem.dimension();
  const std::size_t order = o.step.target_order;
  NeoResult result;
  double radius = std::min(o.initial_radius, o.trust.max_radius);
  std::vector<std::vector<T>> guesses;
  bool failed = false;

  auto hessian_vector = [&](const std::vector<T>& v) { return problem.hessianVector(v); };

  int iteration = 0;
  std::vector<T> g;
  for (;;) {
    const double e0 = problem.energy();
    g = problem.gradient();
    if (g.size() != n) throw std::runtime_error("neoOptimize: gradient size differs from dimension()");
    const double gmax = maxAbs(g);
    result.energy = e0;
    result.gradient_max = gmax;
    result.iterations = iteration;
    if (gmax <= o.gradient_tolerance) {
      result.converged = true;
      break;
    }
    if (iteration >= o.max_iterations || failed) break;

    NeoStepSolver<T> solver(g, hessian_vector, problem.hessianDiagonal(), o.step, guesses);
    NeoIteration record;
    record.energy = e0;
    record.gradient_max = gmax;
    int rejections = 0;
    for (;;) {
      const NeoStep<T> step = solver.solve(radius);
      const double e1 = problem.trialEnergy(step.step);
      const double d_e = e1 - e0;
      const double d_q = step.predicted_change;
      double ratio;
      if (std::abs(d_q) < o.ratio_floor && std::abs(d_e) < o.ratio_floor) {
        ratio = 1.0;  // roundoff-level step: accept, radius unchanged
      } else if (std::abs(d_q) < o.ratio_floor) {
        ratio = -1.0;
      } else {
        ratio = d_e / d_q;
      }
      const NeoTrustDecision decision = neoTrustDecision(order, ratio, radius, o.trust);
      if (o.verbose) {
        std::cout << "  NEO it " << std::setw(3) << iteration + 1 << "  E = " << std::setprecision(12)
                  << e0 << "  |g|max = " << std::scientific << std::setprecision(3) << gmax
                  << "  h = " << std::fixed << std::setprecision(3) << radius
                  << "  |d| = " << step.step_norm << "  shift = " << std::scientific
                  << step.shift << "  alpha = " << std::fixed << step.alpha
                  << "  r = " << ratio << (decision.accept ? "" : "  REJECTED") << "\n";
      }
      if (decision.accept) {
        problem.accept(step.step);
        record.radius = radius;
        record.step_norm = step.step_norm;
        record.shift = step.shift;
        record.alpha = step.alpha;
        record.ratio = ratio;
        record.predicted_change = d_q;
        record.actual_change = d_e;
        record.rejections = rejections;
        record.micro_iterations = step.micro_iterations;
        record.hessian_products = solver.hessianProducts();
        radius = decision.radius;
        break;
      }
      radius = decision.radius;
      if (++rejections > o.max_rejections || radius < o.min_radius) {
        failed = true;
        break;
      }
    }
    result.hessian_products += solver.hessianProducts();
    if (failed) {
      result.history.push_back(record);
      continue;  // top of loop re-evaluates E/g at the unchanged CEP, then stops
    }
    result.history.push_back(record);
    guesses = solver.lowestRoots();
    ++iteration;
  }

  if (o.verify_index && result.converged) {
    const NeoEigenResult<T> eig = neoLowestHessianEigenpairs<T>(
        n, hessian_vector, std::min(n, order + 1), problem.hessianDiagonal());
    result.hessian_products += eig.hessian_products;
    result.lowest_hessian_eigenvalues = eig.eigenvalues;
    result.negative_eigenvalues = 0;
    for (double w : eig.eigenvalues) {
      if (w < -o.verify_tolerance) ++result.negative_eigenvalues;
    }
  }
  return result;
}

template class NeoStepSolver<double>;
template class NeoStepSolver<std::complex<double>>;
template NeoResult neoOptimize<double>(NeoProblem<double>&, const NeoOptions&);
template NeoResult neoOptimize<std::complex<double>>(NeoProblem<std::complex<double>>&,
                                                    const NeoOptions&);
template NeoEigenResult<double> neoLowestHessianEigenpairs<double>(
    std::size_t, const NeoHessianVectorFn<double>&, std::size_t, const std::vector<double>&,
    const NeoEigenOptions&);
template NeoEigenResult<std::complex<double>> neoLowestHessianEigenpairs<std::complex<double>>(
    std::size_t, const NeoHessianVectorFn<std::complex<double>>&, std::size_t,
    const std::vector<double>&, const NeoEigenOptions&);

}  // namespace rerdmft
