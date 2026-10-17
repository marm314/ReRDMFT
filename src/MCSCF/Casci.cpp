#include "Casci.h"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

inline double conjugate(double x) { return x; }
inline std::complex<double> conjugate(const std::complex<double>& x) { return std::conj(x); }
inline double realPart(double x) { return x; }
inline double realPart(const std::complex<double>& x) { return x.real(); }

// <a|b> = sum conj(a) b
template <typename T>
T dot(const std::vector<T>& a, const std::vector<T>& b) {
  T s{};
  for (std::size_t i = 0; i < a.size(); ++i) s += conjugate(a[i]) * b[i];
  return s;
}
template <typename T>
double norm2(const std::vector<T>& a) {
  double s = 0.0;
  for (const T& x : a) s += std::norm(x);
  return std::sqrt(s);
}
inline double norm2(const std::vector<double>& a) {
  double s = 0.0;
  for (double x : a) s += x * x;
  return std::sqrt(s);
}

// C(m x n) = alpha * op(A) * op(B), row-major, for the two element types in production.
void gemm(bool trans_b_conj, int m, int n, int k, double alpha, const double* a, const double* b, double* c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, trans_b_conj ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, k, b,
              trans_b_conj ? k : n, 0.0, c, n);
}
void gemm(bool trans_b_conj, int m, int n, int k, double alpha, const std::complex<double>* a,
          const std::complex<double>* b, std::complex<double>* c) {
  const std::complex<double> al(alpha, 0.0), be(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, trans_b_conj ? CblasConjTrans : CblasNoTrans, m, n, k, &al, a, k, b,
              trans_b_conj ? k : n, &be, c, n);
}

// Lowest eigenpair of a small Hermitian (symmetric) matrix.
void lowestEigenpair(const Matrix<double>& s, double& theta, std::vector<double>& y) {
  const SymmetricEigenResult r = diagonalizeSymmetric(s);
  theta = r.eigenvalues[0];
  y.resize(s.rows());
  for (std::size_t i = 0; i < s.rows(); ++i) y[i] = r.eigenvectors(i, 0);
}
void lowestEigenpair(const Matrix<std::complex<double>>& s, double& theta, std::vector<std::complex<double>>& y) {
  const HermitianEigenResult r = diagonalizeHermitian(s);
  theta = r.eigenvalues[0];
  y.resize(s.rows());
  for (std::size_t i = 0; i < s.rows(); ++i) y[i] = r.eigenvectors(i, 0);
}

// The k lowest eigenpairs of a small Hermitian (symmetric) matrix (columns of the eigenvector matrix are orthonormal).
template <typename T, typename R>
void lowestEigenpairs(const R& r, std::size_t n, std::size_t k, std::vector<double>& thetas, std::vector<std::vector<T>>& ys) {
  thetas.assign(r.eigenvalues.begin(), r.eigenvalues.begin() + static_cast<std::ptrdiff_t>(k));
  ys.assign(k, std::vector<T>(n));
  for (std::size_t j = 0; j < k; ++j)
    for (std::size_t i = 0; i < n; ++i) ys[j][i] = r.eigenvectors(i, j);
}
void lowestEigenpairs(const Matrix<double>& s, std::size_t k, std::vector<double>& thetas, std::vector<std::vector<double>>& ys) {
  lowestEigenpairs<double>(diagonalizeSymmetric(s), s.rows(), k, thetas, ys);
}
void lowestEigenpairs(const Matrix<std::complex<double>>& s, std::size_t k, std::vector<double>& thetas,
                      std::vector<std::vector<std::complex<double>>>& ys) {
  lowestEigenpairs<std::complex<double>>(diagonalizeHermitian(s), s.rows(), k, thetas, ys);
}

std::uint64_t binomial(std::size_t n, std::size_t k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  long double r = 1.0L;
  for (std::size_t i = 1; i <= k; ++i) r = r * static_cast<long double>(n - k + i) / static_cast<long double>(i);
  return static_cast<std::uint64_t>(std::llround(r));
}

inline int popcount(std::uint64_t x) { return __builtin_popcountll(x); }

}  // namespace

template <typename T>
Casci<T>::Casci(std::size_t n_orbitals, std::size_t n_electrons) : m_(n_orbitals), n_(n_electrons) {
  if (m_ == 0 || m_ > 64) throw std::runtime_error("Casci: the number of active spinors must be in 1..64");
  if (n_ > m_) throw std::runtime_error("Casci: more electrons than active spinors");
  const std::uint64_t count = binomial(m_, n_);
  if (count > (std::uint64_t{1} << 31)) throw std::runtime_error("Casci: determinant count exceeds 2^31");

  // Determinants in ascending bitmask order (Gosper's hack); the lookup below uses the
  // combinatorial-number-system rank, which is monotonic in the bitmask.
  dets_.reserve(count);
  if (n_ == 0) {
    dets_.push_back(0);
  } else {
    std::uint64_t d = (n_ == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << n_) - 1);
    const std::uint64_t limit = (m_ == 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << m_) - 1);
    for (;;) {
      dets_.push_back(d);
      if (dets_.size() == count) break;
      const std::uint64_t c = d & (~d + 1);
      const std::uint64_t r = d + c;
      d = (((r ^ d) >> 2) / c) | r;
      if (d > limit) break;
    }
  }
  if (dets_.size() != count) throw std::runtime_error("Casci: determinant enumeration failed");

  std::vector<std::vector<std::uint64_t>> binom(m_ + 1, std::vector<std::uint64_t>(n_ + 1, 0));
  for (std::size_t i = 0; i <= m_; ++i) {
    binom[i][0] = 1;
    for (std::size_t j = 1; j <= std::min(i, n_); ++j) binom[i][j] = binom[i - 1][j - 1] + (j <= i - 1 ? binom[i - 1][j] : 0);
  }
  auto rank = [&](std::uint64_t det) {
    std::uint64_t r = 0;
    std::size_t j = 1;
    for (std::size_t p = 0; p < m_; ++p)
      if ((det >> p) & 1u) r += binom[p][j++];
    return r;
  };

  links_.assign(m_ * m_, {});
  for (std::size_t pq = 0; pq < m_ * m_; ++pq) links_[pq].reserve(static_cast<std::size_t>(count * n_ / m_ + 1));
  for (std::uint32_t i = 0; i < dets_.size(); ++i) {
    const std::uint64_t det = dets_[i];
    for (std::size_t q = 0; q < m_; ++q) {
      if (!((det >> q) & 1u)) continue;
      const std::uint64_t without_q = det & ~(std::uint64_t{1} << q);
      const int sign_q = (popcount(det & ((std::uint64_t{1} << q) - 1)) & 1) ? -1 : 1;
      for (std::size_t p = 0; p < m_; ++p) {
        if (p != q && ((without_q >> p) & 1u)) continue;
        const std::uint64_t result = without_q | (std::uint64_t{1} << p);
        const int sign_p = (popcount(without_q & ((std::uint64_t{1} << p) - 1)) & 1) ? -1 : 1;
        links_[p * m_ + q].push_back({i, static_cast<std::uint32_t>(rank(result)), sign_q * sign_p});
      }
    }
  }
}

template <typename T>
std::vector<T> Casci<T>::excitedVectors(const std::vector<T>& c) const {
  const std::size_t dim = dets_.size();
  std::vector<T> d(m_ * m_ * dim, T{});
#pragma omp parallel for schedule(static)
  for (std::size_t pq = 0; pq < m_ * m_; ++pq) {
    T* row = d.data() + pq * dim;
    for (const Link& l : links_[pq]) row[l.target] = static_cast<double>(l.sign) * c[l.source];
  }
  return d;
}

template <typename T>
std::vector<T> Casci<T>::sigma(const CasciHamiltonian<T>& ham, const std::vector<T>& c) const {
  const std::size_t dim = dets_.size();
  const std::size_t m2 = m_ * m_;
  if (c.size() != dim) throw std::runtime_error("Casci::sigma: wrong vector length");
  if (ham.n_orbitals != m_ || ham.n_electrons != n_ || ham.eri.size() != m2 * m2)
    throw std::runtime_error("Casci::sigma: Hamiltonian does not match the CI space");

  // h'(p,s) = h(p,s) - 1/2 sum_q (pq|qs): the normal-ordering correction of the two-body term.
  Matrix<T> hp(m_, m_);
  for (std::size_t p = 0; p < m_; ++p)
    for (std::size_t s = 0; s < m_; ++s) {
      T v = ham.h(p, s);
      for (std::size_t q = 0; q < m_; ++q) v -= 0.5 * ham.eri[((p * m_ + q) * m_ + q) * m_ + s];
      hp(p, s) = v;
    }

  // sigma = sum_pq E_pq G_pq,  G_pq = 1/2 sum_rs (pq|rs) E_rs c + h'_pq c.
  const std::vector<T> d = excitedVectors(c);
  std::vector<T> g(m2 * dim);
  gemm(false, static_cast<int>(m2), static_cast<int>(dim), static_cast<int>(m2), 0.5, ham.eri.data(), d.data(), g.data());
  for (std::size_t p = 0; p < m_; ++p)
    for (std::size_t q = 0; q < m_; ++q) {
      T* row = g.data() + (p * m_ + q) * dim;
      const T h = hp(p, q);
      for (std::size_t i = 0; i < dim; ++i) row[i] += h * c[i];
    }

  std::vector<T> out(dim, T{});
  for (std::size_t pq = 0; pq < m2; ++pq) {
    const T* row = g.data() + pq * dim;
    const std::vector<Link>& ls = links_[pq];
#pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < ls.size(); ++k) out[ls[k].target] += static_cast<double>(ls[k].sign) * row[ls[k].source];
  }
  return out;
}

template <typename T>
std::vector<double> Casci<T>::diagonal(const CasciHamiltonian<T>& ham) const {
  std::vector<double> diag(dets_.size());
  for (std::size_t i = 0; i < dets_.size(); ++i) {
    const std::uint64_t det = dets_[i];
    double e = 0.0;
    for (std::size_t p = 0; p < m_; ++p) {
      if (!((det >> p) & 1u)) continue;
      e += realPart(ham.h(p, p));
      for (std::size_t q = 0; q < m_; ++q) {
        if (!((det >> q) & 1u)) continue;
        e += 0.5 * realPart(ham.eri[((p * m_ + p) * m_ + q) * m_ + q] - ham.eri[((p * m_ + q) * m_ + q) * m_ + p]);
      }
    }
    diag[i] = e;
  }
  return diag;
}

template <typename T>
std::vector<T> Casci<T>::timeReversed(const std::vector<T>& c) const {
  const std::size_t dim = dets_.size();
  if (tr_target_.size() != dim) {
    if (m_ % 2 != 0) throw std::runtime_error("Casci: time-reversal projection needs an even number of spinors");
    std::vector<std::uint32_t> target(dim);
    std::vector<std::int8_t> sign(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      // images of the occupied spinors in ascending order: 2k -> 2k+1 (+1), 2k+1 -> 2k (-1)
      int img[64], fac = 1, count = 0;
      std::uint64_t mask = 0;
      for (std::size_t t = 0; t < m_; ++t)
        if ((dets_[i] >> t) & 1u) {
          const std::size_t j = (t % 2 == 0) ? t + 1 : t - 1;
          if (t % 2 == 1) fac = -fac;
          img[count++] = static_cast<int>(j);
          mask |= std::uint64_t(1) << j;
        }
      int inversions = 0;
      for (int a = 0; a < count; ++a)
        for (int b = a + 1; b < count; ++b)
          if (img[a] > img[b]) ++inversions;
      if (inversions % 2 != 0) fac = -fac;
      target[i] = static_cast<std::uint32_t>(std::lower_bound(dets_.begin(), dets_.end(), mask) - dets_.begin());
      sign[i] = static_cast<std::int8_t>(fac);
    }
    tr_target_ = std::move(target);
    tr_sign_ = std::move(sign);
  }
  std::vector<T> out(dim, T{});
  for (std::size_t i = 0; i < dim; ++i) out[tr_target_[i]] = static_cast<double>(tr_sign_[i]) * conjugate(c[i]);
  return out;
}

template <typename T>
CasciResult<T> Casci<T>::solve(const CasciHamiltonian<T>& ham, const CasciOptions& options,
                               const std::vector<T>* start) const {
  const std::size_t dim = dets_.size();
  const std::vector<double> hdiag = diagonal(ham);

  const bool project = options.time_reversal_projection && !std::is_same_v<T, double>;
  const auto trEven = [&](std::vector<T>& v) {
    if (!project) return;
    const std::vector<T> tv = timeReversed(v);
    for (std::size_t k = 0; k < dim; ++k) v[k] = 0.5 * (v[k] + tv[k]);
  };

  std::vector<T> x0(dim, T{});
  if (start != nullptr && start->size() == dim) {
    x0 = *start;
  } else {
    x0[static_cast<std::size_t>(std::min_element(hdiag.begin(), hdiag.end()) - hdiag.begin())] = T(1.0);
  }
  trEven(x0);
  const double n0 = norm2(x0);
  if (n0 == 0.0) throw std::runtime_error("Casci::solve: zero starting vector");
  for (T& v : x0) v /= n0;

  std::vector<std::vector<T>> basis, hbasis;
  basis.push_back(x0);
  hbasis.push_back(sigma(ham, x0));

  CasciResult<T> result;
  double theta = 0.0;
  std::vector<double> theta_history;
  std::vector<T> x(dim);
  for (int iter = 1; iter <= options.max_iterations; ++iter) {
    result.iterations = iter;
    const std::size_t nb = basis.size();
    Matrix<T> s(nb, nb);
    for (std::size_t i = 0; i < nb; ++i)
      for (std::size_t j = i; j < nb; ++j) {
        const T v = dot(basis[i], hbasis[j]);
        s(i, j) = v;
        s(j, i) = conjugate(v);
      }
    // the diagonal must be exactly real for the Hermitian eigensolver
    for (std::size_t i = 0; i < nb; ++i) s(i, i) = T(realPart(s(i, i)));
    std::vector<double> thetas;
    std::vector<std::vector<T>> ys;
    lowestEigenpairs(s, std::min<std::size_t>(nb, static_cast<std::size_t>(std::max(1, options.n_keep))), thetas, ys);
    theta = thetas[0];
    const std::vector<T>& y = ys[0];

    std::vector<T> hx(dim, T{});
    std::fill(x.begin(), x.end(), T{});
    for (std::size_t i = 0; i < nb; ++i)
      for (std::size_t k = 0; k < dim; ++k) {
        x[k] += y[i] * basis[i][k];
        hx[k] += y[i] * hbasis[i][k];
      }
    std::vector<T> r(dim);
    for (std::size_t k = 0; k < dim; ++k) r[k] = hx[k] - theta * x[k];
    result.residual_norm = norm2(r);
    if (result.residual_norm < options.residual_tolerance) {
      result.converged = true;
      break;
    }
    // Stagnation inside a near-degenerate cluster: the Ritz energy no longer moves (to stagnation_energy over
    // stagnation_window iterations) and the residual is already small -- the state is an eigenvector to within the
    // cluster's splitting, and further steps only wander among its members.
    theta_history.push_back(theta);
    if (options.stagnation_window > 0 && theta_history.size() > static_cast<std::size_t>(options.stagnation_window) &&
        result.residual_norm < options.stagnation_residual &&
        std::abs(theta_history[theta_history.size() - 1 - static_cast<std::size_t>(options.stagnation_window)] - theta) <
            options.stagnation_energy) {
      result.converged = true;
      break;
    }

    // Diagonal (Davidson) preconditioner with a guard on small denominators.
    std::vector<T> t(dim);
    for (std::size_t k = 0; k < dim; ++k) {
      double den = theta - hdiag[k];
      if (std::abs(den) < 1e-4) den = (den < 0.0) ? -1e-4 : 1e-4;
      t[k] = r[k] / den;
    }

    if (basis.size() >= static_cast<std::size_t>(options.max_subspace)) {
      // thick restart: keep the n_keep lowest Ritz vectors (orthonormal, with their H-images), not only the lowest, so the
      // near-degenerate neighbours of the target state stay in the search space
      std::vector<std::vector<T>> nb_new(ys.size(), std::vector<T>(dim, T{})), nh_new(ys.size(), std::vector<T>(dim, T{}));
      for (std::size_t j = 0; j < ys.size(); ++j)
        for (std::size_t i = 0; i < nb; ++i)
          for (std::size_t k = 0; k < dim; ++k) {
            nb_new[j][k] += ys[j][i] * basis[i][k];
            nh_new[j][k] += ys[j][i] * hbasis[i][k];
          }
      basis = std::move(nb_new);
      hbasis = std::move(nh_new);
    }
    for (int pass = 0; pass < 2; ++pass)
      for (const std::vector<T>& b : basis) {
        const T ov = dot(b, t);
        for (std::size_t k = 0; k < dim; ++k) t[k] -= ov * b[k];
      }
    const double nt = norm2(t);
    if (nt < 1e-12) break;  // the Krylov space is exhausted
    for (T& v : t) v /= nt;
    basis.push_back(t);
    hbasis.push_back(sigma(ham, t));
  }

  const double nx = norm2(x);
  for (T& v : x) v /= nx;
  result.ci = x;
  result.energy = theta + ham.core_energy;
  return result;
}

template <typename T>
Matrix<T> Casci<T>::oneRdm(const std::vector<T>& c) const {
  if (c.size() != dets_.size()) throw std::runtime_error("Casci::oneRdm: wrong vector length");
  Matrix<T> g(m_, m_, T{});
  for (std::size_t pq = 0; pq < m_ * m_; ++pq) {
    T sum{};
    for (const Link& l : links_[pq]) sum += conjugate(c[l.target]) * (static_cast<double>(l.sign) * c[l.source]);
    g(pq / m_, pq % m_) = sum;
  }
  return g;
}

template <typename T>
void Casci<T>::densities(const std::vector<T>& c, Matrix<T>& rdm1, std::vector<T>& rdm2) const {
  const std::size_t dim = dets_.size();
  const std::size_t m2 = m_ * m_;
  if (c.size() != dim) throw std::runtime_error("Casci::densities: wrong vector length");
  const std::vector<T> d = excitedVectors(c);

  rdm1 = Matrix<T>(m_, m_, T{});
  for (std::size_t pq = 0; pq < m2; ++pq) {
    T s{};
    const T* row = d.data() + pq * dim;
    for (std::size_t j = 0; j < dim; ++j) s += conjugate(c[j]) * row[j];
    rdm1(pq / m_, pq % m_) = s;
  }

  // K(a,b) = sum_J D(a,J) conj(D(b,J));  <E_pq E_rs> = sum_J conj(D(qp,J)) D(rs,J) = K(rs,qp)
  std::vector<T> k(m2 * m2);
  gemm(true, static_cast<int>(m2), static_cast<int>(m2), static_cast<int>(dim), 1.0, d.data(), d.data(), k.data());
  rdm2.assign(m2 * m2, T{});
  for (std::size_t p = 0; p < m_; ++p)
    for (std::size_t q = 0; q < m_; ++q)
      for (std::size_t r = 0; r < m_; ++r)
        for (std::size_t s = 0; s < m_; ++s) {
          T v = k[(r * m_ + s) * m2 + (q * m_ + p)];
          if (q == r) v -= rdm1(p, s);
          rdm2[((p * m_ + q) * m_ + r) * m_ + s] = v;
        }
}

template class Casci<double>;
template class Casci<std::complex<double>>;

}  // namespace rerdmft
