#include "KramersPairing.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

using C = std::complex<double>;
using Vec = std::vector<C>;
using VecFn = std::function<Vec(const Vec&)>;

Vec columnOf(const Matrix<C>& m, std::size_t col) {
  Vec v(m.rows());
  for (std::size_t r = 0; r < m.rows(); ++r) v[r] = m(r, col);
  return v;
}

C metricInner(const VecFn& metric, const Vec& a, const Vec& b) {
  const Vec mb = metric(b);
  C sum{};
  for (std::size_t r = 0; r < a.size(); ++r) sum += std::conj(a[r]) * mb[r];
  return sum;
}

// max_k || Theta c_2k - c_2k+1 ||_metric.
double partnerError(const Matrix<C>& c, const VecFn& theta_of, const VecFn& metric) {
  double err = 0.0;
  for (std::size_t k = 0; k + 1 < c.cols(); k += 2) {
    Vec d = theta_of(columnOf(c, k));
    const Vec odd = columnOf(c, k + 1);
    for (std::size_t r = 0; r < d.size(); ++r) d[r] -= odd[r];
    err = std::max(err, std::sqrt(std::max(0.0, std::real(metricInner(metric, d, d)))));
  }
  return err;
}

// The algorithm, in terms of Theta and the metric acting on length-n_orig vectors
// (see KramersPairing.h for the description).
Matrix<C> pairingCore(const Matrix<C>& c_matrix, const std::vector<double>& energies,
                      const VecFn& theta_of, const VecFn& metric, double cluster_tolerance,
                      KramersPairingReport* report) {
  const std::size_t n_orig = c_matrix.rows();
  const std::size_t n_cols = c_matrix.cols();
  if (energies.size() != n_cols || n_cols % 2 != 0) {
    throw std::runtime_error("fixKramersPairing: inconsistent input dimensions");
  }
  KramersPairingReport local;
  local.partner_error_before = partnerError(c_matrix, theta_of, metric);
  Matrix<C> result = c_matrix;

  const std::size_t n_pairs = n_cols / 2;
  std::size_t pair = 0;
  while (pair < n_pairs) {
    // Cluster of consecutive pairs [pair, last].
    std::size_t last = pair;
    while (last + 1 < n_pairs && energies[2 * (last + 1)] - energies[2 * last + 1] < cluster_tolerance) {
      ++last;
    }
    const std::size_t k0 = 2 * pair;
    const std::size_t mc = 2 * (last - pair + 1);
    if (mc > 2) {
      ++local.n_multi_pair_clusters;
      local.largest_cluster_pairs = std::max(local.largest_cluster_pairs, mc / 2);
    }

    // T = V^dagger S (Theta V) in the cluster's own coefficient space:
    // Theta(V x) = (Theta V) conj(x) -> coefficients T conj(x).
    std::vector<Vec> theta_cols(mc);
    for (std::size_t j = 0; j < mc; ++j) theta_cols[j] = metric(theta_of(columnOf(c_matrix, k0 + j)));
    Matrix<C> t(mc, mc, C{});
    for (std::size_t a = 0; a < mc; ++a)
      for (std::size_t b = 0; b < mc; ++b) {
        C sum{};
        for (std::size_t r = 0; r < n_orig; ++r) sum += std::conj(c_matrix(r, k0 + a)) * theta_cols[b][r];
        t(a, b) = sum;
      }

    // Current remaining subspace: orthonormal columns of B (mc x d).
    Matrix<C> basis(mc, mc, C{});
    for (std::size_t i = 0; i < mc; ++i) basis(i, i) = 1.0;
    std::size_t d = mc;
    Matrix<C> new_coeff(mc, mc, C{});  // columns: x_0, y_0, x_1, y_1, ... in cluster coefficients
    for (std::size_t q = 0; q < mc / 2; ++q) {
      Vec x(mc), y(mc, C{});
      for (std::size_t i = 0; i < mc; ++i) x[i] = basis(i, 0);
      // y = P_B T conj(x)
      Vec ty(mc, C{});
      for (std::size_t a = 0; a < mc; ++a)
        for (std::size_t b = 0; b < mc; ++b) ty[a] += t(a, b) * std::conj(x[b]);
      for (std::size_t col = 0; col < d; ++col) {
        C overlap{};
        for (std::size_t a = 0; a < mc; ++a) overlap += std::conj(basis(a, col)) * ty[a];
        for (std::size_t a = 0; a < mc; ++a) y[a] += basis(a, col) * overlap;
      }
      double ynorm = 0.0;
      for (const C& v : y) ynorm += std::norm(v);
      ynorm = std::sqrt(ynorm);
      // Part of Theta v outside the whole cluster subspace, computed directly in the
      // original representation (1 - |y|^2 would be roundoff-limited to ~1e-8 after the sqrt).
      {
        Vec vx(n_orig, C{});
        for (std::size_t r = 0; r < n_orig; ++r)
          for (std::size_t a = 0; a < mc; ++a) vx[r] += c_matrix(r, k0 + a) * x[a];
        Vec tv = theta_of(vx);
        const Vec s_tv = metric(tv);
        for (std::size_t a = 0; a < mc; ++a) {
          C o{};
          for (std::size_t r = 0; r < n_orig; ++r) o += std::conj(c_matrix(r, k0 + a)) * s_tv[r];
          for (std::size_t r = 0; r < n_orig; ++r) tv[r] -= o * c_matrix(r, k0 + a);
        }
        const double res2 = std::real(metricInner(metric, tv, tv));
        local.max_invariance_residual = std::max(local.max_invariance_residual, std::sqrt(std::max(0.0, res2)));
      }
      if (ynorm < 1e-8) throw std::runtime_error("fixKramersPairing: cluster is not time-reversal invariant");
      // Orthogonalize against x (Theta^2 = -1 makes it exact up to noise), normalize.
      C xy{};
      for (std::size_t a = 0; a < mc; ++a) xy += std::conj(x[a]) * y[a];
      for (std::size_t a = 0; a < mc; ++a) y[a] -= xy * x[a];
      double n2 = 0.0;
      for (const C& v : y) n2 += std::norm(v);
      n2 = std::sqrt(n2);
      for (C& v : y) v /= n2;
      for (std::size_t a = 0; a < mc; ++a) {
        new_coeff(a, 2 * q) = x[a];
        new_coeff(a, 2 * q + 1) = y[a];
      }
      if (d <= 2) break;
      // New remaining subspace: span(B) minus span(x, y): diagonalize the d x d projector
      // I - xb xb^dagger - yb yb^dagger (coordinates in B) and keep its eigenvalue-1 vectors.
      Vec xb(d), yb(d);
      for (std::size_t col = 0; col < d; ++col) {
        C ox{}, oy{};
        for (std::size_t a = 0; a < mc; ++a) {
          ox += std::conj(basis(a, col)) * x[a];
          oy += std::conj(basis(a, col)) * y[a];
        }
        xb[col] = ox;
        yb[col] = oy;
      }
      Matrix<C> proj(d, d, C{});
      for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
          proj(i, j) = (i == j ? 1.0 : 0.0) - xb[i] * std::conj(xb[j]) - yb[i] * std::conj(yb[j]);
      const HermitianEigenResult eig = diagonalizeHermitian(proj);
      // Eigenvalues ascending: the (d-2) largest are ~1, the two smallest ~0.
      Matrix<C> next(mc, d - 2, C{});
      for (std::size_t k = 0; k < d - 2; ++k) {
        const std::size_t e = k + 2;
        for (std::size_t a = 0; a < mc; ++a) {
          C sum{};
          for (std::size_t col = 0; col < d; ++col) sum += basis(a, col) * eig.eigenvectors(col, e);
          next(a, k) = sum;
        }
      }
      basis = next;
      d -= 2;
    }

    // Back to the original representation.
    for (std::size_t j = 0; j < mc; ++j) {
      Vec col(n_orig, C{});
      for (std::size_t a = 0; a < mc; ++a)
        for (std::size_t r = 0; r < n_orig; ++r) col[r] += c_matrix(r, k0 + a) * new_coeff(a, j);
      // change relative to the old column, modulo a phase
      const C ov = metricInner(metric, columnOf(c_matrix, k0 + j), col);
      local.max_column_change =
          std::max(local.max_column_change, std::sqrt(std::max(0.0, 1.0 - std::norm(ov))));
      for (std::size_t r = 0; r < n_orig; ++r) result(r, k0 + j) = col[r];
    }
    pair = last + 1;
  }

  // Final, GLOBAL enforcement: the cluster step leaves y = the normalized projection of Theta x
  // onto the cluster subspace, i.e. Theta x only up to that subspace's invariance residual (the
  // SCF-noise-level time-reversal asymmetry of the eigenvectors, ~1e-8). Make the structure
  // exact: keep every EVEN column (S-orthogonalized against the earlier columns), set its odd
  // partner to Theta(even) exactly. Because the earlier set is then exactly Theta-invariant,
  // Theta(even) is automatically orthogonal to it, so the set stays orthonormal.
  {
    Matrix<C> exact = result;
    for (std::size_t pr = 0; pr < n_pairs; ++pr) {
      Vec v = columnOf(result, 2 * pr);
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t j = 0; j < 2 * pr; ++j) {
          const Vec cj = columnOf(exact, j);
          const C o = metricInner(metric, cj, v);
          for (std::size_t r = 0; r < n_orig; ++r) v[r] -= o * cj[r];
        }
      }
      double nrm = std::sqrt(std::max(0.0, std::real(metricInner(metric, v, v))));
      if (nrm < 1e-6) {
        throw std::runtime_error(
            "fixKramersPairing: an even column lies in the span of the earlier pairs -- "
            "a degenerate cluster was not detected (raise cluster_tolerance)");
      }
      for (C& x : v) x /= nrm;
      const Vec w = theta_of(v);
      double change = 0.0;
      for (std::size_t r = 0; r < n_orig; ++r) {
        change = std::max(change, std::abs(v[r] - result(r, 2 * pr)));
        change = std::max(change, std::abs(w[r] - result(r, 2 * pr + 1)));
        exact(r, 2 * pr) = v[r];
        exact(r, 2 * pr + 1) = w[r];
      }
      local.max_enforcement_change = std::max(local.max_enforcement_change, change);
    }
    result = exact;
  }
  local.partner_error_after = partnerError(result, theta_of, metric);
  if (report) *report = local;
  return result;
}

}  // namespace

Matrix<C> fixKramersPairingLarge(const Matrix<C>& c_matrix, const std::vector<double>& energies,
                                 const Matrix<double>& s_large, double cluster_tolerance,
                                 KramersPairingReport* report) {
  const std::size_t n_large = s_large.rows();
  const std::size_t n_orig = 2 * n_large;
  if (s_large.cols() != n_large || c_matrix.rows() != n_orig) {
    throw std::runtime_error("fixKramersPairingLarge: inconsistent input dimensions");
  }
  // Theta psi: (Theta psi)_beta = conj(psi_alpha), (Theta psi)_alpha = -conj(psi_beta).
  const VecFn theta_of = [n_large](const Vec& v) {
    Vec out(2 * n_large);
    for (std::size_t i = 0; i < n_large; ++i) {
      out[n_large + i] = std::conj(v[i]);
      out[i] = -std::conj(v[n_large + i]);
    }
    return out;
  };
  // Block-diagonal S = diag(s_large, s_large).
  const VecFn metric = [&s_large, n_large](const Vec& v) {
    Vec out(2 * n_large, C{});
    for (std::size_t i = 0; i < n_large; ++i) {
      C a{}, b{};
      for (std::size_t j = 0; j < n_large; ++j) {
        a += s_large(i, j) * v[j];
        b += s_large(i, j) * v[n_large + j];
      }
      out[i] = a;
      out[n_large + i] = b;
    }
    return out;
  };
  return pairingCore(c_matrix, energies, theta_of, metric, cluster_tolerance, report);
}

Matrix<C> fixKramersPairingOrthonormal(const Matrix<C>& c, const std::vector<double>& energies,
                                       const Matrix<C>& theta_matrix, double cluster_tolerance,
                                       KramersPairingReport* report) {
  const std::size_t n = c.rows();
  if (theta_matrix.rows() != n || theta_matrix.cols() != n) {
    throw std::runtime_error("fixKramersPairingOrthonormal: inconsistent input dimensions");
  }
  const VecFn theta_of = [&theta_matrix, n](const Vec& v) {
    Vec out(n, C{});
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t b = 0; b < n; ++b) out[a] += theta_matrix(a, b) * std::conj(v[b]);
    return out;
  };
  const VecFn metric = [](const Vec& v) { return v; };
  return pairingCore(c, energies, theta_of, metric, cluster_tolerance, report);
}

double kramersOneBodyDeviation(const Matrix<C>& h, double* scale) {
  double dev = 0.0, sc = 0.0;
  for (std::size_t p = 0; p < h.rows(); ++p) {
    for (std::size_t q = 0; q < h.cols(); ++q) {
      const double s = ((p % 2 == 0) ? 1.0 : -1.0) * ((q % 2 == 0) ? 1.0 : -1.0);
      sc = std::max(sc, std::abs(h(p, q)));
      dev = std::max(dev, std::abs(h(p ^ 1, q ^ 1) - s * std::conj(h(p, q))));
    }
  }
  if (scale) *scale = sc;
  return dev;
}

double kramersAoOneBodyDeviation(const Matrix<C>& h, double* scale) {
  if (h.rows() != h.cols() || h.rows() % 2 != 0) {
    throw std::runtime_error("kramersAoOneBodyDeviation: expected a square matrix of even dimension");
  }
  const std::size_t nl = h.rows() / 2;
  double dev = 0.0, sc = 0.0;
  for (std::size_t i = 0; i < nl; ++i)
    for (std::size_t j = 0; j < nl; ++j) {
      dev = std::max(dev, std::abs(h(nl + i, nl + j) - std::conj(h(i, j))));
      dev = std::max(dev, std::abs(h(nl + i, j) + std::conj(h(i, nl + j))));
    }
  for (std::size_t i = 0; i < h.rows(); ++i)
    for (std::size_t j = 0; j < h.cols(); ++j) sc = std::max(sc, std::abs(h(i, j)));
  if (scale) *scale = sc;
  return dev;
}

double kramersTwoBodyDeviation(const Tensor4<C>& eri, double* scale) {
  const std::size_t n = eri.dim0();
  const auto sgn = [](std::size_t i) { return (i % 2 == 0) ? 1.0 : -1.0; };
  double dev = 0.0, sc = 0.0;
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) {
          const C v = eri(a, b, c, d);
          sc = std::max(sc, std::abs(v));
          dev = std::max(dev, std::abs(eri(a ^ 1, b ^ 1, c ^ 1, d ^ 1) - sgn(a) * sgn(b) * sgn(c) * sgn(d) * std::conj(v)));
        }
  if (scale) *scale = sc;
  return dev;
}

}  // namespace rerdmft
