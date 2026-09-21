#include "DIIS.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rerdmft {

namespace {

inline double realOfProduct(double a, double b) { return a * b; }
inline double realOfProduct(const std::complex<double>& a, const std::complex<double>& b) {
  return std::real(std::conj(a) * b);
}

// Solves the dense n x n system m x = rhs by Gaussian elimination with partial pivoting.
// Returns false if the pivot ratio min|pivot| / max|pivot| falls below `tolerance`.
bool solveDense(std::vector<double> m, std::vector<double> rhs, std::size_t n, double tolerance,
                std::vector<double>& x) {
  std::vector<double> pivots(n);
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t best = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(m[i * n + k]) > std::abs(m[best * n + k])) best = i;
    }
    if (best != k) {
      for (std::size_t j = 0; j < n; ++j) std::swap(m[k * n + j], m[best * n + j]);
      std::swap(rhs[k], rhs[best]);
    }
    pivots[k] = std::abs(m[k * n + k]);
    if (pivots[k] == 0.0) return false;
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = m[i * n + k] / m[k * n + k];
      for (std::size_t j = k; j < n; ++j) m[i * n + j] -= f * m[k * n + j];
      rhs[i] -= f * rhs[k];
    }
  }
  const double pmax = *std::max_element(pivots.begin(), pivots.end());
  const double pmin = *std::min_element(pivots.begin(), pivots.end());
  if (!(pmin > tolerance * pmax)) return false;
  x.assign(n, 0.0);
  for (std::size_t ii = n; ii-- > 0;) {
    double s = rhs[ii];
    for (std::size_t j = ii + 1; j < n; ++j) s -= m[ii * n + j] * x[j];
    x[ii] = s / m[ii * n + ii];
  }
  return true;
}

}  // namespace

template <typename T>
Matrix<T> scfCommutatorError(const Matrix<T>& fock, const Matrix<T>& density,
                             const Matrix<T>& overlap) {
  const Matrix<T> fps = fock * (density * overlap);
  const Matrix<T> spf = (overlap * density) * fock;
  Matrix<T> e(fps.rows(), fps.cols());
  for (std::size_t i = 0; i < e.rows() * e.cols(); ++i) e.data()[i] = fps.data()[i] - spf.data()[i];
  return e;
}

template Matrix<double> scfCommutatorError(const Matrix<double>&, const Matrix<double>&,
                                           const Matrix<double>&);
template Matrix<std::complex<double>> scfCommutatorError(const Matrix<std::complex<double>>&,
                                                         const Matrix<std::complex<double>>&,
                                                         const Matrix<std::complex<double>>&);

template <typename T>
Diis<T>::Diis(std::size_t max_vectors) : max_vectors_(max_vectors) {}

template <typename T>
void Diis<T>::clear() {
  errors_.clear();
  values_.clear();
  last_weights_.clear();
}

template <typename T>
Matrix<T> Diis<T>::extrapolate(const Matrix<T>& error, const Matrix<T>& value) {
  last_weights_.clear();
  if (max_vectors_ < 2) return value;
  if (!errors_.empty() && (errors_.front().rows() != error.rows() || errors_.front().cols() != error.cols() ||
                           values_.front().rows() != value.rows() || values_.front().cols() != value.cols())) {
    throw std::runtime_error("Diis::extrapolate: shapes differ from the stored history");
  }
  // Newest first, at most max_vectors_ entries.
  errors_.push_front(error);
  values_.push_front(value);
  while (errors_.size() > max_vectors_) {
    errors_.pop_back();
    values_.pop_back();
  }

  const std::size_t total = errors_.size();
  const std::size_t n_err = error.rows() * error.cols();
  // B_ij = Re <e_i|e_j> over the full history, scaled below.
  std::vector<double> b_full(total * total);
  for (std::size_t i = 0; i < total; ++i) {
    for (std::size_t j = i; j < total; ++j) {
      double s = 0.0;
      const T* ei = errors_[i].data();
      const T* ej = errors_[j].data();
      for (std::size_t k = 0; k < n_err; ++k) s += realOfProduct(ei[k], ej[k]);
      b_full[i * total + j] = b_full[j * total + i] = s;
    }
  }

  // Use the newest `m` pairs; drop the oldest one whenever the system is singular.
  for (std::size_t m = total; m >= 2; --m) {
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) scale = std::max(scale, b_full[i * total + i]);
    if (!(scale > 0.0)) return value;  // zero error: already consistent, nothing to extrapolate
    const std::size_t n = m + 1;
    std::vector<double> a(n * n, 0.0), rhs(n, 0.0), x;
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < m; ++j) a[i * n + j] = b_full[i * total + j] / scale;
    for (std::size_t i = 0; i < m; ++i) a[i * n + m] = a[m * n + i] = -1.0;
    rhs[m] = -1.0;
    if (!solveDense(std::move(a), std::move(rhs), n, 1e-14, x)) continue;

    Matrix<T> result(value.rows(), value.cols(), T{});
    for (std::size_t i = 0; i < m; ++i) {
      const T* vi = values_[i].data();
      T* r = result.data();
      for (std::size_t k = 0; k < result.rows() * result.cols(); ++k) r[k] += x[i] * vi[k];
    }
    last_weights_.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m));
    return result;
  }
  return value;
}

template class Diis<double>;
template class Diis<std::complex<double>>;

}  // namespace rerdmft
