#include "CholeskyEri.h"

#include <algorithm>
#include <stdexcept>

#include "Cholesky_Decomposition.h"

namespace rerdmft {

namespace {

inline double conjugate(double x) { return x; }
inline std::complex<double> conjugate(const std::complex<double>& x) { return std::conj(x); }

}  // namespace

template <typename T>
CholeskyEri<T> CholeskyEri<T>::fromDense(const Tensor4<T>& eri, double threshold, std::size_t max_batch) {
  const std::size_t n = eri.dim0();
  if (eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error("CholeskyEri::fromDense: eri must have four equal dimensions");
  }
  // R(C,A,B,D) = eri(A,B,C,D): Coulomb grouping (C,A),(B,D).
  Tensor4<T> coulomb(n, n, n, n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b)
      for (std::size_t c = 0; c < n; ++c)
        for (std::size_t d = 0; d < n; ++d) coulomb(c, a, b, d) = eri(a, b, c, d);
  return fromVectors(choleskyDecomposeEri(coulomb, threshold, 0, max_batch));
}

template <typename T>
CholeskyEri<T> CholeskyEri<T>::fromVectors(const std::vector<Matrix<T>>& vectors) {
  if (vectors.empty()) throw std::runtime_error("CholeskyEri::fromVectors: no vectors");
  CholeskyEri out;
  out.n_ = vectors.front().rows();
  out.nchol_ = vectors.size();
  out.w_.assign(out.n_ * out.n_ * out.nchol_, T{});
  for (std::size_t l = 0; l < out.nchol_; ++l) {
    const Matrix<T>& v = vectors[l];
    if (v.rows() != out.n_ || v.cols() != out.n_) {
      throw std::runtime_error("CholeskyEri::fromVectors: inconsistent vector dimensions");
    }
    for (std::size_t x = 0; x < out.n_; ++x)
      for (std::size_t y = 0; y < out.n_; ++y) out.w_[(x * out.n_ + y) * out.nchol_ + l] = v(x, y);
  }
  return out;
}

template <typename T>
CholeskyEri<T> CholeskyEri<T>::restricted(std::size_t offset) const {
  if (offset > n_) throw std::runtime_error("CholeskyEri::restricted: offset exceeds the dimension");
  CholeskyEri out;
  out.n_ = n_ - offset;
  out.nchol_ = nchol_;
  out.w_.assign(out.n_ * out.n_ * nchol_, T{});
  for (std::size_t x = 0; x < out.n_; ++x)
    for (std::size_t y = 0; y < out.n_; ++y)
      std::copy(&w_[((x + offset) * n_ + (y + offset)) * nchol_], &w_[((x + offset) * n_ + (y + offset)) * nchol_] + nchol_,
                &out.w_[(x * out.n_ + y) * nchol_]);
  return out;
}

template <typename T>
T CholeskyEri<T>::operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const {
  const T* left = &w_[(c * n_ + a) * nchol_];
  const T* right = &w_[(b * n_ + d) * nchol_];
  T sum{};
  for (std::size_t l = 0; l < nchol_; ++l) sum += left[l] * conjugate(right[l]);
  return sum;
}

template <typename T>
CholeskyEri<T> CholeskyEri<T>::rotated(const Matrix<T>& u) const {
  if (u.rows() != n_ || u.cols() != n_) {
    throw std::runtime_error("CholeskyEri::rotated: U must be n x n");
  }
  CholeskyEri out;
  out.n_ = n_;
  out.nchol_ = nchol_;
  out.w_.assign(w_.size(), T{});
#pragma omp parallel for
  for (std::size_t l = 0; l < nchol_; ++l) {
    // tmp(x',y) = sum_y' V(x',y') conj(U(y',y)); V'(x,y) = sum_x' U(x',x) tmp(x',y).
    Matrix<T> tmp(n_, n_, T{});
    for (std::size_t xp = 0; xp < n_; ++xp)
      for (std::size_t yp = 0; yp < n_; ++yp) {
        const T v = w_[(xp * n_ + yp) * nchol_ + l];
        for (std::size_t y = 0; y < n_; ++y) tmp(xp, y) += v * conjugate(u(yp, y));
      }
    for (std::size_t x = 0; x < n_; ++x)
      for (std::size_t xp = 0; xp < n_; ++xp) {
        const T uxp = u(xp, x);
        for (std::size_t y = 0; y < n_; ++y) out.w_[(x * n_ + y) * nchol_ + l] += uxp * tmp(xp, y);
      }
  }
  return out;
}

template <typename T>
Tensor4<T> CholeskyEri<T>::toDense() const {
  Tensor4<T> dense(n_, n_, n_, n_);
#pragma omp parallel for collapse(2)
  for (std::size_t c = 0; c < n_; ++c)
    for (std::size_t a = 0; a < n_; ++a)
      for (std::size_t b = 0; b < n_; ++b)
        for (std::size_t d = 0; d < n_; ++d) dense(a, b, c, d) = (*this)(a, b, c, d);
  return dense;
}

template class CholeskyEri<double>;
template class CholeskyEri<std::complex<double>>;

}  // namespace rerdmft
