#include "JkOnlyHessian.h"

#include "CholeskyEri.h"
#include "SymmetricEri.h"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rerdmft {

namespace {

// The "bare" second-derivative coefficient G_pq,rs = d C_pq / d kappa_rs
// -- see JkOnlyHessian.h for the derivation and the formula.
template <typename T, typename Eri>
T rawJkOnlyG(const Matrix<T>& h, const Eri& eri, const std::vector<double>& occupations,
             const Matrix<double>& hc, const Matrix<double>& xc, std::size_t n, std::size_t p,
             std::size_t q, std::size_t r, std::size_t s) {
  T term{};

  if (q == r) term += T(occupations[p] - occupations[q]) * h(s, p);
  if (s == p) term -= T(occupations[p] - occupations[q]) * h(q, r);

  if (q == r) {
    T sum{};
    for (std::size_t t = 0; t < n; ++t) {
      sum += T(hc(p, t) - hc(q, t)) * eri(t, s, t, p) - T(xc(p, t) - xc(q, t)) * eri(s, t, t, p);
    }
    term += sum;
  }
  if (s == p) {
    T sum{};
    for (std::size_t t = 0; t < n; ++t) {
      sum += -T(hc(p, t) - hc(q, t)) * eri(t, q, t, r) + T(xc(p, t) - xc(q, t)) * eri(q, t, t, r);
    }
    term += sum;
  }

  term += T(hc(p, r) - hc(q, r) - hc(p, s) + hc(q, s)) * eri(s, q, r, p);
  term += T(xc(p, s) - xc(p, r) - xc(q, s) + xc(q, r)) * eri(q, s, r, p);
  return term;
}

template <typename T, typename Eri>
void checkJkOnlyArgs(const Matrix<T>& h, const Eri& eri, const std::vector<double>& occ,
                      const Matrix<double>& hc, const Matrix<double>& xc, const char* caller) {
  const std::size_t n = h.rows();
  if (h.cols() != n || eri.dim0() != n || eri.dim1() != n || eri.dim2() != n ||
      eri.dim3() != n || occ.size() != n || hc.rows() != n || hc.cols() != n ||
      xc.rows() != n || xc.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": inconsistent input dimensions");
  }
}

}  // namespace

template <typename T, typename Eri>
T jkOnlyHessianElement(const Matrix<T>& h, const Eri& eri,
                        const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                        const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q,
                        std::size_t r, std::size_t s) {
  const std::size_t n = h.rows();
  const char* caller = "jkOnlyHessianElement";
  if (h.cols() != n) throw std::runtime_error(std::string(caller) + ": h is not square");
  if (eri.dim0() != n || eri.dim1() != n || eri.dim2() != n || eri.dim3() != n) {
    throw std::runtime_error(std::string(caller) + ": eri dimensions inconsistent with h");
  }
  if (occupations.size() != n) {
    throw std::runtime_error(std::string(caller) + ": occupations size inconsistent with h");
  }
  if (two_rdm_h.rows() != n || two_rdm_h.cols() != n || two_rdm_x.rows() != n ||
      two_rdm_x.cols() != n) {
    throw std::runtime_error(std::string(caller) + ": coupling matrices inconsistent with h");
  }
  if (p >= n || q >= n || r >= n || s >= n) {
    throw std::runtime_error(std::string(caller) + ": index out of range");
  }

  return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, p, q, r, s) -
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, p, q, s, r) -
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, q, p, r, s) +
         rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, q, p, s, r);
}

template <typename T>
Matrix<T> jkOnlyHessianMatrix(const Matrix<T>& h, const Tensor4<T>& eri,
                               const std::vector<double>& occupations,
                               const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                               const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  const std::size_t n_pairs = pair_indices.size();
  Matrix<T> hess(n_pairs, n_pairs, T{});
#pragma omp parallel for collapse(2)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = 0; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      hess(big_i, big_j) = jkOnlyHessianElement(h, eri, occupations, two_rdm_h, two_rdm_x, p, q, r, s);
    }
  }
  return hess;
}

template Matrix<double> jkOnlyHessianMatrix(
    const Matrix<double>&, const Tensor4<double>&, const std::vector<double>&,
    const Matrix<double>&, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);
template Matrix<std::complex<double>> jkOnlyHessianMatrix(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&);

template double jkOnlyHessianElement(const Matrix<double>& h, const Tensor4<double>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, std::size_t p,
                                      std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> jkOnlyHessianElement(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q, std::size_t r, std::size_t s);
template double jkOnlyHessianElement(const Matrix<double>& h, const CholeskyEri<double>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, std::size_t p,
                                      std::size_t q, std::size_t r, std::size_t s);
template double jkOnlyHessianElement(const Matrix<double>& h, const SymmetricEri<double>& eri,
                                      const std::vector<double>& occupations,
                                      const Matrix<double>& two_rdm_h,
                                      const Matrix<double>& two_rdm_x, std::size_t p,
                                      std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> jkOnlyHessianElement(
    const Matrix<std::complex<double>>& h, const CholeskyEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q, std::size_t r, std::size_t s);
template std::complex<double> jkOnlyHessianElement(
    const Matrix<std::complex<double>>& h, const SymmetricEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x, std::size_t p, std::size_t q, std::size_t r, std::size_t s);

template <typename T, typename Eri>
T jkOnlyHessianElementImag(const Matrix<T>& h, const Eri& eri,
                            const std::vector<double>& occupations,
                            const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                            std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  checkJkOnlyArgs(h, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlyHessianElementImag");
  const std::size_t n = h.rows();
  auto g = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, a, b, c, d);
  };
  return -(g(p, q, r, s) + g(p, q, s, r) + g(q, p, r, s) + g(q, p, s, r));
}

template <typename T, typename Eri>
T jkOnlyHessianElementMixed(const Matrix<T>& h, const Eri& eri,
                             const std::vector<double>& occupations,
                             const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x,
                             std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  checkJkOnlyArgs(h, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlyHessianElementMixed");
  const std::size_t n = h.rows();
  auto g = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, a, b, c, d);
  };
  return T(0.0, 1.0) * (g(r, s, p, q) + g(r, s, q, p) - g(s, r, p, q) - g(s, r, q, p));
}

Matrix<double> jkOnlyJointHessianMatrix(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  using C = std::complex<double>;
  checkJkOnlyArgs(h, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlyJointHessianMatrix");
  const std::size_t n = h.rows();
  const std::size_t n_pairs = pair_indices.size();
  Matrix<double> hess(2 * n_pairs, 2 * n_pairs, 0.0);
  const C im(0.0, 1.0);
  // Each unordered (I,J), I <= J, owns four disjoint output positions.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    for (std::size_t big_j = big_i; big_j < n_pairs; ++big_j) {
      const auto& [p, q] = pair_indices[big_i];
      const auto& [r, s] = pair_indices[big_j];
      auto g = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
        return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, a, b, c, d);
      };
      const C g_pq_rs = g(p, q, r, s), g_pq_sr = g(p, q, s, r);
      const C g_qp_rs = g(q, p, r, s), g_qp_sr = g(q, p, s, r);
      const C g_rs_pq = g(r, s, p, q), g_rs_qp = g(r, s, q, p);
      const C g_sr_pq = g(s, r, p, q), g_sr_qp = g(s, r, q, p);

      const double tt_ij = (g_pq_rs - g_pq_sr - g_qp_rs + g_qp_sr).real();
      const double tt_ji = (g_rs_pq - g_rs_qp - g_sr_pq + g_sr_qp).real();
      const double yy_ij = -(g_pq_rs + g_pq_sr + g_qp_rs + g_qp_sr).real();
      const double yy_ji = -(g_rs_pq + g_rs_qp + g_sr_pq + g_sr_qp).real();
      const double ty_ij = 0.5 * (im * (g_pq_rs + g_pq_sr - g_qp_rs - g_qp_sr) +
                                   im * (g_rs_pq - g_rs_qp + g_sr_pq - g_sr_qp)).real();
      const double ty_ji = 0.5 * (im * (g_rs_pq + g_rs_qp - g_sr_pq - g_sr_qp) +
                                   im * (g_pq_rs - g_pq_sr + g_qp_rs - g_qp_sr)).real();

      hess(big_i, big_j) = hess(big_j, big_i) = 0.5 * (tt_ij + tt_ji);
      hess(n_pairs + big_i, n_pairs + big_j) = hess(n_pairs + big_j, n_pairs + big_i) =
          0.5 * (yy_ij + yy_ji);
      hess(big_i, n_pairs + big_j) = hess(n_pairs + big_j, big_i) = ty_ij;
      hess(big_j, n_pairs + big_i) = hess(n_pairs + big_i, big_j) = ty_ji;
    }
  }
  return hess;
}

template std::complex<double> jkOnlyHessianElementImag(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);
template std::complex<double> jkOnlyHessianElementMixed(
    const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);
template std::complex<double> jkOnlyHessianElementImag(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);
template std::complex<double> jkOnlyHessianElementImag(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);
template std::complex<double> jkOnlyHessianElementMixed(
    const Matrix<std::complex<double>>&, const CholeskyEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);
template std::complex<double> jkOnlyHessianElementMixed(
    const Matrix<std::complex<double>>&, const SymmetricEri<std::complex<double>>&,
    const std::vector<double>&, const Matrix<double>&, const Matrix<double>&, std::size_t,
    std::size_t, std::size_t, std::size_t);

template <typename Eri>
std::vector<double> jkOnlyJointHessianVector(
    const Matrix<std::complex<double>>& h, const Eri& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v) {
  using C = std::complex<double>;
  checkJkOnlyArgs(h, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlyJointHessianVector");
  const std::size_t n = h.rows();
  const std::size_t n_pairs = pair_indices.size();
  if (v.size() != 2 * n_pairs) {
    throw std::runtime_error("jkOnlyJointHessianVector: v has the wrong size");
  }
  std::vector<double> w(2 * n_pairs, 0.0);
  const C im(0.0, 1.0);
#pragma omp parallel
  {
    std::vector<double> w_local(2 * n_pairs, 0.0);
#pragma omp for schedule(dynamic)
    for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
      for (std::size_t big_j = big_i; big_j < n_pairs; ++big_j) {
        const auto& [p, q] = pair_indices[big_i];
        const auto& [r, s] = pair_indices[big_j];
        auto g = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
          return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, a, b, c, d);
        };
        const C g_pq_rs = g(p, q, r, s), g_pq_sr = g(p, q, s, r);
        const C g_qp_rs = g(q, p, r, s), g_qp_sr = g(q, p, s, r);
        const C g_rs_pq = g(r, s, p, q), g_rs_qp = g(r, s, q, p);
        const C g_sr_pq = g(s, r, p, q), g_sr_qp = g(s, r, q, p);

        const double tt_ij = (g_pq_rs - g_pq_sr - g_qp_rs + g_qp_sr).real();
        const double tt_ji = (g_rs_pq - g_rs_qp - g_sr_pq + g_sr_qp).real();
        const double yy_ij = -(g_pq_rs + g_pq_sr + g_qp_rs + g_qp_sr).real();
        const double yy_ji = -(g_rs_pq + g_rs_qp + g_sr_pq + g_sr_qp).real();
        const double ty_ij = 0.5 * (im * (g_pq_rs + g_pq_sr - g_qp_rs - g_qp_sr) +
                                     im * (g_rs_pq - g_rs_qp + g_sr_pq - g_sr_qp)).real();
        const double ty_ji = 0.5 * (im * (g_rs_pq + g_rs_qp - g_sr_pq - g_sr_qp) +
                                     im * (g_pq_rs - g_pq_sr + g_qp_rs - g_qp_sr)).real();
        const double tt = 0.5 * (tt_ij + tt_ji);
        const double yy = 0.5 * (yy_ij + yy_ji);

        if (big_i == big_j) {
          w_local[big_i] += tt * v[big_i] + ty_ij * v[n_pairs + big_i];
          w_local[n_pairs + big_i] += ty_ij * v[big_i] + yy * v[n_pairs + big_i];
        } else {
          w_local[big_i] += tt * v[big_j] + ty_ij * v[n_pairs + big_j];
          w_local[big_j] += tt * v[big_i] + ty_ji * v[n_pairs + big_i];
          w_local[n_pairs + big_i] += ty_ji * v[big_j] + yy * v[n_pairs + big_j];
          w_local[n_pairs + big_j] += ty_ij * v[big_i] + yy * v[n_pairs + big_i];
        }
      }
    }
#pragma omp critical
    for (std::size_t k = 0; k < w.size(); ++k) w[k] += w_local[k];
  }
  return w;
}

template std::vector<double> jkOnlyJointHessianVector(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);
template std::vector<double> jkOnlyJointHessianVector(
    const Matrix<std::complex<double>>& h, const CholeskyEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);
template std::vector<double> jkOnlyJointHessianVector(
    const Matrix<std::complex<double>>& h, const SymmetricEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices,
    const std::vector<double>& v);

template <typename Eri>
std::vector<double> jkOnlyJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const Eri& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices) {
  using C = std::complex<double>;
  checkJkOnlyArgs(h, eri, occupations, two_rdm_h, two_rdm_x, "jkOnlyJointHessianDiagonal");
  const std::size_t n = h.rows();
  const std::size_t n_pairs = pair_indices.size();
  std::vector<double> d(2 * n_pairs, 0.0);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t big_i = 0; big_i < n_pairs; ++big_i) {
    const auto& [p, q] = pair_indices[big_i];
    auto g = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t e) {
      return rawJkOnlyG(h, eri, occupations, two_rdm_h, two_rdm_x, n, a, b, c, e);
    };
    const C g_pq_pq = g(p, q, p, q), g_pq_qp = g(p, q, q, p);
    const C g_qp_pq = g(q, p, p, q), g_qp_qp = g(q, p, q, p);
    d[big_i] = (g_pq_pq - g_pq_qp - g_qp_pq + g_qp_qp).real();
    d[n_pairs + big_i] = -(g_pq_pq + g_pq_qp + g_qp_pq + g_qp_qp).real();
  }
  return d;
}

template std::vector<double> jkOnlyJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const Tensor4<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);
template std::vector<double> jkOnlyJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const CholeskyEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);
template std::vector<double> jkOnlyJointHessianDiagonal(
    const Matrix<std::complex<double>>& h, const SymmetricEri<std::complex<double>>& eri,
    const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
    const Matrix<double>& two_rdm_x,
    const std::vector<std::pair<std::size_t, std::size_t>>& pair_indices);

}  // namespace rerdmft
