#include "RiSigmaHessianToolkit.h"

#include <complex>
#include <stdexcept>

namespace rerdmft {

template <typename T>
RiSigmaHessianToolkit<T> buildRiSigmaHessianToolkit(std::size_t n, const Matrix<T>& b,
                                                     const Matrix<double>& K) {
  if (b.cols() != n * n) throw std::runtime_error("buildRiSigmaHessianToolkit: b has the wrong shape");
  if (K.rows() != n || K.cols() != n) throw std::runtime_error("buildRiSigmaHessianToolkit: K has the wrong shape");

  RiSigmaHessianToolkit<T> t;
  t.n = n;
  t.n_aux = b.rows();
  t.b = b;
  t.kb1 = Matrix<T>(t.n_aux, n * n, T{});
  t.kb2 = Matrix<T>(t.n_aux, n * n, T{});
  t.bdiag.assign(t.n_aux * n, T{});
  t.kb1diag.assign(t.n_aux * n, T{});
  t.kb2diag.assign(t.n_aux * n, T{});

#pragma omp parallel for
  for (std::size_t P = 0; P < t.n_aux; ++P) {
    for (std::size_t a = 0; a < n; ++a)
      for (std::size_t c = 0; c < n; ++c) {
        T acc1{}, acc2{};
        for (std::size_t s = 0; s < n; ++s) acc1 += T(K(a, s)) * b(P, s * n + c);  // KB1[P](a,c)=sum_s K(a,s)*B(P,s,c)
        for (std::size_t r = 0; r < n; ++r) acc2 += b(P, a * n + r) * T(K(r, c));  // KB2[P](a,c)=sum_r B(P,a,r)*K(r,c)
        t.kb1(P, a * n + c) = acc1;
        t.kb2(P, a * n + c) = acc2;
      }
    for (std::size_t x = 0; x < n; ++x) {
      t.bdiag[P * n + x] = b(P, x * n + x);
      t.kb1diag[P * n + x] = t.kb1(P, x * n + x);
      t.kb2diag[P * n + x] = t.kb2(P, x * n + x);
    }
  }
  return t;
}

template RiSigmaHessianToolkit<double> buildRiSigmaHessianToolkit(std::size_t, const Matrix<double>&,
                                                                   const Matrix<double>&);
template RiSigmaHessianToolkit<std::complex<double>> buildRiSigmaHessianToolkit(
    std::size_t, const Matrix<std::complex<double>>&, const Matrix<double>&);

}  // namespace rerdmft
