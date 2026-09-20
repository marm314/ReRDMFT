#ifndef RERDMFT_UTILS_CHOLESKYERI_H
#define RERDMFT_UTILS_CHOLESKYERI_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// A two-electron integral tensor in physics notation, eri(a,b,c,d) = <ab|cd>, kept as CHOLESKY
// VECTORS of the Coulomb metric instead of as a dense n^4 tensor, with element access on demand.
//
// The PSD (Hermitian, positive semidefinite) matrix is the Coulomb matrix
//   G[(C,A),(B,D)] = (CA|BD) = eri(A,B,C,D)
// over the pair indices x = (C,A), y = (B,D) -- NOT the (AB),(CD) bra-ket grouping that
// Utils/Cholesky_Decomposition.h's basis-transform helpers use. In the AO/spinor basis the
// Coulomb grouping has rank O(n) (Nchol ~ a few times n_AO; for closed-shell spin-orbitals it
// is the SPATIAL rank), whereas the bra-ket grouping is generally close to full rank O(n^2) --
// the reason the MO-transform Cholesky path shows little compression. With
//   G = sum_L V_L conj(V_L)^T   (choleskyDecomposeEri applied to G):
//   eri(a,b,c,d) = sum_L V_L(c,a) conj(V_L(b,d)),
// so V_L(x,y) behaves like the matrix <x|O_L|y> of an operator O_L, and an orbital rotation
// C_new = C U acts on each vector as
//   V'_L = U^T V_L conj(U)       (bra legs conjugated: <a'b'|c'd'> = sum conj(U_aa') conj(U_bb')
//                                 U_cc' U_dd' <ab|cd>, physics notation).
// An orbital rotation therefore costs O(Nchol n^3), an element O(Nchol), and the storage is
// O(Nchol n^2) -- against O(n^5) for a dense rotation and O(n^4) storage. Every function of the
// RDMFT model layer that takes `const Tensor4<T>& eri` is templated on the tensor type and
// accepts this class (it needs only dim0..dim3 and operator()).
//
// The truncation error of the decomposition is set by `threshold` (Cholesky_Decomposition.h);
// rotations do NOT add to it (a unitary change of basis acts exactly on every vector).
template <typename T>
class CholeskyEri {
 public:
  CholeskyEri() = default;

  // Decomposes a dense tensor (which must have the Hermitian symmetry <ab|cd> = conj <dc|ba> and
  // be PSD in the Coulomb grouping, as any physical two-electron tensor is).
  static CholeskyEri fromDense(const Tensor4<T>& eri, double threshold = 1e-10);
  // From vectors V_L(x,y) (each n x n) already in the convention above.
  static CholeskyEri fromVectors(const std::vector<Matrix<T>>& vectors);

  std::size_t dim0() const { return n_; }
  std::size_t dim1() const { return n_; }
  std::size_t dim2() const { return n_; }
  std::size_t dim3() const { return n_; }
  std::size_t nVectors() const { return nchol_; }

  // <ab|cd>, O(Nchol).
  T operator()(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const;

  // The integrals in the rotated basis C_new = C U (U unitary, n x n): V' = U^T V conj(U).
  CholeskyEri rotated(const Matrix<T>& u) const;

  // The dense tensor, O(Nchol n^4) time and O(n^4) memory -- for tests and one-off checks only.
  Tensor4<T> toDense() const;

 private:
  std::size_t n_ = 0;
  std::size_t nchol_ = 0;
  // w_[(x*n + y)*nchol + L] = V_L(x,y): the L-sum of an element is a contiguous dot product.
  std::vector<T> w_;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_CHOLESKYERI_H
