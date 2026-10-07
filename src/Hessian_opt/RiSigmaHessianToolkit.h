#ifndef RERDMFT_RISIGMAHESSIANTOOLKIT_H
#define RERDMFT_RISIGMAHESSIANTOOLKIT_H

#include <cstddef>

#include "Matrix.h"

namespace rerdmft {

// The shared RI-native building blocks every functional's sigma-vector orbital-rotation
// Hessian-vector product reduces to (see [[project-ri-hessian-neo-design]]'s Stage 1d/2d --
// proven in Python for BOTH JK_only's rawJkOnlyG and PNOF/pCCD's shared
// rawHartreeExchangeHessianTerm/rawL1L2HessianTerm; this is the C++ port of that shared piece,
// not functional-specific). Given an RI tensor `B(P,p,q)` (P = auxiliary index, p,q = the SAME
// orbital space the Hessian-vector product acts on -- for NON_REL's spin-orbital Hessian this
// means the caller has already embedded the spin-selection rule into B as a block-diagonal
// matrix, since the toolkit itself has no spin-selection logic) and the antisymmetric kappa(v)
// matrix `K` (ALWAYS real, even when `B`/`T` is complex -- kappa(v)'s real/imaginary parts are
// parametrized by real t/y, see JkOnlySigmaHessianVector.h's joint-case comment), builds:
//
//   KB1[P](a,b) := sum_s K(a,s)*B(P,s,b)  == (K @ B[P])(a,b)
//   KB2[P](a,b) := sum_r B(P,a,r)*K(r,b)  == (B[P] @ K)(a,b)
//   Bdiag[P,t]   := B(P,t,t)
//   KB1diag[P,r] := KB1[P](r,r),  KB2diag[P,s] := KB2[P](s,s)
//
// `T` is `B`'s own scalar type -- `double` for NON_REL's real RI tensor, `complex<double>` for
// C4_SPINOR/X2C's `RiMoEri` (whose own `eri(A,B,C,D)=sum_P B(P,A,C)*B(P,B,D)` reconstruction is
// UNCONJUGATED, i.e. the same plain algebraic identity as the real case -- confirmed directly
// from `RiMoEri::operator()`'s own code before trusting this generalizes; `K` multiplies `B`
// un-conjugated throughout, matching that). Every term of every functional's bare Hessian
// coefficient reduces to a contraction of `B` against these objects, substituting
// eri(A,B,C,D) = sum_P B(P,A,C)*B(P,B,D) and resumming over the pair index FIRST (see the project
// memory for the full per-term derivation) -- so this is the ONE piece of RI-specific machinery
// every functional needs, built once per Hessian-vector product call.
//
// COST (see [[feedback-target-large-systems]] -- storage and compute stated separately, not as
// one exponent): `B`/`KB1`/`KB2` are each O(n_aux*n^2) STORAGE (same class as the already-shipped
// `RiMoEri`/`buildRiMoEri`'s own tensor); building `KB1`/`KB2` costs O(n_aux*n^3) FLOPs (one n x n
// matrix product per auxiliary index). `Bdiag`/`KB1diag`/`KB2diag` are O(n_aux*n), free by
// comparison. This toolkit does NOT itself allocate any O(n^3)-or-larger object -- those
// (per-term intermediates like `M_A`/`V_h`) are each functional's own responsibility, built and
// freed one at a time.
template <typename T>
struct RiSigmaHessianToolkit {
  std::size_t n = 0;
  std::size_t n_aux = 0;
  Matrix<T> b;         // (n_aux, n*n), b(P, p*n+q) = B(P,p,q)
  Matrix<T> kb1;       // (n_aux, n*n), kb1(P, a*n+b) = KB1[P](a,b)
  Matrix<T> kb2;       // (n_aux, n*n), kb2(P, a*n+b) = KB2[P](a,b)
  std::vector<T> bdiag;     // size n_aux*n, bdiag[P*n+t] = Bdiag[P,t]
  std::vector<T> kb1diag;   // size n_aux*n
  std::vector<T> kb2diag;   // size n_aux*n

  T B(std::size_t P, std::size_t p, std::size_t q) const { return b(P, p * n + q); }
  T KB1(std::size_t P, std::size_t a, std::size_t c) const { return kb1(P, a * n + c); }
  T KB2(std::size_t P, std::size_t a, std::size_t c) const { return kb2(P, a * n + c); }
  T Bdiag(std::size_t P, std::size_t t) const { return bdiag[P * n + t]; }
  T KB1diag(std::size_t P, std::size_t r) const { return kb1diag[P * n + r]; }
  T KB2diag(std::size_t P, std::size_t s) const { return kb2diag[P * n + s]; }
};

// `b` must be (n_aux, n*n), `b(P, p*n+q) == B(P,p,q)`; `K` must be n x n and REAL (the
// antisymmetric or symmetric kappa(v) matrix, even when `T` is complex). O(n_aux*n^3) FLOPs,
// O(n_aux*n^2) additional storage (kb1+kb2).
template <typename T>
RiSigmaHessianToolkit<T> buildRiSigmaHessianToolkit(std::size_t n, const Matrix<T>& b,
                                                     const Matrix<double>& K);

}  // namespace rerdmft

#endif  // RERDMFT_RISIGMAHESSIANTOOLKIT_H
