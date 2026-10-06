#ifndef RERDMFT_NONRELFOCKMATRIXRI_H
#define RERDMFT_NONRELFOCKMATRIXRI_H

#include <cstddef>

#include "Matrix.h"

namespace rerdmft {

// USE_RI counterpart of NonRelHartreeFock.h's nonRelFockMatrix: builds the standard (restricted,
// closed-shell) nonrelativistic Fock matrix from a pre-built RI (resolution-of-identity) 3-center
// tensor, instead of the dense/packed/Cholesky two-electron representation. The simplest of the
// three USE_RI cases -- real (T=double, no complex/conjugation anywhere), a SINGLE spatial AO
// sector (no small component, no spin-block structure at all: closed-shell RHF folds both spin
// channels into one density/Fock matrix with the standard factor of 2 for double occupancy), so
// this is a direct real-arithmetic, single-block special case of UKB/UkbFockMatrixRi.h's own
// riFockTwoElectronDirect -- same occupied-orbital recovery discipline (reuse the SCF loop's OWN
// `x_large`/`overlap`, never build a fresh orthogonalization of the AO overlap).
//
//   P = 2 * sum_i C_i C_i^T  (C^T S C = I, each column DOUBLY occupied) -- D := P/2 is the genuine
//   idempotent object (D S D = D); recovered via X^dagger(S P S)X / 2 = Y Y^T (X = eri.x_large,
//   S = eri.overlap), eigenvalue ~1 marks occupied, C_occ = X * Y_occ. c_t(i,q) = sqrt(2) *
//   C_occ(q,i) (the sqrt(2) folds the factor-of-2 occupancy directly into the half-transform, so
//   the resulting K already matches nonRelFockMatrix's OWN convention, which reads the FULL
//   (factor-of-2-included) P -- the caller still applies the SAME external (1/2) on K).
//
//   J(p,r) = sum_P (pr|P) * X_P,  X_P = sum_{q,s} (qs|P) * P(s,q)
//   K(p,r) = sum_i sum_P tmp(i,p,P) * tmp(i,r,P),  tmp(i,p,P) = sum_q c_t(i,q) * (pq|P)
//   F = H_core + J - (1/2) K   (the (1/2) applied by the CALLER, same as the dense path)
struct RiNonRelEriSource {
  std::size_t nl = 0;
  Matrix<double> eri3_L;  // (n_aux, nl*nl), real -- AO_ints/ThreeCenterIntegrals.h's convention
  // The SAME x_large_cart/s_large_cart already used by the dense NON_REL SCF loop -- reused, never
  // rebuilt, for the occupied-orbital recovery above.
  Matrix<double> x_large;
  Matrix<double> overlap;
};

Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const RiNonRelEriSource& eri,
                                 const Matrix<double>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_NONRELFOCKMATRIXRI_H
