#ifndef RERDMFT_X2C_DHF_X2C_FOCKMATRIXRI_H
#define RERDMFT_X2C_DHF_X2C_FOCKMATRIXRI_H

#include <complex>
#include <cstddef>

#include "Matrix.h"

namespace rerdmft {

// USE_RI counterpart of X2C_FockMatrix.h's x2cFockMatrix: builds the (approximate) X2C-HF Fock
// matrix in the Large-component spin-orbital basis [Large-alpha, Large-beta] from a pre-built RI
// (resolution-of-identity) 3-center tensor, instead of the dense/packed/Cholesky two-electron
// representation. Only ONE spatial AO sector exists here (X2C has no small component at all --
// it was already decoupled away into h_x2c; same reason MOLGW's own setup_exchange_ri_x2c_1/_2,
// read directly this session, never needs one either -- see project memory
// project-ri-pauto-kr-validation), so this is structurally a 2-block (not 4-block) special case of
// UKB/UkbFockMatrixRi.h's own riFockTwoElectronDirect -- same derivation, same occupied-orbital
// recovery discipline (reuse the SCF loop's OWN `x_large`/`overlap`, exactly as they are passed to
// runX2CHartreeFockScf -- never build a fresh orthogonalization of the AO overlap; see
// UkbFockMatrixRi.h's header comment for why that was the one real lesson from the C4_SPINOR
// case), just written out directly for 2 blocks since there is no Small sector to carry.
//
//   J(p,r)      = sum_P (pr|P) * X_P,  X_P = sum_{q,s} (qs|P) * Re(P_total(s,q))  (P_total = P_aa+P_bb)
//   K_fu(p,r)   = -sum_i conj(tmp_f(i,p)) * tmp_u(i,r),  tmp_f(i,p) = sum_q c_t_f(i,q) * (pq|P)
// for f,u in {alpha, beta}, with c_t_f(i,q) = conj(C_occ(offset[f]+q, i)) * sqrt(occupation_i),
// C_occ recovered from `density_matrix` via `eri.x_large`/`eri.overlap` (see UkbFockMatrixRi.h's
// header comment for the full derivation of why this, not a fresh orthogonalization, is required).
struct RiX2cEriSource {
  std::size_t nl = 0;
  Matrix<double> eri3_L;  // (n_aux, nl*nl), real -- see AO_ints/ThreeCenterIntegrals.h's convention
  // The SAME x_large/overlap already passed to runX2CHartreeFockScf (X2C_HF.h) -- reused, never
  // rebuilt, for the occupied-orbital recovery above.
  Matrix<std::complex<double>> x_large;
  Matrix<std::complex<double>> overlap;
};

Matrix<std::complex<double>> x2cFockMatrix(const Matrix<std::complex<double>>& h_x2c,
                                            const RiX2cEriSource& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_FOCKMATRIXRI_H
