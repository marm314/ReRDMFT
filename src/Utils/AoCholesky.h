#ifndef RERDMFT_UTILS_AOCHOLESKY_H
#define RERDMFT_UTILS_AOCHOLESKY_H

#include <complex>
#include <cstddef>
#include <vector>

#include "CholeskyEri.h"
#include "Cholesky_Decomposition.h"
#include "Matrix.h"

namespace rerdmft {

class PackedTwoElectronTensor;

// The AO-basis two-electron integrals as Cholesky vectors, (pq|rs) = sum_L B_L(p,q) B_L(r,s) in
// CHEMIST notation with real symmetric n x n vectors B_L (n = number of Large AO functions). Built once
// from the packed AO tensor; with CHOLESKY TRUE this is all the program keeps of the AO integrals:
// the NON_REL and X2C SCF Fock matrices, and the MO-basis vectors used by every later stage, are
// generated from it (the dense/packed AO integrals are only needed again under DEBUG, for the
// dense-vs-Cholesky checks).
struct AoCholesky {
  std::size_t n = 0;
  std::vector<Matrix<double>> vectors;

  std::size_t nVectors() const { return vectors.size(); }
  // Decomposes the packed AO tensor with choleskyDecomposeEriChecked (batch retry, sampled
  // reconstruction check against `eri`).
  static AoCholesky fromPacked(const PackedTwoElectronTensor& eri, double threshold,
                               CholeskyCheckReport* report = nullptr);
};

// Fock matrices from the vectors (same definitions as NonRelHartreeFock.h's nonRelFockMatrix and
// X2C_FockMatrix.h's x2cFockMatrix, O(N_chol n^3) instead of O(n^4)):
//   NON_REL:  F = h + J - K/2,  J = sum_L B_L tr(B_L P),  K = sum_L B_L P B_L         (closed-shell P)
//   X2C:      F = h + J - K over the [alpha; beta] blocks,  J_{ss} = sum_L B_L tr(B_L (P_aa+P_bb)),
//             K_{st} = sum_L B_L P_{st} B_L  (P a general Hermitian 2n x 2n density).
Matrix<double> nonRelFockMatrix(const Matrix<double>& h_core, const AoCholesky& ao,
                                 const Matrix<double>& density_matrix);
Matrix<std::complex<double>> x2cFockMatrix(const Matrix<std::complex<double>>& h_x2c, const AoCholesky& ao,
                                            const Matrix<std::complex<double>>& density_matrix);

// MO-basis vectors in the CholeskyEri convention (eri(a,b,c,d) = <ab|cd> = (ac|bd)):
//  * NON_REL closed-shell SPIN-ORBITAL vectors (dimension 2 n_mo, block layout [alpha; beta] like
//    closedShellSpinOrbitalTwoElectron): W_L = blockdiag(B'_L, B'_L), B'_L = C^T B_L C, C the real
//    n x n_mo spatial coefficients.
//  * X2C spinor vectors (dimension n_mo): B'_L = C_a^dagger B_L C_a + C_b^dagger B_L C_b with C_a/C_b the
//    alpha/beta AO blocks (n x n_mo each) of the 2n x n_mo spinor coefficients, W_L = B'_L^T.
CholeskyEri<double> aoCholeskyToMoSpinOrbital(const AoCholesky& ao, const Matrix<double>& c_spatial);
CholeskyEri<std::complex<double>> aoCholeskyToMoSpinor(const AoCholesky& ao,
                                                         const Matrix<std::complex<double>>& c_spinor);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_AOCHOLESKY_H
