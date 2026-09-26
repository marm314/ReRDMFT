#ifndef RERDMFT_C4_DHF_RKBCHOLESKY_H
#define RERDMFT_C4_DHF_RKBCHOLESKY_H

#include <complex>
#include <cstddef>
#include <vector>

#include "CholeskyEri.h"
#include "Cholesky_Decomposition.h"
#include "MolecularBasis.h"
#include "Matrix.h"

namespace rerdmft {

// The 4-component (RKB) two-electron integrals as Cholesky vectors, from ONE decomposition of the real AO
// Coulomb matrix over the pair set {LL pairs} u {SS pairs} (Large-Large and unrestricted-kinetic-balance
// Small-Small AO products). That matrix is a principal submatrix of the full Coulomb matrix, hence PSD, and
// its blocks are exactly the three real AO tensors the dense RKB build uses: (LL|LL), (LL|SS), (SS|SS). Each
// vector L splits into a Large part B_L (n_large x n_large, real symmetric) and a Small part (n_small x
// n_small); the Small part is projected into the two RKB-Small partner flavors y (0/1) exactly as
// rkbTwoElectronIntegrals projects the (SS|SS) vectors, W^y_L = sum over spin blocks conj(c) S_L c^T
// (Hermitian, n_large x n_large). The RKB integrals are then, with pair densities that are diagonal in the
// spinor flavor (Large-alpha, Large-beta, Small(alpha-partner), Small(beta-partner)),
//   <AB|CD> = (AC|BD) = sum_L B_L(A,C) B_L(B,D),   B_L = blockdiag(B^LL_L, B^LL_L, W^0_L, W^1_L)
// (the same value for every alpha/beta slot combination of two Large pairs, as in the dense build).
// Storage is O(N_chol n_large^2) -- no n^4 object and no packed RKB tensor. With CHOLESKY TRUE this is all
// that is kept of the AO integrals: the DHF SCF Fock matrices and the MO-basis vectors come from it.
struct RkbCholesky {
  std::size_t n_large = 0;
  std::vector<Matrix<double>> large;                  // B^LL_L
  std::vector<Matrix<std::complex<double>>> small[2]; // W^0_L, W^1_L
  std::size_t nVectors() const { return large.size(); }
  std::size_t dim() const { return 4 * n_large; }

  static RkbCholesky build(const std::vector<BasisFunction>& large_basis,
                           const std::vector<BasisFunction>& small_basis,
                           const Matrix<std::complex<double>>& rkb_coefficients, double threshold,
                           CholeskyCheckReport* report = nullptr);
};

// F = h + J - K over the 4 RKB flavor blocks: J_ff = sum_L B^f_L tr(sum_g B^g_L P_gg), K_fg = sum_L B^f_L
// P_fg B^g_L (P a general Hermitian 4 n_large density, as rkbFockMatrix).
Matrix<std::complex<double>> rkbFockMatrix(const Matrix<std::complex<double>>& h_rkb, const RkbCholesky& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

// MO-basis vectors in the CholeskyEri convention (eri(a,b,c,d) = <ab|cd> = (ac|bd)): B'_L = C^dagger B_L C
// (C the 4 n_large x n_mo spinor coefficients), W_L = B'_L^T.
//
// `n_negative` > 0 (the first n_negative MO indices are the negative-energy Dirac-sea branch, as in
// FULL_OPTIMIZATION's no-pair treatment): the vectors are restricted to the positive-energy block (every
// element touching the negative branch is set to zero -- it only ever meets an exactly-zero occupation
// coefficient) and then RECOMPRESSED: with the negative block gone the N_chol vectors are nearly linearly
// dependent, so the eigen-decomposition of their Gram matrix keeps only the directions with eigenvalue
// above threshold/N_chol (element error <= threshold). Pure linear algebra on the vectors, no integrals are
// touched again; on LiH/6-31G it takes 662 vectors back to about 60.
CholeskyEri<std::complex<double>> rkbCholeskyToMo(const RkbCholesky& eri,
                                                   const Matrix<std::complex<double>>& c_dhf,
                                                   std::size_t n_negative = 0, double threshold = 1e-10);

}  // namespace rerdmft

#endif  // RERDMFT_C4_DHF_RKBCHOLESKY_H
