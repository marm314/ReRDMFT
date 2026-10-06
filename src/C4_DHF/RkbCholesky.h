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
  // Cross-flavor projections W^cross_L, needed because RKB-Small(0) and RKB-Small(1) are NOT
  // spinor-orthogonal (sigma.p mixes both elementary spin blocks into each partner), so
  // <RKBSmall(0)...|RKBSmall(1)...> is generally nonzero -- small_cross[0] = bra:partner(0),
  // ket:partner(1) (forward); small_cross[1] = bra:partner(1), ket:partner(0) (backward). See
  // RkbTwoElectron.h's rkbProjectSmallVectorCross and rkbTwoElectronIntegrals's own cross-term
  // comment for the derivation; this is the SAME fix applied to this struct's independent
  // Cholesky-vector pipeline.
  std::vector<Matrix<std::complex<double>>> small_cross[2];
  std::size_t nVectors() const { return large.size(); }
  std::size_t dim() const { return 4 * n_large; }

  // `large_transform` (Utils/SphericalTransform.h's buildSphericalTransform(large_basis).transform,
  // n_large_cart x n_large_sph) converts each AO Cholesky vector's Large block to the spherical large
  // basis right after the raw (Cartesian) AO integrals are decomposed; `rkb_coefficients` must already
  // be the SAME transform's row-projected version (see main.cpp's X2C/C4_SPINOR construction block),
  // so that `n_large` below (= large_transform.cols()) matches rkb_coefficients.rows()/2 and the
  // resulting Large/Small flavor blocks come out the same (spherical) size, exactly as the one-electron
  // H_RKB side already does.
  //
  // `on_demand` (Input.h's ON_DEMAND_ERI, default TRUE): TRUE evaluates every (LL|LL)/(LL|SS)/(SS|SS)
  // quadruplet ON DEMAND (OnDemandUnionCoulombPairs below) -- no AO tensor is ever materialized,
  // trading speed for the O(n^2)-only memory footprint that makes this safe even for a basis (e.g.
  // Xe/dyall.v2z) whose (SS|SS) alone would otherwise need tens of GB (UnionCoulombPairs, FALSE).
  static RkbCholesky build(const std::vector<BasisFunction>& large_basis,
                           const std::vector<BasisFunction>& small_basis,
                           const Matrix<std::complex<double>>& rkb_coefficients,
                           const Matrix<double>& large_transform, double threshold, bool on_demand,
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
// FULL_OPTIMIZATION's no-pair treatment): the transform is restricted to the positive-energy columns of `c`
// FROM THE START (never materializes the full n_mo x n_mo vectors -- see the .cpp's own comment on why this
// is bit-identical to transforming in full and slicing afterward) and the result is then RECOMPRESSED: with
// the negative block gone the N_chol vectors are nearly linearly dependent, so the eigen-decomposition of
// their Gram matrix keeps only the directions with eigenvalue above threshold/N_chol (element error <=
// threshold). Pure linear algebra on the vectors, no integrals are touched again; on LiH/6-31G it takes 662
// vectors back to about 60. The returned CholeskyEri is still n_mo x n_mo (zero-padded on the negative block,
// which only ever meets an exactly-zero occupation coefficient).
CholeskyEri<std::complex<double>> rkbCholeskyToMo(const RkbCholesky& eri,
                                                   const Matrix<std::complex<double>>& c_dhf,
                                                   std::size_t n_negative = 0, double threshold = 1e-10);

// FUNCTIONAL_POS_CHO_4C: RkbCholesky::build + rkbCholeskyToMo fused into one pass, for the
// trimmed (n_negative > 0, i.e. non-MINMAX) case only -- runs the SAME AO-pair decomposition
// (same pair matrix, same on_demand choice) but RKB-projects and MO-transforms each accepted
// AO-pair vector immediately (as a local temporary) instead of ever materializing the full
// RkbCholesky struct (O(N_chol_AO * n_large^2), the memory this exists to avoid). Same inputs as
// RkbCholesky::build plus the DHF spinor coefficients `c_dhf` and the number of negative-energy
// columns to skip (`n_negative`, must be > 0). The MINMAX (FULL_OPTIMIZATION_4C_NEG) stage still
// needs the untrimmed (n_negative = 0) transform, which still goes through
// RkbCholesky::build + rkbCholeskyToMo, unchanged.
CholeskyEri<std::complex<double>> rkbCholeskyToMoFused(const std::vector<BasisFunction>& large_basis,
                                                        const std::vector<BasisFunction>& small_basis,
                                                        const Matrix<std::complex<double>>& rkb_coefficients,
                                                        const Matrix<double>& large_transform,
                                                        const Matrix<std::complex<double>>& c_dhf,
                                                        std::size_t n_negative, double threshold, bool on_demand,
                                                        CholeskyCheckReport* report = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_C4_DHF_RKBCHOLESKY_H
