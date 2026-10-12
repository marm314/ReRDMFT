#ifndef RERDMFT_CC_GAUSS_RDM_H
#define RERDMFT_CC_GAUSS_RDM_H

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Density matrices of the (complex spin-orbital) CCSD in the basis of the Cartesian Gaussian PRIMITIVES, in the formats read by
// RHO2_OPS (a .basis file and a .dm2 file; the .dm1 uses the same record layout). The MO-basis spinors are expanded over the
// distinct primitives of the contracted Cartesian AOs; the Small component of a 4-component spinor is a second, independent set
// of primitives (the raw RKB small AOs), and the density of each electron is summed over the spinor components (alpha/beta, and
// Large/Small), so only products of two primitives of the same component enter.
using GaussScalar = std::complex<double>;

// Spinors over the contracted Cartesian AOs, one column per spinor in the output order of the density matrix files.
struct CartesianSpinors {
  std::vector<BasisFunction> large_basis;           // contracted Cartesian AOs of the Large component
  std::vector<BasisFunction> small_basis;           // ... of the Small component (empty for NON_REL / X2C)
  Matrix<GaussScalar> large_alpha, large_beta;      // (n_large x n_spinors): alpha / beta coefficients over large_basis
  Matrix<GaussScalar> small_alpha, small_beta;      // (n_small x n_spinors), only with a Small component
};

struct GaussRdmSettings {
  std::string basis_path, dm1_path, dm2_path;
  double threshold = 1e-10;        // stored 2-RDM values below this are not written
  double max_memory_gb = 4.0;      // budget for the block buffers of the transform
  bool unnormalized_primitives = false;  // true: D contains the primitive normalization (RHO2_OPS keyword $CNORM1)
};

struct GaussRdmResult {
  std::size_t n_primitives = 0, n_primitives_large = 0;
  std::size_t n_dm1 = 0, n_dm2 = 0;     // records written (without the terminating zero record)
  double trace_dm1 = 0.0;               // sum gamma_{mu lambda} S_{mu lambda}  (= N)
  double trace_dm2 = 0.0;               // sum D_{mu nu lambda sigma} S_{mu lambda} S_{nu sigma} (= N(N-1)), over the stored elements
  double max_imag = 0.0;                // largest imaginary part of a fully symmetrized element (should vanish)
  double vee_primitives = std::numeric_limits<double>::quiet_NaN();  // 1/2 sum D (mu lambda|nu sigma) with primitive integrals (RERDMFT_CC_RDM_GAU_CHECK)
  int passes = 0;                       // sweeps over the MO-basis 2-RDM
  double seconds = 0.0;
};

// Slab of the MO-basis 2-RDM: out[(q*n + r)*n + s] = Gamma_{p q r s} for all q,r,s, Gamma_pqrs = <a_p^+ a_q^+ a_s a_r>, p 0-based.
using GammaSlabFn = std::function<void(std::size_t p, GaussScalar* out)>;

// Writes <basis_path>, <dm1_path>, <dm2_path>.
//   gamma1(p,q) = <a_p^+ a_q> over the same n spinors as the columns of `spinors`.
//   .basis: header line, then one line "Z  x y z  exponent  nx ny nz" per primitive (Large first, then Small).
//   .dm1  : Fortran-unformatted-style records (int32 length, int32 mu, int32 lambda, float64 value, int32 length) for mu >= lambda,
//           the real symmetrized gamma_{mu lambda}; ended by a record mu = lambda = 0.
//   .dm2  : records (int32 length, int32 p q r s, float64 value, int32 length), 1-based primitive indices, with r <= p and s <= q
//           (both D_{pqrs} and its electron-exchange partner D_{qpsr} are present), value = sum of the symmetry-equivalent
//           elements D_{pqrs} + D_{rqps} + D_{psrq} + D_{rspq} of the real, symmetrized 2-RDM (the convention of RHO2_OPS without
//           $SYMRED); ended by a record of four zero indices.
// D_{pqrs} multiplies chi_p(r1) chi_q(r2) chi_r(r1) chi_s(r2); sum D S S = N(N-1).
GaussRdmResult writeGaussRdm(const CartesianSpinors& spinors, const Matrix<GaussScalar>& gamma1, std::size_t n_spinors,
                             const GammaSlabFn& slab, const GaussRdmSettings& settings, std::ostream& log);

}  // namespace rerdmft

#endif  // RERDMFT_CC_GAUSS_RDM_H
