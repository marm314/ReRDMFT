#ifndef RERDMFT_FULLOPTIMIZATION_H
#define RERDMFT_FULLOPTIMIZATION_H

#include <complex>
#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "IntegralRotation.h"
#include "JK_only.h"
#include "Matrix.h"
#include "PNOFs.h"
#include "Tensor4.h"

namespace rerdmft {

// FULL (macro-iteration) RDMFT optimization at the level of the MO-basis
// integrals: alternate
//   1) ADAM (Utils/ADAM.h, the DoNOF orbital optimizer) over the orbital
//      rotations at FIXED occupation numbers, then
//   2) re-optimization of the occupation numbers at the new (fixed)
//      orbitals, exactly as done once before this loop starts,
// until the energy stops changing -- the structure of standalone_donof's
// m_noft_driver/m_optorb loop: converged when |E - E_old| < tolE AND ADAM
// did not request a restart (a failed ADAM run lowers its learning rate
// and asks for more iterations instead).
//
// Scope: NON_REL (real spin-orbitals) and X2C (complex spinors, orbital
// rotations restricted to the time-reversal-symmetric Kramers-restricted
// subspace via Utils/KramersRestriction.h). NOT for the 4-component path.
// Everything acts on the (h, eri) MO integrals passed in, rotating them
// with the EXACT O(n^5) leg transform (`rotateIntegralsExact`) -- no
// AO-basis re-transformation is ever needed.

struct FullOptSettings {
  bool enabled = false;
  int max_macro_iterations = 1000;
  double energy_tolerance = 1e-9;     // tolE of the Fortran
  double gradient_tolerance = 1e-5;   // 10**-itolLambda of the Fortran
};

struct RdmftOccupationResult {
  std::vector<double> occupations;   // full (n_total) occupation vector
  double electronic_energy = 0.0;
  bool converged = false;
  int iterations = 0;
};

// The functional-specific pieces the macro loop needs, all expressed for
// integrals (h, eri) in the CURRENT orbital basis. Built by makeJkOnlyModel /
// makePnofModel below (the same energy/gradient/occupation optimizer that
// the fixed-orbital occupation optimization in main.cpp uses).
template <typename T>
struct RdmftModel {
  // Electronic energy (no nuclear repulsion) at fixed occupations.
  std::function<double(const Matrix<T>&, const Tensor4<T>&, const std::vector<double>&)> energy;
  // Orbital-rotation gradient matrix (OrbitalGradient.h's orbitalGradient: g_pq,
  // lower triangle p >= q) at fixed occupations.
  std::function<Matrix<T>(const Matrix<T>&, const Tensor4<T>&, const std::vector<double>&)> gradient;
  // Re-optimizes the occupation numbers at fixed integrals. `state` is the
  // model's own warm-start variable (JK_only: active occupations; PNOF
  // L-BFGS: gamma angles; PNOF SQP: frontier occupations), updated in place.
  std::function<RdmftOccupationResult(const Matrix<T>&, const Tensor4<T>&, std::vector<double>&)>
      optimize_occupations;
  // Optional (PNOF): the energy through the Kramers/spin-pair-SYMMETRIC shortcut
  // (Occ_opt/PNOFs.h's pnofElectronicEnergy), valid only while the integrals keep
  // that symmetry. The end-of-run check compares it with `energy`: agreement
  // shows the rotated integrals still have the pairing symmetry.
  std::function<double(const Matrix<T>&, const Tensor4<T>&, const std::vector<double>&)>
      symmetric_shortcut_energy;
};

// JK_only functionals (Occ_opt/JK_only.h): SQP over the active occupations
// with sum(n) = n_electrons (Occ_opt/OccupationEnergy.h + Utils/SQP.h),
// window [n_inactive_below, n_inactive_below + n_active).
template <typename T>
RdmftModel<T> makeJkOnlyModel(JkFunctional functional, std::size_t f_l, double n_electrons,
                              std::size_t n_total, std::size_t n_inactive_below,
                              std::size_t n_active);

// PNOF functionals (Occ_opt/PNOFs.h): `geminals`/`n_core` as built by
// buildPnofGeminals (indices already converted to actual array indices);
// `sqp_occupations` selects the SQP branch (SQP_PNOF_OCC TRUE), otherwise
// L-BFGS over the unconstrained gamma angles.
template <typename T>
RdmftModel<T> makePnofModel(PnofFunctional functional, std::vector<PnofGeminal> geminals,
                            std::size_t n_core, int pnof_subspaces, int pnof_coupling,
                            bool relativistic, bool sqp_occupations, std::size_t n_total);

// Exact rotation of the MO integrals into the basis C_new = C_old * U:
// h' = U^dagger h U, eri'(pqrs) = sum conj(U_ap) conj(U_bq) U_cr U_ds
// eri(abcd) (physics notation <ab|cd>, bra legs conjugated) -- four successive one-leg O(n^5) transforms, no Cholesky
// truncation (unlike IntegralRotation.h's rotateIntegrals, whose 1e-10
// decomposition threshold is comparable to the 1e-9 energy tolerance of
// this loop).
template <typename T>
RotatedIntegrals<T> rotateIntegralsExact(const Matrix<T>& h, const Tensor4<T>& eri,
                                         const Matrix<T>& u);

struct FullOptResult {
  bool checks_passed = false;
  bool converged = false;
  int iterations = 0;
  double electronic_energy = 0.0;   // at the final orbitals/occupations
  double gradient_max = 0.0;        // max |g_pq| (p > q) at the final point
  std::vector<double> occupations;  // final full occupation vector
};

// Runs (a) the validation of the ADAM/Kramers-restriction machinery on the
// actual data -- exact rotation vs the Cholesky one, orbital gradient vs
// finite differences of the energy, and (kramers_restricted) occupation
// equality within Kramers pairs, time-reversal symmetry of the gradient
// and of a probe rotation -- and, only if all pass, (b) the macro loop.
// `occupations`/`state` are the result of the occupation optimization
// already done at the starting orbitals. `kramers_restricted` (X2C):
// requires complex T and consecutive Kramers pairs (2k, 2k+1). Progress is
// written to `log`.
// `spin_partner` (real T only, NON_REL): partner[i] = the opposite-spin twin of
// spin-orbital i (block layout: k <-> n_spatial + k). If non-empty the orbital
// optimization is SPIN-RESTRICTED: the gradient entries of the alpha pair (p,q)
// and its beta twin are averaged before every ADAM step (they are equal up to
// roundoff, but ADAM normalizes each entry by its own magnitude and would
// amplify that roundoff into independent alpha/beta steps of size ~ the learning
// rate for redundant rotations), so both spins rotate identically and the
// pair-symmetric energy shortcut of the occupation optimizer stays valid.
template <typename T>
FullOptResult runFullOptimization(const Matrix<T>& h, const Tensor4<T>& eri,
                                  const std::vector<double>& occupations,
                                  const std::vector<double>& state, const RdmftModel<T>& model,
                                  const FullOptSettings& settings, bool kramers_restricted,
                                  double nuclear_repulsion_energy, std::ostream& log,
                                  const std::vector<std::size_t>& spin_partner = {});

}  // namespace rerdmft

#endif  // RERDMFT_FULLOPTIMIZATION_H
