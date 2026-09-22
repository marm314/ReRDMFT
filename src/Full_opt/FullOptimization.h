#ifndef RERDMFT_FULLOPTIMIZATION_H
#define RERDMFT_FULLOPTIMIZATION_H

#include <complex>
#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "HartreeExchangeHessian.h"
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

// Which method drives the orbital-rotation step of the macro loop (ORBITAL_OPTIMIZER keyword).
enum class OrbitalOptimizer { kAdam, kNeo };

struct FullOptSettings {
  bool enabled = false;
  int max_macro_iterations = 1000;
  double energy_tolerance = 1e-9;     // tolE of the Fortran
  double gradient_tolerance = 1e-5;   // 10**-itolLambda of the Fortran
  // CHOLESKY TRUE: the loop keeps the two-electron integrals as Cholesky vectors of the Coulomb
  // metric (Utils/CholeskyEri.h) instead of a dense n^4 tensor: rotations act on the vectors,
  // elements are evaluated on demand.
  bool cholesky = false;
  double cholesky_threshold = 1e-10;
  // ADAM (default) or NEO (Utils/NEO.h, second-order trust-region Newton, matrix-free
  // Hessian-vector products -- Hessian_opt/JkOnlyHessian.h's/HartreeExchangeHessian.h's
  // ...JointHessianVector or a plain symmetrized per-element sum for real orbitals, see
  // RdmftModel::hessian_vector). Falls back to ADAM with a printed note when the model has no
  // hessian_vector (a Cholesky-vector model: CHOLESKY TRUE has no cheap per-element Hessian).
  // NEO always targets the GROUND STATE (target_order = 0, a minimum) here -- no saddle-point
  // search. Every macro-iteration's Newton descent is run to ITS OWN full convergence
  // (ORBITAL_GRADIENT_TOLERANCE), up to `neo_max_iterations` Newton steps -- NOT an ADAM-style
  // small-then-growing budget: that was tried and measured to only ever hurt, never help. Cutting
  // a Newton descent short mid-iteration hands the next occupation re-optimization a
  // not-actually-stationary orbital point, and on some PNOF/GNOF systems (found by comparing
  // against ADAM, e.g. lih_pnof7's X2C branch: 2e-5 Ha off, truncated, vs machine precision, not
  // truncated) that locks the whole macro loop into a WORSE final answer with no way back --
  // ADAM's own many small, cheap steps never have this failure mode, since a partial ADAM step is
  // still along the true (not budget-cut) direction. A single generous `neo_max_iterations` avoids
  // it: NEO is a quadratically convergent Newton method, so genuinely needing more than a few tens
  // of steps per macro-iteration would itself be a sign of trouble.
  OrbitalOptimizer orbital_optimizer = OrbitalOptimizer::kAdam;
  int neo_max_iterations = 100;
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
template <typename T, typename Eri = Tensor4<T>>
struct RdmftModel {
  // Electronic energy (no nuclear repulsion) at fixed occupations.
  std::function<double(const Matrix<T>&, const Eri&, const std::vector<double>&)> energy;
  // Orbital-rotation gradient matrix (OrbitalGradient.h's orbitalGradient: g_pq,
  // lower triangle p >= q) at fixed occupations.
  std::function<Matrix<T>(const Matrix<T>&, const Eri&, const std::vector<double>&)> gradient;
  // Re-optimizes the occupation numbers at fixed integrals. `state` is the
  // model's own warm-start variable (JK_only: active occupations; PNOF
  // L-BFGS: gamma angles; PNOF SQP: frontier occupations), updated in place.
  std::function<RdmftOccupationResult(const Matrix<T>&, const Eri&, std::vector<double>&)>
      optimize_occupations;
  // Optional (PNOF): the energy through the Kramers/spin-pair-SYMMETRIC shortcut
  // (Occ_opt/PNOFs.h's pnofElectronicEnergy), valid only while the integrals keep
  // that symmetry. The end-of-run check compares it with `energy`: agreement
  // shows the rotated integrals still have the pairing symmetry.
  std::function<double(const Matrix<T>&, const Eri&, const std::vector<double>&)>
      symmetric_shortcut_energy;
  // Prints a full occupation vector in the SAME format main.cpp uses right after the
  // first occupation optimization (JK_only: index/occupation table, two Kramers columns
  // for X2C; PNOF: one line per geminal, core frozen at 1). Called after the macro loop.
  std::function<void(const std::vector<double>&, std::ostream&)> print_occupations;
  // Orbital-rotation Hessian-VECTOR product w = H v at fixed occupations, over the SAME pair list
  // `hessianPairIndices(h.rows())` jointOrbitalGradient/the joint Hessian builders use --
  // ROW-based (Hessian_opt/JkOnlyHessian.h's/HartreeExchangeHessian.h's ...JointHessianVector, or
  // a plain per-element sum for real orbitals): v/w never cost more than O(n_pairs) memory, the
  // O(n_pairs^2 x n) element cost is paid on every call (no dense matrix is ever cached) -- this
  // is what Utils/NEO.h's NeoProblem::hessianVector needs. T = double: v/w size = n_pairs (real,
  // t only). T = complex<double>: v/w size = 2*n_pairs (joint [t;y], complex spinors). Works for
  // `Eri = Tensor4<T>` AND `Eri = CholeskyEri<T>` (the underlying element/joint-vector functions
  // only ever touch `eri` through `operator()` and `dim0..dim3()`, which CholeskyEri provides
  // too) -- CHOLESKY TRUE no longer disables NEO.
  std::function<std::vector<double>(const Matrix<T>&, const Eri&, const std::vector<double>&,
                                     const std::vector<double>&)>
      hessian_vector;
  // The SAME computation as `hessian_vector`, but ALWAYS over `Tensor4<T>` regardless of `Eri`
  // (set alongside `hessian_vector` unconditionally). When `Eri = CholeskyEri<T>`,
  // `CholeskyEri::operator()` costs O(n_chol) per element instead of O(1), and a single Newton
  // step's Davidson solve calls `hessian_vector` many times -- NeoOrbitalProblem instead
  // materializes ONE dense Tensor4 per accepted step (`CholeskyEri::toDense()`, O(n_chol n^4),
  // paid once) and calls this field for every one of that step's Hessian-vector products, which
  // are then as cheap as the native-Tensor4 case. When `Eri` already IS `Tensor4<T>` this is the
  // identical computation as `hessian_vector` (kept as a separate field only for a uniform call
  // site in NeoOrbitalProblem, not because the two ever differ there).
  std::function<std::vector<double>(const Matrix<T>&, const Tensor4<T>&, const std::vector<double>&,
                                     const std::vector<double>&)>
      hessian_vector_dense;
};

// JK_only functionals (Occ_opt/JK_only.h): SQP over the active occupations
// with sum(n) = n_electrons (Occ_opt/OccupationEnergy.h + Utils/SQP.h),
// window [n_inactive_below, n_inactive_below + n_active).
// `two_columns`: print the occupations as even/odd Kramers pairs side by side (X2C),
// otherwise one orbital per line (NON_REL).
template <typename T, typename Eri = Tensor4<T>>
RdmftModel<T, Eri> makeJkOnlyModel(JkFunctional functional, std::size_t f_l, double n_electrons,
                              std::size_t n_total, std::size_t n_inactive_below,
                              std::size_t n_active, bool two_columns = false);

// PNOF functionals (Occ_opt/PNOFs.h): `geminals`/`n_core` as built by
// buildPnofGeminals (indices already converted to actual array indices);
// `sqp_occupations` selects the SQP branch (SQP_PNOF_OCC TRUE), otherwise
// L-BFGS over the unconstrained gamma angles.
template <typename T, typename Eri = Tensor4<T>>
RdmftModel<T, Eri> makePnofModel(PnofFunctional functional, std::vector<PnofGeminal> geminals,
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
  // Final warm-start variable of the occupation optimizer (JK_only: active occupations; PNOF:
  // frontier occupations for SQP, gamma angles for L-BFGS).
  std::vector<double> occupation_state;
  // Accumulated orbital rotation of the macro loop, C_final = C_start * total_rotation (n x n,
  // stored complex also for real orbitals). EMPTY when the loop did not run (a validation check
  // failed): the orbitals are then the starting ones.
  Matrix<std::complex<double>> total_rotation;
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
template <typename T, typename Eri = Tensor4<T>>
FullOptResult runFullOptimization(const Matrix<T>& h, const Eri& eri,
                                  const std::vector<double>& occupations,
                                  const std::vector<double>& state, const RdmftModel<T, Eri>& model,
                                  const FullOptSettings& settings, bool kramers_restricted,
                                  double nuclear_repulsion_energy, std::ostream& log,
                                  const std::vector<std::size_t>& spin_partner = {});

// Entry points used by main.cpp: same as building the model and calling runFullOptimization, but
// honouring FullOptSettings::cholesky (the dense MO integrals are decomposed into Cholesky vectors
// and the loop runs on them). `spin_partner` as in runFullOptimization.
template <typename T>
FullOptResult runFullOptimizationJk(const Matrix<T>& h, const Tensor4<T>& eri,
                                    const std::vector<double>& occupations,
                                    const std::vector<double>& state, JkFunctional functional,
                                    std::size_t f_l, double n_electrons, std::size_t n_total,
                                    std::size_t n_inactive_below, std::size_t n_active,
                                    bool two_columns, const FullOptSettings& settings,
                                    bool kramers_restricted, double nuclear_repulsion_energy,
                                    std::ostream& log,
                                    const std::vector<std::size_t>& spin_partner = {});

template <typename T>
FullOptResult runFullOptimizationPnof(const Matrix<T>& h, const Tensor4<T>& eri,
                                      const std::vector<double>& occupations,
                                      const std::vector<double>& state, PnofFunctional functional,
                                      const std::vector<PnofGeminal>& geminals, std::size_t n_core,
                                      int pnof_subspaces, int pnof_coupling, bool relativistic,
                                      bool sqp_occupations, std::size_t n_total,
                                      const FullOptSettings& settings, bool kramers_restricted,
                                      double nuclear_repulsion_energy, std::ostream& log,
                                      const std::vector<std::size_t>& spin_partner = {});

}  // namespace rerdmft

#endif  // RERDMFT_FULLOPTIMIZATION_H
