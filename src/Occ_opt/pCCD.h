#ifndef RERDMFT_OCC_OPT_PCCD_H
#define RERDMFT_OCC_OPT_PCCD_H

#include <cstddef>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// Kramers-restricted pCCD (doc/kr_pccd.tex: generalization of Henderson, Bulik, Stein,
// Scuseria, J. Chem. Phys. 141, 244104 (2014) to a spin-WITH/Kramers-restricted spinor
// basis). For FIXED one-/two-electron integrals (the converged HF/X2C-HF/4C-DHF natural
// MO/spinor basis), this solves the t-/z-amplitude residue equations and builds the
// pair-level 1-/2-RDM -- ONE real-amplitude formalism that covers NON_REL (spin-with
// spin-orbitals), X2C and C4_DHF (Kramers-restricted spinors) uniformly: the opposite-
// Kramers exchange integrals K_{p\bar q} that distinguish the relativistic case from the
// nonrelativistic one are just zero automatically for NON_REL's spin-pure pairs (tex
// doc, Sec. "Recovery of the nonrelativistic limit"), so eps_p/G_pq/W_pq below reduce to
// the same numbers either way with no special-cased branch. Orbitals are NOT optimized
// here (that is a separate, later task): this module only ever reads the (h, eri) it is
// given.
//
// "Pairs" here are Kramers/spin pairs in the SAME sense as Occ_opt/Orb_subspaces.h and
// Occ_opt/PNOFs.h: one representative spinor/spin-orbital index p, plus its partner
// \bar p (NON_REL: the opposite-spin twin of the same spatial orbital; X2C/C4_DHF: the
// Kramers partner). `eps`/`g`/`w` are built once over a combined pair list (frozen-
// occupied "core" pairs, pinned at n=1, followed by the "active" occupied and virtual
// pairs the amplitudes are solved over); `n_core` says where the core block ends.

struct PccdCoefficients {
  // eps_p, G_pq, W_pq (Eqs. (eps)/(G)/(W), Coulomb case, of doc/kr_pccd.tex), over the
  // combined pair list core ++ occupied ++ virtual (size n_core + n_occ + n_vir). `g`/`w`
  // are symmetric with an unused (zero) diagonal.
  std::vector<double> eps;
  Matrix<double> g;
  Matrix<double> w;
  std::size_t n_core = 0;
  std::size_t n_occ = 0;
  std::size_t n_vir = 0;
};

// Builds `PccdCoefficients` from the MO integrals (h, eri) and a pair list: `reps` is the
// combined list of representative indices (core pairs first, then the n_occ occupied
// pairs, then the n_vir virtual pairs -- `n_core`/`n_occ` say where each block ends/
// begins), `bar` is `reps`' own Kramers/spin partner, index-for-index (bar.size() ==
// reps.size()). Both are ACTUAL array indices into `h`/`eri` (already resolved by the
// caller -- main.cpp's buildPnofFunctionalReport's own toActualIndex/pair_of convention
// for NON_REL's interleaved block layout vs X2C/C4_DHF's direct adjacent-pair layout).
// `g`/`w` are explicitly symmetrized (M <- (M + M^T)/2) to guard against roundoff
// asymmetry in the underlying (generally complex) integrals -- time-reversal symmetry
// guarantees exact symmetry only in exact arithmetic (tex doc, Sec. "Conventions").
template <typename T, typename Eri>
PccdCoefficients buildPccdCoefficients(const Matrix<T>& h, const Eri& eri,
                                       const std::vector<std::size_t>& reps,
                                       const std::vector<std::size_t>& bar,
                                       std::size_t n_core, std::size_t n_occ,
                                       std::size_t n_vir);

// Which algorithm drives the amplitude residue equations to zero (PCCD_AMPLITUDE_SOLVER
// keyword, Input.h). kNewton: exact Newton-Raphson, rebuilding the analytic Jacobian and
// solving it directly (LinearAlgebra.h's `invert`) every iteration -- for the (linear) z
// equation this is EXACT in one iteration, since the z-residual's own Jacobian (see
// pccdZJacobian) is already the converged t-Jacobian's transpose, a genuine constant.
// kLbfgs: minimize 0.5*||residual||^2 (Utils/LBFGS.h) with the exact chain-rule gradient
// J^T * residual -- never forms/solves the Jacobian, at the cost of needing many more
// (cheap) iterations than Newton's quadratic convergence.
enum class PccdAmplitudeSolver { kNewton, kLbfgs };

struct PccdSettings {
  PccdAmplitudeSolver solver = PccdAmplitudeSolver::kNewton;
  int max_iterations = 200;
  // Converged once the residual's 2-norm falls below this (kNewton) or the L-BFGS
  // gradient's inf-norm does (kLbfgs's own, separate `LbfgsOptions::gradient_tolerance`
  // is derived from this -- see the .cpp).
  double residual_tolerance = 1e-10;
};

struct PccdAmplitudeResult {
  Matrix<double> t;  // n_occ x n_vir
  Matrix<double> z;  // n_occ x n_vir
  bool t_converged = false;
  bool z_converged = false;
  int t_iterations = 0;
  int z_iterations = 0;
  double t_residual_norm = 0.0;
  double z_residual_norm = 0.0;
};

// R_ia(T) itself (Eq. (tamp), boxed, of doc/kr_pccd.tex) -- exposed (not just used
// internally by `solvePccdAmplitudes`) so tests can finite-difference it directly against
// `pccdTJacobian` below.
Matrix<double> pccdTResidual(const PccdCoefficients& coeff, const Matrix<double>& t);

// The analytic Jacobian dR_ia/dT_kd of `pccdTResidual`, re-derived directly from Eq.
// (tamp) (not ported from a reference Fortran implementation) and flattened row-major
// over (i,a) -- index (i*n_vir+a, k*n_vir+d) -- for `invert`/Utils/LBFGS.h's gradient
// chain rule. Block-sparse by construction: entry (i,a),(k,d) is zero unless k==i or
// d==a (a pair-transfer amplitude only couples to others sharing its occupied OR its
// virtual pair), which this function exploits directly rather than filling then zeroing.
Matrix<double> pccdTJacobian(const PccdCoefficients& coeff, const Matrix<double>& t);

// Solves the t-amplitude equations (R_ia(T) = 0) via `settings.solver`, starting from
// `t0` (size n_occ x n_vir -- a zero matrix is always a valid, if slow-converging,
// starting point; the caller may supply a warm start, e.g. a previous geometry's/
// macro-iteration's converged t).
PccdAmplitudeResult solvePccdTAmplitudes(const PccdCoefficients& coeff, const Matrix<double>& t0,
                                          const PccdSettings& settings);

// Solves the (linear) z-amplitude equations at the GIVEN, already-converged `t`: the
// Lagrangian L(t,z) = E_ref + sum_ia G_ia t_ia + sum_ia z_ia R_ia(t) is stationary in t
// (identically in t AND z, since E(t) is itself exactly linear in t -- see
// pccdCorrelationEnergy) exactly when G_kd + sum_ia z_ia * dR_ia/dT_kd(t) = 0 for every
// (k,d) -- i.e. J(t)^T z = -G_ov, J = pccdTJacobian(coeff, t) -- the standard CC-Lambda
// adjoint-equation structure (not re-derived/ported separately from any reference
// implementation: it follows directly from pccdTJacobian once E(t)'s own linearity in t
// is used). `settings.solver` still selects Newton (exact, one iteration, since this
// system is linear in z) or L-BFGS (iterative) for consistency with `solvePccdTAmplitudes`,
// and for a cross-check between the two at validation time.
PccdAmplitudeResult solvePccdZAmplitudes(const PccdCoefficients& coeff, const Matrix<double>& t,
                                          const Matrix<double>& z0, const PccdSettings& settings);

// E_ref (Eq. (Eref) of doc/kr_pccd.tex): the ordinary single-determinant (HF-like)
// energy of the full occupied set (frozen core AND active-occupied pairs alike) --
// Sum_p eps_p + 1/2 Sum_{p!=q} W_pq over p, q in [0, n_core+n_occ).
double pccdReferenceEnergy(const PccdCoefficients& coeff);

// E_corr = sum_ia G_ia t^a_i (the pCCD correlation energy, Eq. in Sec. "pCCD energy and
// amplitude equations" -- EXACTLY linear in t, the fact `solvePccdZAmplitudes`'s own
// header comment uses). E_pCCD = pccdReferenceEnergy(coeff) + pccdCorrelationEnergy(...).
double pccdCorrelationEnergy(const PccdCoefficients& coeff, const Matrix<double>& t);

// Pair-level 1-/2-RDM quantities (Sec. "Density matrices" of doc/kr_pccd.tex), over the
// ACTIVE (occupied + virtual) pair list ONLY -- core pairs are not covered (the frozen-
// pair/fpCCSD extension needed for that is explicitly flagged as future work by the tex
// doc itself, Sec. "Remarks and checks"): a caller wanting a FULL occupation vector over
// core+active+deep-virtual should embed `n_occ`/`n_vir` at 1/[these]/0 directly, which
// needs no RDM formula at all for the core/deep-virtual entries.
struct PccdRdm {
  std::vector<double> n_occ;  // size n_occ: n_i = 1 - sum_a T(i,a) Z(i,a)
  std::vector<double> n_vir;  // size n_vir: n_a = sum_i T(i,a) Z(i,a)
  // D_pq (pair-transfer, D_pp == n_p by convention) and Q_pq (density-density, Q_aa' == 0
  // for two DIFFERENT virtual pairs by construction -- see the tex doc), both
  // (n_occ+n_vir) x (n_occ+n_vir), index 0..n_occ-1 occupied then n_occ..n_occ+n_vir-1
  // virtual, matching `PccdCoefficients`' own active-block ordering.
  Matrix<double> d;
  Matrix<double> q;
};

PccdRdm buildPccdRdm(const Matrix<double>& t, const Matrix<double>& z);

// Cross-check (Eq. (Epair) of doc/kr_pccd.tex): recomputes the ACTIVE-block electronic
// energy from `rdm` alone (eps_p n_p + G_pq D_pq + 1/2 W_pq Q_pq, p,q ranging over the
// active occupied+virtual pairs only -- i.e. excluding the core contribution already in
// pccdReferenceEnergy) and returns it; a caller compares this against
// pccdCorrelationEnergy(coeff, t) + (the active-occupied-only part of
// pccdReferenceEnergy) as an independent validation that T, Z and the RDM formulas agree
// (see tests/test_pccd_amplitudes.cpp).
double pccdPairEnergyFromRdm(const PccdCoefficients& coeff, const PccdRdm& rdm);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_PCCD_H
