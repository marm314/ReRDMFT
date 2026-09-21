#ifndef RERDMFT_OCC_OPT_PNOFS_H
#define RERDMFT_OCC_OPT_PNOFS_H

#include <cstddef>
#include <string>
#include <vector>

#include "Matrix.h"
#include "Orb_subspaces.h"
#include "Tensor4.h"

namespace rerdmft {

// The four Piris-type functionals of doc/rel_pnofs.tex (relativistic
// PNOF5, PNOF7, PNOF7s, GNOF -- explicit 2-RDM elements and full
// algebraic derivation from the no-pair relativistic 2-RDM
// reconstruction of SciPost Chem. 1, 004 (2022)). This file implements
// the ENERGY expression (doc/rel_pnofs.tex Eq. 36, `eq:E64`) only;
// orbital-rotation gradient/Hessian support is a separate, later task.
enum class PnofFunctional { kPnof5, kPnof7, kPnof7s, kGnof };

// Case-insensitive: "PNOF5", "PNOF7", "PNOF7S", "GNOF". Throws
// std::runtime_error otherwise.
PnofFunctional parsePnofFunctional(const std::string& name);

// One "geminal": a single degenerate orbital pair (i, ibar) -- NON_REL's
// alpha/beta spin partner, or X2C/C4_DHF's Kramers partner (see
// Occ_opt/Orb_subspaces.h, whose `pair_of` convention this reuses) --
// tagged with which subspace (Om_p in doc/rel_pnofs.tex notation) it
// belongs to and whether it is that subspace's own principal
// (occupied, reference) pair. Every occupied pair in the system is
// exactly one PnofGeminal: either a deep-core pair (its own trivial,
// uncoupled subspace, is_principal=true) or a pair belonging to a
// PNOF_SUBSPACES-built frontier subspace (Occ_opt/Orb_subspaces.h) --
// its own principal (is_principal=true) or one of that subspace's
// coupled virtual pairs (is_principal=false). Deep, uncoupled virtual
// pairs (occ=0) are NOT included at all: with occupation exactly zero
// they contribute nothing to any Hartree/exchange/pairing term.
struct PnofGeminal {
  std::size_t i;
  std::size_t ibar;
  std::size_t subspace_id;
  bool is_principal;
};

// Builds the full geminal list from an OrbitalSubspaceTable: one entry
// per pair in `table.frozen_occupied` (each its own trivial, uncoupled
// subspace, `subspace_id` values not shared with any other geminal)
// plus one entry per pair (principal or coupled virtual) in every
// `table.subspaces[k]` (all pairs of one subspaces[k] share one
// `subspace_id`, itself not shared with any other subspace's pairs or
// any frozen_occupied pair). Subspace ids are only ever compared for
// equality (same subspace or not, see buildPnofTwoRdm below), never
// interpreted numerically.
std::vector<PnofGeminal> buildPnofGeminals(const OrbitalSubspaceTable& table);

// Pi^intra_ij (doc/rel_pnofs.tex Eq. 10, `eq:Pi-intra`): common to all
// four functionals, used for two DISTINCT geminals i,j WITHIN the same
// subspace (exactly one of them is that subspace's principal, since a
// subspace has only one). `involves_principal` is true iff i or j is
// that subspace's principal pair (always true in the current
// Occ_opt/Orb_subspaces.h construction, where every non-principal pair
// of a subspace is a direct virtual coupled to its ONE principal --
// there are never two non-principal pairs of the same subspace paired
// with each other -- but the sign still depends on this flag per the
// source equation, so it is taken explicitly rather than assumed).
double pnofPiIntra(double n_i, double n_j, bool involves_principal);

// Pi^inter_ij;pq, functional-specific (doc/rel_pnofs.tex Eqs. 30-32,
// `eq:98-pnof5`/`eq:98-pnof7`/`eq:98-pnof7s`), used for two geminals in
// DIFFERENT subspaces. `functional` must be kPnof5, kPnof7, or kPnof7s
// (GNOF needs pnofPiInterGnof below, which requires more than just
// n_i,n_j); throws std::runtime_error if given kGnof.
double pnofPiInter(PnofFunctional functional, double n_i, double n_j);

// GNOF's own piecewise Pi^inter (doc/rel_pnofs.tex Eq. 34, `eq:Pi-gnof`).
// Needs FOUR occupations, not two: `n_i`/`n_j` are i's and j's own
// occupations (used directly, e.g. in the sqrt(n_i*n_j*h_i*h_j) term),
// while `n_principal_i`/`n_principal_j` are the occupation of i's OWN
// subspace's principal pair and of j's OWN subspace's principal pair
// respectively (used to build n_i^d/n_j^d via that subspace's h_p^d/h_p
// -- these differ from n_i/n_j themselves whenever i or j is a
// non-principal, coupled-virtual pair of its subspace). `i_is_principal`/
// `j_is_principal` select the piecewise branch.
double pnofPiInterGnof(double n_i, double n_j, double n_principal_i, double n_principal_j,
                        bool i_is_principal, bool j_is_principal);

// The four Eq.-36 2-RDM coefficient "matrices" (doc/rel_pnofs.tex,
// Gaunt-free per the project's own convention -- J^G/K^G/L^G are simply
// never computed), one entry per ORDERED PAIR of DISTINCT geminals
// (i != j) -- i.e. Nrep x Nrep, Nrep = geminals.size(), NOT the full
// n_active actual-orbital dimension Hessian_opt/HartreeExchangeGradient.h's
// own (separate, later-task) two_rdm_h/x/l1/l2 convention uses for the
// orbital-rotation gradient/Hessian: Eq. 36's own energy expression is
// naturally and unambiguously written directly in this geminal-
// representative index space, with no need to unfold it into the full
// actual-orbital space for the energy alone.
//
//   two_rdm_h(i,j)  = 2*(n_i*n_j - Delta_ij)   [feeds row 2, J_ij]
//   two_rdm_x(i,j)  =   (n_i*n_j - Delta_ij)   [feeds rows 4 and 5, K_ij and L_ij]
//   two_rdm_l1(i,j) =   Pi_ij                  [feeds row 6, K_ij]
//   two_rdm_l2(i,j) =   Pi_ij                  [feeds row 7, L_ij -- relativistic only]
//
// for i != j, where Delta_ij = n_i*n_j if i,j are in the SAME subspace
// (making two_rdm_h/x vanish there -- ordinary Hartree/exchange never
// applies WITHIN one subspace, only the Pi-pairing term does), or 0 if
// i,j are in DIFFERENT subspaces (ordinary Hartree/exchange applies at
// full strength there, same coefficient as plain HF/JK_only's own
// n_i*n_j). Diagonal entries (i=i) are all exactly 0: PNOFs are
// self-interaction-free by construction (confirmed against
// doc/rel_pnofs.tex's own diagonal row-2/row-4 terms, which this
// project's own convention does not reproduce -- there is no separate
// "self" Hartree/exchange contribution at all here).
//
// two_rdm_l2 is left empty (0x0) if `relativistic` is false: row 5
// (opposite-spin exchange, needing L_ij = <ibar j|j ibar>, a genuinely
// relativistic/spinor construct) and row 7 (its pairing counterpart) do
// not exist for NON_REL's ordinary spin-orbital basis, so two_rdm_x
// there is JUST the row-4 (same-spin exchange) coefficient (still
// n_i*n_j - Delta_ij, unchanged), and two_rdm_l2/row 7 are unused
// entirely.
struct PnofTwoRdm {
  Matrix<double> two_rdm_h;
  Matrix<double> two_rdm_x;
  Matrix<double> two_rdm_l1;
  Matrix<double> two_rdm_l2;
};

PnofTwoRdm buildPnofTwoRdm(PnofFunctional functional, const std::vector<PnofGeminal>& geminals,
                            const std::vector<double>& occupations, bool relativistic);

// The total electronic energy (one- plus two-electron; caller adds
// nuclear repulsion separately, matching every other *_result convention
// in this project), doc/rel_pnofs.tex Eq. 36 with Gaunt dropped:
//   E = sum_i n_i*(2*h_ii + J_ii)
//       + sum_{i!=j geminals} [ two_rdm_h(i,j)*J_ij - two_rdm_x(i,j)*K_ij
//                                - (relativistic ? two_rdm_x(i,j)*L_ij : 0)
//                                + two_rdm_l1(i,j)*K_ij
//                                + (relativistic ? two_rdm_l2(i,j)*L_ij : 0) ]
// where the FIRST sum runs over geminal REPRESENTATIVES i only (one
// term per pair, not per actual orbital -- `2*h_ii` already equals
// h_ii*(n_i+n_ibar) under Kramers/spin-pair occupation symmetry
// n_ibar=n_i, matching Eq. 36's own h-block, and `J_ii` is Eq. 36's own,
// separate, SELF-INTERACTION-FREE-BY-CONSTRUCTION diagonal Hartree term
// -- NOT expressible via two_rdm_h/x, whose diagonal is exactly 0 (see
// PnofTwoRdm's own comment); this diagonal J_ii term is REAL and always
// present, common to every functional and independent of subspace
// membership (doc/rel_pnofs.tex `eq:master`'s own leading term,
// unchanged by the later intra/inter-subspace split in `eq:Ep`)), and
// J_ij = eri(i,j,i,j), K_ij = eri(i,j,j,i), L_ij = eri(ibar,j,j,ibar)
// (all real by Hermiticity -- verified, not merely assumed, via an
// internal check that the imaginary part is negligible before it is
// discarded). `h` is the converged natural-orbital/natural-spinor
// one-electron matrix, `eri` the converged two-electron tensor (physics
// notation, eri(A,B,C,D) == <AB|CD>), and `occupations` is indexed over
// the SAME active-orbital space as `geminals`/the OrbitalSubspaceTable
// they came from (main.cpp's own `occupations_active` convention -- see
// Occ_opt/Orb_subspaces.h). `relativistic` must match what was passed
// to buildPnofTwoRdm to produce `two_rdm`.
//
// Throws std::runtime_error if `geminals` is empty, or if any J_ij/
// K_ij/L_ij integral used has a non-negligible imaginary part.
template <typename T, typename Eri>
double pnofElectronicEnergy(PnofFunctional functional, const Matrix<T>& h, const Eri& eri,
                             const std::vector<double>& occupations,
                             const std::vector<PnofGeminal>& geminals, const PnofTwoRdm& two_rdm,
                             bool relativistic);

// Independent CROSS-CHECK of pnofElectronicEnergy above: computes the
// SAME total energy directly from doc/rel_pnofs.tex's own already-
// simplified, purely occupation-number closed forms -- `eq:Ep`
// (Eq. 55, common intra-subspace term) plus the functional-specific
// boxed inter-subspace expression (`eq:PNOF5-final`/`eq:PNOF7-final`/
// `eq:PNOF7s-final`/`eq:GNOF-final`, Eqs. 57-60), Gaunt dropped -- built
// WITHOUT ever constructing a two_rdm_h/x/l1/l2 matrix at all, so that
// agreement between the two functions (verified numerically before
// either is trusted, matching this project's own established practice)
// confirms the 2-RDM-element-based derivation above (this file's
// PnofTwoRdm/pnofElectronicEnergy) is not just internally consistent
// but genuinely equivalent to the source paper's own final, independent
// formulas:
//   E_p   = sum_{i in Om_p} n_i*(2h_ii+J_ii)
//           + sum_{i,j in Om_p, i!=j} Pi^intra_ij*(K_ij+L_ij)
//   E_qp  = sum_{i in Om_q}sum_{j in Om_p} n_i*n_j*[2J_ij-K_ij-(relativistic?L_ij:0)]
//           + sum_{i in Om_q}sum_{j in Om_p} Pi^inter_ij;pq*(K_ij+(relativistic?L_ij:0))
//   E     = sum_p E_p + sum_{q!=p} E_qp
// with Pi^inter_ij;pq given by Eqs. 30-32/34 (see pnofPiInter/
// pnofPiInterGnof) for the requested `functional`. Same throw
// conditions as pnofElectronicEnergy.
template <typename T>
double pnofElectronicEnergyDirect(PnofFunctional functional, const Matrix<T>& h,
                                   const Tensor4<T>& eri, const std::vector<double>& occupations,
                                   const std::vector<PnofGeminal>& geminals, bool relativistic);

// ---------------------------------------------------------------------
// Occupation-number gradient/Hessian, for SQP-optimizing GEMINAL
// occupations at FIXED orbitals (Utils/SQP.h), mirroring
// Occ_opt/OccupationEnergy.h's jkFunctionalGradient/jkFunctionalHessian
// for the existing JK_only functionals. Kramers/spin-pair occupation
// symmetry (n_ibar = n_i) is enforced BY CONSTRUCTION here: there is one
// gradient/Hessian entry per GEMINAL (geminals.size() of them), never
// per actual orbital -- i's and ibar's own partial derivatives are
// always summed together into geminal i's single slot (see the .cpp).
// ---------------------------------------------------------------------

// d(pnofElectronicEnergy)/d(n_a), one entry per geminal a. `occupations`
// is the SAME full active-orbital-indexed vector pnofElectronicEnergy
// itself takes (only occupations[geminals[*].i] is ever read).
//
// For GNOF specifically, Pi^inter's dependence on each pair's OWN
// subspace-principal occupation (pnofPiInterGnof's n_principal_i/
// n_principal_j, see that function's own comment) means a term between
// two NON-principal (virtual) geminals in different subspaces
// contributes to FOUR gradient entries, not two: the two geminals
// directly involved, AND each one's own subspace principal (chain rule
// through n_i^d/n_j^d) -- handled explicitly here, verified against
// central finite differences of pnofElectronicEnergy before being
// trusted (not merely assumed correct by construction, given how easy
// this specific cross-coupling is to get wrong).
template <typename T, typename Eri>
std::vector<double> pnofOccupationGradient(PnofFunctional functional, const Matrix<T>& h,
                                            const Eri& eri,
                                            const std::vector<double>& occupations,
                                            const std::vector<PnofGeminal>& geminals,
                                            bool relativistic);

// Analytic Hessian, geminals.size() x geminals.size(). NOT implemented
// for GNOF (throws std::runtime_error) -- its Pi^inter cross-subspace
// coupling makes a fully general analytic second derivative
// substantially more involved and error-prone to hand-derive than
// PNOF5/PNOF7/PNOF7s's own self-contained (n_i,n_j)-only Pi^inter; use
// pnofOccupationHessianFD below for GNOF instead (an explicit engineering
// choice, not an oversight).
template <typename T, typename Eri>
Matrix<double> pnofOccupationHessian(PnofFunctional functional, const Matrix<T>& h,
                                      const Eri& eri,
                                      const std::vector<double>& occupations,
                                      const std::vector<PnofGeminal>& geminals,
                                      bool relativistic);

// Central finite-difference Hessian built from pnofOccupationGradient
// (step `h_step` on each geminal occupation in turn, symmetrized to
// guarantee an exactly-symmetric matrix regardless of finite-difference
// truncation error): (grad(n + h_step*e_b) - grad(n - h_step*e_b)) /
// (2*h_step), column b, for every geminal b -- works for ANY functional,
// including GNOF, since it never needs to differentiate Pi^inter's
// functional form directly. `h_step` defaults to 1e-4, appropriate for
// occupations that stay an interior-box `kOccupationEpsilon` away from
// the true [0,1] boundary (same reasoning as main.cpp's existing
// JK_only SQP wiring: several of these functionals have a genuinely
// divergent second derivative AT the true boundary, so a coarser step
// avoids stepping outside the box while still resolving curvature well
// away from it).
template <typename T, typename Eri>
Matrix<double> pnofOccupationHessianFD(PnofFunctional functional, const Matrix<T>& h,
                                        const Eri& eri,
                                        const std::vector<double>& occupations,
                                        const std::vector<PnofGeminal>& geminals,
                                        bool relativistic, double h_step = 1e-4);

// ---------------------------------------------------------------------
// Trigonometric ("gamma") occupation-number guess, adapted from
// standalone_donof's reference restricted/spatial-orbital DoNOF code
// (m_gammatodm2.F90::gamma_to_2rdm, m_rdmd.F90's Ngammas=Ncoupled*Npairs,
// m_optocc.F90's own GAMMAs=pi/4 default guess) that this project's PNOF
// functionals are themselves based on. Independent, UNCONSTRAINED angles
// gamma_i map to occupations that sum to exactly 1 within a subspace and
// lie in (0,1) for any REAL gamma_i -- i.e. an unconstrained
// reparameterization of the sum=1 simplex -- used HERE only to build a
// feasible SQP starting guess (main.cpp's buildPnofFunctionalReport),
// not as optimization variables in their own right (DoNOF itself
// optimizes directly over gamma; this project instead optimizes the
// occupations subject to explicit box+equality constraints via
// Utils/SQP.h, so gamma is only ever used up front, for x0).
//
// One subspace with `pnof_coupling` geminals (1 principal + coupling-1
// unoccupied, Occ_opt/Orb_subspaces.h's own convention) needs exactly
// `pnof_coupling - 1` independent gamma angles (DoNOF's own Ncoupled =
// pnof_coupling-1, and Ngammas per subspace = Ncoupled): the first sets
// the principal's own occupation via
//   n_principal = 1/2 + 1/2*cos^2(gamma_0)
// (DoNOF's own convention -- ranges [0.5, 1] as gamma_0 varies, since a
// principal pair is by definition at least half-occupied); the
// remaining `pnof_coupling - 2` angles then split the leftover hole
// (1 - n_principal) among the subspace's virtuals via a nested
// sin^2/cos^2 "stick-breaking" scheme (DoNOF's Ncoupled>1 branch,
// m_gammatodm2.F90 lines ~138-232): virtual k (0-indexed) takes a
// sin^2(gamma_{k+1}) fraction of whatever hole remains after virtuals
// 0..k-1, and the LAST virtual simply receives whatever hole is left
// over (no gamma of its own -- DoNOF reuses the second-to-last split's
// own angle via its complementary cos^2 term, since sin^2+cos^2=1
// already accounts for it exactly). Every partial sum telescopes
// exactly regardless of the gamma values, which is the entire point of
// the parameterization; for pnof_coupling=2 (DoNOF's Ncoupled=1,
// "perfect pairing") there are no split angles at all and the single
// virtual simply gets the whole hole, matching DoNOF's own dedicated
// Ncoupled==1 branch exactly.
//
// Used in main.cpp's buildPnofFunctionalReport for TWO purposes: (1) a
// feasible SQP starting guess (pnofDefaultGuessGammas below, occupations
// only, no derivatives needed), and (2) as the actual UNCONSTRAINED
// optimization variables of a second, alternative occupation-number
// optimization via Utils/LBFGS.h (pnofSubspaceOccupationsFromGammasWithGradient
// below, providing the d(occ)/d(gamma) Jacobian the chain rule needs) --
// DoNOF's own approach (m_optocc.F90 optimizes directly over GAMMAs),
// avoiding Utils/SQP.h's explicit box+equality machinery entirely, since
// gamma already guarantees sum(n)=1 and 0<n<1 for ANY real gamma value.
std::size_t pnofGammasPerSubspace(int pnof_coupling);

// One occupation value per geminal of a single subspace (size
// pnof_coupling: index 0 is the principal, indices 1..pnof_coupling-1
// are its virtuals in order), from exactly
// pnofGammasPerSubspace(pnof_coupling) angles. Throws
// std::runtime_error if `pnof_coupling < 2` or `gammas.size() !=
// pnof_coupling - 1`.
std::vector<double> pnofSubspaceOccupationsFromGammas(int pnof_coupling,
                                                       const std::vector<double>& gammas);

// Inverse of pnofSubspaceOccupationsFromGammas: the pnof_coupling-1 angles (each in [0, pi/2])
// whose occupations are `occ` (size pnof_coupling, index 0 the principal geminal, sum = 1 to
// within roundoff; the last virtual is implied by the others). Used to write the GAMMAs of an
// SQP-optimized (occupation-space) result to the RESTART file. A stick-breaking share whose
// remaining hole is numerically zero has no defined angle and gets pi/4. Throws
// std::runtime_error if `occ.size() != pnof_coupling` or `pnof_coupling < 2`.
std::vector<double> pnofSubspaceGammasFromOccupations(int pnof_coupling,
                                                       const std::vector<double>& occ);

// DoNOF's own default occupation guess (m_optocc.F90: `GAMMAs=pi/four`,
// its comment reading "Perturbed occ. numbers (i.e. pi/4) -> occ(i<Fermi
// level) = 0.75") -- every gamma angle set to pi/4, giving n_principal =
// 0.75 and each split taking half of whatever hole remains (so the
// virtuals decay geometrically: hole/2, hole/4, ..., with the last
// virtual receiving whatever is left, equal to the second-to-last
// share). Always strictly interior to (0,1) for any pnof_coupling in
// the range this project's examples use, unlike a "start at exactly
// n=1" HF-like guess, which is the whole reason DoNOF itself calls this
// a "perturbed" rather than an aufbau start.
std::vector<double> pnofDefaultGuessGammas(int pnof_coupling);

// occ, plus the Jacobian d(occ[i])/d(gammas[k]) (size pnof_coupling x
// (pnof_coupling-1)), needed to gamma-optimize occupations via
// Utils/LBFGS.h's chain rule: d(E)/d(gamma_k) =
// sum_i d(E)/d(occ[i]) * docc_dgamma(i,k). Derived by propagating the
// SAME nested "remaining hole" recursion pnofSubspaceOccupationsFromGammas
// itself uses, tracking d(remaining)/d(gamma_m) alongside `remaining`
// at every step (verified against central finite differences of
// pnofSubspaceOccupationsFromGammas before being trusted -- see this
// file's own gamma/LBFGS cross-check). Same throw conditions as
// pnofSubspaceOccupationsFromGammas.
struct PnofSubspaceOccupationsWithGradient {
  std::vector<double> occ;
  Matrix<double> docc_dgamma;
};

PnofSubspaceOccupationsWithGradient pnofSubspaceOccupationsFromGammasWithGradient(
    int pnof_coupling, const std::vector<double>& gammas);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_PNOFS_H
