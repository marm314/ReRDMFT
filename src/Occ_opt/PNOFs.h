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
template <typename T>
double pnofElectronicEnergy(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
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

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_PNOFS_H
