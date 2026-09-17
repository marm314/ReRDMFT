#ifndef RERDMFT_OCC_OPT_ORB_SUBSPACES_H
#define RERDMFT_OCC_OPT_ORB_SUBSPACES_H

#include <array>
#include <cstddef>
#include <vector>

namespace rerdmft {

// One PNOF-style (Piris) coupling subspace: ONE occupied degenerate
// orbital pair (`occupied`, the subspace's "reference" pair -- e.g. a
// HOMO-side spin/Kramers pair) coupled with `unoccupied.size()`
// unoccupied degenerate orbital pairs (e.g. LUMO, LUMO+1, ... side
// pairs, closest to the Fermi level first) -- the standard Piris
// extended-pairing construction, generalized here to a spin-WITH
// orbital basis (spin-orbitals for NON_REL, spinors for X2C/C4_DHF)
// rather than the doubly-occupied spatial orbitals PNOF is usually
// written for. Every array/pair is stored in ascending index order.
// This struct only records WHICH orbitals are grouped together -- it
// says nothing about how their occupations are functionally related
// (that is a separate, later task).
struct OrbitalSubspace {
  std::array<std::size_t, 2> occupied;
  std::vector<std::array<std::size_t, 2>> unoccupied;
};

// The full partition of an "active" orbital window (matching main.cpp's
// own buildFunctionalReport terminology: index i here is exactly that
// function's `orbital_energies_active`/`occupations_active` index i,
// i.e. NOT the full h/eri dimension when n_inactive_below > 0, as for
// C4_DHF's excluded negative-energy branch) into:
//   - `frozen_occupied`: degenerate PAIRS to hold at EXACTLY occ = 1
//     each (deep core, below the coupled frontier), excluded from
//     occupation-number optimization entirely -- stored as pairs (not a
//     flat index list) because PNOFs.h treats every occupied pair,
//     core or frontier, as its own "geminal"/trivial subspace when
//     computing ordinary Hartree/exchange energy between it and every
//     other pair (core-core and core-frontier alike)
//   - `frozen_unoccupied`: degenerate PAIRS to hold at EXACTLY occ = 0
//     each (deep virtual, above the coupled frontier and disjoint from
//     every subspace's own virtual pairs), likewise excluded -- these
//     contribute nothing to any Hartree/exchange/pairing energy (their
//     occupation is exactly 0), so unlike `frozen_occupied` no caller
//     currently needs to iterate them; kept as pairs anyway for
//     symmetry with `frozen_occupied`
//   - `subspaces`: exactly `pnof_subspaces` OrbitalSubspace entries,
//     subspaces[0] anchored on HOMO, subspaces[1] on HOMO-1, etc. Each
//     subspace's virtual pairs are a DISJOINT block of the available
//     unoccupied pairs (no virtual pair is ever shared between two
//     subspaces): subspaces[0] gets the (pnof_coupling-1) pairs closest
//     to the Fermi level (LUMO, LUMO+1, ...), subspaces[1] gets the
//     NEXT (pnof_coupling-1) pairs, and so on.
// This partition is EXHAUSTIVE and non-overlapping: every occupied
// active index appears in exactly one of {a `frozen_occupied` pair,
// some subspace's `occupied` pair}, and every unoccupied active index
// appears in exactly one of {a `frozen_unoccupied` pair, some
// subspace's `unoccupied` pairs}.
struct OrbitalSubspaceTable {
  std::vector<std::array<std::size_t, 2>> frozen_occupied;
  std::vector<std::array<std::size_t, 2>> frozen_unoccupied;
  std::vector<OrbitalSubspace> subspaces;
};

// Builds the table above from `pair_of` (length `n_active`, pair_of[p] =
// the index of p's own degenerate pair partner -- an involution with no
// fixed points: pair_of[pair_of[p]] == p and pair_of[p] != p for every
// p -- EXACTLY Hessian_opt/HartreeExchangeGradient.h's own `pair_of`
// convention, so a future caller can build one `pair_of` vector and feed
// it to both this function and hartreeExchangeFockMatrix's L1/L2 hooks).
// The pairing convention itself is basis-dependent and is entirely the
// caller's responsibility to encode into `pair_of` (see
// HartreeExchangeGradient.h's own header comment): NON_REL's block
// offset p <-> p + n_spatial (mod 2*n_spatial), or X2C/C4_DHF's adjacent
// Kramers pairs p <-> p xor 1. This function does not care WHICH
// convention `pair_of` encodes, only that it is a valid involution and
// (see below) that it never pairs an occupied orbital with an
// unoccupied one.
//
// `n_electrons` is rounded to the nearest integer (matching
// Occ_opt/OccupationInit.h's aufbauOccupations) to get n_occ, the number
// of occupied active orbitals -- occupied = active indices
// [0, n_occ), unoccupied = [n_occ, n_active). A closed-shell aufbau
// reference is assumed throughout this file: every degenerate pair must
// be either FULLY occupied or FULLY empty, never split across the Fermi
// level, so n_occ must be even and, for every active index p,
// (p < n_occ) must equal (pair_of[p] < n_occ).
//
// `pnof_subspaces` (the PNOF_SUBSPACES input keyword, default 1) is how
// many independent frontier subspaces to build, one per occupied pair
// outward from HOMO (subspaces[0]=HOMO, subspaces[1]=HOMO-1, ...). Must
// be >= 1 and <= the number of occupied pairs available (n_occ/2).
//
// `pnof_coupling` (the PNOF_COUPLING input keyword, default 2) is the
// SIZE of each subspace, counted in pairs: 1 occupied pair +
// (pnof_coupling-1) unoccupied pairs. PNOF_COUPLING 2 is plain perfect
// pairing (HOMO with LUMO alone); PNOF_COUPLING 3 couples HOMO with
// BOTH LUMO and LUMO+1; and so on. Must be >= 2, and
// pnof_subspaces*(pnof_coupling-1) cannot exceed the number of
// unoccupied pairs available ((n_active-n_occ)/2), since every
// subspace's virtual block is disjoint from every other's (see
// OrbitalSubspaceTable's own comment) -- if it does, this function
// throws a std::runtime_error explaining exactly how many pairs were
// requested vs. how many the basis actually has available (see below).
//
// The default PNOF_SUBSPACES=1, PNOF_COUPLING=2 is itself a genuine,
// meaningful (non-degenerate) configuration -- plain HOMO/LUMO perfect
// pairing -- not a "disabled" sentinel; a caller not wanting ANY PNOF
// coupling at all should simply not call this function.
//
// IMPORTANT, DISCOVERED WHILE BUILDING THIS: NON_REL's own
// `orbital_energies_active`/`occupations_active` (main.cpp, built via
// ClosedShellSpinOrbitals.h's block layout [alpha_0..alpha_{n-1},
// beta_0..beta_{n-1}]) do NOT currently satisfy the "no pair crosses the
// Fermi level" precondition above in general: Occ_opt/OccupationInit.h's
// aufbauOccupations fills the first round(n_electrons) ACTIVE-INDEX
// POSITIONS, which for NON_REL's block layout is NOT the same set as
// "every alpha/beta pair below a spatial HOMO cutoff" (e.g. 10 electrons,
// n_spatial=7 gives occupied active indices {0..9} = alpha_0..alpha_6
// plus beta_0..beta_2, splitting spatial orbitals 3-6's pairs across the
// boundary) -- this function will correctly THROW if handed that array
// as-is. It is harmless today only because every EXISTING JK_only
// functional treats each orbital independently (degenerate exchange
// symmetry makes the exact index assignment irrelevant to the converged
// SQP energy), but PNOF-style pairing genuinely needs a well-defined
// HOMO/LUMO PAIR, so NON_REL's active-orbital ordering will need to be
// reworked (e.g. interleaved by spatial index) before PNOF_COUPLING can
// be wired up for NON_REL specifically -- X2C and C4_DHF's existing
// adjacent-Kramers-pair ordering already satisfies this precondition as
// they stand today.
//
// Throws std::runtime_error, with a specific diagnostic naming both the
// requested and the available count, if: `pair_of` is not length
// `n_active`; `pair_of` is not a fixed-point-free involution; n_occ is
// odd or out of [0, n_active]; any pair crosses the occupied/unoccupied
// boundary; `pnof_subspaces` is < 1 or exceeds the number of available
// occupied pairs; `pnof_coupling` is < 2; or the total disjoint
// virtual-pair demand (pnof_subspaces*(pnof_coupling-1)) exceeds the
// number of available unoccupied pairs -- i.e. whenever PNOF_SUBSPACES/
// PNOF_COUPLING together ask for more occupied or unoccupied orbitals
// than this basis actually has.
OrbitalSubspaceTable buildOrbitalSubspaces(const std::vector<std::size_t>& pair_of,
                                            std::size_t n_active, double n_electrons,
                                            int pnof_subspaces, int pnof_coupling);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_ORB_SUBSPACES_H
