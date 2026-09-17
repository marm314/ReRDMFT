#include "Orb_subspaces.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace rerdmft {

namespace {

void validatePairOf(const std::vector<std::size_t>& pair_of, std::size_t n_active) {
  if (pair_of.size() != n_active) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: pair_of.size() does not match n_active");
  }
  for (std::size_t p = 0; p < n_active; ++p) {
    const std::size_t partner = pair_of[p];
    if (partner >= n_active) {
      throw std::runtime_error("buildOrbitalSubspaces: pair_of contains an out-of-range index");
    }
    if (partner == p) {
      throw std::runtime_error(
          "buildOrbitalSubspaces: pair_of has a fixed point (orbital paired with itself) -- "
          "every orbital must have a DISTINCT degenerate partner");
    }
    if (pair_of[partner] != p) {
      throw std::runtime_error(
          "buildOrbitalSubspaces: pair_of is not an involution (p's partner's partner must be p "
          "again)");
    }
  }
}

// Greedily walks `order` (a permutation of the indices in [range_begin,
// range_end)), grouping each not-yet-visited index with its pair_of
// partner, in the order encountered -- see this file's header comment
// for why this correctly enumerates HOMO-first (occupied) / LUMO-first
// (unoccupied) pairs whenever every pair lies entirely within one side
// of the occupied/unoccupied boundary (validated by the caller before
// this runs).
std::vector<std::array<std::size_t, 2>> enumeratePairs(const std::vector<std::size_t>& pair_of,
                                                         std::vector<bool>& visited,
                                                         std::size_t range_begin,
                                                         std::size_t range_end, bool descending) {
  std::vector<std::array<std::size_t, 2>> pairs;
  if (range_begin >= range_end) return pairs;
  const std::size_t count = range_end - range_begin;
  for (std::size_t k = 0; k < count; ++k) {
    const std::size_t i = descending ? (range_end - 1 - k) : (range_begin + k);
    if (visited[i]) continue;
    const std::size_t partner = pair_of[i];
    visited[i] = true;
    visited[partner] = true;
    std::array<std::size_t, 2> pair = {i, partner};
    if (pair[0] > pair[1]) std::swap(pair[0], pair[1]);
    pairs.push_back(pair);
  }
  return pairs;
}

}  // namespace

OrbitalSubspaceTable buildOrbitalSubspaces(const std::vector<std::size_t>& pair_of,
                                            std::size_t n_active, double n_electrons,
                                            int pnof_subspaces, int pnof_coupling) {
  validatePairOf(pair_of, n_active);

  const long n_occ_signed = std::lround(n_electrons);
  if (n_occ_signed < 0 || static_cast<std::size_t>(n_occ_signed) > n_active) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: n_electrons rounds to a value outside [0, n_active]");
  }
  const std::size_t n_occ = static_cast<std::size_t>(n_occ_signed);
  if (n_occ % 2 != 0) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: n_electrons must round to an EVEN number of occupied active "
        "orbitals -- PNOF-style pairing assumes a closed-shell aufbau reference where every "
        "degenerate pair is either fully occupied or fully empty");
  }

  for (std::size_t p = 0; p < n_active; ++p) {
    const bool p_occ = p < n_occ;
    const bool partner_occ = pair_of[p] < n_occ;
    if (p_occ != partner_occ) {
      throw std::runtime_error(
          "buildOrbitalSubspaces: orbital " + std::to_string(p) + " and its degenerate partner " +
          std::to_string(pair_of[p]) +
          " sit on opposite sides of the occupied/unoccupied boundary -- PNOF-style HOMO/LUMO "
          "pairing requires every degenerate pair to be entirely occupied or entirely "
          "unoccupied in the reference state (see this file's header comment for the known "
          "NON_REL indexing caveat)");
    }
  }

  if (pnof_subspaces < 1) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: PNOF_SUBSPACES must be >= 1 (a caller not using PNOF coupling "
        "should simply not call this function)");
  }
  if (pnof_coupling < 2) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: PNOF_COUPLING must be >= 2 (1 occupied pair + at least 1 "
        "unoccupied pair)");
  }

  std::vector<bool> visited(n_active, false);
  const auto occupied_pairs = enumeratePairs(pair_of, visited, 0, n_occ, /*descending=*/true);
  const auto unoccupied_pairs =
      enumeratePairs(pair_of, visited, n_occ, n_active, /*descending=*/false);

  const std::size_t n_subspaces = static_cast<std::size_t>(pnof_subspaces);
  const std::size_t n_virtual_pairs_each = static_cast<std::size_t>(pnof_coupling - 1);
  const std::size_t n_virtual_pairs_total = n_subspaces * n_virtual_pairs_each;
  if (n_subspaces > occupied_pairs.size()) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: PNOF_SUBSPACES=" + std::to_string(pnof_subspaces) +
        " requires that many occupied orbital pairs (one HOMO-side pair per subspace), but this "
        "basis only has " + std::to_string(occupied_pairs.size()) +
        " occupied pair(s) available -- reduce PNOF_SUBSPACES or use a basis/system with more "
        "occupied orbitals");
  }
  if (n_virtual_pairs_total > unoccupied_pairs.size()) {
    throw std::runtime_error(
        "buildOrbitalSubspaces: PNOF_SUBSPACES=" + std::to_string(pnof_subspaces) +
        " with PNOF_COUPLING=" + std::to_string(pnof_coupling) + " (" +
        std::to_string(n_virtual_pairs_each) + " unoccupied pair(s) per subspace) requires " +
        std::to_string(n_virtual_pairs_total) +
        " DISJOINT unoccupied orbital pairs in total (every subspace's virtual pairs are "
        "separate from every other's), but this basis only has " +
        std::to_string(unoccupied_pairs.size()) +
        " unoccupied pair(s) available -- reduce PNOF_SUBSPACES/PNOF_COUPLING or use a larger "
        "basis");
  }

  OrbitalSubspaceTable table;
  table.subspaces.reserve(n_subspaces);
  for (std::size_t k = 0; k < n_subspaces; ++k) {
    OrbitalSubspace subspace;
    subspace.occupied = occupied_pairs[k];
    subspace.unoccupied.reserve(n_virtual_pairs_each);
    for (std::size_t j = 0; j < n_virtual_pairs_each; ++j) {
      subspace.unoccupied.push_back(unoccupied_pairs[k * n_virtual_pairs_each + j]);
    }
    table.subspaces.push_back(std::move(subspace));
  }
  for (std::size_t k = n_subspaces; k < occupied_pairs.size(); ++k) {
    table.frozen_occupied.push_back(occupied_pairs[k]);
  }
  for (std::size_t k = n_virtual_pairs_total; k < unoccupied_pairs.size(); ++k) {
    table.frozen_unoccupied.push_back(unoccupied_pairs[k]);
  }
  std::sort(table.frozen_occupied.begin(), table.frozen_occupied.end());
  std::sort(table.frozen_unoccupied.begin(), table.frozen_unoccupied.end());
  return table;
}

}  // namespace rerdmft
