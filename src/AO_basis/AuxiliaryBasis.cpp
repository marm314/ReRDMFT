#include "AuxiliaryBasis.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

#include "Element.h"
#include "Shell.h"

namespace rerdmft {

namespace {

// (exponent, angular momentum) candidate pair.
struct Candidate {
  double exponent;
  int am;
};

// lval(Z) exactly as MOLGW's own get_lmax_abs (m_basis_set.f90): a coarse, Z-only bracket.
int lval(int z) {
  if (z <= 2) return 0;
  if (z <= 18) return 1;
  if (z <= 54) return 2;
  return 3;
}

int lmaxAbs(int lmax_obs, int z, int lmax_inc) { return std::max(lmax_obs + lmax_inc, 2 * lval(z)); }

// One physical shell at a center: its angular momentum and primitive exponents (coefficients are
// irrelevant here -- the recipe only ever looks at exponents/angular momenta, never contraction
// weights).
struct CenterShell {
  int am;
  std::vector<double> exponents;
};

// Builds the PAUTO candidate list: every ordered pair of shells at this center (including a shell
// with itself) and every pair of their primitives, exponent summed, am summed.
std::vector<Candidate> pautoCandidates(const std::vector<CenterShell>& shells) {
  std::vector<Candidate> out;
  for (const auto& si : shells) {
    for (double ei : si.exponents) {
      for (const auto& sj : shells) {
        for (double ej : sj.exponents) {
          out.push_back({ei + ej, si.am + sj.am});
        }
      }
    }
  }
  return out;
}

// Builds the AUTO candidate list: each primitive's own exponent/am doubled.
std::vector<Candidate> autoCandidates(const std::vector<CenterShell>& shells) {
  std::vector<Candidate> out;
  for (const auto& s : shells) {
    for (double e : s.exponents) out.push_back({2.0 * e, 2 * s.am});
  }
  return out;
}

// Greedy clustering shared by AUTO and PAUTO (only the candidate list differs): repeatedly take
// the largest remaining exponent, absorb every remaining candidate within ratio f_sam of it, take
// the UNIQUE (exponent, am) pairs' geometric mean (log-space -- see below), emit one aux shell per
// am in [0, am_selected]. Returns (exponent, am_selected) per cluster, in the SAME order MOLGW's
// own loop produces them (largest exponent first).
std::vector<std::pair<double, int>> clusterCandidates(std::vector<Candidate> candidates, double f_sam,
                                                        int lmax_abs) {
  std::vector<std::pair<double, int>> groups;
  std::vector<bool> remaining(candidates.size(), true);
  int am_current = 0;
  std::size_t n_left = candidates.size();
  while (n_left > 0) {
    std::size_t idx = 0;
    double best = -1.0;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      if (remaining[k] && candidates[k].exponent > best) {
        best = candidates[k].exponent;
        idx = k;
      }
    }
    const double exponent_current = candidates[idx].exponent;
    std::vector<std::size_t> trial = {idx};
    remaining[idx] = false;
    --n_left;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      if (remaining[k] && exponent_current / candidates[k].exponent < f_sam) {
        trial.push_back(k);
        remaining[k] = false;
        --n_left;
      }
    }

    // Unique (exponent, am) pairs within the trial set (tolerance matches MOLGW's own 1e-6).
    std::vector<Candidate> unique;
    for (std::size_t t : trial) {
      const Candidate& c = candidates[t];
      bool is_new = true;
      for (const Candidate& u : unique) {
        if (u.am == c.am && std::abs(u.exponent - c.exponent) < 1e-6) {
          is_new = false;
          break;
        }
      }
      if (is_new) unique.push_back(c);
    }

    // Log-space geometric mean: a running product of many large exponents (deep-core sums,
    // O(1e6)-O(1e7), in clusters that can reach dozens of members for a PAUTO candidate pool)
    // overflows IEEE754 double BEFORE the n-th root is applied -- found this session validating
    // this exact recipe for Kr's small-component sector (project memory
    // project-ri-pauto-kr-validation). Summing logs and exponentiating once is stable regardless
    // of cluster size.
    double log_sum = 0.0;
    for (const Candidate& u : unique) log_sum += std::log(u.exponent);
    const double exponent_selected = std::exp(log_sum / static_cast<double>(unique.size()));

    int am_max_trial = 0;
    for (std::size_t t : trial) am_max_trial = std::max(am_max_trial, candidates[t].am);
    const int am_selected = std::min(std::max(am_max_trial, am_current), lmax_abs);
    am_current = am_selected;

    groups.emplace_back(exponent_selected, am_selected);
  }
  return groups;
}

}  // namespace

std::vector<BasisFunction> buildAutoAuxiliaryBasis(const std::vector<BasisFunction>& orbital_basis,
                                                    AuxBasisType type, double f_sam, int lmax_inc) {
  if (orbital_basis.empty()) {
    throw std::runtime_error("buildAutoAuxiliaryBasis: empty orbital basis");
  }

  // Deduplicate into UNIQUE physical shells (center, l, exponent list), WITHOUT assuming
  // `groupIntoShells`'s consecutive-layout invariant -- deliberately NOT using that function.
  // An ordinary (large-component) orbital basis has each shell repeated once per cartesian
  // component (all sharing the same l/exponents/center), which this collapses back to one entry
  // per physical shell; the UKB small component is NOT shell-consecutive at all and genuinely
  // contains DUPLICATE (l, exponent) shells from different parent raising/lowering operations
  // (confirmed this session, project memory project-ri-pauto-kr-validation: Kr's small component
  // has 366 raw AOs but only 51 UNIQUE (am, exponent) shells) -- deduplicating by the full
  // (center, l, exponents) key handles both cases uniformly, matching the already-PySCF-validated
  // Python recipe's own `sorted(set(...))` dedup exactly.
  using CenterKey = std::tuple<double, double, double>;
  std::map<CenterKey, std::map<std::pair<int, std::vector<double>>, const BasisFunction*>> shells_by_center;
  for (const BasisFunction& fn : orbital_basis) {
    shells_by_center[{fn.x, fn.y, fn.z}][{fn.l, fn.exponents}] = &fn;
  }

  std::vector<BasisFunction> aux_basis;
  for (const auto& [center, shell_map] : shells_by_center) {
    const BasisFunction& center_rep = *shell_map.begin()->second;
    const int z = atomicNumber(center_rep.element);

    std::vector<CenterShell> center_shells;
    int lmax_obs = 0;
    for (const auto& [key, rep_ptr] : shell_map) {
      center_shells.push_back({key.first, rep_ptr->exponents});
      lmax_obs = std::max(lmax_obs, key.first);
    }

    const int lmax_abs = lmaxAbs(lmax_obs, z, lmax_inc);
    std::vector<Candidate> candidates =
        (type == AuxBasisType::kPauto) ? pautoCandidates(center_shells) : autoCandidates(center_shells);
    const std::vector<std::pair<double, int>> groups = clusterCandidates(std::move(candidates), f_sam, lmax_abs);

    for (const auto& [exponent, am_selected] : groups) {
      for (int am = 0; am <= am_selected; ++am) {
        for (const CartesianExponents& c : cartesianComponents(am)) {
          BasisFunction fn;
          fn.element = center_rep.element;
          fn.x = center_rep.x;
          fn.y = center_rep.y;
          fn.z = center_rep.z;
          fn.l = am;
          fn.cartesian = c;
          fn.exponents = {exponent};
          fn.coefficients = {1.0};
          aux_basis.push_back(std::move(fn));
        }
      }
    }
  }
  return aux_basis;
}

}  // namespace rerdmft
