#include "SmallComponentBasis.h"

#include <set>
#include <utility>

namespace rerdmft {

void SmallComponentBasis::build(const std::vector<Atom>& geometry,
                                 const BasisSet& basis_set) {
  functions_.clear();

  for (const auto& atom : geometry) {
    const std::vector<Shell>& large_shells =
        basis_set.shellsForElement(atom.symbol);
    // (small_l, exponent) pairs already emitted for THIS atom, so a
    // primitive exponent shared by more than one large shell on the same
    // center only produces one small shell for a given target small_l.
    std::set<std::pair<int, double>> seen_exponents;
    for (const auto& large_shell : large_shells) {
      std::vector<int> small_ls;
      if (large_shell.l > 0) small_ls.push_back(large_shell.l - 1);
      small_ls.push_back(large_shell.l + 1);

      for (int small_l : small_ls) {
        // d/dx_k of a primitive with angular momentum l decomposes (same
        // exponent a) into an l-1 piece and an l+1 piece whose relative
        // per-primitive weight (proportional to a) differs from the
        // parent shell's own contraction weights. Rather than try to
        // reproduce that per-primitive reweighting on a contracted shell,
        // build one UNCONTRACTED, single-primitive small shell per
        // (exponent, small_l) pair -- exactly the convention used by an
        // independently-verified reference implementation (M. Rodriguez-
        // Mayorga's m_relativistic.f90, MOLGW), which sidesteps the
        // reweighting question entirely (a single primitive has no
        // relative-weight ambiguity) at the cost of a larger, linearly-
        // safe small basis. Cross-shell duplicates (the same exponent
        // appearing in more than one large shell, e.g. a standalone S
        // primitive whose exponent also appears inside a contracted S
        // shell) are only added once, matching the reference exactly.
        for (std::size_t p = 0; p < large_shell.exponents.size(); ++p) {
          const double exponent = large_shell.exponents[p];
          const auto key = std::make_pair(small_l, exponent);
          if (!seen_exponents.insert(key).second) continue;

          for (const auto& cart : cartesianComponents(small_l)) {
            BasisFunction fn;
            fn.element = atom.symbol;
            fn.x = atom.x;
            fn.y = atom.y;
            fn.z = atom.z;
            fn.l = small_l;
            fn.cartesian = cart;
            fn.exponents = {exponent};
            fn.coefficients = {1.0};
            functions_.push_back(std::move(fn));
          }
        }
      }
    }
  }
}

}  // namespace rerdmft
