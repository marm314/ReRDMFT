#ifndef RERDMFT_AUXILIARYBASIS_H
#define RERDMFT_AUXILIARYBASIS_H

#include <vector>

#include "MolecularBasis.h"

namespace rerdmft {

// AUX_BASIS_TYPE (Input.h): which automatic RI auxiliary-basis construction recipe to use.
enum class AuxBasisType { kAuto, kPauto };

// Builds an automatically-generated resolution-of-identity (RI) auxiliary basis from
// `orbital_basis`, per atomic center -- Yang, Rendell, Frisch, J. Chem. Phys. 127, 074102 (2007),
// as implemented by MOLGW's own `init_auxil_basis_set_auto` (m_basis_set.f90), ported and
// VALIDATED against it this session (project memory project-ri-pauto-kr-validation: PySCF-checked
// RI accuracy ~4e-5 Hartree for Kr/dyall-v2z, both the Large-only and Small-component-only
// sectors, with the defaults below).
//
// Recipe, per center: build a candidate list of (exponent, angular momentum) pairs --
//   kAuto:  each orbital PRIMITIVE's own (exponent, am) doubled (self-products phi_i*phi_i only).
//   kPauto: every ORDERED PAIR of orbital shells at this center (including a shell with itself)
//           and every pair of their primitives: (exponent_i+exponent_j, am_i+am_j) -- the full set
//           that genuine orbital PRODUCTS phi_i*phi_j produce, not just self-products. Richer
//           candidate set than kAuto, same O(1) extra implementation cost.
// Then repeatedly: take the largest remaining candidate exponent, pull in every remaining
// candidate within a ratio `f_sam` of it, take the (log-space, overflow-safe -- see this file's
// .cpp for why a plain running-product geometric mean is NOT safe here) geometric mean of the
// UNIQUE (exponent, am) pairs in that cluster, and emit ONE aux shell per angular momentum
// `0..am_selected` at that exponent, where `am_selected = min(max(am in cluster, running
// am_current), lmax_abs)`. `lmax_abs = max(lmax_obs + lmax_inc, 2*lval(Z))`, `lval` depending on
// the center's atomic number alone (0 for Z<=2, 1 for Z<=18, 2 for Z<=54, 3 otherwise) -- a HARD
// cap applied regardless of how high the raw candidate angular momentum goes.
//
// Returned shells are single-primitive (uncontracted), coefficient 1.0 (NOT yet individually
// normalized -- run `normalizeCartesianBasis` on the result first, exactly as for any other fresh
// BasisFunction list; this function does not call it itself so a caller building a LARGE+SMALL
// combined aux basis, as C4_SPINOR needs, can normalize the whole combined list in one pass).
// `f_sam`/`lmax_inc` are MOLGW's own defaults (1.5, 1) and not yet exposed as separate keywords.
std::vector<BasisFunction> buildAutoAuxiliaryBasis(const std::vector<BasisFunction>& orbital_basis,
                                                    AuxBasisType type, double f_sam = 1.5,
                                                    int lmax_inc = 1);

}  // namespace rerdmft

#endif  // RERDMFT_AUXILIARYBASIS_H
