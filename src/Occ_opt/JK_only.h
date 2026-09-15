#ifndef RERDMFT_OCC_OPT_JK_ONLY_H
#define RERDMFT_OCC_OPT_JK_ONLY_H

#include <cstddef>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// The JK-only ("K-functional") density matrix functional approximations
// of Rodriguez-Mayorga et al., "Comprehensive benchmarking of density
// matrix functional approximations", Phys. Chem. Chem. Phys. (2017),
// doi:10.1039/C7CP03349D, Table 1 (f(n_i,n_j) functions, see the paper's
// eqn (5)). These functionals only modify the EXCHANGE part of the
// 2-RDM relative to the trivial single-determinant (SD) approximation
// -- the Hartree/Coulomb part is always the plain occupation-number
// product n_i*n_j, for every functional listed here (see the paper's
// own text: "those that only modify the exchange part of the
// functional and those that modify both" -- Table 1 covers only the
// FORMER group; PNOFs, the latter group, are NOT covered by this file).
enum class JkFunctional {
  kSd,      // Single-determinant / HF: f = n_i*n_j.
  kMbb,     // Muller / Buijse-Baerends: f = sqrt(n_i*n_j).
  kBbc2,    // BBC2 (== BBC1 for a two-electron closed-shell system).
  kCa,      // Csanyi-Arias.
  kCga,     // Csanyi-Goedecker-Arias.
  kMl,      // Marques-Lathiotakis.
  kMlsic,   // Marques-Lathiotakis, self-interaction corrected.
  kGu,      // Goedecker-Umrigar.
  kPower,   // (n_i*n_j)^alpha.
};

// f(n_i,n_j) itself (Table 1), for the exchange part of the 2-RDM. `i`,
// `j` are the (0-based) orbital indices of n_i, n_j themselves -- needed
// (not just the occupation VALUES) because BBC2's own definition splits
// on whether i, j lie in the "strongly occupied" set [1,F_L] or the
// "weakly occupied"/virtual set (F_L,infinity), a partition by ORBITAL
// INDEX, not by occupation number. `f_l` is the (0-based) SIZE of the
// strongly-occupied set -- orbital index p is "in [1,F_L]" (the paper's
// own 1-based notation) iff `p < f_l` here, and "in (F_L,infinity)" iff
// `p >= f_l`; only meaningful for kBbc2, ignored otherwise. `power_alpha`
// is the POWER functional's own exponent (fitted per system in the
// paper -- the caller's job to supply, not derived here); only
// meaningful for kPower, ignored otherwise.
double jkExchangeFunction(JkFunctional functional, double n_i, double n_j, std::size_t i,
                           std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// two_rdm_H(p,q) = n_p * n_q for every (p,q) -- identical for all
// functionals in this file (see the class comment above): the plain
// Hartree/Coulomb coupling matrix expected by
// Hessian_opt/HartreeExchangeGradient.h's/HartreeExchangeHessian.h's
// `two_rdm_h` parameter.
Matrix<double> jkHartreeCoupling(const std::vector<double>& occupations);

// two_rdm_X(p,q) = jkExchangeFunction(functional, n_p, n_q, p, q, f_l,
// power_alpha) for every (p,q) -- Table 1's own f(n_i,n_j), fed
// DIRECTLY (neither doubled nor halved) into
// Hessian_opt/HartreeExchangeGradient.h's/HartreeExchangeHessian.h's
// existing `two_rdm_x` parameter and its
// `two_rdm_pqrs = (1/2)[two_rdm_h(p,q) delta_pr delta_qs -
// two_rdm_x(p,q) delta_ps delta_qr]` assembly formula. That formula
// ALREADY carries a (1/2) prefactor (this project's own N(N-1)/2 2-RDM
// normalization, see [[project-hessian-opt]]), so the assembled
// off-diagonal exchange element works out to
// `D_pq,qp = -(1/2) * jkExchangeFunction(...)`, e.g.
// `-(1/2)*sqrt(n_p*n_q)` for Muller/BBB -- DELIBERATELY the SAME
// convention already used (and validated against real HF/DHF energies)
// for ordinary HF/DHF, where `two_rdm_x(p,q) = n_p*n_q` reduces to
// EXACTLY the kSd row here (`f = n_i*n_j`) with no extra scaling.
// Confirmed with the user explicitly (2026-09-16) rather than matching
// the source paper's own un-halved 2D convention, which would instead
// need `2 * jkExchangeFunction(...)` here.
//
// We use this SAME table (including for opposite-flavor exchange, e.g.
// occupied-positive-energy/negative-energy or Kramers-partner exchange
// in the relativistic C4_DHF case) even where the paper's own
// derivation assumed a nonrelativistic same-spin exchange -- an
// explicit simplifying choice for this project, not a claim that the
// paper itself covers the relativistic case.
Matrix<double> jkExchangeCoupling(JkFunctional functional, const std::vector<double>& occupations,
                                   std::size_t f_l = 0, double power_alpha = 1.0);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_JK_ONLY_H
