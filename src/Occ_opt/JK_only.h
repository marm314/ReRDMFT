#ifndef RERDMFT_OCC_OPT_JK_ONLY_H
#define RERDMFT_OCC_OPT_JK_ONLY_H

#include <cstddef>
#include <string>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// The JK-only ("K-functional") density matrix functional approximations
// of Rodriguez-Mayorga et al., "Comprehensive benchmarking of density
// matrix functional approximations", Phys. Chem. Chem. Phys. (2017),
// doi:10.1039/C7CP03349D, Table 1 (f(n_i,n_j) functions, see the paper's
// eqn (5)). These functionals only modify the EXCHANGE part of the
// 2-RDM relative to the trivial single-determinant (SD) approximation
// -- the Hartree/Coulomb part is the plain occupation-number product
// n_i*n_j for EVERY functional here EXCEPT `kMullerAs` (see its own
// comment below) -- see the paper's own text: "those that only modify
// the exchange part of the functional and those that modify both" --
// Table 1 covers only the FORMER group; PNOFs, the latter group, are
// NOT covered by this file.
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
  // Muller, ANTISYMMETRIZED: g(n_i,n_j) = (n_i*n_j - sqrt(n_i*n_j))/2,
  // used for BOTH the Hartree AND exchange coupling (two_rdm_h(p,q) =
  // two_rdm_x(p,q) = g(n_p,n_q)), NOT just Table 1's own exchange-only
  // f(n_i,n_j) = sqrt(n_i*n_j). Project-internal, not from the PCCP
  // 2017 paper: g is Hartree's own n_i*n_j minus Muller's own
  // sqrt(n_i*n_j) (an earlier variant used their AVERAGE instead --
  // see project memory, project_jk_only.md, for why this form was
  // tried next: the averaged version's SQP occupation optimization did
  // not converge for water/STO-3G, landing on the box boundary after
  // 100 iterations, unlike every other functional here), and setting
  // two_rdm_h == two_rdm_x this way
  // makes the resulting dense 2-RDM ansatz Gamma_pqrs = g(n_p,n_q)
  // [delta_pr delta_qs - delta_ps delta_qr] GENUINELY antisymmetric
  // (Gamma_pqrs = -Gamma_qprs EXACTLY, for ANY occupations, fractional
  // or not) -- confirmed to be exactly what
  // GeneralizedHessian.h's boxed orbital-rotation Hessian (Eq. 9)
  // needs to reproduce the true numerical Hessian for a fractional-
  // occupation ansatz (see project memory,
  // project_jk_only_relativistic_hessian_gap.md's "DEFINITIVE, fully
  // controlled confirmation" entry: plain Muller's own two_rdm_h !=
  // two_rdm_x breaks antisymmetry and gives a large, confirmed Hessian
  // defect; substituting two_rdm_x=two_rdm_h at the SAME fractional
  // occupations fixes it exactly). `kMullerAs` is the ONLY functional
  // in this file whose Hartree coupling is NOT plain n_i*n_j -- see
  // `jkHartreeFunction`'s own comment.
  kMullerAs,
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

// f_H(n_i,n_j): the Hartree/Coulomb coupling function itself. For
// every functional here EXCEPT `kMullerAs`, this is plain n_i*n_j
// (matching `jkExchangeFunction`'s own `kSd` row exactly). For
// `kMullerAs`, it is the SAME g(n_i,n_j) = (n_i*n_j - sqrt(n_i*n_j))/2
// as `jkExchangeFunction`'s own `kMullerAs` row -- i.e. `kMullerAs` is
// the one functional in this file where `jkHartreeFunction ==
// jkExchangeFunction` identically, by deliberate construction (see the
// enum's own comment for why). `i`,`j`,`f_l`,`power_alpha` are threaded
// through purely for signature parity with `jkExchangeFunction` --
// none of them affect this function's value for ANY functional listed
// here (unlike `jkExchangeFunction`, where `i`,`j`,`f_l` matter for
// `kBbc2` specifically).
double jkHartreeFunction(JkFunctional functional, double n_i, double n_j, std::size_t i,
                          std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// D1/D11/D12 of `jkHartreeFunction`, mirroring `jkExchangeFunctionD1`/
// `D11`/`D12`'s own meaning exactly (partials of the Hartree coupling
// function itself now, not the exchange one). For every functional
// except `kMullerAs`: D1=n_j, D11=0, D12=1 (the trivial derivatives of
// n_i*n_j). For `kMullerAs`: identical to `jkExchangeFunctionD1`/`D11`/
// `D12`'s own `kMullerAs` values (same underlying function g).
double jkHartreeFunctionD1(JkFunctional functional, double n_i, double n_j, std::size_t i,
                            std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);
double jkHartreeFunctionD11(JkFunctional functional, double n_i, double n_j, std::size_t i,
                             std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);
double jkHartreeFunctionD12(JkFunctional functional, double n_i, double n_j, std::size_t i,
                             std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// two_rdm_H(p,q) = jkHartreeFunction(functional, n_p, n_q, p, q, f_l,
// power_alpha) for every (p,q) -- n_p*n_q for every functional here
// except `kMullerAs` (see that function's own comment): the Hartree/
// Coulomb coupling matrix expected by
// Hessian_opt/HartreeExchangeGradient.h's/HartreeExchangeHessian.h's
// `two_rdm_h` parameter.
Matrix<double> jkHartreeCoupling(JkFunctional functional, const std::vector<double>& occupations,
                                  std::size_t f_l = 0, double power_alpha = 1.0);

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

// Partial derivatives of jkExchangeFunction's own f(n_i,n_j), needed to
// optimize occupation numbers (Utils/SQP.h) at FIXED orbitals for a
// given functional -- see Occ_opt/OccupationEnergy.h, which is the only
// caller. All three take EXACTLY jkExchangeFunction's own arguments
// (same per-branch index/f_l/power_alpha dispatch, so each piecewise
// functional's derivative uses the SAME branch as the value itself --
// each branch is a smooth, ordinary function of (n_i,n_j) once the
// branch is fixed by the DISCRETE indices i,j,f_l, so differentiating
// within a branch is unambiguous; only WHICH branch applies depends on
// the indices, never on the occupation values themselves).
//
// D1 = df/dn_i (partial wrt the FIRST argument only). Every functional
// here has f(n_i,n_j) = f(n_j,n_i) (symmetric), so df/dn_j at (n_i,n_j)
// equals D1(functional, n_j, n_i, j, i, f_l, power_alpha) -- callers
// needing the second partial should call D1 with arguments swapped,
// not a separate function.
double jkExchangeFunctionD1(JkFunctional functional, double n_i, double n_j, std::size_t i,
                             std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// D11 = d^2f/dn_i^2 (pure second partial wrt the first argument).
double jkExchangeFunctionD11(JkFunctional functional, double n_i, double n_j, std::size_t i,
                              std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// D12 = d^2f/(dn_i dn_j) (mixed second partial). Symmetric in the sense
// that D12(f,n_i,n_j,i,j,...) == D12(f,n_j,n_i,j,i,...) for every
// functional here (Schwarz's theorem plus f's own argument symmetry).
double jkExchangeFunctionD12(JkFunctional functional, double n_i, double n_j, std::size_t i,
                              std::size_t j, std::size_t f_l = 0, double power_alpha = 1.0);

// Maps Input.h's FUNCTIONAL keyword string (already validated there
// against this exact name list, case-insensitive but stored uppercase)
// to the corresponding JkFunctional -- kept here, not in Input.h, so
// that Input.h stays independent of Occ_opt (see its own `functional()`
// comment). Throws std::runtime_error on an unrecognized name (should
// not happen for a string that already passed Input::read's own
// validation).
JkFunctional parseJkFunctional(const std::string& name);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_JK_ONLY_H
