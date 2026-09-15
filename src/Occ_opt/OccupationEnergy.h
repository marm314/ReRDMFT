#ifndef RERDMFT_OCC_OPT_OCCUPATION_ENERGY_H
#define RERDMFT_OCC_OPT_OCCUPATION_ENERGY_H

#include <cstddef>
#include <vector>

#include "JK_only.h"
#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// The SAME Hartree/exchange-ansatz RDMFT electronic energy as
// Hessian_opt/HartreeExchangeGradient.h's hartreeExchangeEnergy, but
// re-expressed as an EXPLICIT function of the occupation numbers
// themselves (rather than requiring the caller to pre-build
// two_rdm_h/two_rdm_x) for a GIVEN Occ_opt/JK_only.h functional, with
// `h`/`eri` (the converged, FIXED one-/two-electron integrals in the
// natural-orbital/natural-spinor MO basis) held constant. This is the
// objective for OPTIMIZING OCCUPATION NUMBERS ALONE at fixed orbitals
// (Occ_opt/SQP.h): value/gradient/hessian below all take exactly the
// shape SqpValueFn/SqpGradientFn/SqpHessianFn expect (a function of
// `occupations` only).
//
//   E(n) = sum_p n_p*h_pp
//          + (1/2)*sum_pq [ J_pq*n_p*n_q - K_pq*f_X(n_p,n_q) ]
// where J_pq = Re[eri(p,q,p,q)] (Coulomb), K_pq = Re[eri(p,q,q,p)]
// (exchange), f_X = JK_only.h's jkExchangeFunction for `functional` --
// IDENTICAL to hartreeExchangeEnergy's own formula with
// two_rdm_h(p,q)=n_p*n_q, two_rdm_x(p,q)=f_X(n_p,n_q) substituted in
// (see JK_only.h's own two_rdm_H/two_rdm_X comments). An O(n^2) sum.
template <typename T>
double jkFunctionalEnergy(const Matrix<T>& h, const Tensor4<T>& eri,
                           const std::vector<double>& occupations, JkFunctional functional,
                           std::size_t f_l = 0, double power_alpha = 1.0);

// dE/dn_r = h_rr + sum_q J_rq*n_q - sum_q K_rq*D1(f_X)(n_r,n_q)
// where D1 = JK_only.h's jkExchangeFunctionD1 (partial wrt the FIRST
// argument) -- derived by differentiating jkFunctionalEnergy's own sum
// termwise (each (p,q) pair contributes via BOTH the p=r and q=r
// occurrences, which combine into a single sum over q using J/K's own
// p<->q symmetry and f_X's own argument symmetry -- see the .cpp for
// the full derivation). An O(n^2) computation, returned as one
// std::vector<double> entry per orbital.
template <typename T>
std::vector<double> jkFunctionalGradient(const Matrix<T>& h, const Tensor4<T>& eri,
                                          const std::vector<double>& occupations,
                                          JkFunctional functional, std::size_t f_l = 0,
                                          double power_alpha = 1.0);

// d^2E/dn_r dn_s = J_rs - K_rs*D12(f_X)(n_r,n_s)
//                  - delta_rs * sum_q K_rq*D11(f_X)(n_r,n_q)
// where D11/D12 = JK_only.h's jkExchangeFunctionD11/D12 -- derived by
// differentiating the gradient formula above wrt n_s (see the .cpp).
// An O(n^2) computation (the diagonal's own inner sum over q is
// O(n) per row, O(n^2) total over all rows -- never O(n^3)).
template <typename T>
Matrix<double> jkFunctionalHessian(const Matrix<T>& h, const Tensor4<T>& eri,
                                    const std::vector<double>& occupations,
                                    JkFunctional functional, std::size_t f_l = 0,
                                    double power_alpha = 1.0);

}  // namespace rerdmft

#endif  // RERDMFT_OCC_OPT_OCCUPATION_ENERGY_H
