#ifndef RERDMFT_RIJKONLYFOCK_H
#define RERDMFT_RIJKONLYFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "JK_only.h"
#include "Matrix.h"
#include "UkbFockMatrixRi.h"

namespace rerdmft {

// RI_4C counterpart of UkbJkOnlyFock.h's ukbJkOnlyFockMatrix (the GENERIC, not the separable-sum
// "Fast" variant -- see this file's own comment for why no RI-specific "Fast" path is needed):
// builds a JK_only functional's generalized Fock matrix from a RESOLUTION-OF-IDENTITY 3-center
// tensor transformed into the CURRENT orbital basis (UKB/RiMoEri.h's buildRiMoEri), instead of
// on-demand AO integrals. Reuses `jkHartreeCoupling`/`jkExchangeCoupling`'s own already-validated
// coupling matrices UNCHANGED, and feeds them into Hessian_opt/JkOnlyFock.h's jkOnlyFockMatrix<T,
// Eri> (generalized to accept any Eri this session, see that header) with `Eri = RiMoEri` -- no
// JK_only-specific math is re-derived here, only the integral SOURCE changes.
//
// Why no "Fast" RI variant is needed: UkbJkOnlyFockFast.h's own separable-density trick exists to
// avoid paying the FULL O(n_mo) AO-direct integral pass; RiMoEri's own MO-RI transform (buildRiMoEri)
// already costs O(n_aux*n_total*(nl^2+ns^2)) ONCE per call, independent of how many densities a
// generic (non-separable) formula would otherwise need -- the RI tensor's aux-index contraction
// already amortizes that cost uniformly across EVERY one of the 10 JK_only functionals, so there is
// nothing left to optimize by functional-specific separability here.
//
// `eri` supplies large_basis/small_basis/eri3_LL/eri3_SS/v_total (UKB/UkbFockMatrixRi.h's
// RiDirectEriSource). `c_rkb` is the RKB-basis (n_rkb x n_total) natural-spinor coefficient
// matrix; `h_rkb` is the RKB-basis one-electron Hamiltonian; `occupations`/`functional`/`f_l`/
// `power_alpha` are exactly `jkHartreeCoupling`/`jkExchangeCoupling`'s own arguments. Returns the
// n_total x n_total GenFock matrix, same convention `jkOnlyFockMatrix`/`ukbJkOnlyFockMatrix`
// already use.
Matrix<std::complex<double>> riJkOnlyFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                                  const RiDirectEriSource& eri,
                                                  const Matrix<std::complex<double>>& c_rkb,
                                                  const std::vector<double>& occupations,
                                                  JkFunctional functional, std::size_t f_l = 0,
                                                  double power_alpha = 1.0);

}  // namespace rerdmft

#endif  // RERDMFT_RIJKONLYFOCK_H
