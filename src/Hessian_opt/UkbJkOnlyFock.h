#ifndef RERDMFT_UKBJKONLYFOCK_H
#define RERDMFT_UKBJKONLYFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "JK_only.h"
#include "Matrix.h"
#include "UkbFockMatrixDirect.h"

namespace rerdmft {

// Builds JK_only's generalized Fock matrix (GenFock, Dyall Eq. 8.30 convention) DIRECTLY from
// on-demand UKB AO integrals (UkbGenFockPrimitives.h's ukbGenFockBuild) -- never a stored
// two-electron tensor of any kind. See project memory project-ukb-direct-genfock-scheme for the
// derivation.
//
// This is the GENERIC (always-correct) form: builds one UKB-AO density per OUTPUT row `q`, for the
// Hartree term (weighted by two_rdm_h(:,q)) and the exchange term (weighted by two_rdm_x(q,:)),
// using jkHartreeCoupling/jkExchangeCoupling's own already-validated coupling matrices directly --
// no per-functional separable-sum decomposition is hand-coded here, so there is no risk of
// misderiving one. This costs O(n_mo) densities per call regardless of functional (the "expensive"
// regime even for functionals that are cheaply separable, e.g. SD/MBB/GU/POWER/CA/CGA/MULLER_AS/
// BBC2 per that memory file's own table) -- a deliberate first-correctness-pass choice; exploiting
// the cheap separable cases is a later, cross-validated-against-this-baseline optimization, not
// done here.
//
// `eri` supplies large_basis/small_basis/v_total (UkbFockMatrixDirect.h's UkbDirectEriSource,
// SCF_DIRECT_4C's own struct, unchanged, reused as-is). `c_rkb` is the RKB-basis (n_rkb x n_mo)
// natural-spinor coefficient matrix; `h_rkb` is the RKB-basis one-electron Hamiltonian (n_rkb x
// n_rkb); `occupations` has size n_mo. Returns the n_mo x n_mo GenFock matrix, same convention
// `hartreeExchangeFockMatrix`/`jkOnlyFockMatrix` already use (feed into OrbitalGradient.h's
// orbitalGradient the same way).
Matrix<std::complex<double>> ukbJkOnlyFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                                  const UkbDirectEriSource& eri,
                                                  const Matrix<std::complex<double>>& c_rkb,
                                                  const std::vector<double>& occupations,
                                                  JkFunctional functional, std::size_t f_l = 0,
                                                  double power_alpha = 1.0);

}  // namespace rerdmft

#endif  // RERDMFT_UKBJKONLYFOCK_H
