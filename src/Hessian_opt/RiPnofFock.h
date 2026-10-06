#ifndef RERDMFT_RIPNOFFOCK_H
#define RERDMFT_RIPNOFFOCK_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "PNOFs.h"
#include "UkbFockMatrixRi.h"

namespace rerdmft {

// RI_4C counterpart of UkbPnofFock.h's ukbPnofFockMatrix: builds PNOF's generalized Fock matrix
// (GenFock, Dyall Eq. 8.30 convention) from a RESOLUTION-OF-IDENTITY 3-center tensor transformed
// into the CURRENT orbital basis (UKB/RiMoEri.h's buildRiMoEri), instead of either a stored
// MO-basis two-electron tensor (rotated every ADAM step, O(n^5)) or on-demand AO integrals
// (UkbGenFockPrimitives.h's ukbGenFockBuild, O(n^4)-shaped). Reuses `buildPnofFullTwoRdm`'s own
// already-validated 2-RDM unfold UNCHANGED, and feeds it into the SAME generic
// Hessian_opt/PnofFock.h's pnofFockMatrix<T,Eri> the dense/Cholesky path already uses, with
// `Eri = RiMoEri` -- no PNOF-specific math is re-derived here, only the integral SOURCE changes.
//
// `eri` supplies large_basis/small_basis/eri3_LL/eri3_SS/v_total (UKB/UkbFockMatrixRi.h's
// RiDirectEriSource). `c_rkb` is the RKB-basis (n_rkb x n_total) natural-spinor coefficient
// matrix; `h_rkb` is the RKB-basis one-electron Hamiltonian; `geminals`/`occupations`/
// `relativistic` are exactly `buildPnofFullTwoRdm`'s own arguments. Returns the n_total x n_total
// GenFock matrix, same convention `pnofFockMatrix`/`ukbPnofFockMatrix` already use.
Matrix<std::complex<double>> riPnofFockMatrix(PnofFunctional functional,
                                                const Matrix<std::complex<double>>& h_rkb,
                                                const RiDirectEriSource& eri,
                                                const Matrix<std::complex<double>>& c_rkb,
                                                const std::vector<PnofGeminal>& geminals,
                                                const std::vector<double>& occupations,
                                                bool relativistic);

}  // namespace rerdmft

#endif  // RERDMFT_RIPNOFFOCK_H
