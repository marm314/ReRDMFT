#ifndef RERDMFT_RIPCCDFOCK_H
#define RERDMFT_RIPCCDFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "UkbFockMatrixRi.h"
#include "pCCD.h"

namespace rerdmft {

// RI_4C counterpart of UkbPccdFock.h's ukbPccdFockMatrix: builds pCCD's generalized Fock matrix
// from a RESOLUTION-OF-IDENTITY 3-center tensor transformed into the CURRENT orbital basis
// (UKB/RiMoEri.h's buildRiMoEri), instead of on-demand AO integrals. Reuses
// `buildPccdFullTwoRdm`'s own already-validated unfold UNCHANGED, and feeds it into the SAME
// generic Hessian_opt/PccdFock.h's pccdFockMatrix<T,Eri> the dense/Cholesky path already uses,
// with `Eri = RiMoEri` -- no pCCD-specific math is re-derived here, only the integral SOURCE
// changes.
//
// `eri` supplies large_basis/small_basis/eri3_LL/eri3_SS/v_total (UKB/UkbFockMatrixRi.h's
// RiDirectEriSource). `c_rkb` is the RKB-basis (n_rkb x n_total) natural-spinor coefficient
// matrix; `h_rkb` is the RKB-basis one-electron Hamiltonian; `reps`/`bar`/`n_core`/`n_occ`/
// `n_vir`/`rdm`/`occupations` are exactly `buildPccdFullTwoRdm`'s own arguments. Returns the
// n_total x n_total GenFock matrix, same convention `pccdFockMatrix`/`ukbPccdFockMatrix` already
// use.
Matrix<std::complex<double>> riPccdFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                                const RiDirectEriSource& eri,
                                                const Matrix<std::complex<double>>& c_rkb,
                                                const std::vector<std::size_t>& reps,
                                                const std::vector<std::size_t>& bar, std::size_t n_core,
                                                std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                                const std::vector<double>& occupations);

}  // namespace rerdmft

#endif  // RERDMFT_RIPCCDFOCK_H
