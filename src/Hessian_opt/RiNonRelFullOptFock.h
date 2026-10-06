#ifndef RERDMFT_RINONRELFULLOPTFOCK_H
#define RERDMFT_RINONRELFULLOPTFOCK_H

#include <cstddef>
#include <vector>

#include "JK_only.h"
#include "Matrix.h"
#include "PNOFs.h"
#include "pCCD.h"

namespace rerdmft {

// USE_RI counterparts of Hessian_opt/PnofFock.h/JkOnlyFock.h/PccdFock.h for NON_REL's
// FULL_OPTIMIZATION (the real, spin-orbital-doubled case, T=double) -- the direct analogue of
// Hessian_opt/RiPnofFock.h/RiJkOnlyFock.h/RiPccdFock.h (C4_SPINOR, complex, UKB-sourced), but
// built from NON_REL/NonRelSpinRiMoEri.h's RiNonRelSpinMoEri instead of UKB/RiMoEri.h's RiMoEri:
// ONE real, spatial-only RI half-transform (n_spatial, not 2*n_spatial -- see
// NonRelSpinRiMoEri.h's own comment on why alpha/beta are always tied here), then the SAME
// generic pnofFockMatrix<T,Eri>/jkOnlyFockMatrix<T,Eri>/pccdFockMatrix<T,Eri> the dense/Cholesky
// NON_REL path already uses, with `Eri = RiNonRelSpinMoEri`.
//
// `h_core_ao` is the FIXED, bare SPATIAL AO one-electron Hamiltonian (main.cpp's own
// h_core_nonrel, n_spatial x n_spatial -- NEVER the already spin-orbital-expanded h_spin: c_current
// has n_spatial rows, so multiplying by the n_total x n_total h_spin would silently read past the
// end of its rows). `eri3_L`/`n_spatial` are the Large-AO-basis RI 3-center tensor and its spatial
// dimension (AO_ints/ThreeCenterIntegrals.h's convention); `c_current` is the CURRENT spin-orbital
// coefficient matrix (n_spatial x n_total -- bare spatial AO rows, spin-orbital columns, see
// Full_opt/FullOptimization.h's RdmftModel::ukb_direct_fock comment for how the ADAM sub-loop
// builds it from `ukb_direct_c0 * accumulated_rotation`). The one-electron MO transform built from
// `h_core_ao`/`c_current` explicitly zeroes the alpha-beta cross-spin blocks (a spin-free
// one-electron operator never connects different spins, exactly ClosedShellSpinOrbitals.h's
// closedShellSpinOrbitalOneElectron's own rule) -- a plain C^T h_core_ao C would NOT do this on
// its own (`c_current`'s tied-but-spin-agnostic columns leak nonzero cross-spin entries), so this
// is not merely a dimension fix. Returns the n_total x n_total GenFock matrix, same convention
// pnofFockMatrix/jkOnlyFockMatrix/pccdFockMatrix already use.
Matrix<double> riPnofFockMatrixNonRel(PnofFunctional functional, const Matrix<double>& h_core_ao,
                                       const Matrix<double>& eri3_L, std::size_t n_spatial,
                                       const Matrix<double>& c_current,
                                       const std::vector<PnofGeminal>& geminals,
                                       const std::vector<double>& occupations);

Matrix<double> riJkOnlyFockMatrixNonRel(const Matrix<double>& h_core_ao, const Matrix<double>& eri3_L,
                                         std::size_t n_spatial, const Matrix<double>& c_current,
                                         const std::vector<double>& occupations,
                                         JkFunctional functional, std::size_t f_l = 0,
                                         double power_alpha = 1.0);

Matrix<double> riPccdFockMatrixNonRel(const Matrix<double>& h_core_ao, const Matrix<double>& eri3_L,
                                       std::size_t n_spatial, const Matrix<double>& c_current,
                                       const std::vector<std::size_t>& reps,
                                       const std::vector<std::size_t>& bar, std::size_t n_core,
                                       std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                       const std::vector<double>& occupations);

}  // namespace rerdmft

#endif  // RERDMFT_RINONRELFULLOPTFOCK_H
