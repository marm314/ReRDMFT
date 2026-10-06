#include "RiNonRelFullOptFock.h"

#include <stdexcept>

#include "JkOnlyFock.h"
#include "NonRelSpinRiMoEri.h"
#include "PccdFock.h"
#include "PnofFock.h"

namespace rerdmft {

namespace {

// The alpha-block (first n_spatial) columns of `c_current` -- identical to the beta-block
// columns throughout NON_REL's FULL_OPTIMIZATION, see NonRelSpinRiMoEri.h's own comment.
Matrix<double> spatialHalf(const Matrix<double>& c_current, std::size_t n_spatial) {
  const std::size_t n_total = 2 * n_spatial;
  if (c_current.rows() != n_spatial || c_current.cols() != n_total) {
    throw std::runtime_error("RiNonRelFullOptFock: c_current dimensions do not match n_spatial/n_total");
  }
  Matrix<double> c_spatial(n_spatial, n_spatial);
  for (std::size_t mu = 0; mu < n_spatial; ++mu)
    for (std::size_t p = 0; p < n_spatial; ++p) c_spatial(mu, p) = c_current(mu, p);
  return c_spatial;
}

// h_mo(p,q) at the CURRENT (possibly rotated) c_current, from the FIXED bare-spatial-AO one-
// electron Hamiltonian `h_core_ao` (n_spatial x n_spatial -- main.cpp's own h_core_nonrel, the
// SAME role UKB/RiPnofFock.h's `h_rkb` plays for C4_SPINOR). `c_current` is n_spatial x n_total,
// so a plain C^T h_core_ao C is dimensionally valid (unlike multiplying by the ALREADY
// spin-orbital-expanded h_spin, which is n_total x n_total and would silently read past the end
// of `c_current`'s n_spatial rows) -- but it is not yet CORRECT: with nothing to keep the two
// spin blocks apart, it also produces nonzero alpha-beta cross-spin entries from the tied-but-
// spin-agnostic `c_current`, which the exact `closedShellSpinOrbitalOneElectron` convention
// (main.cpp's own `h_spin`, which every non-RI call site builds this way) explicitly zeroes --
// a spin-free one-electron operator never connects different spins. Enforcing that same rule
// here, explicitly, is what makes this match `h_spin` at the identity rotation.
Matrix<double> buildSpinOneElectron(const Matrix<double>& h_core_ao, const Matrix<double>& c_current,
                                     std::size_t n_spatial) {
  const std::size_t n_total = 2 * n_spatial;
  const Matrix<double> h_mo_raw = transpose(c_current) * (h_core_ao * c_current);
  Matrix<double> h_mo(n_total, n_total, 0.0);
  for (std::size_t p = 0; p < n_total; ++p) {
    for (std::size_t q = 0; q < n_total; ++q) {
      if (p / n_spatial == q / n_spatial) h_mo(p, q) = h_mo_raw(p, q);
    }
  }
  return h_mo;
}

}  // namespace

Matrix<double> riPnofFockMatrixNonRel(PnofFunctional functional, const Matrix<double>& h_core_ao,
                                       const Matrix<double>& eri3_L, std::size_t n_spatial,
                                       const Matrix<double>& c_current,
                                       const std::vector<PnofGeminal>& geminals,
                                       const std::vector<double>& occupations) {
  if (h_core_ao.rows() != n_spatial || h_core_ao.cols() != n_spatial) {
    throw std::runtime_error("riPnofFockMatrixNonRel: h_core_ao dimensions do not match n_spatial");
  }
  const Matrix<double> c_spatial = spatialHalf(c_current, n_spatial);
  const RiNonRelSpinMoEri ri_mo = buildRiNonRelSpinMoEri(eri3_L, n_spatial, c_spatial);
  const Matrix<double> h_mo = buildSpinOneElectron(h_core_ao, c_current, n_spatial);
  return pnofFockMatrix<double, RiNonRelSpinMoEri>(functional, h_mo, ri_mo, geminals, occupations,
                                                    /*relativistic=*/false);
}

Matrix<double> riJkOnlyFockMatrixNonRel(const Matrix<double>& h_core_ao, const Matrix<double>& eri3_L,
                                         std::size_t n_spatial, const Matrix<double>& c_current,
                                         const std::vector<double>& occupations,
                                         JkFunctional functional, std::size_t f_l,
                                         double power_alpha) {
  if (h_core_ao.rows() != n_spatial || h_core_ao.cols() != n_spatial) {
    throw std::runtime_error("riJkOnlyFockMatrixNonRel: h_core_ao dimensions do not match n_spatial");
  }
  const Matrix<double> two_rdm_h = jkHartreeCoupling(functional, occupations, f_l, power_alpha);
  const Matrix<double> two_rdm_x = jkExchangeCoupling(functional, occupations, f_l, power_alpha);
  const Matrix<double> c_spatial = spatialHalf(c_current, n_spatial);
  const RiNonRelSpinMoEri ri_mo = buildRiNonRelSpinMoEri(eri3_L, n_spatial, c_spatial);
  const Matrix<double> h_mo = buildSpinOneElectron(h_core_ao, c_current, n_spatial);
  return jkOnlyFockMatrix<double, RiNonRelSpinMoEri>(h_mo, ri_mo, occupations, two_rdm_h, two_rdm_x);
}

Matrix<double> riPccdFockMatrixNonRel(const Matrix<double>& h_core_ao, const Matrix<double>& eri3_L,
                                       std::size_t n_spatial, const Matrix<double>& c_current,
                                       const std::vector<std::size_t>& reps,
                                       const std::vector<std::size_t>& bar, std::size_t n_core,
                                       std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                                       const std::vector<double>& occupations) {
  if (h_core_ao.rows() != n_spatial || h_core_ao.cols() != n_spatial) {
    throw std::runtime_error("riPccdFockMatrixNonRel: h_core_ao dimensions do not match n_spatial");
  }
  const Matrix<double> c_spatial = spatialHalf(c_current, n_spatial);
  const RiNonRelSpinMoEri ri_mo = buildRiNonRelSpinMoEri(eri3_L, n_spatial, c_spatial);
  const Matrix<double> h_mo = buildSpinOneElectron(h_core_ao, c_current, n_spatial);
  return pccdFockMatrix<double, RiNonRelSpinMoEri>(h_mo, ri_mo, reps, bar, n_core, n_occ, n_vir, rdm,
                                                    occupations);
}

}  // namespace rerdmft
