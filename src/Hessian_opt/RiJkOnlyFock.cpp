#include "RiJkOnlyFock.h"

#include <stdexcept>

#include "JkOnlyFock.h"
#include "RiMoEri.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> riJkOnlyFockMatrix(const Matrix<C>& h_rkb, const RiDirectEriSource& eri, const Matrix<C>& c_rkb,
                              const std::vector<double>& occupations, JkFunctional functional, std::size_t f_l,
                              double power_alpha) {
  const std::size_t n_mo = occupations.size();
  if (c_rkb.cols() != n_mo) {
    throw std::runtime_error("riJkOnlyFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("riJkOnlyFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  const Matrix<double> two_rdm_h = jkHartreeCoupling(functional, occupations, f_l, power_alpha);
  const Matrix<double> two_rdm_x = jkExchangeCoupling(functional, occupations, f_l, power_alpha);

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_mo
  const RiMoEri ri_mo = buildRiMoEri(eri, c_ukb);
  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  return jkOnlyFockMatrix<C, RiMoEri>(h_mo, ri_mo, occupations, two_rdm_h, two_rdm_x);
}

}  // namespace rerdmft
