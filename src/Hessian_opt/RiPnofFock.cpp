#include "RiPnofFock.h"

#include <stdexcept>

#include "PnofFock.h"
#include "RiMoEri.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> riPnofFockMatrix(PnofFunctional functional, const Matrix<C>& h_rkb, const RiDirectEriSource& eri,
                            const Matrix<C>& c_rkb, const std::vector<PnofGeminal>& geminals,
                            const std::vector<double>& occupations, bool relativistic) {
  const std::size_t n_total = occupations.size();
  if (c_rkb.cols() != n_total) {
    throw std::runtime_error("riPnofFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("riPnofFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_total
  const RiMoEri ri_mo = buildRiMoEri(eri, c_ukb);
  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  return pnofFockMatrix<C, RiMoEri>(functional, h_mo, ri_mo, geminals, occupations, relativistic);
}

}  // namespace rerdmft
