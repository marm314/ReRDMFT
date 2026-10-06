#include "RiPccdFock.h"

#include <stdexcept>

#include "PccdFock.h"
#include "RiMoEri.h"

namespace rerdmft {

namespace {
using C = std::complex<double>;
}  // namespace

Matrix<C> riPccdFockMatrix(const Matrix<C>& h_rkb, const RiDirectEriSource& eri, const Matrix<C>& c_rkb,
                            const std::vector<std::size_t>& reps, const std::vector<std::size_t>& bar,
                            std::size_t n_core, std::size_t n_occ, std::size_t n_vir, const PccdRdm& rdm,
                            const std::vector<double>& occupations) {
  const std::size_t n_total = occupations.size();
  if (c_rkb.cols() != n_total) {
    throw std::runtime_error("riPccdFockMatrix: c_rkb column count must match occupations.size()");
  }
  if (h_rkb.rows() != c_rkb.rows() || h_rkb.cols() != c_rkb.rows()) {
    throw std::runtime_error("riPccdFockMatrix: h_rkb dimensions inconsistent with c_rkb");
  }

  const Matrix<C> c_ukb = eri.v_total * c_rkb;  // n_ukb x n_total
  const RiMoEri ri_mo = buildRiMoEri(eri, c_ukb);
  const Matrix<C> h_mo = dagger(c_rkb) * (h_rkb * c_rkb);

  return pccdFockMatrix<C, RiMoEri>(h_mo, ri_mo, reps, bar, n_core, n_occ, n_vir, rdm, occupations);
}

}  // namespace rerdmft
