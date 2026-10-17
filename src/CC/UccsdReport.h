#ifndef RERDMFT_CC_UCCSDREPORT_H
#define RERDMFT_CC_UCCSDREPORT_H

// Included at the end of Uccsd.h.
#include <iomanip>

#include "OutputFormat.h"

namespace rerdmft {

template <typename T, typename Eri>
UccsdResult runUccsdReport(const std::string& label, const Matrix<T>& h_mo, const Eri& eri, std::size_t eri_shift,
                           const CcOrbitalSpaces& spaces, const UccsdSettings& settings, double scf_total_energy,
                           std::ostream& log) {
  Matrix<CcScalar> h(h_mo.rows(), h_mo.cols());
  for (std::size_t i = 0; i < h_mo.rows(); ++i)
    for (std::size_t j = 0; j < h_mo.cols(); ++j) h(i, j) = CcScalar(h_mo(i, j));
  const auto source = makeCcIntegrals(eri, eri_shift);
  log << "\n=== UCCSD (" << label << ") ===\n";
  const UccsdResult result = runUccsd(h, *source, spaces.occ_frozen, spaces.occ_active, spaces.vir, settings, log);
  printEnergyLine(log, 2, "UMP2 correlation energy", result.e_mp2);
  printEnergyLine(log, 2, "UMP2 total energy", scf_total_energy + result.e_mp2);
  printEnergyLine(log, 2, "UCCSD correlation energy", result.e_corr, result.converged ? " Hartree" : " Hartree   (NOT CONVERGED)");
  printEnergyLine(log, 2, "SCF total energy", scf_total_energy);
  printEnergyLine(log, 2, "UCCSD total energy", scf_total_energy + result.e_corr);
  return result;
}

}  // namespace rerdmft

#endif  // RERDMFT_CC_UCCSDREPORT_H
