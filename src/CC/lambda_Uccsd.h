#ifndef RERDMFT_CC_LAMBDA_UCCSD_H
#define RERDMFT_CC_LAMBDA_UCCSD_H

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string>

#include "Matrix.h"
#include "Uccsd.h"
#include "gauss_rdm.h"

namespace rerdmft {

// Left-hand (Lambda) equations of the complex spin-orbital CCSD and the one-/two-particle density matrices built from
// (T, Lambda) -- doc/uccsd.tex, Section 9. The equations are PySCF's gccsd_lambda.py / gccsd_rdm.py in this note's
// conventions, validated against PySCF with complex orbitals (Lagrangian stationarity, finite differences of the energy
// with respect to h and to the integrals).
struct CcRdmSettings {
  int max_iterations = 100;  // Lambda iterations
  double convergence = 1e-8; // Lambda amplitude-change threshold is 100 * convergence (as for T)
  int diis_size = 8;
  double level_shift = 0.0;
  double threshold = 1e-10;  // matrix elements with |value| below this are not written
  std::string dm1_path, dm2_path;
  const CartesianSpinors* gauss = nullptr;  // when set, the density matrices are also written in the Gaussian primitive basis
  GaussRdmSettings gauss_settings;
  double reference_total_energy = 0.0;  // SCF total energy (E_0 + nuclear repulsion), to report the total energy from the densities
};

// Global spinor index (position in the one-electron/integral index space) of every output spinor, in output order.
std::vector<std::size_t> fileOrderGlobals(const CcOrbitalSpaces& spaces);

struct CcRdmResult {
  bool lambda_converged = false;
  int lambda_iterations = 0;
  std::size_t n_dm1 = 0, n_dm2 = 0;  // records written (excluding the terminating zero record)
  double trace_dm1 = 0.0;            // sum_p gamma_pp (the number of electrons)
  double e_elec_rdm = 0.0;           // sum h gamma + 1/2 sum <pq|rs> Gamma, from the elements actually written (no nuclear repulsion)
  double e_elec_cc = 0.0;            // Re(E_0 + E_c) of the converged amplitudes
  double e_total_rdm = 0.0;          // e_elec_rdm + nuclear repulsion (= reference_total_energy - E_0)
  double trace_dm2 = 0.0;            // sum_pq Gamma_pqpq (= N(N-1))
};

// One-particle density gamma_pq = <a_p^+ a_q> (Hermitian part) written as records (int32 p, int32 q, float64 re, float64 im) and
// two-particle density Gamma_pqrs = <a_p^+ a_q^+ a_s a_r> as (int32 p, q, r, s, float64 re, float64 im) for p<q, r<s only
// (antisymmetry gives the rest); indices are 1-based in Kramers-pair order (CcOrbitalSpaces::file_index); each file ends with
// an all-zero record. Frozen-core spinors are included (their mean-field contributions). Energy: E = sum h_pq gamma_pq +
// 1/2 sum <pq|rs> Gamma_pqrs.
CcRdmResult runUccsdRdm(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const CcOrbitalSpaces& spaces,
                        const UccsdResult& amplitudes, const CcRdmSettings& settings, std::ostream& log);

// UCCSD followed by the Lambda equations and the density matrix files (main.cpp's CC_RDM path); same report as
// runUccsdReport plus the CC_RDM block.
template <typename T, typename Eri>
UccsdResult runUccsdRdmReport(const std::string& label, const Matrix<T>& h_mo, const Eri& eri, std::size_t eri_shift,
                              const CcOrbitalSpaces& spaces, const UccsdSettings& settings, const CcRdmSettings& rdm_settings,
                              double scf_total_energy, std::ostream& log) {
  Matrix<CcScalar> h(h_mo.rows(), h_mo.cols());
  for (std::size_t i = 0; i < h_mo.rows(); ++i)
    for (std::size_t j = 0; j < h_mo.cols(); ++j) h(i, j) = CcScalar(h_mo(i, j));
  const auto source = makeCcIntegrals(eri, eri_shift);
  log << "\n=== UCCSD (" << label << ") ===\n";
  const UccsdResult result = runUccsd(h, *source, spaces.occ_frozen, spaces.occ_active, spaces.vir, settings, log);
  log << std::fixed << std::setprecision(10);
  log << "  UCCSD correlation energy : " << result.e_corr << " Eh" << (result.converged ? "" : "   (NOT CONVERGED)") << "\n";
  log << "  SCF total energy         : " << scf_total_energy << " Eh\n";
  log << "  UCCSD total energy       : " << scf_total_energy + result.e_corr << " Eh\n";
  if (result.converged) {
    CcRdmSettings rs = rdm_settings;
    rs.reference_total_energy = scf_total_energy;
    runUccsdRdm(h, *source, spaces, result, rs, log);
  }
  else log << "  CC_RDM skipped: the amplitudes did not converge\n";
  return result;
}

}  // namespace rerdmft

#endif  // RERDMFT_CC_LAMBDA_UCCSD_H
