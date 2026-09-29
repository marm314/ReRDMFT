#ifndef RERDMFT_UTILS_FCIDUMP_H
#define RERDMFT_UTILS_FCIDUMP_H

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "Matrix.h"

namespace rerdmft {

// Writes an FCIDUMP file (the MOLPRO/PySCF convention used to hand one- and two-electron MO
// integrals to an external CI/DMRG/FCI code): an "&FCI ... &END" namelist header (NORB, NELEC,
// MS2, ORBSYM all 1 -- ReRDMFT tracks no point-group symmetry, ISYM=1), then one line per
// integral as "VALUE I J K L" (1-indexed): two-electron (IJ|KL) with K,L != 0, one-electron h_IJ
// with K=L=0, and the core/nuclear-repulsion energy last with I=J=K=L=0.
//
// Real orbitals only (MS2 = 0 always -- no spin symmetry breaking to report). `h_mo` and
// `eri_mo` must already be in the final MO basis, in whatever ORDER that basis was actually used
// in (the caller's job): only the unique elements permutational symmetry implies are written --
// (IJ|KL) for I>=J, K>=L, and the composite pair I(I-1)/2+J >= K(K-1)/2+L; h_IJ for I>=J. `Eri` is
// templated on any type with a physics-notation eri_mo(a,b,c,d) == <ab|cd> == chemist(a,c,b,d)
// element interface -- SymmetricEri<double> (direct AO integrals) and CholeskyEri<double> (AO
// Cholesky vectors, CHOLESKY TRUE) both work unchanged, so the caller need not densify either.
template <typename Eri>
void writeFcidump(const std::string& path, std::size_t norb, int n_electrons, int ms2,
                  const Matrix<double>& h_mo, const Eri& eri_mo, double core_energy) {
  if (h_mo.rows() != norb || h_mo.cols() != norb) {
    throw std::runtime_error("writeFcidump: h_mo size does not match norb");
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("writeFcidump: could not open '" + path + "' for writing");
  }
  out << " &FCI NORB=" << norb << ",NELEC=" << n_electrons << ",MS2=" << ms2 << ",\n";
  out << "  ORBSYM=";
  for (std::size_t i = 0; i < norb; ++i) out << "1,";
  out << "\n  ISYM=1,\n &END\n";

  out << std::scientific << std::setprecision(16);
  const auto writeLine = [&](double value, std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    out << std::setw(24) << value << std::setw(4) << i << std::setw(4) << j << std::setw(4) << k << std::setw(4)
        << l << "\n";
  };

  // Two-electron integrals (IJ|KL), the standard 8-fold-unique enumeration for real orbitals:
  // I>=J, K>=I, and if K==I then L<=J else L<=K (equivalently the composite pair I(I-1)/2+J >=
  // K(K-1)/2+L). chemist(I,J,K,L) == physics(I,K,J,L) (SymmetricEri.h/MoIntegralTransform.h).
  for (std::size_t i = 1; i <= norb; ++i) {
    for (std::size_t j = 1; j <= i; ++j) {
      for (std::size_t k = 1; k <= i; ++k) {
        const std::size_t lmax = (k == i) ? j : k;
        for (std::size_t l = 1; l <= lmax; ++l) {
          writeLine(eri_mo(i - 1, k - 1, j - 1, l - 1), i, j, k, l);
        }
      }
    }
  }
  // One-electron integrals h_IJ, I>=J (real orbitals: h is symmetric).
  for (std::size_t i = 1; i <= norb; ++i) {
    for (std::size_t j = 1; j <= i; ++j) {
      writeLine(h_mo(i - 1, j - 1), i, j, 0, 0);
    }
  }
  // Core/nuclear-repulsion energy, last, with all indices 0.
  writeLine(core_energy, 0, 0, 0, 0);
}

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_FCIDUMP_H
