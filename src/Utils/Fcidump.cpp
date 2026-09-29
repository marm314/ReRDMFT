#include "Fcidump.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace rerdmft {

void writeFcidump(const std::string& path, std::size_t norb, int n_electrons, int ms2,
                  const Matrix<double>& h_mo, const SymmetricEri<double>& eri_mo, double core_energy) {
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
