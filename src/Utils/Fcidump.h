#ifndef RERDMFT_UTILS_FCIDUMP_H
#define RERDMFT_UTILS_FCIDUMP_H

#include <cstddef>
#include <string>

#include "Matrix.h"
#include "SymmetricEri.h"

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
// (IJ|KL) for I>=J, K>=L, and the composite pair I(I-1)/2+J >= K(K-1)/2+L; h_IJ for I>=J. `eri_mo`
// is read through its physics-notation interface (SymmetricEri.h): physics(A,C,B,D) is the
// chemist integral (AB|CD) this function writes, i.e. FCIDUMP's own convention.
void writeFcidump(const std::string& path, std::size_t norb, int n_electrons, int ms2,
                  const Matrix<double>& h_mo, const SymmetricEri<double>& eri_mo, double core_energy);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_FCIDUMP_H
