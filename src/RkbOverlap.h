#ifndef RERDMFT_RKBOVERLAP_H
#define RERDMFT_RKBOVERLAP_H

#include <complex>
#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds the Small-component overlap matrix in the restricted-kinetic-
// balance (RKB) basis:
//   S_small = C^dagger S_uKB C
// where C = rkb_coefficients (RkbTransformation.h) and S_uKB is the
// overlap matrix of the unrestricted-kinetic-balance Small AOs (the ones
// built from e.g. a Large P AO producing Small S and D AOs, etc. --
// SmallComponentBasis).
//
// C is stored (2*nLarge x 2*nSmall), rows indexed by Large spin-orbitals
// and columns by Small spin-orbitals (see RkbTransformation.h), so
// "C^dagger S_uKB C" -- read as the bilinear form <RKB_p|S|RKB_q> that
// C^dagger S_uKB C denotes -- is built here as conj(C) * S_uKB * C^T:
// conj(C) supplies the (implicitly conjugated) bra side and C^T the ket
// side, both derived from C's actual (Large-row, Small-column) storage.
// S_uKB itself is the block-diagonal repeat of the scalar Small-AO
// overlap matrix (overlapMatrix, Integrals.h) across the alpha/beta spin
// blocks, matching every other place this project embeds a spin-diagonal
// scalar quantity (e.g. diracRestEnergyMatrix, vextMatrix).
//
// The result is (2*nLarge x 2*nLarge) and, since C is complex, generally
// complex Hermitian (not real symmetric).
Matrix<std::complex<double>> rkbSmallOverlapMatrix(
    const std::vector<BasisFunction>& small_basis,
    const Matrix<std::complex<double>>& rkb_coefficients);

// Same construction as rkbSmallOverlapMatrix, but for the nuclear-electron
// attraction matrix instead of the overlap: F_small = C^dagger V_uKB C,
// where V_uKB is the block-diagonal repeat (across the alpha/beta spin
// blocks) of nuclearAttractionMatrix(small_basis, geometry)
// (NuclearAttraction.h). Both this and S_small (rkbSmallOverlapMatrix) are
// entirely independent of the speed of light, which is what lets
// RkbPositiveEnergyHamiltonian.h reconstruct the RKB Small-Small
// Hamiltonian block's exact -2c^2*S_small + F_small structure without ever
// forming that (badly cancelling, for large c) sum in floating point.
Matrix<std::complex<double>> rkbSmallVextMatrix(const std::vector<BasisFunction>& small_basis,
                                                 const Matrix<std::complex<double>>& rkb_coefficients,
                                                 const std::vector<Atom>& geometry);

}  // namespace rerdmft

#endif  // RERDMFT_RKBOVERLAP_H
