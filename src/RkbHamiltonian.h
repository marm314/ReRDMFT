#ifndef RERDMFT_RKBHAMILTONIAN_H
#define RERDMFT_RKBHAMILTONIAN_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// Builds the restricted-kinetic-balance (RKB) core Hamiltonian by
// transforming the unrestricted-kinetic-balance one,
//
//   H_RKB = W^dagger H_UKB W,   W = [[I, 0], [0, C^T]],
//
// where I is the (2*nLarge x 2*nLarge) identity (the Large block is
// unchanged) and C^T is the transpose of the RKB transformation
// coefficients from rkbCoefficients (RkbTransformation.h) -- C itself is
// (2*nLarge x 2*nSmall) with rows indexed by Large spin-orbitals and
// columns by Small spin-orbitals, so the W block that right-multiplies
// H_UKB's (2*nSmall)-dimensional Small-component rows needs the transpose,
// (2*nSmall x 2*nLarge), to project them onto the RKB small basis (which
// has exactly 2*nLarge functions, one per Large spin-orbital). The zero
// blocks are sized to match: (2*nLarge x 2*nLarge) top-right, (2*nSmall x
// 2*nLarge) bottom-left.
//
// H_UKB must be square with dimension 2*nLarge + 2*nSmall, matching
// rkb_coefficients' (2*nLarge x 2*nSmall) shape; throws std::runtime_error
// otherwise. The result is (4*nLarge x 4*nLarge): Large and (now RKB)
// Small blocks both have dimension 2*nLarge.
Matrix<std::complex<double>> rkbHamiltonianMatrix(
    const Matrix<std::complex<double>>& h_ukb,
    const Matrix<std::complex<double>>& rkb_coefficients);

// Builds W = [[I, 0], [0, C^T]] on its own (see rkbHamiltonianMatrix for
// the full derivation): the ((2*nLarge + 2*nSmall) x 4*nLarge) embedding
// of the RKB core Hamiltonian's basis back into the original
// [Large-alpha, Large-beta, uKB-Small-alpha, uKB-Small-beta] spinor-AO
// basis. Used both by rkbHamiltonianMatrix and by anything that needs to
// map an RKB-basis vector back to that original representation (e.g. a
// Kramers-symmetry check).
Matrix<std::complex<double>> rkbEmbeddingMatrix(
    const Matrix<std::complex<double>>& rkb_coefficients);

}  // namespace rerdmft

#endif  // RERDMFT_RKBHAMILTONIAN_H
