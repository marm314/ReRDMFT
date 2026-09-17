#ifndef RERDMFT_VEXT_H
#define RERDMFT_VEXT_H

#include <complex>
#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds the external nuclear-electron attraction matrix
//   spinor_a^dagger Vext(r) I_4x4 spinor_b,
// where Vext(r) = -sum_A Z_A/|r-R_A| is the usual Coulomb attraction to
// every nucleus in `geometry`, in the 4-component spinor basis
// [Large-alpha, Large-beta, Small-alpha, Small-beta] (see SpinorBasis).
// Since it is a scalar potential (identity across all 4 spinor
// components), it is block-diagonal with the same <Large_i|Vext|Large_j>
// matrix repeated for both Large spin blocks and the same
// <Small_i|Vext|Small_j> matrix repeated for both Small spin blocks, with
// zero Large-Small coupling.
Matrix<std::complex<double>> vextMatrix(const std::vector<BasisFunction>& large_basis,
                                         const std::vector<BasisFunction>& small_basis,
                                         const std::vector<Atom>& geometry);

}  // namespace rerdmft

#endif  // RERDMFT_VEXT_H
