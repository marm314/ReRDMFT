#ifndef RERDMFT_RKBTRANSFORMATION_H
#define RERDMFT_RKBTRANSFORMATION_H

#include <complex>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Builds the restricted-kinetic-balance (RKB) expansion coefficients C that
// express sigma.p acting on each Large spin-orbital as a linear
// combination of the (larger, redundant) unrestricted-kinetic-balance
// Small spin-orbitals already built by SmallComponentBasis:
//
//   sigma.p |Large_p> = sum_q C_pq |Small_q>
//
// (p ranges over the 2*nLarge Large spin-orbitals, q over the 2*nSmall
// Small ones, sigma the Pauli matrices, p = -i grad_r the momentum
// operator). Projecting onto <Small_t| and using the Small-component
// overlap matrix S_tq = <Small_t|Small_q> gives M = C S, i.e. C = M S^-1,
// with M_tp = <Small_t|sigma.p|Large_p>. S^-1 is computed via LAPACK (see
// LinearAlgebra.h).
//
// Returned as a (2*nLarge x 2*nSmall) matrix following the same spin block
// ordering as SpinorBasis: rows [Large-alpha, Large-beta], columns
// [Small-alpha, Small-beta].
Matrix<std::complex<double>> rkbCoefficients(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTRANSFORMATION_H
