#ifndef RERDMFT_RKBTRANSFORMATION_H
#define RERDMFT_RKBTRANSFORMATION_H

#include <complex>
#include <vector>

#include "LinearAlgebra.h"
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
// overlap matrix S_tq = <Small_t|Small_q> gives M = C S, i.e. C = M S^+,
// with M_tp = <Small_t|sigma.p|Large_p>. S^+ is the PSEUDO-inverse
// (pseudoInverseSymmetric, LinearAlgebra.h), not a literal inverse: a large,
// fully uncontracted basis (e.g. a heavy element's dyall basis) can make
// the raw Small-component overlap S genuinely, legitimately near-singular
// (true near-linear-dependency among its tightest functions, not a scaling
// artifact -- confirmed by its diagonal already being exactly 1, i.e.
// individually normalized). A plain LU inverse (invert()) has no
// conditioning safeguard at all and silently returns garbage in that case
// (observed: |S * S^-1 - I| ~ 1e3 on Xe/dyall.v2z) -- the pseudo-inverse
// instead drops near-null eigendirections rather than amplifying them,
// which changes nothing about C's shape (still 2*nLarge x 2*nSmall, same
// downstream RKB dimension) since the dropped directions represent
// information nothing downstream could safely use anyway. Pass `report` to
// see how many directions were dropped, if any -- 0 for any well-behaved
// (segmented/contracted, or lighter-element) basis.
//
// Returned as a (2*nLarge x 2*nSmall) matrix following the same spin block
// ordering as SpinorBasis: rows [Large-alpha, Large-beta], columns
// [Small-alpha, Small-beta].
Matrix<std::complex<double>> rkbCoefficients(const std::vector<BasisFunction>& large_basis,
                                              const std::vector<BasisFunction>& small_basis,
                                              PseudoInverseReport* report = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_RKBTRANSFORMATION_H
