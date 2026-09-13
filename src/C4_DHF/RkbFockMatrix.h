#ifndef RERDMFT_RKBFOCKMATRIX_H
#define RERDMFT_RKBFOCKMATRIX_H

#include <complex>

#include "Matrix.h"
#include "RkbTwoElectron.h"

namespace rerdmft {

// Builds the 4-component Dirac-Hartree-Fock Fock matrix in the RKB spinor
// AO basis,
//   F(p,r) = H_RKB(p,r) + J(p,r) - K_same(p,r) - K_opp(p,r),
// where H_RKB is the (non-orthonormalized) RKB core Hamiltonian
// (RkbHamiltonian.h -- NOT H_RKB_ortho) and, writing <A B|C D> for the RKB
// two-electron tensor (RkbTwoElectron.h) and P for the RKB density matrix
// (RkbDensityMatrix.h):
//
//   J(p,r)      = sum_{q,s} P(s,q) <p q|r s>              (Hartree/Coulomb)
//   K_same(p,r) = sum_{q,s: same flavor} P(s,q) <p q|s r>  (exchange, same spin)
//   K_opp(p,r)  = sum_{q,s: different flavor} P(s,q) <p q|s r>  (exchange,
//                                                            opposite spin)
//
// "Flavor" is which of the four RKB spinor blocks (Large-alpha, Large-
// beta, RKB-Small-alpha-partner, RKB-Small-beta-partner -- see
// RkbHamiltonian.h) an index falls in. J needs no such split: <p q|r s>
// is already exactly zero unless q and s share a flavor, so summing over
// all q,s picks up only the physically meaningful (same-flavor) terms
// automatically. The exchange sum <p q|s r>, however, requires p and s to
// share a flavor and q and r to share a flavor (independently) -- so for
// a FIXED (p,r), the sum over q,s splits into contributions where q and s
// happen to share a flavor with each other (K_same, the only kind of
// exchange that exists in a spin-block-diagonal density, e.g. ordinary
// unrestricted Hartree-Fock) and contributions where they do not
// (K_opp) -- which is nonzero here specifically because the RKB density
// matrix is a single, general (not spin-block-diagonal) object: nothing
// prevents P from having nonzero elements connecting, say, a Large-alpha
// index to a Large-beta one, and K_opp is exactly the exchange
// contribution such "spin-off-diagonal" density elements produce. Both
// terms are always well-defined regardless of whether P actually has
// such elements (K_opp is simply zero if it does not); they are kept
// separate here to make each contribution's physical origin explicit and
// independently checkable, matching the standard textbook split for
// general (non-spin-restricted) Hartree-Fock in a spinor basis.
//
// Throws std::runtime_error if h_rkb, the ERI tensor, and density_matrix
// have inconsistent dimensions, or if that common dimension is not a
// multiple of 4 (4*nLarge).
Matrix<std::complex<double>> rkbFockMatrix(const Matrix<std::complex<double>>& h_rkb,
                                            const RkbTwoElectronTensor& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_RKBFOCKMATRIX_H
