#ifndef RERDMFT_X2C_DHF_X2C_FOCKMATRIX_H
#define RERDMFT_X2C_DHF_X2C_FOCKMATRIX_H

#include <complex>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Builds the (approximate) X2C-HF Fock matrix in the Large-component
// spin-orbital basis (dimension n = 2*nLarge, ordered
// [Large-alpha, Large-beta]):
//   F(p,r) = h_x2c(p,r) + J(p,r) - K_same(p,r) - K_opp(p,r)
// where h_x2c is the FIXED, one-electron-only, complex Hermitian,
// spin-orbit-coupled exact X2C Hamiltonian (X2C_hamiltonian.h) and,
// writing <A B|C D> for the ORDINARY (real, non-relativistic, spin-
// independent) physics-notation two-electron Coulomb tensor over the
// SAME spin-orbital basis (NON_REL/ClosedShellSpinOrbitals.h's own
// closedShellSpinOrbitalTwoElectron, reused here even though the
// density below need not be closed-shell -- that function only
// performs the mechanical spin-orbital expansion of a spin-free
// operator, it is agnostic to what density it is later contracted
// with) and P for the X2C density matrix:
//   J(p,r)      = sum_{q,s} P(s,q) <p q|r s>
//   K_same(p,r) = sum_{q,s: same spin} P(s,q) <p q|s r>
//   K_opp(p,r)  = sum_{q,s: different spin} P(s,q) <p q|s r>
//
// Unlike ordinary closed-shell (or even unrestricted) NON_REL HF, P
// here is a GENERAL complex Hermitian matrix, not assumed spin-block-
// diagonal: h_x2c's own spin-orbit coupling already mixes alpha/beta
// at the ONE-ELECTRON level, so the converged SCF density inherits
// spin-off-diagonal elements too, making K_opp (exchange between
// opposite-spin density elements) generically NONZERO here -- unlike
// ordinary HF, where it vanishes identically by construction. This
// mirrors exactly how C4_DHF/RkbFockMatrix.h's own same/opposite
// exchange split works, just in the smaller, 2-flavor (not 4-flavor)
// Large-component-only space, with REAL (not RKB-relativistic) two-
// electron integrals -- i.e. no two-electron picture-change correction
// is applied anywhere (see X2C_DHF/X2C_HF.h).
//
// Throws std::runtime_error on inconsistent dimensions, or if the
// common dimension is not even (2*nLarge).
Matrix<std::complex<double>> x2cFockMatrix(const Matrix<std::complex<double>>& h_x2c,
                                            const Tensor4<double>& eri,
                                            const Matrix<std::complex<double>>& density_matrix);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_FOCKMATRIX_H
