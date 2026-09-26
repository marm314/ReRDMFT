#ifndef RERDMFT_CLOSEDSHELLSPINORBITALS_H
#define RERDMFT_CLOSEDSHELLSPINORBITALS_H

#include <cstddef>

#include "Matrix.h"
#include "SymmetricEri.h"
#include "Tensor4.h"

namespace rerdmft {

// Expands a real, closed-shell (restricted) spatial-MO basis of size
// `n_spatial` into a 2*n_spatial spin-orbital basis, block-ordered as
// [alpha_0..alpha_{n_spatial-1}, beta_0..beta_{n_spatial-1}] (spin-
// orbital index p has spatial index p % n_spatial and spin p /
// n_spatial, 0=alpha/1=beta) -- the input for
// Hessian_opt/GeneralizedFock.h and OrbitalGradient.h, both of which
// work in a general spin-orbital basis, never a closed-shell spatial
// one. Used to validate that module against a converged NON_REL
// (restricted closed-shell Hartree-Fock) reference, whose orbital
// gradient should vanish at convergence.

// h_spin(p,q) = h_mo(spatial(p), spatial(q)) if spin(p) == spin(q),
// else 0 -- the spin-free one-electron Hamiltonian does not connect
// different spins.
Matrix<double> closedShellSpinOrbitalOneElectron(const Matrix<double>& h_mo,
                                                  std::size_t n_spatial);

// eri_spin(A,B,C,D) = eri_mo_physics(spatial(A),spatial(B),spatial(C),
// spatial(D)) if spin(A)==spin(C) AND spin(B)==spin(D), else 0 (the
// standard "each electron's bra/ket spin must match" rule for a
// spin-free two-electron operator, physics notation: A,C electron 1;
// B,D electron 2). `eri_mo_physics` must already be in physics
// notation (see MoIntegralTransform.h's moTwoElectronTransformPhysics).
Tensor4<double> closedShellSpinOrbitalTwoElectron(const Tensor4<double>& eri_mo_physics,
                                                   std::size_t n_spatial);

// The same spin-orbital expansion for a unique-element spatial store, returning a unique-element spin-orbital
// store (no dense (2n)^4 array): only the elements with a <= c are evaluated (the store rebuilds the rest).
SymmetricEri<double> closedShellSpinOrbitalTwoElectron(const SymmetricEri<double>& eri_mo_physics,
                                                        std::size_t n_spatial);

// The idempotent 1-RDM of the closed-shell single determinant that
// doubly occupies the lowest `n_occupied_spatial` spatial MOs: diagonal
// 1 for both spin-orbital copies (alpha and beta) of each occupied
// spatial MO, 0 elsewhere.
Matrix<double> closedShellSpinOrbitalDensity(std::size_t n_spatial, int n_occupied_spatial);

}  // namespace rerdmft

#endif  // RERDMFT_CLOSEDSHELLSPINORBITALS_H
