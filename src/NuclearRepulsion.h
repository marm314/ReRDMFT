#ifndef RERDMFT_NUCLEARREPULSION_H
#define RERDMFT_NUCLEARREPULSION_H

#include <vector>

#include "Input.h"

namespace rerdmft {

// Sum_{A<B} Z_A*Z_B / |R_A - R_B| -- the classical nucleus-nucleus Coulomb
// repulsion energy, constant for a fixed geometry (Bohr, as stored in
// Input::geometry()). Z_A from Element.h's atomicNumber. Shared by both
// the 4-component (C4_DHF/C4_DHF.h) and nonrelativistic
// (NON_REL/NonRelHartreeFock.h) Hartree-Fock energy expressions, since
// this classical term does not depend on the electronic structure method.
double nuclearRepulsionEnergy(const std::vector<Atom>& geometry);

}  // namespace rerdmft

#endif  // RERDMFT_NUCLEARREPULSION_H
