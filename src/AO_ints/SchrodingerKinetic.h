#ifndef RERDMFT_SCHRODINGERKINETIC_H
#define RERDMFT_SCHRODINGERKINETIC_H

#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Computes the full (real, symmetric) nonrelativistic (Schrodinger)
// kinetic-energy matrix <AO_i| -nabla^2/2 |AO_j> for an already-normalized
// cartesian AO basis, via libcint's cint1e_kin_cart, which computes
// 0.5<i|p.p|j> = 0.5<i|-nabla^2|j> = <i|-nabla^2/2|j> (p = -i grad_r).
Matrix<double> schrodingerKineticMatrix(const std::vector<BasisFunction>& basis);

}  // namespace rerdmft

#endif  // RERDMFT_SCHRODINGERKINETIC_H
