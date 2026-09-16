#ifndef RERDMFT_X2C_DHF_X2C_DENSITYMATRIX_H
#define RERDMFT_X2C_DHF_X2C_DENSITYMATRIX_H

#include <complex>

#include "Matrix.h"

namespace rerdmft {

// Builds the X2C-HF density matrix
//   P(i,j) = sum_{c occupied} C(i,c) * conj(C(j,c)),
// occupying the LOWEST n_electrons columns of `c_matrix` (ascending
// orbital energy, columns 0..n_electrons-1) -- unlike
// C4_DHF/RkbDensityMatrix.h's own rkbDensityMatrix, there is no
// negative-energy branch to skip here: X2C's own decoupling already
// eliminated it entirely (X2C_hamiltonian.h), so EVERY column of
// `c_matrix` is already a genuine positive-energy ("electronic")
// state, and the lowest n_electrons are exactly the ones to occupy
// (the ordinary Aufbau rule, same as NON_REL's own density).
//
// Throws std::runtime_error if c_matrix is not square, or n_electrons
// is not a positive number of spinors that fits within it.
Matrix<std::complex<double>> x2cDensityMatrix(const Matrix<std::complex<double>>& c_matrix,
                                               int n_electrons);

}  // namespace rerdmft

#endif  // RERDMFT_X2C_DHF_X2C_DENSITYMATRIX_H
