#ifndef RERDMFT_LINEARALGEBRA_H
#define RERDMFT_LINEARALGEBRA_H

#include "Matrix.h"

namespace rerdmft {

// Inverts a square, nonsingular real matrix via LAPACK (LU factorization
// with LAPACKE_dgetrf, then LAPACKE_dgetri). Throws std::runtime_error if
// `a` is singular (or not square).
Matrix<double> invert(const Matrix<double>& a);

}  // namespace rerdmft

#endif  // RERDMFT_LINEARALGEBRA_H
