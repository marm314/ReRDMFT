#ifndef RERDMFT_UTILS_RESTARTLOADER_H
#define RERDMFT_UTILS_RESTARTLOADER_H

#include <complex>
#include <cstddef>
#include <ostream>
#include <string>

#include "Matrix.h"
#include "Restart.h"

namespace rerdmft {

// READ_RESTART support: reads a RESTART.<method> file (Utils/Restart.h) and prepares its orbitals for the CURRENT
// calculation, which may be at another geometry (a potential-energy-curve scan): the file's MO coefficients C are
// only orthonormal in the overlap S of the geometry they were written at.
struct RestartOrbitals {
  RestartData data;                       // the file's contents (occupations, gammas, window, ...)
  Matrix<std::complex<double>> c;         // the coefficients, orthonormal in the current overlap
  double overlap_deviation_read = 0.0;    // max |C_read^dagger S C_read - 1|
  double overlap_deviation_final = 0.0;   // max |C^dagger S C - 1| after the (optional) Loewdin step
  double min_overlap_eigenvalue = 0.0;    // smallest eigenvalue of C_read^dagger S C_read
  bool lowdin_applied = false;
};

// Symmetric (Loewdin) orthonormalization of the columns of `c` in the metric `s`: with S_check = C^dagger S C
// (Hermitian, positive definite), C' = C S_check^(-1/2) satisfies C'^dagger S C' = S_check^(-1/2) S_check S_check^(-1/2)
// = 1 and is the orthonormal set closest to C. Throws std::runtime_error if S_check has an eigenvalue below
// `min_eigenvalue` (linearly dependent columns). Outputs the deviations from the identity before/after and the
// smallest eigenvalue when the pointers are non-null.
Matrix<std::complex<double>> lowdinOrthonormalize(const Matrix<std::complex<double>>& c,
                                                  const Matrix<std::complex<double>>& s,
                                                  double* deviation_before = nullptr, double* deviation_after = nullptr,
                                                  double* min_eigenvalue = nullptr, double min_eigenvalue_floor = 1e-8);

// Reads `path` and validates it against the current run (method string, number of electrons, coefficient matrix
// `expected_rows` x `expected_cols`, real/complex storage, one occupation per column), then orthonormalizes the
// coefficients in `s_ao` (the AO metric in the file's own representation: NON_REL blockdiag(S, S), X2C
// blockdiag(S, S) over [alpha; beta], 4C the RKB overlap) when the deviation exceeds `tolerance`. Prints what it
// found and did to `log`. Throws std::runtime_error on any inconsistency. NON_REL passes `n_electrons` of the run.
RestartOrbitals readRestartOrbitals(const std::string& path, const std::string& method, long long n_electrons,
                                    std::size_t expected_rows, std::size_t expected_cols, bool expected_complex,
                                    const Matrix<std::complex<double>>& s_ao, std::ostream& log,
                                    double tolerance = 1e-10);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_RESTARTLOADER_H
