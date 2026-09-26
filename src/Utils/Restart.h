#ifndef RERDMFT_UTILS_RESTART_H
#define RERDMFT_UTILS_RESTART_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// RESTART file: the final result of an RDMFT run (NON_REL, X2C_HF, 4C or 4C_NEG), written in BINARY so that
// a later run can start from it: the occupation numbers (JK_only functionals) or the GAMMA
// angles (PNOF functionals) and the final molecular-orbital coefficients. Only the WRITER is
// used by the program for now; readRestart exists to verify the file (main.cpp reads every file
// back right after writing it) and for the future restart itself.
//
// File layout (all integers little-endian, doubles IEEE-754 binary64 little-endian):
//   8 bytes   magic "RERDMFT\0"
//   uint32    format version (kRestartVersion)
//   uint32    byte-order marker 0x01020304 (a reader on a big-endian host refuses the file)
//   string    method            ("NON_REL" | "X2C_HF" | "4C" | "4C_NEG")           string = uint64 length + bytes
//   string    functional        (the FUNCTIONAL keyword, upper case)
//   string    kind              ("OCCUPATIONS" | "GAMMAS")
//   uint64    basis fingerprint (BasisFingerprint.h's basisFingerprint of the Large AO basis)
//   int64     n_electrons
//   int64     pnof_subspaces, pnof_coupling, n_core   (PNOF; 0 for JK_only)
//   int64     jk_frozen_pairs, jk_active_pairs        (JK_only: the RESOLVED occupation window -- pairs pinned at
//                                                      occupation 1, and pairs in the fractional-occupation window
//                                                      above them, JK_ACTIVE_PAIRS absent = all the rest; 0 for PNOF)
//   double    final total energy (Hartree, nuclear repulsion included)
//   uint8     orbitals_optimized (1: FULL_OPTIMIZATION rotated the orbitals away from the SCF ones)
//   uint8     converged          (1: the optimization that produced the file reported convergence)
//   vector    occupations   (uint64 count + doubles): the FULL occupation vector, one entry per
//                            spin-orbital/spinor of the MO basis, always present
//   vector    gammas        (uint64 count + doubles): PNOF only, subspace after subspace,
//                            pnof_coupling-1 angles each (Occ_opt/PNOFs.h's trigonometric
//                            parameterization); empty for JK_only
//   uint8     complex_coefficients (0: real orbitals, 1: complex spinors)
//   uint64    rows, cols of the coefficient matrix
//   doubles   rows*cols coefficients, ROW-major; complex: (real, imag) pairs
//
// The coefficient matrix C maps the MO basis to the AO SPIN-ORBITAL basis of the method:
//   NON_REL: rows 2 n_ao   ordered [alpha AO_0..AO_{n-1}, beta AO_0..AO_{n-1}],
//            cols 2 n_mo   ordered [alpha MO_0.., beta MO_0..] (the block layout of the MO
//            integrals): C = blockdiag(C_scf, C_scf) * U_total, real;
//   X2C_HF:  rows/cols 2 n_large (Large-component spin-orbital AO basis / spinors ordered as
//            the MO integrals, Kramers pairs (2k, 2k+1)): C = C_scf * U_total, complex;
//   4C, 4C_NEG: rows/cols 4 n_large (RKB spinor basis [Large-alpha; Large-beta; Small; Small] /
//            MOs in ascending energy, the negative-energy branch first with occupation 0): C = C_dhf * U_total,
//            complex; 4C's U_total leaves the negative branch untouched (no-pair), 4C_NEG's mixes it;
// with U_total the accumulated FULL_OPTIMIZATION rotation (identity when no orbital
// optimization was done). Column j of C is MO j: |MO_j> = sum_mu C(mu, j) |AO_mu>; its
// occupation is occupations[j].
struct RestartData {
  std::string method;
  std::string functional;
  std::string kind;  // "OCCUPATIONS" or "GAMMAS"
  std::uint64_t basis_fingerprint = 0;
  std::int64_t n_electrons = 0;
  std::int64_t pnof_subspaces = 0;
  std::int64_t pnof_coupling = 0;
  std::int64_t n_core = 0;
  std::int64_t jk_frozen_pairs = 0;  // JK_only: resolved window, see the file layout
  std::int64_t jk_active_pairs = 0;
  double total_energy = 0.0;
  bool orbitals_optimized = false;
  bool converged = false;
  std::vector<double> occupations;
  std::vector<double> gammas;
  bool complex_coefficients = false;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::vector<double> coefficients;  // row-major; complex: interleaved (re, im)

  // Stores `c` (real or complex) as the coefficient matrix.
  void setCoefficients(const Matrix<double>& c);
  void setCoefficients(const Matrix<std::complex<double>>& c);
  // The coefficient matrix back as a complex Matrix (imaginary parts 0 for real files).
  Matrix<std::complex<double>> coefficientsComplex() const;
};

// What the functional reports (main.cpp) hand back for the RESTART file: the final RDMFT state
// of one method, before the SCF coefficients are attached.
struct RestartCapture {
  bool valid = false;
  std::string kind;                  // "OCCUPATIONS" or "GAMMAS"
  std::vector<double> occupations;   // full occupation vector
  std::vector<double> gammas;        // PNOF
  std::int64_t n_core = 0;           // PNOF: number of frozen core geminals
  std::int64_t jk_frozen_pairs = 0;  // JK_only: resolved JK_FROZEN_PAIRS / active window (pairs)
  std::int64_t jk_active_pairs = 0;
  double electronic_energy = 0.0;    // final electronic energy (no nuclear repulsion)
  bool orbitals_optimized = false;
  bool converged = false;
  Matrix<std::complex<double>> total_rotation;  // FULL_OPTIMIZATION rotation; empty = identity
};

constexpr std::uint32_t kRestartVersion = 2;

// Writes `data` to `path` (overwriting). Throws std::runtime_error if the file cannot be
// written or the data are inconsistent (empty/mismatched sizes, unknown kind).
void writeRestart(const std::string& path, const RestartData& data);

// Reads a file written by writeRestart. Throws std::runtime_error on a missing file, a wrong
// magic/version/byte order, or a truncated/inconsistent file.
RestartData readRestart(const std::string& path);

// The FULL_OPTIMIZATION state -> the full-size coefficient matrix: C_scf (n_rows x n_mo) times
// the rotation `u` (n_mo x n_mo; an EMPTY `u` means the identity). NON_REL passes the
// block-diagonal spin-orbital C_scf, see above.
Matrix<std::complex<double>> restartCoefficients(const Matrix<std::complex<double>>& c_scf,
                                                 const Matrix<std::complex<double>>& u);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_RESTART_H
