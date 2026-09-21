#ifndef RERDMFT_UTILS_DIIS_H
#define RERDMFT_UTILS_DIIS_H

#include <complex>
#include <cstddef>
#include <deque>
#include <vector>

#include "Matrix.h"

namespace rerdmft {

// Pulay's DIIS (direct inversion in the iterative subspace) for the Hartree-Fock SCF loops of
// this project: NON_REL (real, T = double), X2C and 4-component (complex, T = std::complex).
//
// The SCF error vector is the commutator of the Fock matrix with the density matrix in the
// (non-orthogonal) atomic-orbital basis,
//   e = F P S - S P F                                                       (scfCommutatorError)
// which vanishes exactly when F and P are consistent (P built from eigenvectors of F C = S C E).
// S is the AO overlap of the space F lives in: S_Large for NON_REL, the Large-component
// spin-orbital block of S_full for X2C, S_full (Large, Large, Small) for the 4-component path.
//
// Usage per SCF iteration n (see the three HF loops): build F_n from the density P_current of the
// previous iteration, e_n = scfCommutatorError(F_n, P_current, S), and
//   F_n' = diis.extrapolate(e_n, F_n)
// which stores the pair (e_n, F_n) -- keeping at most `max_vectors` of them, the oldest dropped
// first -- and returns F_n' = sum_i w_i F_i with the weights that minimize |sum_i w_i e_i|^2
// subject to sum_i w_i = 1:
//   [ B  -1 ] [ w      ]   [ 0 ]
//   [-1   0 ] [ lambda ] = [-1 ],        B_ij = Re <e_i|e_j> = Re sum conj(e_i) e_j.
// Diagonalizing F_n' instead of F_n gives the next density; no linear density mixing is applied
// on top. With a single stored pair (the first iteration) nothing can be extrapolated and F_n is
// returned unchanged, so the extrapolation effectively starts at the second iteration.
//
// Differences from QuAcK's complex_DIIS_extrapolation.f90 (deliberate): B uses the REAL part of
// the Hermitian inner product and the weights are REAL. A complex combination of Hermitian Fock
// matrices would not be Hermitian and would break the time-reversal (Kramers) structure of the
// X2C / 4-component Fock matrices; real weights keep both, and B is real symmetric for both
// scalar types, so a single real solver serves NON_REL and the relativistic paths.
//
// The B matrix is scaled by its largest diagonal element before the solve (the errors shrink to
// ~1e-8 during convergence). If the system is numerically singular (nearly linearly dependent
// errors, pivot ratio below 1e-14 like QuAcK's rcond test) the oldest stored pair is dropped and
// the solve repeated; with a single pair left the input is returned unchanged.
template <typename T>
Matrix<T> scfCommutatorError(const Matrix<T>& fock, const Matrix<T>& density,
                             const Matrix<T>& overlap);

template <typename T>
class Diis {
 public:
  // `max_vectors` = number of (error, value) pairs kept (typically 5); values < 2 disable the
  // extrapolation (extrapolate() returns its input).
  explicit Diis(std::size_t max_vectors);

  // Stores (error, value) and returns the extrapolated value (see above). `error` and `value`
  // may have different shapes but every call must use the same shapes.
  Matrix<T> extrapolate(const Matrix<T>& error, const Matrix<T>& value);

  void clear();
  std::size_t size() const { return errors_.size(); }
  std::size_t maxVectors() const { return max_vectors_; }
  // Weights of the last extrapolation (empty if it returned its input unchanged) and the number
  // of stored pairs that entered it.
  const std::vector<double>& lastWeights() const { return last_weights_; }

 private:
  std::size_t max_vectors_;
  std::deque<Matrix<T>> errors_;
  std::deque<Matrix<T>> values_;
  std::vector<double> last_weights_;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_DIIS_H
