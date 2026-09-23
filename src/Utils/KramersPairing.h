#ifndef RERDMFT_UTILS_KRAMERSPAIRING_H
#define RERDMFT_UTILS_KRAMERSPAIRING_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "Tensor4.h"

namespace rerdmft {

// Exact Kramers pairing of converged two-component (X2C) spinors, robust to
// NEAR-DEGENERATE pairs.
//
// A Fock diagonalization returns the Kramers pairs as consecutive columns
// (2k, 2k+1) only as long as every level is separated from the others by
// more than the numerical noise. When two (or more) Kramers pairs are
// (nearly) degenerate -- e.g. the spin-orbit-split p shell of a stretched or
// light-atom molecule, gaps ~1e-6 Ha -- the eigenvectors of the whole cluster
// are ill-conditioned: consecutive columns are still an orthonormal basis of
// the (time-reversal-invariant) cluster subspace, but NOT necessarily
// (psi, Theta psi) pairs, and the MO integrals then violate the Kramers
// structure by ~noise/gap (3e-6 seen for LiH at 5 A) or even O(1) for an
// exactly degenerate cluster. fixKramersPhaseLarge only fixes the phase of
// each consecutive pair and cannot repair that.
//
// This routine groups consecutive Kramers pairs whose orbital energies differ
// by less than `cluster_tolerance` into clusters and, inside each cluster,
// rebuilds the pairs exactly: take the first remaining vector v, its partner
// Theta v projected onto the cluster subspace (Theta|k alpha> = |k beta>,
// Theta|k beta> = -|k alpha>, antilinear; the convention of
// fixKramersPhaseLarge), remove the pair from the subspace and repeat. The
// result spans the same subspace, is orthonormal in the S metric, satisfies
// Theta|2k> = |2k+1> to the invariance residual of the cluster, and for a
// non-degenerate pair reduces to the phase fix (column 2k unchanged). A final
// global step makes the structure EXACT: every odd column is set to Theta of
// its (re-orthogonalized) even column, removing the remaining SCF-noise-level
// (~1e-8) time-reversal asymmetry of the eigenvectors; the change made there is
// reported as max_enforcement_change.
struct KramersPairingReport {
  std::size_t n_multi_pair_clusters = 0;  // clusters with more than one Kramers pair
  std::size_t largest_cluster_pairs = 1;
  double max_invariance_residual = 0.0;   // max sqrt(1 - |P_cluster Theta v|^2): how far a
                                          // cluster is from being time-reversal invariant
  double max_column_change = 0.0;         // max sqrt(1 - |<old|new>|^2) over columns: how much
                                          // the cluster re-pairing moved a column beyond a phase
  double max_enforcement_change = 0.0;    // max |element| change of the final exact enforcement
                                          // (odd := Theta even): the SCF-noise-level asymmetry
  double partner_error_before = 0.0;      // max_k || Theta c_2k - c_2k+1 ||  on the input ...
  double partner_error_after = 0.0;       // ... and on the output (linear in the error, unlike
                                          // 1 - |<odd|Theta even>| which is quadratic)
};

// `c_matrix`: columns = spinors in the [alpha AO; beta AO] representation
// (2 nL rows), sorted by `energies` (ascending), consecutive pairs (2k,2k+1)
// the Kramers pairs; C^dagger S C = 1 with S = diag(s_large, s_large).
// Throws std::runtime_error on inconsistent dimensions or an odd column count.
Matrix<std::complex<double>> fixKramersPairingLarge(const Matrix<std::complex<double>>& c_matrix,
                                                    const std::vector<double>& energies,
                                                    const Matrix<double>& s_large,
                                                    double cluster_tolerance = 1e-4,
                                                    KramersPairingReport* report = nullptr);

// The same repair for spinors given as coefficients in an ORTHONORMAL basis (identity
// metric; columns of `c` orthonormal), where time reversal acts as
// Theta c = theta_matrix * conj(c) (theta_matrix unitary with theta conj(theta) = -1).
// Used for the 4-component (RKB) eigenvectors, whose orthonormal Fock_ortho basis is mapped
// back to the original spinor-AO representation only to BUILD theta_matrix
// (KramersSymmetry.h's fixKramersPairing).
Matrix<std::complex<double>> fixKramersPairingOrthonormal(
    const Matrix<std::complex<double>>& c, const std::vector<double>& energies,
    const Matrix<std::complex<double>>& theta_matrix, double cluster_tolerance = 1e-4,
    KramersPairingReport* report = nullptr);

// Kramers structure of MO integrals in a basis whose spinors are the consecutive pairs
// (2k, 2k+1) with Theta|2k> = |2k+1>, Theta|2k+1> = -|2k>: a time-reversal-even operator
// obeys M(P p, P q) = s_p s_q conj M(p,q) (one-body) and <Pa Pb|Pc Pd> = s_a s_b s_c s_d
// conj <ab|cd> (two-body physics notation), P i = i^1, s = +1 (even) / -1 (odd). Each returns
// max |lhs - rhs| and stores max |element| in *scale (if non-null).
double kramersOneBodyDeviation(const Matrix<std::complex<double>>& h, double* scale = nullptr);

// The same test for a one-body matrix in the two-component AO representation
// [alpha AO (nL); beta AO (nL)] where Theta|k alpha> = |k beta>, Theta|k beta> = -|k alpha>
// (e.g. X2C's h_x2c): a time-reversal-even operator obeys
//   M(beta_i, beta_j) = conj M(alpha_i, alpha_j),  M(beta_i, alpha_j) = -conj M(alpha_i, beta_j).
// Returns the max deviation over both relations; *scale = max |element|.
double kramersAoOneBodyDeviation(const Matrix<std::complex<double>>& h, double* scale = nullptr);

// Time-reversal-even part of a matrix in the two-component AO representation [alpha AO (nL);
// beta AO (nL)] (the projection matching kramersAoOneBodyDeviation's relations):
//   even(a,b) = (M(a,b) + s_a s_b conj M(P a, P b)) / 2,  P = alpha <-> beta, s = +1 (alpha) / -1 (beta).
// An exact no-op on an already even matrix; keeps Hermiticity; *removed = max |M - even|. Throws on
// an odd dimension. Used to keep the X2C-HF SCF Kramers-restricted.
Matrix<std::complex<double>> kramersSymmetrizeAo(const Matrix<std::complex<double>>& m,
                                                  double* removed = nullptr);

// The same test and projection for a matrix in the 4-component RKB spinor AO representation
// [Large-alpha (nL); Large-beta (nL); Small-alpha (nL); Small-beta (nL)] (dimension 4 nL, one
// RKB small function sigma.p|Large_p> per large spin-orbital p). sigma.p is time-reversal even,
// so Theta maps the small block exactly like the large one: Theta|k alpha> = |k beta>,
// Theta|k beta> = -|k alpha> in BOTH blocks. With partner P(a) (alpha <-> beta of the same AO,
// same block) and sign s(a) = +1 (alpha) / -1 (beta), a time-reversal-EVEN matrix obeys
//   M(P a, P b) = s(a) s(b) conj M(a, b)
// (Large-Large, Small-Small and Large-Small sectors alike). kramersRkbAoDeviation returns
// max |M(P a, P b) - s(a) s(b) conj M(a, b)| (and max |element| in *scale);
// kramersSymmetrizeRkbAo returns the time-reversal-even part (M + Theta M Theta^-1)/2 -- an
// exact no-op on an already even matrix, keeps Hermiticity, and (S being even) the trace with
// the overlap -- and stores max |M - even part| in *removed. Both throw on a dimension that is
// not a multiple of 4.
double kramersRkbAoDeviation(const Matrix<std::complex<double>>& m, double* scale = nullptr);
Matrix<std::complex<double>> kramersSymmetrizeRkbAo(const Matrix<std::complex<double>>& m,
                                                     double* removed = nullptr);
double kramersTwoBodyDeviation(const Tensor4<std::complex<double>>& eri, double* scale = nullptr);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_KRAMERSPAIRING_H
