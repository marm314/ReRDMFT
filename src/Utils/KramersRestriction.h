#ifndef RERDMFT_UTILS_KRAMERSRESTRICTION_H
#define RERDMFT_UTILS_KRAMERSRESTRICTION_H

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "ADAM.h"
#include "NEO.h"

namespace rerdmft {

// Kramers-restricted (KR) orbital rotations: the subspace of the joint
// real rotation parameters [t_I; y_I] (kappa_pq = t + iy, kappa_qp =
// -conj(kappa_pq), pairs I = (p,q) with p > q; U = exp(-kappa)) that
// commutes with time reversal, so that rotated spinors remain Kramers
// pairs -- the KR-MCSCF restriction of Saue et al. (Jensen/Dyall/Saue/
// Faegri 1996; Thyssen/Fleig/Jensen 2008), here derived from Theta kappa
// Theta^-1 = kappa.
//
// Spinors are numbered in Kramers pairs (2k, 2k+1) with Theta|2k> =
// |2k+1>, Theta|2k+1> = -|2k> (Utils/KramersSymmetry.h; run
// fixKramersPhase(Large) first). With partner P(i) = i^1 and sign s(i) =
// +1 (even) / -1 (odd), a time-reversal-even operator M obeys
//   M(P p, P q) = s(p) s(q) conj(M(p,q))
// (quaternion structure: kappa_{abbar} = kappa_ab^*, kappa_{abar b} =
// -kappa_{a bbar}^*). On the joint parameters this is a signed
// permutation R: pair (p,q) -> pair (P p, P q) with z' = s_p s_q conj(z)
// if P p > P q, else (pair swapped) z' = -s_p s_q z, i.e. t' = a t,
// y' = b y with (a,b) = (ss,-ss) or (-ss,-ss). The KR parameters are the
// fixed subspace R x = x. Its structure:
//   * a pair (2k+1, 2k) inside one Kramers pair maps to ITSELF with
//     a = b = +1 (the free SU(2) rotation within the pair): one t and
//     one y coordinate each;
//   * every other pair (I) is tied to exactly one partner pair (J):
//     x_J = a x_I (t) / b x_I (y): one t and one y coordinate per PAIR of
//     pairs -- half the full parameters, as in KR-MCSCF.
// Consequently reducedSize() = 2 * nOrbits() with nOrbits() = (m +
// nFixedPairs())/2 for m pairs.
//
// Two normalizations of the reduced coordinates are provided:
//   * ISOMETRIC (expand/contract, x_full = Q u with Q^T Q = 1): the
//     reduced gradient is Q^T g, the reduced Hessian Q^T H Q -- exactly
//     the restriction of H to the time-reversal-even sector (its spectrum
//     is the subset of the full spectrum belonging to R = +1), and step
//     norms equal the full-space norms (so a NEO trust radius means the
//     same in both spaces). Used by KramersNeoProblem.
//   * REPRESENTATIVE (expandRepresentative/contractRepresentative,
//     x_full = J u with J = sqrt(#)Q): the reduced coordinate IS the
//     parameter of the lower-index pair of each orbit, copied with signs to
//     its partner. ADAM normalizes each coordinate by its own gradient
//     magnitude, so in this form its steps equal the full-space ADAM steps
//     exactly (each full entry moves by lr). Used by KramersAdamProblem.
// The joint layout everywhere is [t_0..t_{m-1}; y_0..y_{m-1}] over the
// caller's pair list, i.e. the ordering of jointOrbitalGradient and
// jkOnlyJointHessianMatrix/pnofJointHessianMatrix. Only pairs with p > q
// are handled (a diagonal phase rotation has identically zero gradient).
class KramersRestriction {
 public:
  struct Orbit {
    std::size_t first;   // canonical (lower-index) pair
    std::size_t second;  // its time-reversal partner pair; == first for a within-Kramers-pair rotation
    double sign_t;       // t_second = sign_t * t_first
    double sign_y;       // y_second = sign_y * y_first
  };

  // Throws std::runtime_error if n_spinors is odd, a pair has p <= q or
  // p >= n_spinors, a pair is repeated, or the list is not closed under
  // Kramers partners (every pair's image must be in the list).
  KramersRestriction(std::size_t n_spinors, const std::vector<std::pair<std::size_t, std::size_t>>& pairs);

  std::size_t nSpinors() const { return n_spinors_; }
  std::size_t nPairs() const { return n_pairs_; }
  std::size_t fullSize() const { return 2 * n_pairs_; }
  std::size_t nOrbits() const { return orbits_.size(); }
  std::size_t reducedSize() const { return 2 * orbits_.size(); }
  std::size_t nFixedPairs() const { return n_fixed_; }
  const std::vector<Orbit>& orbits() const { return orbits_; }

  // Isometric embedding / restriction.
  std::vector<double> expand(const std::vector<double>& reduced) const;
  std::vector<double> contract(const std::vector<double>& full) const;
  // Representative parametrization (see above).
  std::vector<double> expandRepresentative(const std::vector<double>& reduced) const;
  std::vector<double> contractRepresentative(const std::vector<double>& full) const;

  // Time-reversal image R x of a full joint vector, its symmetric part
  // (x + R x)/2 (= Q Q^T x) and the asymmetry max_i |x - R x|.
  std::vector<double> timeReversed(const std::vector<double>& full) const;
  std::vector<double> project(const std::vector<double>& full) const;
  double asymmetry(const std::vector<double>& full) const;

  // Q^T M Q for a full joint-space matrix M (Hessian).
  Matrix<double> contractMatrix(const Matrix<double>& full) const;
  // Approximate diagonal of Q^T H Q from diag(H) alone (H_IJ between an
  // orbit's two pairs neglected): the average of the two diagonal entries.
  // Meant as a Davidson/ADAM preconditioner only.
  std::vector<double> contractDiagonal(const std::vector<double>& full_diagonal) const;

  // Per-pair complex form c_I = t_I + i y_I <-> joint [Re c; Im c].
  static std::vector<double> jointFromComplex(const std::vector<std::complex<double>>& c);
  static std::vector<std::complex<double>> complexFromJoint(const std::vector<double>& joint);

 private:
  std::size_t n_spinors_;
  std::size_t n_pairs_;
  std::size_t n_fixed_ = 0;
  std::vector<Orbit> orbits_;
};

// Largest |n_a - n_abar| over the Kramers pairs (2k,2k+1): the KR
// restriction (and the time-reversal symmetry of the gradient/Hessian)
// requires equal occupations within each pair.
double kramersOccupationDeviation(const std::vector<double>& occupations);

// NEO in the KR sector: wraps a NeoProblem over the FULL joint parameters
// [t;y] (dimension() == restriction.fullSize()) into a NeoProblem over the
// reduced isometric coordinates. The wrapped problem must outlive this
// object. gradient = Q^T g, H*v = Q^T H (Q v), trial steps and accepted
// steps are expanded back with Q, so every step is exactly
// time-reversal-symmetric. Note target_order then counts negative
// eigenvalues of the KR-sector Hessian.
class KramersNeoProblem : public NeoProblem<double> {
 public:
  KramersNeoProblem(NeoProblem<double>& full, KramersRestriction restriction);
  std::size_t dimension() const override { return restriction_.reducedSize(); }
  double energy() override { return full_.energy(); }
  std::vector<double> gradient() override;
  std::vector<double> hessianVector(const std::vector<double>& v) override;
  std::vector<double> hessianDiagonal() override;
  double trialEnergy(const std::vector<double>& d) override;
  void accept(const std::vector<double>& d) override;
  const KramersRestriction& restriction() const { return restriction_; }

 private:
  NeoProblem<double>& full_;
  KramersRestriction restriction_;
};

// ADAM in the KR sector: wraps an AdamProblem<complex> whose gradient
// entries are g_I = dE/dt_I + i dE/dy_I over the restriction's pair list
// (dimension() == restriction.nPairs()) into one over the representative
// coordinates (one complex entry per orbit). The reduced gradient of an
// orbit is g_first + (a, b)-signed g_second; ADAM's per-entry normalization
// then reproduces the full-space step for every representative pair and
// the expanded step is time-reversal symmetric by construction.
class KramersAdamProblem : public AdamProblem<std::complex<double>> {
 public:
  KramersAdamProblem(AdamProblem<std::complex<double>>& full, KramersRestriction restriction);
  std::size_t dimension() const override { return restriction_.nOrbits(); }
  double energy() override { return full_.energy(); }
  std::vector<std::complex<double>> gradient() override;
  void rotate(const std::vector<std::complex<double>>& step) override;
  void saveBest() override { full_.saveBest(); }
  void restoreBest() override { full_.restoreBest(); }
  const KramersRestriction& restriction() const { return restriction_; }

 private:
  AdamProblem<std::complex<double>>& full_;
  KramersRestriction restriction_;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_KRAMERSRESTRICTION_H
