#include "CholeskyEri.h"
#include "SymmetricEri.h"
#include "PNOFs.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>

#include "StringUtils.h"

namespace rerdmft {

namespace {

double realPartOf(double x) { return x; }
double realPartOf(std::complex<double> x) { return x.real(); }

void checkNegligibleImag(double /*x*/, const char* /*context*/) {}
void checkNegligibleImag(std::complex<double> x, const char* context) {
  constexpr double kTol = 1e-8;
  if (std::abs(x.imag()) > kTol * std::max(1.0, std::abs(x.real()))) {
    throw std::runtime_error(std::string("PNOFs: integral ") + context +
                              " has a non-negligible imaginary part (" + std::to_string(x.imag()) +
                              ") -- expected real by Hermiticity");
  }
}

// GNOF's n_i^d = n_i * h_p^d/h_p, with h_p^d = h_p*exp[-(h_p/(0.02*sqrt2))^2]
// (doc/rel_pnofs.tex, GNOF section). The h_p factor cancels algebraically
// (h_p^d/h_p = exp[-(h_p/(0.02*sqrt2))^2]), so this is computed directly,
// with no division (and hence no 0/0 risk as h_p -> 0).
double gnofOccD(double n_i, double h_principal) {
  constexpr double kC = 0.02 * 1.4142135623730951;  // 0.02*sqrt(2)
  const double ratio = h_principal / kC;
  return n_i * std::exp(-(ratio * ratio));
}

// Looks up, for each subspace_id appearing in `geminals`, the occupation
// of that subspace's own principal (occupied) pair -- needed by GNOF's
// Pi^inter (every pair's n^d depends on its OWN subspace's principal
// hole, not necessarily its own).
std::unordered_map<std::size_t, double> principalOccupationBySubspace(
    const std::vector<PnofGeminal>& geminals, const std::vector<double>& occupations) {
  std::unordered_map<std::size_t, double> result;
  for (const auto& g : geminals) {
    if (g.is_principal) result[g.subspace_id] = occupations[g.i];
  }
  return result;
}

// Same idea, but mapping subspace_id -> that principal's own GEMINAL
// INDEX (not its occupation) -- needed so GNOF's gradient/Hessian can
// find WHICH gradient/Hessian slot to add a "via principal" chain-rule
// contribution to.
std::unordered_map<std::size_t, std::size_t> principalIndexBySubspace(
    const std::vector<PnofGeminal>& geminals) {
  std::unordered_map<std::size_t, std::size_t> result;
  for (std::size_t k = 0; k < geminals.size(); ++k) {
    if (geminals[k].is_principal) result[geminals[k].subspace_id] = k;
  }
  return result;
}

// d(Pi_intra)/d(n_i), holding n_j fixed (doc/rel_pnofs.tex `eq:Pi-intra`,
// Pi_intra = involves_principal ? -sqrt(n_i*n_j) : +sqrt(n_i*n_j)).
double pnofPiIntraD1(double n_i, double n_j, bool involves_principal) {
  const double d = (n_i > 0.0) ? 0.5 * std::sqrt(n_j / n_i) : 0.0;
  return involves_principal ? -d : d;
}

// d(Pi_inter)/d(n_i), holding n_j fixed. PNOF5/7/7s only (see
// pnofPiInter's own comment on why GNOF needs a different pair of
// functions).
double pnofPiInterD1(PnofFunctional functional, double n_i, double n_j) {
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  switch (functional) {
    case PnofFunctional::kPnof5:
      return 0.0;
    case PnofFunctional::kPnof7: {
      // Pi = -sqrt(p*q), p=n_i*h_i, q=n_j*h_j; d(p)/d(n_i) = h_i - n_i.
      const double p = n_i * h_i;
      const double q = n_j * h_j;
      if (p <= 0.0 || q <= 0.0) return 0.0;
      return -(h_i - n_i) * std::sqrt(q) / (2.0 * std::sqrt(p));
    }
    case PnofFunctional::kPnof7s:
      return -4.0 * n_j * h_j * (h_i - n_i);
    case PnofFunctional::kGnof:
      throw std::runtime_error(
          "pnofPiInterD1: GNOF needs pnofPiInterGnofPartials, not pnofPiInterD1");
  }
  throw std::runtime_error("pnofPiInterD1: unhandled PnofFunctional value");
}

// GNOF's Pi^inter, split into its two independent partial derivatives
// (see PNOFs.h's own pnofOccupationGradient comment for why GNOF needs
// two separate partials instead of one): *f1 = d(Pi)/d(n_i) holding
// n_principal_i FIXED (the "direct" dependence -- through h_i=1-n_i and
// n_i^d's own n_i prefactor); *f2 = d(Pi)/d(n_principal_i) holding n_i
// FIXED (the "indirect" dependence -- through n_i^d's own h_p^d/h_p
// exponential factor only; the sqrt(n_i*n_j*h_i*h_j) term never
// references n_principal_i at all, so contributes nothing to *f2).
void pnofPiInterGnofPartials(double n_i, double n_j, double n_principal_i, double n_principal_j,
                              bool i_is_principal, bool j_is_principal, double* f1, double* f2) {
  if (i_is_principal && j_is_principal) {
    *f1 = 0.0;
    *f2 = 0.0;
    return;
  }
  constexpr double kC2 = (0.02 * 1.4142135623730951) * (0.02 * 1.4142135623730951);
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  const double h_principal_i = 1.0 - n_principal_i;
  const double h_principal_j = 1.0 - n_principal_j;
  const double e_pi = std::exp(-(h_principal_i * h_principal_i) / kC2);
  const double n_i_d = n_i * e_pi;
  const double n_j_d = gnofOccD(n_j, h_principal_j);
  const double sign = (i_is_principal != j_is_principal) ? -1.0 : 1.0;  // base -cross vs +cross

  // d(n_i^d)/d(n_i), n_principal_i fixed: = e_pi (computed directly, not
  // via n_i_d/n_i, to stay well-defined at n_i=0).
  const double dnid_dni = e_pi;
  const double p = n_i * h_i;
  const double q = n_j * h_j;
  const double dsqrt_dni =
      (p > 0.0 && q > 0.0) ? (h_i - n_i) * std::sqrt(q) / (2.0 * std::sqrt(p)) : 0.0;
  const double dbase_dni = dnid_dni * n_j_d - dsqrt_dni;

  // d(n_i^d)/d(n_principal_i), n_i fixed: = n_i_d * 2*h_principal_i/c^2
  // (the sqrt(...) term does not depend on n_principal_i at all).
  const double dnid_dnpi = n_i_d * 2.0 * h_principal_i / kC2;
  const double dbase_dnpi = dnid_dnpi * n_j_d;

  const double cross = std::sqrt(n_i_d * n_j_d);
  const double dcross_dni = (cross > 0.0) ? dnid_dni * n_j_d / (2.0 * cross) : 0.0;
  const double dcross_dnpi = (cross > 0.0) ? dnid_dnpi * n_j_d / (2.0 * cross) : 0.0;

  *f1 = dbase_dni + sign * dcross_dni;
  *f2 = dbase_dnpi + sign * dcross_dnpi;
}

}  // namespace

PnofFunctional parsePnofFunctional(const std::string& name) {
  const std::string upper = toUpper(name);
  if (upper == "PNOF5") return PnofFunctional::kPnof5;
  if (upper == "PNOF7") return PnofFunctional::kPnof7;
  if (upper == "PNOF7S") return PnofFunctional::kPnof7s;
  if (upper == "GNOF") return PnofFunctional::kGnof;
  throw std::runtime_error("parsePnofFunctional: unrecognized functional name '" + name + "'");
}

std::vector<PnofGeminal> buildPnofGeminals(const OrbitalSubspaceTable& table) {
  std::vector<PnofGeminal> geminals;
  std::size_t next_id = 0;
  for (const auto& pair : table.frozen_occupied) {
    geminals.push_back(PnofGeminal{pair[0], pair[1], next_id, true});
    ++next_id;
  }
  for (const auto& subspace : table.subspaces) {
    const std::size_t id = next_id++;
    geminals.push_back(PnofGeminal{subspace.occupied[0], subspace.occupied[1], id, true});
    for (const auto& virt : subspace.unoccupied) {
      geminals.push_back(PnofGeminal{virt[0], virt[1], id, false});
    }
  }
  return geminals;
}

double pnofPiIntra(double n_i, double n_j, bool involves_principal) {
  const double s = std::sqrt(n_i * n_j);
  return involves_principal ? -s : s;
}

double pnofPiInter(PnofFunctional functional, double n_i, double n_j) {
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  switch (functional) {
    case PnofFunctional::kPnof5:
      return 0.0;
    case PnofFunctional::kPnof7:
      return -std::sqrt(n_i * n_j * h_i * h_j);
    case PnofFunctional::kPnof7s:
      return -4.0 * n_i * n_j * h_i * h_j;
    case PnofFunctional::kGnof:
      throw std::runtime_error(
          "pnofPiInter: GNOF needs pnofPiInterGnof (its Pi^inter depends on each pair's OWN "
          "subspace-principal occupation, not just n_i,n_j)");
  }
  throw std::runtime_error("pnofPiInter: unhandled PnofFunctional value");
}

double pnofPiInterGnof(double n_i, double n_j, double n_principal_i, double n_principal_j,
                        bool i_is_principal, bool j_is_principal) {
  if (i_is_principal && j_is_principal) return 0.0;
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  const double h_principal_i = 1.0 - n_principal_i;
  const double h_principal_j = 1.0 - n_principal_j;
  const double n_i_d = gnofOccD(n_i, h_principal_i);
  const double n_j_d = gnofOccD(n_j, h_principal_j);
  const double base = n_i_d * n_j_d - std::sqrt(n_i * n_j * h_i * h_j);
  const double cross = std::sqrt(n_i_d * n_j_d);
  return (i_is_principal != j_is_principal) ? (base - cross) : (base + cross);
}

PnofTwoRdm buildPnofTwoRdm(PnofFunctional functional, const std::vector<PnofGeminal>& geminals,
                            const std::vector<double>& occupations, bool relativistic) {
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("buildPnofTwoRdm: no geminals given");
  }
  std::vector<double> n_gem(n);
  for (std::size_t k = 0; k < n; ++k) n_gem[k] = occupations[geminals[k].i];
  const auto principal_occ = principalOccupationBySubspace(geminals, occupations);

  PnofTwoRdm result;
  result.two_rdm_h = Matrix<double>(n, n, 0.0);
  result.two_rdm_x = Matrix<double>(n, n, 0.0);
  result.two_rdm_l1 = Matrix<double>(n, n, 0.0);
  if (relativistic) result.two_rdm_l2 = Matrix<double>(n, n, 0.0);

  // Diagonal (a=a): PNOFs are self-interaction-free by construction --
  // two_rdm_h(a,a)=n_a, two_rdm_x(a,a)=0 (NOT n_a^2/n_a^2 as plain
  // HF/JK_only's own diagonal convention would give). Contracted (see
  // pnofElectronicEnergy) against J_aa=K_aa=eri(i,i,i,i) -- no bar
  // partner is involved at a=a, since a and b are GEMINAL
  // (representative-pair) indices, not actual-orbital indices -- this
  // reproduces exactly doc/rel_pnofs.tex's own `eq:Ep` diagonal term
  // n_i*(2h_ii+J_ii), with no separate diagonal term needed elsewhere.
  // two_rdm_l1/l2 are left at 0 on the diagonal: Eq. 36's rows 6/7 only
  // ever sum over i != j geminals.
  for (std::size_t a = 0; a < n; ++a) {
    result.two_rdm_h(a, a) = n_gem[a];
  }

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      if (a == b) continue;
      const double n_i = n_gem[a];
      const double n_j = n_gem[b];
      const bool same_subspace = geminals[a].subspace_id == geminals[b].subspace_id;
      const double delta = same_subspace ? n_i * n_j : 0.0;
      result.two_rdm_h(a, b) = 2.0 * (n_i * n_j - delta);
      result.two_rdm_x(a, b) = n_i * n_j - delta;

      double pi;
      if (same_subspace) {
        const bool involves_principal = geminals[a].is_principal || geminals[b].is_principal;
        pi = pnofPiIntra(n_i, n_j, involves_principal);
      } else if (functional == PnofFunctional::kGnof) {
        pi = pnofPiInterGnof(n_i, n_j, principal_occ.at(geminals[a].subspace_id),
                              principal_occ.at(geminals[b].subspace_id), geminals[a].is_principal,
                              geminals[b].is_principal);
      } else {
        pi = pnofPiInter(functional, n_i, n_j);
      }
      result.two_rdm_l1(a, b) = pi;
      if (relativistic) result.two_rdm_l2(a, b) = pi;
    }
  }
  return result;
}

template <typename T, typename Eri>
double pnofElectronicEnergy(PnofFunctional /*functional*/, const Matrix<T>& h,
                             const Eri& eri, const std::vector<double>& occupations,
                             const std::vector<PnofGeminal>& geminals, const PnofTwoRdm& two_rdm,
                             bool relativistic) {
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("pnofElectronicEnergy: no geminals given");
  }

  double energy = 0.0;
  // One-electron term: sum_p n_p*h_pp over every ACTUAL orbital (both
  // members i and ibar of every geminal) -- equals
  // sum_a 2*n_a*h_ii under Kramers/spin-pair occupation symmetry
  // (n_ibar=n_i, h_ibarbar=h_ii), matching doc/rel_pnofs.tex's own
  // h-block, but summed directly over actual orbitals here rather than
  // via that identity.
  for (std::size_t a = 0; a < n; ++a) {
    const T h_i = h(geminals[a].i, geminals[a].i);
    const T h_ibar = h(geminals[a].ibar, geminals[a].ibar);
    checkNegligibleImag(h_i, "h_ii");
    checkNegligibleImag(h_ibar, "h_ibarbar");
    energy += occupations[geminals[a].i] * realPartOf(h_i) +
              occupations[geminals[a].ibar] * realPartOf(h_ibar);
  }

  // Two-electron H/X term: ALL ordered geminal pairs (a,b), INCLUDING
  // a=b -- at a=b, J_aa=K_aa=eri(i,i,i,i) trivially (no bar partner
  // involved, since a,b are geminal/representative indices, not actual
  // orbital indices), and two_rdm_h(a,a)=n_a, two_rdm_x(a,a)=0 (see
  // buildPnofTwoRdm) give exactly n_a*J_aa, doc/rel_pnofs.tex's own
  // `eq:Ep` diagonal Hartree term -- no separate diagonal term needed.
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      const std::size_t i = geminals[a].i;
      const std::size_t j = geminals[b].i;
      const T J_val = eri(i, j, i, j);
      const T K_val = eri(i, j, j, i);
      checkNegligibleImag(J_val, "J_ij");
      checkNegligibleImag(K_val, "K_ij");
      const double J_ij = realPartOf(J_val);
      const double K_ij = realPartOf(K_val);

      energy += two_rdm.two_rdm_h(a, b) * J_ij;
      energy -= two_rdm.two_rdm_x(a, b) * K_ij;
      if (a == b) continue;  // L1/L2 (Pi-pairing) terms are i != j only

      energy += two_rdm.two_rdm_l1(a, b) * K_ij;

      if (relativistic) {
        const std::size_t ibar = geminals[a].ibar;
        const T L_val = eri(ibar, j, j, ibar);
        checkNegligibleImag(L_val, "L_ij");
        const double L_ij = realPartOf(L_val);
        energy -= two_rdm.two_rdm_x(a, b) * L_ij;
        energy += two_rdm.two_rdm_l2(a, b) * L_ij;
      }
    }
  }
  return energy;
}

template <typename T>
double pnofElectronicEnergyDirect(PnofFunctional functional, const Matrix<T>& h,
                                   const Tensor4<T>& eri, const std::vector<double>& occupations,
                                   const std::vector<PnofGeminal>& geminals, bool relativistic) {
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("pnofElectronicEnergyDirect: no geminals given");
  }
  const auto principal_occ = principalOccupationBySubspace(geminals, occupations);

  double energy = 0.0;
  for (std::size_t a = 0; a < n; ++a) {
    const std::size_t i = geminals[a].i;
    const T h_ii = h(i, i);
    const T J_ii = eri(i, i, i, i);
    checkNegligibleImag(h_ii, "h_ii");
    checkNegligibleImag(J_ii, "J_ii");
    energy += occupations[i] * (2.0 * realPartOf(h_ii) + realPartOf(J_ii));
  }

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      if (a == b) continue;
      const auto& ga = geminals[a];
      const auto& gb = geminals[b];
      const std::size_t i = ga.i;
      const std::size_t j = gb.i;
      const double n_i = occupations[i];
      const double n_j = occupations[j];

      const T K_val = eri(i, j, j, i);
      checkNegligibleImag(K_val, "K_ij");
      const double K_ij = realPartOf(K_val);
      double L_ij = 0.0;
      if (relativistic) {
        const T L_val = eri(ga.ibar, j, j, ga.ibar);
        checkNegligibleImag(L_val, "L_ij");
        L_ij = realPartOf(L_val);
      }

      if (ga.subspace_id == gb.subspace_id) {
        const bool involves_principal = ga.is_principal || gb.is_principal;
        const double pi_intra = pnofPiIntra(n_i, n_j, involves_principal);
        energy += pi_intra * (K_ij + L_ij);
      } else {
        const T J_val = eri(i, j, i, j);
        checkNegligibleImag(J_val, "J_ij");
        const double J_ij = realPartOf(J_val);
        energy += n_i * n_j * (2.0 * J_ij - K_ij - L_ij);

        double pi_inter;
        if (functional == PnofFunctional::kGnof) {
          pi_inter = pnofPiInterGnof(n_i, n_j, principal_occ.at(ga.subspace_id),
                                      principal_occ.at(gb.subspace_id), ga.is_principal,
                                      gb.is_principal);
        } else {
          pi_inter = pnofPiInter(functional, n_i, n_j);
        }
        energy += pi_inter * (K_ij + L_ij);
      }
    }
  }
  return energy;
}

namespace {

// d^2(Pi_intra)/d(n_i)^2, holding n_j fixed.
double pnofPiIntraD11(double n_i, double n_j, bool involves_principal) {
  if (n_i <= 0.0) return 0.0;
  const double d = -0.25 * std::sqrt(n_j) / (n_i * std::sqrt(n_i));  // -sqrt(n_j)/(4*n_i^1.5)
  return involves_principal ? -d : d;
}

// d^2(Pi_intra)/d(n_i)d(n_j) (mixed).
double pnofPiIntraD12(double n_i, double n_j, bool involves_principal) {
  if (n_i <= 0.0 || n_j <= 0.0) return 0.0;
  const double d = 0.25 / std::sqrt(n_i * n_j);
  return involves_principal ? -d : d;
}

// d^2(Pi_inter)/d(n_i)^2, holding n_j fixed. PNOF5/7/7s only.
double pnofPiInterD11(PnofFunctional functional, double n_i, double n_j) {
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  switch (functional) {
    case PnofFunctional::kPnof5:
      return 0.0;
    case PnofFunctional::kPnof7: {
      const double p = n_i * h_i;
      const double q = n_j * h_j;
      if (p <= 0.0 || q <= 0.0) return 0.0;
      // Derived by hand (verified against finite differences before
      // trusting): using 4p+(h_i-n_i)^2 = (h_i+n_i)^2 = 1 identically,
      // this simplifies to a single term with no cancellation risk.
      return std::sqrt(q) / (4.0 * p * std::sqrt(p));
    }
    case PnofFunctional::kPnof7s:
      return 8.0 * n_j * h_j;
    case PnofFunctional::kGnof:
      throw std::runtime_error("pnofPiInterD11: GNOF's analytic Hessian is not implemented -- "
                                "use pnofOccupationHessianFD instead");
  }
  throw std::runtime_error("pnofPiInterD11: unhandled PnofFunctional value");
}

// d^2(Pi_inter)/d(n_i)d(n_j) (mixed). PNOF5/7/7s only.
double pnofPiInterD12(PnofFunctional functional, double n_i, double n_j) {
  const double h_i = 1.0 - n_i;
  const double h_j = 1.0 - n_j;
  switch (functional) {
    case PnofFunctional::kPnof5:
      return 0.0;
    case PnofFunctional::kPnof7: {
      const double p = n_i * h_i;
      const double q = n_j * h_j;
      if (p <= 0.0 || q <= 0.0) return 0.0;
      return -(h_i - n_i) * (h_j - n_j) / (4.0 * std::sqrt(p * q));
    }
    case PnofFunctional::kPnof7s:
      return -4.0 * (h_i - n_i) * (h_j - n_j);
    case PnofFunctional::kGnof:
      throw std::runtime_error("pnofPiInterD12: GNOF's analytic Hessian is not implemented -- "
                                "use pnofOccupationHessianFD instead");
  }
  throw std::runtime_error("pnofPiInterD12: unhandled PnofFunctional value");
}

}  // namespace

template <typename T, typename Eri>
std::vector<double> pnofOccupationGradient(PnofFunctional functional, const Matrix<T>& h,
                                            const Eri& eri,
                                            const std::vector<double>& occupations,
                                            const std::vector<PnofGeminal>& geminals,
                                            bool relativistic) {
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("pnofOccupationGradient: no geminals given");
  }
  std::vector<double> n_gem(n);
  for (std::size_t k = 0; k < n; ++k) n_gem[k] = occupations[geminals[k].i];
  const auto principal_occ = principalOccupationBySubspace(geminals, occupations);
  const auto principal_idx = principalIndexBySubspace(geminals);

  std::vector<double> grad(n, 0.0);

  // Diagonal (one-electron + self-interaction-free diagonal Hartree):
  // d/dn_a[n_a*(2h_aa+J_aa)] = 2h_aa+J_aa.
  for (std::size_t a = 0; a < n; ++a) {
    const std::size_t i = geminals[a].i;
    const T h_ii = h(i, i);
    const T J_ii = eri(i, i, i, i);
    checkNegligibleImag(h_ii, "h_ii");
    checkNegligibleImag(J_ii, "J_ii");
    grad[a] += 2.0 * realPartOf(h_ii) + realPartOf(J_ii);
  }

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = a + 1; b < n; ++b) {
      const std::size_t i = geminals[a].i;
      const std::size_t j = geminals[b].i;
      const T J_val = eri(i, j, i, j);
      const T K_val = eri(i, j, j, i);
      checkNegligibleImag(J_val, "J_ij");
      checkNegligibleImag(K_val, "K_ij");
      const double J_ab = realPartOf(J_val);
      const double K_ab = realPartOf(K_val);
      double L_ab = 0.0;
      if (relativistic) {
        const T L_val = eri(geminals[a].ibar, j, j, geminals[a].ibar);
        checkNegligibleImag(L_val, "L_ij");
        L_ab = realPartOf(L_val);
      }

      const double n_a = n_gem[a];
      const double n_b = n_gem[b];
      const bool same_subspace = geminals[a].subspace_id == geminals[b].subspace_id;
      const double contrib = K_ab + (relativistic ? L_ab : 0.0);

      // NOTE the overall factor of 2 throughout this block: the energy
      // sum (pnofElectronicEnergy) loops over BOTH ordered pairs (a,b)
      // AND (b,a) separately, and buildPnofTwoRdm's two_rdm_h(a,b) =
      // 2*(n_a*n_b-delta) (an extra factor of 2 relative to
      // two_rdm_x/two_rdm_l1/l2) -- so the TOTAL unordered-pair energy
      // contribution is 2*[2*n_a*n_b*J_ab - n_a*n_b*contrib +
      // Pi(n_a,n_b)*contrib] for inter pairs (delta=0), and
      // 2*[Pi_intra(n_a,n_b)*contrib] for intra pairs (delta=n_a*n_b
      // makes the H/X bilinear part vanish exactly). Verified against
      // central finite differences of pnofElectronicEnergy for all 4
      // functionals, relativistic and non-relativistic, before trusting.
      if (same_subspace) {
        const bool involves_principal = geminals[a].is_principal || geminals[b].is_principal;
        grad[a] += 2.0 * pnofPiIntraD1(n_a, n_b, involves_principal) * contrib;
        grad[b] += 2.0 * pnofPiIntraD1(n_b, n_a, involves_principal) * contrib;
      } else {
        grad[a] += 2.0 * (2.0 * n_b * J_ab - n_b * K_ab - (relativistic ? n_b * L_ab : 0.0));
        grad[b] += 2.0 * (2.0 * n_a * J_ab - n_a * K_ab - (relativistic ? n_a * L_ab : 0.0));

        if (functional == PnofFunctional::kGnof) {
          const double n_pa = principal_occ.at(geminals[a].subspace_id);
          const double n_pb = principal_occ.at(geminals[b].subspace_id);
          double f1_a = 0.0, f2_a = 0.0, f1_b = 0.0, f2_b = 0.0;
          pnofPiInterGnofPartials(n_a, n_b, n_pa, n_pb, geminals[a].is_principal,
                                   geminals[b].is_principal, &f1_a, &f2_a);
          pnofPiInterGnofPartials(n_b, n_a, n_pb, n_pa, geminals[b].is_principal,
                                   geminals[a].is_principal, &f1_b, &f2_b);
          grad[a] += 2.0 * f1_a * contrib;
          grad[b] += 2.0 * f1_b * contrib;
          // "Via principal" chain-rule contributions, for EVERY geminal
          // including a principal one: f1 holds n_principal FIXED, but a
          // principal geminal IS its subspace's n_principal, so its own
          // d/dn_a needs f2 as well (its slot principal_idx == a). An earlier
          // version skipped f2 for principal geminals ("already covered by
          // f1", which is false: f1 is the partial at fixed n_principal),
          // giving a GNOF gradient wrong by ~2.5e-4 at principal
          // occupations near 1 (found 2026-09-20: L-BFGS stalled without
          // converging, see tests/test_pnof_occupation_gradient.cpp).
          grad[principal_idx.at(geminals[a].subspace_id)] += 2.0 * f2_a * contrib;
          grad[principal_idx.at(geminals[b].subspace_id)] += 2.0 * f2_b * contrib;
        } else {
          grad[a] += 2.0 * pnofPiInterD1(functional, n_a, n_b) * contrib;
          grad[b] += 2.0 * pnofPiInterD1(functional, n_b, n_a) * contrib;
        }
      }
    }
  }
  return grad;
}

template <typename T, typename Eri>
Matrix<double> pnofOccupationHessian(PnofFunctional functional, const Matrix<T>& /*h*/,
                                      const Eri& eri,
                                      const std::vector<double>& occupations,
                                      const std::vector<PnofGeminal>& geminals,
                                      bool relativistic) {
  if (functional == PnofFunctional::kGnof) {
    throw std::runtime_error(
        "pnofOccupationHessian: GNOF's analytic Hessian is not implemented (its Pi^inter "
        "cross-subspace coupling makes it substantially more involved than PNOF5/7/7s) -- use "
        "pnofOccupationHessianFD instead");
  }
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("pnofOccupationHessian: no geminals given");
  }
  std::vector<double> n_gem(n);
  for (std::size_t k = 0; k < n; ++k) n_gem[k] = occupations[geminals[k].i];

  Matrix<double> hess(n, n, 0.0);

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = a + 1; b < n; ++b) {
      const std::size_t i = geminals[a].i;
      const std::size_t j = geminals[b].i;
      const T J_val = eri(i, j, i, j);
      const T K_val = eri(i, j, j, i);
      checkNegligibleImag(J_val, "J_ij");
      checkNegligibleImag(K_val, "K_ij");
      const double J_ab = realPartOf(J_val);
      const double K_ab = realPartOf(K_val);
      double L_ab = 0.0;
      if (relativistic) {
        const T L_val = eri(geminals[a].ibar, j, j, geminals[a].ibar);
        checkNegligibleImag(L_val, "L_ij");
        L_ab = realPartOf(L_val);
      }
      const double contrib = K_ab + (relativistic ? L_ab : 0.0);

      const double n_a = n_gem[a];
      const double n_b = n_gem[b];
      const bool same_subspace = geminals[a].subspace_id == geminals[b].subspace_id;

      double off_diag;   // Hessian(a,b)
      double d11_a_wrt_b;  // contributes to Hessian(a,a)
      double d11_b_wrt_a;  // contributes to Hessian(b,b)
      if (same_subspace) {
        const bool involves_principal = geminals[a].is_principal || geminals[b].is_principal;
        off_diag = 2.0 * pnofPiIntraD12(n_a, n_b, involves_principal) * contrib;
        d11_a_wrt_b = 2.0 * pnofPiIntraD11(n_a, n_b, involves_principal) * contrib;
        d11_b_wrt_a = 2.0 * pnofPiIntraD11(n_b, n_a, involves_principal) * contrib;
      } else {
        const double bilinear = 2.0 * J_ab - K_ab - (relativistic ? L_ab : 0.0);
        off_diag = 2.0 * (bilinear + pnofPiInterD12(functional, n_a, n_b) * contrib);
        d11_a_wrt_b = 2.0 * pnofPiInterD11(functional, n_a, n_b) * contrib;
        d11_b_wrt_a = 2.0 * pnofPiInterD11(functional, n_b, n_a) * contrib;
      }
      hess(a, b) += off_diag;
      hess(b, a) += off_diag;
      hess(a, a) += d11_a_wrt_b;
      hess(b, b) += d11_b_wrt_a;
    }
  }
  return hess;
}

template <typename T, typename Eri>
Matrix<double> pnofOccupationHessianFD(PnofFunctional functional, const Matrix<T>& h,
                                        const Eri& eri,
                                        const std::vector<double>& occupations,
                                        const std::vector<PnofGeminal>& geminals,
                                        bool relativistic, double h_step) {
  const std::size_t n = geminals.size();
  if (n == 0) {
    throw std::runtime_error("pnofOccupationHessianFD: no geminals given");
  }
  Matrix<double> hess(n, n, 0.0);
  for (std::size_t b = 0; b < n; ++b) {
    auto occ_plus = occupations;
    auto occ_minus = occupations;
    occ_plus[geminals[b].i] += h_step;
    occ_plus[geminals[b].ibar] += h_step;
    occ_minus[geminals[b].i] -= h_step;
    occ_minus[geminals[b].ibar] -= h_step;
    const auto grad_plus =
        pnofOccupationGradient(functional, h, eri, occ_plus, geminals, relativistic);
    const auto grad_minus =
        pnofOccupationGradient(functional, h, eri, occ_minus, geminals, relativistic);
    for (std::size_t a = 0; a < n; ++a) {
      hess(a, b) = (grad_plus[a] - grad_minus[a]) / (2.0 * h_step);
    }
  }
  // Symmetrize: finite-difference truncation error makes the raw result
  // only approximately symmetric; SQP.h's QP subproblem assumes an
  // exactly symmetric Hessian.
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = a + 1; b < n; ++b) {
      const double avg = 0.5 * (hess(a, b) + hess(b, a));
      hess(a, b) = avg;
      hess(b, a) = avg;
    }
  }
  return hess;
}

template double pnofElectronicEnergy(PnofFunctional, const Matrix<double>&, const Tensor4<double>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<double>&, const CholeskyEri<double>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<double>&, const SymmetricEri<double>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<std::complex<double>>&,
                                      const Tensor4<std::complex<double>>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<std::complex<double>>&,
                                      const CholeskyEri<std::complex<double>>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<std::complex<double>>&,
                                      const SymmetricEri<std::complex<double>>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergyDirect(PnofFunctional, const Matrix<double>&,
                                            const Tensor4<double>&, const std::vector<double>&,
                                            const std::vector<PnofGeminal>&, bool);
template double pnofElectronicEnergyDirect(PnofFunctional, const Matrix<std::complex<double>>&,
                                            const Tensor4<std::complex<double>>&,
                                            const std::vector<double>&,
                                            const std::vector<PnofGeminal>&, bool);

template std::vector<double> pnofOccupationGradient(PnofFunctional, const Matrix<double>&,
                                                     const Tensor4<double>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template std::vector<double> pnofOccupationGradient(PnofFunctional, const Matrix<double>&,
                                                     const CholeskyEri<double>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template std::vector<double> pnofOccupationGradient(PnofFunctional, const Matrix<double>&,
                                                     const SymmetricEri<double>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template std::vector<double> pnofOccupationGradient(PnofFunctional,
                                                     const Matrix<std::complex<double>>&,
                                                     const Tensor4<std::complex<double>>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template std::vector<double> pnofOccupationGradient(PnofFunctional,
                                                     const Matrix<std::complex<double>>&,
                                                     const CholeskyEri<std::complex<double>>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template std::vector<double> pnofOccupationGradient(PnofFunctional,
                                                     const Matrix<std::complex<double>>&,
                                                     const SymmetricEri<std::complex<double>>&,
                                                     const std::vector<double>&,
                                                     const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<double>&,
                                               const Tensor4<double>&, const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<double>&,
                                               const CholeskyEri<double>&, const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<double>&,
                                               const SymmetricEri<double>&, const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<std::complex<double>>&,
                                               const Tensor4<std::complex<double>>&,
                                               const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<std::complex<double>>&,
                                               const CholeskyEri<std::complex<double>>&,
                                               const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessian(PnofFunctional, const Matrix<std::complex<double>>&,
                                               const SymmetricEri<std::complex<double>>&,
                                               const std::vector<double>&,
                                               const std::vector<PnofGeminal>&, bool);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional, const Matrix<double>&,
                                                 const Tensor4<double>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional, const Matrix<double>&,
                                                 const CholeskyEri<double>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional, const Matrix<double>&,
                                                 const SymmetricEri<double>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional,
                                                 const Matrix<std::complex<double>>&,
                                                 const Tensor4<std::complex<double>>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional,
                                                 const Matrix<std::complex<double>>&,
                                                 const CholeskyEri<std::complex<double>>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);
template Matrix<double> pnofOccupationHessianFD(PnofFunctional,
                                                 const Matrix<std::complex<double>>&,
                                                 const SymmetricEri<std::complex<double>>&,
                                                 const std::vector<double>&,
                                                 const std::vector<PnofGeminal>&, bool, double);

std::size_t pnofGammasPerSubspace(int pnof_coupling) {
  if (pnof_coupling < 2) {
    throw std::runtime_error("pnofGammasPerSubspace: PNOF_COUPLING must be >= 2");
  }
  return static_cast<std::size_t>(pnof_coupling - 1);
}

std::vector<double> pnofSubspaceOccupationsFromGammas(int pnof_coupling,
                                                       const std::vector<double>& gammas) {
  const std::size_t n_coupled = pnofGammasPerSubspace(pnof_coupling);
  if (gammas.size() != n_coupled) {
    throw std::runtime_error(
        "pnofSubspaceOccupationsFromGammas: expected exactly pnof_coupling-1 gamma angles");
  }
  std::vector<double> occ(1 + n_coupled, 0.0);
  const double c = std::cos(gammas[0]);
  occ[0] = 0.5 + 0.5 * c * c;
  double remaining = 1.0 - occ[0];
  for (std::size_t k = 0; k + 1 < n_coupled; ++k) {
    const double s = std::sin(gammas[k + 1]);
    occ[1 + k] = remaining * s * s;
    remaining -= occ[1 + k];
  }
  occ[n_coupled] = remaining;
  return occ;
}

std::vector<double> pnofSubspaceGammasFromOccupations(int pnof_coupling,
                                                       const std::vector<double>& occ) {
  const std::size_t n_coupled = pnofGammasPerSubspace(pnof_coupling);
  if (occ.size() != n_coupled + 1) {
    throw std::runtime_error(
        "pnofSubspaceGammasFromOccupations: expected exactly pnof_coupling occupations");
  }
  constexpr double kPiOverFour = 0.7853981633974483;
  const auto clamp01 = [](double x) { return std::min(1.0, std::max(0.0, x)); };
  std::vector<double> gammas(n_coupled, kPiOverFour);
  // n_principal = 1/2 + 1/2 cos^2(gamma_0)  ->  cos^2(gamma_0) = 2 n_principal - 1.
  gammas[0] = std::acos(std::sqrt(clamp01(2.0 * occ[0] - 1.0)));
  double remaining = 1.0 - occ[0];
  for (std::size_t k = 0; k + 1 < n_coupled; ++k) {
    // occ[1+k] = remaining * sin^2(gamma_{k+1}).
    if (remaining > 1e-14) gammas[k + 1] = std::asin(std::sqrt(clamp01(occ[1 + k] / remaining)));
    remaining -= occ[1 + k];
  }
  return gammas;
}

std::vector<double> pnofDefaultGuessGammas(int pnof_coupling) {
  const std::size_t n_coupled = pnofGammasPerSubspace(pnof_coupling);
  constexpr double kPiOverFour = 0.7853981633974483;
  return std::vector<double>(n_coupled, kPiOverFour);
}

PnofSubspaceOccupationsWithGradient pnofSubspaceOccupationsFromGammasWithGradient(
    int pnof_coupling, const std::vector<double>& gammas) {
  const std::size_t n_coupled = pnofGammasPerSubspace(pnof_coupling);
  if (gammas.size() != n_coupled) {
    throw std::runtime_error(
        "pnofSubspaceOccupationsFromGammasWithGradient: expected exactly pnof_coupling-1 "
        "gamma angles");
  }
  PnofSubspaceOccupationsWithGradient result;
  result.occ.assign(1 + n_coupled, 0.0);
  result.docc_dgamma = Matrix<double>(1 + n_coupled, n_coupled, 0.0);

  const double c0 = std::cos(gammas[0]);
  const double s0 = std::sin(gammas[0]);
  result.occ[0] = 0.5 + 0.5 * c0 * c0;
  result.docc_dgamma(0, 0) = -c0 * s0;  // d(occ[0])/d(gamma_0) = -0.5*sin(2*gamma_0)

  double remaining = 1.0 - result.occ[0];
  // d(remaining)/d(gamma_m) for every gamma index m, updated in place
  // alongside `remaining` itself as the nested recursion proceeds.
  std::vector<double> dremaining(n_coupled, 0.0);
  dremaining[0] = -result.docc_dgamma(0, 0);

  for (std::size_t j = 1; j < n_coupled; ++j) {
    const double sj = std::sin(gammas[j]);
    const double cj = std::cos(gammas[j]);
    result.occ[j] = remaining * sj * sj;
    for (std::size_t m = 0; m < n_coupled; ++m) {
      result.docc_dgamma(j, m) = (m == j) ? remaining * 2.0 * sj * cj : dremaining[m] * sj * sj;
    }
    std::vector<double> dremaining_next(n_coupled);
    for (std::size_t m = 0; m < n_coupled; ++m) {
      dremaining_next[m] =
          (m == j) ? -result.docc_dgamma(j, j) : dremaining[m] * cj * cj;
    }
    remaining -= result.occ[j];
    dremaining = std::move(dremaining_next);
  }
  result.occ[n_coupled] = remaining;
  for (std::size_t m = 0; m < n_coupled; ++m) result.docc_dgamma(n_coupled, m) = dremaining[m];

  return result;
}

}  // namespace rerdmft
