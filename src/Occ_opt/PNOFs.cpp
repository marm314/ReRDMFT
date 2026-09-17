#include "PNOFs.h"

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

template <typename T>
double pnofElectronicEnergy(PnofFunctional /*functional*/, const Matrix<T>& h,
                             const Tensor4<T>& eri, const std::vector<double>& occupations,
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

template double pnofElectronicEnergy(PnofFunctional, const Matrix<double>&, const Tensor4<double>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergy(PnofFunctional, const Matrix<std::complex<double>>&,
                                      const Tensor4<std::complex<double>>&,
                                      const std::vector<double>&,
                                      const std::vector<PnofGeminal>&, const PnofTwoRdm&, bool);
template double pnofElectronicEnergyDirect(PnofFunctional, const Matrix<double>&,
                                            const Tensor4<double>&, const std::vector<double>&,
                                            const std::vector<PnofGeminal>&, bool);
template double pnofElectronicEnergyDirect(PnofFunctional, const Matrix<std::complex<double>>&,
                                            const Tensor4<std::complex<double>>&,
                                            const std::vector<double>&,
                                            const std::vector<PnofGeminal>&, bool);

}  // namespace rerdmft
