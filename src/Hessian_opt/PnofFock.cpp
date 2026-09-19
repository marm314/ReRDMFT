#include "PnofFock.h"

#include <array>
#include <complex>
#include <stdexcept>
#include <unordered_map>

#include "HartreeExchangeGradient.h"

namespace rerdmft {

namespace {

std::unordered_map<std::size_t, double> principalOccupationBySubspaceLocal(
    const std::vector<PnofGeminal>& geminals, const std::vector<double>& occupations) {
  std::unordered_map<std::size_t, double> result;
  for (const auto& g : geminals) {
    if (g.is_principal) result[g.subspace_id] = occupations[g.i];
  }
  return result;
}

}  // namespace

PnofFullTwoRdm buildPnofFullTwoRdm(PnofFunctional functional,
                                    const std::vector<PnofGeminal>& geminals,
                                    const std::vector<double>& occupations, std::size_t n_total,
                                    bool relativistic) {
  if (geminals.empty()) {
    throw std::runtime_error("buildPnofFullTwoRdm: no geminals given");
  }
  PnofFullTwoRdm result;
  result.two_rdm_h = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_x = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_l1 = Matrix<double>(n_total, n_total, 0.0);
  result.two_rdm_l2 = Matrix<double>(n_total, n_total, 0.0);

  const auto principal_occ = principalOccupationBySubspaceLocal(geminals, occupations);
  const std::size_t n_gem = geminals.size();

  for (std::size_t a = 0; a < n_gem; ++a) {
    const std::size_t i = geminals[a].i;
    const std::size_t ibar = geminals[a].ibar;
    const double n_a = occupations[i];

    // Same geminal (rep(P)=rep(Q)=a): P==Q left at the default 0; the
    // two bar-partner combinations (i,ibar)/(ibar,i) both get n_a, with
    // NO Pi correction (Occ_opt/PNOFs.cpp's own geminal-representative
    // loop explicitly skips a==b before ever touching its Pi terms).
    result.two_rdm_h(i, ibar) = n_a;
    result.two_rdm_h(ibar, i) = n_a;
    result.two_rdm_x(i, ibar) = n_a;
    result.two_rdm_x(ibar, i) = n_a;

    for (std::size_t b = a + 1; b < n_gem; ++b) {
      const std::size_t j = geminals[b].i;
      const std::size_t jbar = geminals[b].ibar;
      const double n_b = occupations[j];
      const bool same_subspace = geminals[a].subspace_id == geminals[b].subspace_id;
      const double delta = same_subspace ? n_a * n_b : 0.0;
      const double h_x_value = n_a * n_b - delta;

      double pi;
      if (same_subspace) {
        const bool involves_principal = geminals[a].is_principal || geminals[b].is_principal;
        pi = pnofPiIntra(n_a, n_b, involves_principal);
      } else if (functional == PnofFunctional::kGnof) {
        pi = pnofPiInterGnof(n_a, n_b, principal_occ.at(geminals[a].subspace_id),
                              principal_occ.at(geminals[b].subspace_id), geminals[a].is_principal,
                              geminals[b].is_principal);
      } else {
        pi = pnofPiInter(functional, n_a, n_b);
      }

      // H (Coulomb, J-type) is uniform across every bar-combination of
      // (i/ibar, j/jbar) -- doc/rel_pnofs.tex's own `eq:coulomb-elems`
      // first relation. X (exchange) is the SAME uniform value at
      // MATCHING bar-parity combinations (P,Q both representatives or
      // both bar-partners), unconditionally; at MISMATCHED bar-parity,
      // only when `relativistic` (the genuinely relativistic-only
      // L_ij-type contribution, doc's own `eq:lflip-elems`). Pi's own
      // pairing contribution is a SEPARATE, independent 2-RDM entry
      // (Hessian_opt/HartreeExchangeGradient.h's own L1/L2 pattern,
      // `eri(P,Pbar,Q,Qbar)`/`eri(P,Pbar,Qbar,Q)`) -- NOT folded into X
      // (see this file's header comment for why an earlier version's
      // "fold into X" simplification, while exactly energy/Fock-
      // equivalent, is NOT Hessian-equivalent).
      const std::array<std::size_t, 2> a_members = {i, ibar};
      const std::array<std::size_t, 2> b_members = {j, jbar};
      for (std::size_t pp = 0; pp < 2; ++pp) {
        for (std::size_t qq = 0; qq < 2; ++qq) {
          const std::size_t P = a_members[pp];
          const std::size_t Q = b_members[qq];
          const bool matching_parity = (pp == qq);

          result.two_rdm_h(P, Q) = h_x_value;
          result.two_rdm_h(Q, P) = h_x_value;

          if (matching_parity || relativistic) {
            result.two_rdm_x(P, Q) = h_x_value;
            result.two_rdm_x(Q, P) = h_x_value;
          }

          const double l1_signed = matching_parity ? (pi / 2.0) : (-pi / 2.0);
          result.two_rdm_l1(P, Q) = l1_signed;
          result.two_rdm_l1(Q, P) = l1_signed;
          result.two_rdm_l2(P, Q) = -l1_signed;
          result.two_rdm_l2(Q, P) = -l1_signed;
        }
      }
    }
  }
  return result;
}

std::vector<std::size_t> buildPnofPairOf(const std::vector<PnofGeminal>& geminals,
                                          std::size_t n_total) {
  std::vector<std::size_t> pair_of(n_total);
  for (std::size_t p = 0; p < n_total; ++p) pair_of[p] = p;
  for (const auto& g : geminals) {
    pair_of[g.i] = g.ibar;
    pair_of[g.ibar] = g.i;
  }
  return pair_of;
}

template <typename T>
Matrix<T> pnofFockMatrix(PnofFunctional functional, const Matrix<T>& h, const Tensor4<T>& eri,
                          const std::vector<PnofGeminal>& geminals,
                          const std::vector<double>& occupations, bool relativistic) {
  const std::size_t n = h.rows();
  const auto full = buildPnofFullTwoRdm(functional, geminals, occupations, n, relativistic);
  const auto pair_of = buildPnofPairOf(geminals, n);
  return hartreeExchangeFockMatrix(h, eri, occupations, full.two_rdm_h, full.two_rdm_x, pair_of,
                                    full.two_rdm_l1, full.two_rdm_l2);
}

template Matrix<double> pnofFockMatrix(PnofFunctional, const Matrix<double>&,
                                        const Tensor4<double>&, const std::vector<PnofGeminal>&,
                                        const std::vector<double>&, bool);
template Matrix<std::complex<double>> pnofFockMatrix(
    PnofFunctional, const Matrix<std::complex<double>>&, const Tensor4<std::complex<double>>&,
    const std::vector<PnofGeminal>&, const std::vector<double>&, bool);

}  // namespace rerdmft
