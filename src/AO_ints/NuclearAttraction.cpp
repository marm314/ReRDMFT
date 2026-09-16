#include "NuclearAttraction.h"

#include <cstddef>

#include "Element.h"

// libcint is a C library and its headers do not guard themselves with
// `extern "C"`, so that is done here to get correct (unmangled) linkage.
extern "C" {
#include <cint.h>
}

namespace rerdmft {

namespace {

// libcint's simplified ("cint2") nuclear-attraction wrapper, in the same
// situation as cint1e_ovlp_cart/cint1e_ipovlp_cart (see Integrals.cpp /
// DiracKinetic.cpp): compiled in but not declared in any installed header.
extern "C" FINT cint1e_nuc_cart(double* out, FINT* shls, FINT* atm_data, FINT natm,
                                 FINT* bas, FINT nbas, double* env);

// <bra|Vext|ket> for one specific pair of individually-normalized cartesian
// AOs, where Vext(r) = -sum_A Z_A/|r-R_A| sums over every atom in
// `geometry` (libcint negates the charge magnitude internally, so real
// atoms are given their plain positive atomic number). Bra and ket get
// their own dedicated, zero-charge placement atoms (as in
// Integrals.cpp's overlapPair / DiracKinetic.cpp's nablaKet) so that the
// real nuclei -- appended after them, some possibly at the very same
// coordinates as bra or ket -- are the only contributions to the Coulomb
// sum, regardless of where the AOs themselves happen to be centered.
double nuclearAttractionPair(const BasisFunction& bra, const BasisFunction& ket,
                              const std::vector<Atom>& geometry) {
  const int bra_index = cartesianComponentIndex(bra.l, bra.cartesian);
  const int ket_index = cartesianComponentIndex(ket.l, ket.cartesian);

  const FINT n_bra_prim = static_cast<FINT>(bra.exponents.size());
  const FINT n_ket_prim = static_cast<FINT>(ket.exponents.size());
  const FINT n_nuclei = static_cast<FINT>(geometry.size());
  const FINT natm = 2 + n_nuclei;

  std::vector<FINT> atm_data(static_cast<std::size_t>(natm) * ATM_SLOTS, 0);
  atm_data[0 * ATM_SLOTS + CHARGE_OF] = 0;
  atm_data[0 * ATM_SLOTS + PTR_COORD] = PTR_ENV_START;
  atm_data[1 * ATM_SLOTS + CHARGE_OF] = 0;
  atm_data[1 * ATM_SLOTS + PTR_COORD] = PTR_ENV_START + 3;

  FINT env_offset = PTR_ENV_START + 6;
  for (FINT a = 0; a < n_nuclei; ++a) {
    const std::size_t atom_slot = static_cast<std::size_t>(2 + a) * ATM_SLOTS;
    atm_data[atom_slot + CHARGE_OF] = atomicNumber(geometry[static_cast<std::size_t>(a)].symbol);
    atm_data[atom_slot + PTR_COORD] = env_offset;
    env_offset += 3;
  }

  FINT bas[2 * BAS_SLOTS] = {0};
  bas[0 * BAS_SLOTS + ATOM_OF] = 0;
  bas[0 * BAS_SLOTS + ANG_OF] = bra.l;
  bas[0 * BAS_SLOTS + NPRIM_OF] = n_bra_prim;
  bas[0 * BAS_SLOTS + NCTR_OF] = 1;
  bas[1 * BAS_SLOTS + ATOM_OF] = 1;
  bas[1 * BAS_SLOTS + ANG_OF] = ket.l;
  bas[1 * BAS_SLOTS + NPRIM_OF] = n_ket_prim;
  bas[1 * BAS_SLOTS + NCTR_OF] = 1;

  bas[0 * BAS_SLOTS + PTR_EXP] = env_offset;
  env_offset += n_bra_prim;
  bas[0 * BAS_SLOTS + PTR_COEFF] = env_offset;
  env_offset += n_bra_prim;
  bas[1 * BAS_SLOTS + PTR_EXP] = env_offset;
  env_offset += n_ket_prim;
  bas[1 * BAS_SLOTS + PTR_COEFF] = env_offset;
  env_offset += n_ket_prim;

  std::vector<double> env(static_cast<std::size_t>(env_offset), 0.0);
  env[static_cast<std::size_t>(atm_data[0 * ATM_SLOTS + PTR_COORD]) + 0] = bra.x;
  env[static_cast<std::size_t>(atm_data[0 * ATM_SLOTS + PTR_COORD]) + 1] = bra.y;
  env[static_cast<std::size_t>(atm_data[0 * ATM_SLOTS + PTR_COORD]) + 2] = bra.z;
  env[static_cast<std::size_t>(atm_data[1 * ATM_SLOTS + PTR_COORD]) + 0] = ket.x;
  env[static_cast<std::size_t>(atm_data[1 * ATM_SLOTS + PTR_COORD]) + 1] = ket.y;
  env[static_cast<std::size_t>(atm_data[1 * ATM_SLOTS + PTR_COORD]) + 2] = ket.z;
  for (FINT a = 0; a < n_nuclei; ++a) {
    const std::size_t atom_slot = static_cast<std::size_t>(2 + a) * ATM_SLOTS;
    const std::size_t coord = static_cast<std::size_t>(atm_data[atom_slot + PTR_COORD]);
    const Atom& nucleus = geometry[static_cast<std::size_t>(a)];
    env[coord + 0] = nucleus.x;
    env[coord + 1] = nucleus.y;
    env[coord + 2] = nucleus.z;
  }
  for (FINT i = 0; i < n_bra_prim; ++i) {
    env[static_cast<std::size_t>(bas[0 * BAS_SLOTS + PTR_EXP] + i)] =
        bra.exponents[static_cast<std::size_t>(i)];
    env[static_cast<std::size_t>(bas[0 * BAS_SLOTS + PTR_COEFF] + i)] =
        bra.coefficients[static_cast<std::size_t>(i)];
  }
  for (FINT j = 0; j < n_ket_prim; ++j) {
    env[static_cast<std::size_t>(bas[1 * BAS_SLOTS + PTR_EXP] + j)] =
        ket.exponents[static_cast<std::size_t>(j)];
    env[static_cast<std::size_t>(bas[1 * BAS_SLOTS + PTR_COEFF] + j)] =
        ket.coefficients[static_cast<std::size_t>(j)];
  }

  FINT shls[2] = {0, 1};
  const FINT ni = CINTcgto_cart(0, bas);
  const FINT nj = CINTcgto_cart(1, bas);
  std::vector<double> buf(static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj));
  cint1e_nuc_cart(buf.data(), shls, atm_data.data(), natm, bas, 2, env.data());

  return buf[static_cast<std::size_t>(bra_index) + static_cast<std::size_t>(ni) *
                                                         static_cast<std::size_t>(ket_index)];
}

}  // namespace

Matrix<double> nuclearAttractionMatrix(const std::vector<BasisFunction>& basis,
                                        const std::vector<Atom>& geometry) {
  const std::size_t n = basis.size();
  Matrix<double> v(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double value = nuclearAttractionPair(basis[i], basis[j], geometry);
      v(i, j) = value;
      v(j, i) = value;
    }
  }
  return v;
}

}  // namespace rerdmft
