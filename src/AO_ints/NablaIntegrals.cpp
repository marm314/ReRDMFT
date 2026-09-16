#include "NablaIntegrals.h"

#include <cstddef>
#include <vector>

// libcint is a C library and its headers do not guard themselves with
// `extern "C"`, so that is done here to get correct (unmangled) linkage.
extern "C" {
#include <cint.h>
}

namespace rerdmft {

namespace {

// libcint's simplified ("cint2") bra-differentiated overlap wrapper
// (<NABLA i|j>, a 3-component X/Y/Z tensor). Compiled into libcint whenever
// it is built WITH_CINT2_INTERFACE (the default, see Integrals.cpp for the
// same situation with cint1e_ovlp_cart), but not declared in any of its
// installed headers, so it is declared here from its known implementation.
extern "C" FINT cint1e_ipovlp_cart(double* out, FINT* shls, FINT* atm,
                                    FINT natm, FINT* bas, FINT nbas,
                                    double* env);

}  // namespace

std::array<double, 3> nablaIntegral(const BasisFunction& bra, const BasisFunction& ket) {
  const int bra_index = cartesianComponentIndex(bra.l, bra.cartesian);
  const int ket_index = cartesianComponentIndex(ket.l, ket.cartesian);

  const FINT n_bra_prim = static_cast<FINT>(bra.exponents.size());
  const FINT n_ket_prim = static_cast<FINT>(ket.exponents.size());

  FINT atm[2 * ATM_SLOTS] = {0};
  atm[0 * ATM_SLOTS + CHARGE_OF] = 0;
  atm[0 * ATM_SLOTS + PTR_COORD] = PTR_ENV_START;
  atm[1 * ATM_SLOTS + CHARGE_OF] = 0;
  atm[1 * ATM_SLOTS + PTR_COORD] = PTR_ENV_START + 3;

  FINT bas[2 * BAS_SLOTS] = {0};
  bas[0 * BAS_SLOTS + ATOM_OF] = 0;
  bas[0 * BAS_SLOTS + ANG_OF] = bra.l;
  bas[0 * BAS_SLOTS + NPRIM_OF] = n_bra_prim;
  bas[0 * BAS_SLOTS + NCTR_OF] = 1;
  bas[1 * BAS_SLOTS + ATOM_OF] = 1;
  bas[1 * BAS_SLOTS + ANG_OF] = ket.l;
  bas[1 * BAS_SLOTS + NPRIM_OF] = n_ket_prim;
  bas[1 * BAS_SLOTS + NCTR_OF] = 1;

  FINT env_offset = PTR_ENV_START + 6;
  bas[0 * BAS_SLOTS + PTR_EXP] = env_offset;
  env_offset += n_bra_prim;
  bas[0 * BAS_SLOTS + PTR_COEFF] = env_offset;
  env_offset += n_bra_prim;
  bas[1 * BAS_SLOTS + PTR_EXP] = env_offset;
  env_offset += n_ket_prim;
  bas[1 * BAS_SLOTS + PTR_COEFF] = env_offset;
  env_offset += n_ket_prim;

  std::vector<double> env(static_cast<std::size_t>(env_offset), 0.0);
  env[static_cast<std::size_t>(atm[0 * ATM_SLOTS + PTR_COORD]) + 0] = bra.x;
  env[static_cast<std::size_t>(atm[0 * ATM_SLOTS + PTR_COORD]) + 1] = bra.y;
  env[static_cast<std::size_t>(atm[0 * ATM_SLOTS + PTR_COORD]) + 2] = bra.z;
  env[static_cast<std::size_t>(atm[1 * ATM_SLOTS + PTR_COORD]) + 0] = ket.x;
  env[static_cast<std::size_t>(atm[1 * ATM_SLOTS + PTR_COORD]) + 1] = ket.y;
  env[static_cast<std::size_t>(atm[1 * ATM_SLOTS + PTR_COORD]) + 2] = ket.z;
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
  std::vector<double> buf(3 * static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj));
  cint1e_ipovlp_cart(buf.data(), shls, atm, 2, bas, 2, env.data());

  // libcint returns <NABLA bra|ket>, a 3-component tensor (X,Y,Z blocks
  // stored consecutively), each block a (ni x nj) matrix in Fortran order
  // (bra index fastest). Integration by parts gives the wanted
  // <bra|d/dx_k|ket> = -<d/dx_k bra|ket>, since real, localized basis
  // functions have no boundary term.
  const std::size_t block = static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj);
  const std::size_t entry =
      static_cast<std::size_t>(bra_index) + static_cast<std::size_t>(ni) * static_cast<std::size_t>(ket_index);
  return {-buf[0 * block + entry], -buf[1 * block + entry], -buf[2 * block + entry]};
}

}  // namespace rerdmft
