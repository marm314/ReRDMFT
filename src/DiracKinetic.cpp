#include "DiracKinetic.h"

#include <array>
#include <cstddef>

#include "Integrals.h"
#include "NuclearAttraction.h"

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

// <bra|d/dx_k|ket> for k=0(x),1(y),2(z), for one specific pair of
// individually-normalized cartesian AOs placed at their real atomic
// centers. Built as a minimal, independent 2-shell/2-atom libcint system
// per pair (rather than a shared multi-shell system) because each cartesian
// AO here carries its own individually rescaled contraction coefficients
// (see normalizeCartesianBasis) -- distinct cartesian components of the
// same physical shell no longer share one coefficient set once normalized,
// so each is its own libcint "shell" of its own angular momentum, and only
// the entry matching its own (lx,ly,lz) is read out of the full shell block
// libcint returns.
std::array<double, 3> nablaKet(const BasisFunction& bra, const BasisFunction& ket) {
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

}  // namespace

Matrix<std::complex<double>> diracKineticMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, double speed_of_light) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> T(n, n, std::complex<double>(0.0, 0.0));

  const std::complex<double> i_unit(0.0, 1.0);
  const std::complex<double> factor = -i_unit * speed_of_light;

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::array<double, 3> d = nablaKet(large_basis[a], small_basis[b]);
      const double dx = d[0];
      const double dy = d[1];
      const double dz = d[2];

      // sigma_x=[[0,1],[1,0]], sigma_y=[[0,-i],[i,0]], sigma_z=[[1,0],[0,-1]]
      const std::complex<double> t_alpha_alpha = factor * dz;
      const std::complex<double> t_alpha_beta = factor * (dx - i_unit * dy);
      const std::complex<double> t_beta_alpha = factor * (dx + i_unit * dy);
      const std::complex<double> t_beta_beta = factor * (-dz);

      T(off_large_alpha + a, off_small_alpha + b) = t_alpha_alpha;
      T(off_large_alpha + a, off_small_beta + b) = t_alpha_beta;
      T(off_large_beta + a, off_small_alpha + b) = t_beta_alpha;
      T(off_large_beta + a, off_small_beta + b) = t_beta_beta;

      // Hermitian conjugate (Small,Large) blocks.
      T(off_small_alpha + b, off_large_alpha + a) = std::conj(t_alpha_alpha);
      T(off_small_beta + b, off_large_alpha + a) = std::conj(t_alpha_beta);
      T(off_small_alpha + b, off_large_beta + a) = std::conj(t_beta_alpha);
      T(off_small_beta + b, off_large_beta + a) = std::conj(t_beta_beta);
    }
  }

  return T;
}

Matrix<std::complex<double>> diracRestEnergyMatrix(
    const std::vector<BasisFunction>& large_basis,
    const std::vector<BasisFunction>& small_basis, double speed_of_light) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> m(n, n, std::complex<double>(0.0, 0.0));

  const Matrix<double> s_large = overlapMatrix(large_basis);
  const Matrix<double> s_small = overlapMatrix(small_basis);
  const double minus_two_c2 = -2.0 * speed_of_light * speed_of_light;

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_large; ++b) {
      const std::complex<double> value(s_large(a, b), 0.0);
      m(off_large_alpha + a, off_large_alpha + b) = value;
      m(off_large_beta + a, off_large_beta + b) = value;
    }
  }
  for (std::size_t a = 0; a < n_small; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::complex<double> value(minus_two_c2 * s_small(a, b), 0.0);
      m(off_small_alpha + a, off_small_alpha + b) = value;
      m(off_small_beta + a, off_small_beta + b) = value;
    }
  }

  return m;
}

Matrix<std::complex<double>> vextMatrix(const std::vector<BasisFunction>& large_basis,
                                         const std::vector<BasisFunction>& small_basis,
                                         const std::vector<Atom>& geometry) {
  const std::size_t n_large = large_basis.size();
  const std::size_t n_small = small_basis.size();
  const std::size_t n = 2 * n_large + 2 * n_small;

  Matrix<std::complex<double>> vext(n, n, std::complex<double>(0.0, 0.0));

  const Matrix<double> v_large = nuclearAttractionMatrix(large_basis, geometry);
  const Matrix<double> v_small = nuclearAttractionMatrix(small_basis, geometry);

  const std::size_t off_large_alpha = 0;
  const std::size_t off_large_beta = n_large;
  const std::size_t off_small_alpha = 2 * n_large;
  const std::size_t off_small_beta = 2 * n_large + n_small;

  for (std::size_t a = 0; a < n_large; ++a) {
    for (std::size_t b = 0; b < n_large; ++b) {
      const std::complex<double> value(v_large(a, b), 0.0);
      vext(off_large_alpha + a, off_large_alpha + b) = value;
      vext(off_large_beta + a, off_large_beta + b) = value;
    }
  }
  for (std::size_t a = 0; a < n_small; ++a) {
    for (std::size_t b = 0; b < n_small; ++b) {
      const std::complex<double> value(v_small(a, b), 0.0);
      vext(off_small_alpha + a, off_small_alpha + b) = value;
      vext(off_small_beta + a, off_small_beta + b) = value;
    }
  }

  return vext;
}

}  // namespace rerdmft
