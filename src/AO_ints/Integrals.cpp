#include "Integrals.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

// libcint is a C library and its headers do not guard themselves with
// `extern "C"`, so that is done here to get correct (unmangled) linkage.
extern "C" {
#include <cint.h>
}

namespace rerdmft {

namespace {

// libcint's simplified ("cint2") overlap wrapper. It is compiled into
// libcint whenever it is built WITH_CINT2_INTERFACE (the default), but is
// not declared in any of its installed headers, so it is declared here
// from its known implementation (src/misc.h's ALL_CINT1E macro).
extern "C" FINT cint1e_ovlp_cart(double* out, FINT* shls, FINT* atm,
                                  FINT natm, FINT* bas, FINT nbas,
                                  double* env);

// Computes the nf x nf self-overlap block of one contracted shell (all of
// its cartesian components), placed at an arbitrary center. A Gaussian
// shell's self overlap does not depend on where it is centered, so the
// origin is used here regardless of the atom's actual position.
std::vector<double> shellSelfOverlap(int l,
                                      const std::vector<double>& exponents,
                                      const std::vector<double>& coefficients) {
  const FINT n_prim = static_cast<FINT>(exponents.size());

  FINT atm[ATM_SLOTS] = {0};
  atm[CHARGE_OF] = 0;
  atm[PTR_COORD] = PTR_ENV_START;

  FINT bas[BAS_SLOTS] = {0};
  bas[ATOM_OF] = 0;
  bas[ANG_OF] = l;
  bas[NPRIM_OF] = n_prim;
  bas[NCTR_OF] = 1;
  bas[PTR_EXP] = PTR_ENV_START + 3;
  bas[PTR_COEFF] = bas[PTR_EXP] + n_prim;

  std::vector<double> env(static_cast<std::size_t>(bas[PTR_COEFF] + n_prim), 0.0);
  for (FINT i = 0; i < n_prim; ++i) {
    env[static_cast<std::size_t>(bas[PTR_EXP] + i)] = exponents[static_cast<std::size_t>(i)];
    env[static_cast<std::size_t>(bas[PTR_COEFF] + i)] = coefficients[static_cast<std::size_t>(i)];
  }

  FINT shls[2] = {0, 0};
  const FINT nf = CINTcgto_cart(0, bas);
  std::vector<double> buf(static_cast<std::size_t>(nf) * static_cast<std::size_t>(nf));
  cint1e_ovlp_cart(buf.data(), shls, atm, 1, bas, 1, env.data());
  return buf;
}

// <bra|ket> for one specific pair of individually-normalized cartesian AOs,
// placed at their real atomic centers. Built as a minimal, independent
// 2-shell/2-atom libcint system per pair (rather than a shared multi-shell
// system) for the same reason as DiracKinetic.cpp's nablaKet: each cartesian
// AO carries its own individually rescaled contraction coefficients once
// normalizeCartesianBasis has run, so distinct cartesian components of one
// physical shell (e.g. d_xx vs d_xy) no longer share a single coefficient
// set that libcint's shell model requires.
double overlapPair(const BasisFunction& bra, const BasisFunction& ket) {
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
  std::vector<double> buf(static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj));
  cint1e_ovlp_cart(buf.data(), shls, atm, 2, bas, 2, env.data());

  return buf[static_cast<std::size_t>(bra_index) + static_cast<std::size_t>(ni) *
                                                         static_cast<std::size_t>(ket_index)];
}

}  // namespace

std::vector<NormalizationCheck> normalizeCartesianBasis(
    std::vector<BasisFunction>& functions, double tolerance) {
  std::vector<NormalizationCheck> report;

  std::size_t i = 0;
  while (i < functions.size()) {
    const int l = functions[i].l;
    if (l > 6) {
      // libcint's cartesian integrals only support angular momentum up to
      // l=6 (i-type); see doc/program_ref.txt.
      throw std::runtime_error(
          "angular momentum l=" + std::to_string(l) +
          " exceeds libcint's supported maximum (l=6, i-type)");
    }
    const int nf = static_cast<int>(cartesianComponents(l).size());

    // Primitive normalization: basis-set-file coefficients are defined for
    // already-normalized primitives, but libcint expects raw ones.
    std::vector<double> exponents = functions[i].exponents;
    std::vector<double> coefficients = functions[i].coefficients;
    for (std::size_t p = 0; p < exponents.size(); ++p) {
      coefficients[p] *= CINTgto_norm(l, exponents[p]);
    }

    const std::vector<double> overlap =
        shellSelfOverlap(l, exponents, coefficients);

    for (int k = 0; k < nf; ++k) {
      const double s_kk = overlap[static_cast<std::size_t>(k) * nf + k];
      const double scale = 1.0 / std::sqrt(s_kk);

      BasisFunction& fn = functions[i + static_cast<std::size_t>(k)];
      fn.exponents = exponents;
      fn.coefficients = coefficients;
      for (double& c : fn.coefficients) c *= scale;

      NormalizationCheck check;
      check.element = fn.element;
      check.l = l;
      check.cartesian = fn.cartesian;
      check.self_overlap_before = s_kk;
      check.was_renormalized = std::abs(s_kk - 1.0) > tolerance;
      report.push_back(check);
    }

    i += static_cast<std::size_t>(nf);
  }

  return report;
}

Matrix<double> overlapMatrix(const std::vector<BasisFunction>& basis) {
  const std::size_t n = basis.size();
  Matrix<double> s(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double value = overlapPair(basis[i], basis[j]);
      s(i, j) = value;
      s(j, i) = value;
    }
  }
  return s;
}

}  // namespace rerdmft
