#include "ThreeCenterIntegrals.h"

#include <cblas.h>

#include <cstddef>
#include <stdexcept>

// libcint is a C library and its headers do not guard themselves with `extern "C"`, so that is
// done here, same as every other AO_ints file.
extern "C" {
#include <cint.h>
}

// cint3c2e_cart/cint2c2e_cart are libcint's RI-specific wrappers (same simplified 8-argument
// "cint2"-interface signature as cint2e_cart, confirmed against libcint-5.1.6's own source,
// src/misc.h's ALL_CINT macro) -- not declared in cint.h itself (same situation this project
// already handles for cint1e_ipovlp_cart in NablaIntegrals.cpp).
extern "C" FINT cint3c2e_cart(double* out, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas,
                               double* env, CINTOpt* opt);
extern "C" FINT cint2c2e_cart(double* out, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas,
                               double* env, CINTOpt* opt);

namespace rerdmft {

std::vector<double> twoCenterAuxShellPair(const std::vector<BasisFunction>& aux_basis,
                                           const ShellInfo& shell_p, const ShellInfo& shell_q) {
  const BasisFunction& rep_p = aux_basis[shell_p.first];
  const BasisFunction& rep_q = aux_basis[shell_q.first];
  const BasisFunction* fns[2] = {&rep_p, &rep_q};

  FINT atm[2 * ATM_SLOTS] = {0};
  FINT bas[2 * BAS_SLOTS] = {0};
  FINT n_prim[2];
  for (int i = 0; i < 2; ++i) {
    n_prim[i] = static_cast<FINT>(fns[i]->exponents.size());
    atm[i * ATM_SLOTS + CHARGE_OF] = 0;
    atm[i * ATM_SLOTS + PTR_COORD] = PTR_ENV_START + 3 * i;
    bas[i * BAS_SLOTS + ATOM_OF] = i;
    bas[i * BAS_SLOTS + ANG_OF] = fns[i]->l;
    bas[i * BAS_SLOTS + NPRIM_OF] = n_prim[i];
    bas[i * BAS_SLOTS + NCTR_OF] = 1;
  }
  FINT env_offset = PTR_ENV_START + 3 * 2;
  for (int i = 0; i < 2; ++i) {
    bas[i * BAS_SLOTS + PTR_EXP] = env_offset;
    env_offset += n_prim[i];
    bas[i * BAS_SLOTS + PTR_COEFF] = env_offset;
    env_offset += n_prim[i];
  }
  std::vector<double> env(static_cast<std::size_t>(env_offset), 0.0);
  for (int i = 0; i < 2; ++i) {
    const std::size_t coord = static_cast<std::size_t>(atm[i * ATM_SLOTS + PTR_COORD]);
    env[coord + 0] = fns[i]->x;
    env[coord + 1] = fns[i]->y;
    env[coord + 2] = fns[i]->z;
    for (FINT k = 0; k < n_prim[i]; ++k) {
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_EXP] + k)] = fns[i]->exponents[static_cast<std::size_t>(k)];
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_COEFF] + k)] = fns[i]->coefficients[static_cast<std::size_t>(k)];
    }
  }

  FINT shls[2] = {0, 1};
  const FINT ni = CINTcgto_cart(0, bas);
  const FINT nj = CINTcgto_cart(1, bas);
  std::vector<double> buf(static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj));
  cint2c2e_cart(buf.data(), shls, atm, 2, bas, 2, env.data(), nullptr);

  const double ref_p = rep_p.coefficients[0];
  const double ref_q = rep_q.coefficients[0];
  std::vector<double> result(shell_p.count * shell_q.count);
  for (std::size_t i = 0; i < shell_p.count; ++i) {
    const double scale_i = aux_basis[shell_p.first + i].coefficients[0] / ref_p;
    for (std::size_t j = 0; j < shell_q.count; ++j) {
      const double scale_j = aux_basis[shell_q.first + j].coefficients[0] / ref_q;
      result[i * shell_q.count + j] =
          buf[i + static_cast<std::size_t>(ni) * j] * scale_i * scale_j;
    }
  }
  return result;
}

std::vector<double> threeCenterPairAuxShell(const BasisFunction& p, const BasisFunction& q,
                                             const std::vector<BasisFunction>& aux_basis,
                                             const ShellInfo& aux_shell) {
  const int p_index = cartesianComponentIndex(p.l, p.cartesian);
  const int q_index = cartesianComponentIndex(q.l, q.cartesian);
  const BasisFunction& aux_rep = aux_basis[aux_shell.first];
  const BasisFunction* fns[3] = {&p, &q, &aux_rep};

  FINT atm[3 * ATM_SLOTS] = {0};
  FINT bas[3 * BAS_SLOTS] = {0};
  FINT n_prim[3];
  for (int i = 0; i < 3; ++i) {
    n_prim[i] = static_cast<FINT>(fns[i]->exponents.size());
    atm[i * ATM_SLOTS + CHARGE_OF] = 0;
    atm[i * ATM_SLOTS + PTR_COORD] = PTR_ENV_START + 3 * i;
    bas[i * BAS_SLOTS + ATOM_OF] = i;
    bas[i * BAS_SLOTS + ANG_OF] = fns[i]->l;
    bas[i * BAS_SLOTS + NPRIM_OF] = n_prim[i];
    bas[i * BAS_SLOTS + NCTR_OF] = 1;
  }
  FINT env_offset = PTR_ENV_START + 3 * 3;
  for (int i = 0; i < 3; ++i) {
    bas[i * BAS_SLOTS + PTR_EXP] = env_offset;
    env_offset += n_prim[i];
    bas[i * BAS_SLOTS + PTR_COEFF] = env_offset;
    env_offset += n_prim[i];
  }
  std::vector<double> env(static_cast<std::size_t>(env_offset), 0.0);
  for (int i = 0; i < 3; ++i) {
    const std::size_t coord = static_cast<std::size_t>(atm[i * ATM_SLOTS + PTR_COORD]);
    env[coord + 0] = fns[i]->x;
    env[coord + 1] = fns[i]->y;
    env[coord + 2] = fns[i]->z;
    for (FINT k = 0; k < n_prim[i]; ++k) {
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_EXP] + k)] = fns[i]->exponents[static_cast<std::size_t>(k)];
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_COEFF] + k)] = fns[i]->coefficients[static_cast<std::size_t>(k)];
    }
  }

  FINT shls[3] = {0, 1, 2};
  const FINT ni = CINTcgto_cart(0, bas);
  const FINT nj = CINTcgto_cart(1, bas);
  const FINT nk = CINTcgto_cart(2, bas);
  std::vector<double> buf(static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(nk));
  cint3c2e_cart(buf.data(), shls, atm, 3, bas, 3, env.data(), nullptr);

  const double ref = aux_rep.coefficients[0];
  std::vector<double> result(static_cast<std::size_t>(nk));
  for (FINT k = 0; k < nk; ++k) {
    const std::size_t idx = static_cast<std::size_t>(p_index) +
                             static_cast<std::size_t>(ni) *
                                 (static_cast<std::size_t>(q_index) + static_cast<std::size_t>(nj) * static_cast<std::size_t>(k));
    const double scale = aux_basis[aux_shell.first + static_cast<std::size_t>(k)].coefficients[0] / ref;
    result[static_cast<std::size_t>(k)] = buf[idx] * scale;
  }
  return result;
}

Matrix<double> auxMetric(const std::vector<BasisFunction>& aux_basis) {
  const std::size_t n_aux = aux_basis.size();
  const std::vector<ShellInfo> aux_shells = groupIntoShells(aux_basis);
  Matrix<double> metric(n_aux, n_aux, 0.0);

#pragma omp parallel for schedule(dynamic)
  for (std::size_t sp = 0; sp < aux_shells.size(); ++sp) {
    for (std::size_t sq = sp; sq < aux_shells.size(); ++sq) {
      const ShellInfo& shell_p = aux_shells[sp];
      const ShellInfo& shell_q = aux_shells[sq];
      const std::vector<double> block = twoCenterAuxShellPair(aux_basis, shell_p, shell_q);
      for (std::size_t i = 0; i < shell_p.count; ++i) {
        for (std::size_t j = 0; j < shell_q.count; ++j) {
          const double v = block[i * shell_q.count + j];
          metric(shell_p.first + i, shell_q.first + j) = v;
          metric(shell_q.first + j, shell_p.first + i) = v;
        }
      }
    }
  }
  return metric;
}

Matrix<double> threeCenterIntegralsRaw(const std::vector<BasisFunction>& basis,
                                        const std::vector<BasisFunction>& aux_basis) {
  const std::size_t n = basis.size();
  const std::size_t n_aux = aux_basis.size();
  const std::vector<ShellInfo> aux_shells = groupIntoShells(aux_basis);
  Matrix<double> raw(n_aux, n * n, 0.0);

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p; q < n; ++q) {
      for (const ShellInfo& aux_shell : aux_shells) {
        const std::vector<double> values = threeCenterPairAuxShell(basis[p], basis[q], aux_basis, aux_shell);
        for (std::size_t k = 0; k < aux_shell.count; ++k) {
          const std::size_t aux_index = aux_shell.first + k;
          raw(aux_index, p * n + q) = values[k];
          raw(aux_index, q * n + p) = values[k];
        }
      }
    }
  }
  return raw;
}

Matrix<double> auxMetricOrthogonalization(const std::vector<BasisFunction>& aux_basis, double threshold,
                                           RankReductionReport* report) {
  return canonicalOrthogonalize(auxMetric(aux_basis), threshold, report);
}

Matrix<double> transformRiThreeCenterAoLegs(const Matrix<double>& eri3_in, std::size_t n_ao, const Matrix<double>& t) {
  if (eri3_in.cols() != n_ao * n_ao) {
    throw std::runtime_error("transformRiThreeCenterAoLegs: eri3_in column count does not match n_ao^2");
  }
  if (t.rows() != n_ao) {
    throw std::runtime_error("transformRiThreeCenterAoLegs: t row count does not match n_ao");
  }
  const std::size_t n_aux = eri3_in.rows();
  const std::size_t n_out = t.cols();
  Matrix<double> eri3_out(n_aux, n_out * n_out, 0.0);
  if (n_ao == 0 || n_out == 0) return eri3_out;

#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_aux; ++p) {
    const double* slice = eri3_in.data() + p * n_ao * n_ao;
    // u = t^T @ slice : (n_out x n_ao) @ (n_ao x n_ao) -> n_out x n_ao.
    Matrix<double> u(n_out, n_ao, 0.0);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_out), static_cast<int>(n_ao),
                static_cast<int>(n_ao), 1.0, t.data(), static_cast<int>(n_out), slice, static_cast<int>(n_ao), 0.0,
                u.data(), static_cast<int>(n_ao));
    // out_P = u @ t : (n_out x n_ao) @ (n_ao x n_out) -> n_out x n_out, written into eri3_out's row p.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(n_out), static_cast<int>(n_out),
                static_cast<int>(n_ao), 1.0, u.data(), static_cast<int>(n_ao), t.data(), static_cast<int>(n_out), 0.0,
                eri3_out.data() + p * n_out * n_out, static_cast<int>(n_out));
  }
  return eri3_out;
}

Matrix<double> riThreeCenterTensor(const std::vector<BasisFunction>& basis,
                                    const std::vector<BasisFunction>& aux_basis,
                                    const Matrix<double>& metric_x) {
  const Matrix<double> raw = threeCenterIntegralsRaw(basis, aux_basis);
  const std::size_t n_aux_raw = metric_x.rows();
  const std::size_t n_aux_kept = metric_x.cols();
  const std::size_t n2 = raw.cols();
  if (raw.rows() != n_aux_raw) {
    throw std::runtime_error("riThreeCenterTensor: metric_x row count does not match the raw aux basis size");
  }

  Matrix<double> result(n_aux_kept, n2, 0.0);
  // result = metric_x^T * raw : (n_aux_kept x n_aux_raw) * (n_aux_raw x n2).
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(n_aux_kept), static_cast<int>(n2),
              static_cast<int>(n_aux_raw), 1.0, metric_x.data(), static_cast<int>(n_aux_kept), raw.data(),
              static_cast<int>(n2), 0.0, result.data(), static_cast<int>(n2));
  return result;
}

}  // namespace rerdmft
