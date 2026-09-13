#include "ElectronRepulsion.h"

#include <cstddef>

// libcint is a C library and its headers do not guard themselves with
// `extern "C"`, so that is done here to get correct (unmangled) linkage.
extern "C" {
#include <cint.h>
}

namespace rerdmft {

namespace {

// (pq|rs) for one specific quadruplet of individually-normalized cartesian
// AOs, each placed at its own real atomic center. Built as a minimal,
// independent 4-shell/4-atom libcint system per quadruplet (rather than a
// shared multi-shell system), for the same reason as Integrals.cpp's
// overlapPair: each cartesian AO carries its own individually rescaled
// contraction coefficients once normalizeCartesianBasis has run, so
// distinct cartesian components of one physical shell (e.g. d_xx vs d_xy)
// no longer share a single coefficient set that libcint's shell model
// requires.
double twoElectronQuadruplet(const BasisFunction& p, const BasisFunction& q,
                              const BasisFunction& r, const BasisFunction& s) {
  const int p_index = cartesianComponentIndex(p.l, p.cartesian);
  const int q_index = cartesianComponentIndex(q.l, q.cartesian);
  const int r_index = cartesianComponentIndex(r.l, r.cartesian);
  const int s_index = cartesianComponentIndex(s.l, s.cartesian);

  const BasisFunction* fns[4] = {&p, &q, &r, &s};

  FINT atm[4 * ATM_SLOTS] = {0};
  FINT bas[4 * BAS_SLOTS] = {0};
  FINT n_prim[4];
  for (int i = 0; i < 4; ++i) {
    n_prim[i] = static_cast<FINT>(fns[i]->exponents.size());
    atm[i * ATM_SLOTS + CHARGE_OF] = 0;
    atm[i * ATM_SLOTS + PTR_COORD] = PTR_ENV_START + 3 * i;
    bas[i * BAS_SLOTS + ATOM_OF] = i;
    bas[i * BAS_SLOTS + ANG_OF] = fns[i]->l;
    bas[i * BAS_SLOTS + NPRIM_OF] = n_prim[i];
    bas[i * BAS_SLOTS + NCTR_OF] = 1;
  }

  FINT env_offset = PTR_ENV_START + 3 * 4;
  for (int i = 0; i < 4; ++i) {
    bas[i * BAS_SLOTS + PTR_EXP] = env_offset;
    env_offset += n_prim[i];
    bas[i * BAS_SLOTS + PTR_COEFF] = env_offset;
    env_offset += n_prim[i];
  }

  std::vector<double> env(static_cast<std::size_t>(env_offset), 0.0);
  for (int i = 0; i < 4; ++i) {
    const std::size_t coord = static_cast<std::size_t>(atm[i * ATM_SLOTS + PTR_COORD]);
    env[coord + 0] = fns[i]->x;
    env[coord + 1] = fns[i]->y;
    env[coord + 2] = fns[i]->z;
    for (FINT k = 0; k < n_prim[i]; ++k) {
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_EXP] + k)] =
          fns[i]->exponents[static_cast<std::size_t>(k)];
      env[static_cast<std::size_t>(bas[i * BAS_SLOTS + PTR_COEFF] + k)] =
          fns[i]->coefficients[static_cast<std::size_t>(k)];
    }
  }

  FINT shls[4] = {0, 1, 2, 3};
  const FINT ni = CINTcgto_cart(0, bas);
  const FINT nj = CINTcgto_cart(1, bas);
  const FINT nk = CINTcgto_cart(2, bas);
  const FINT nl = CINTcgto_cart(3, bas);
  std::vector<double> buf(static_cast<std::size_t>(ni) * static_cast<std::size_t>(nj) *
                           static_cast<std::size_t>(nk) * static_cast<std::size_t>(nl));
  cint2e_cart(buf.data(), shls, atm, 4, bas, 4, env.data(), nullptr);

  // libcint's 2e integrals are returned as a (ni,nj,nk,nl) block in Fortran
  // order (the first index varies fastest), the same convention as every
  // other libcint buffer this project reads (see Integrals.cpp,
  // NablaIntegrals.cpp).
  const std::size_t entry =
      static_cast<std::size_t>(p_index) +
      static_cast<std::size_t>(ni) *
          (static_cast<std::size_t>(q_index) +
           static_cast<std::size_t>(nj) *
               (static_cast<std::size_t>(r_index) + static_cast<std::size_t>(nk) * static_cast<std::size_t>(s_index)));
  return buf[entry];
}

}  // namespace

Tensor4<double> twoElectronIntegrals(const std::vector<BasisFunction>& basis) {
  const std::size_t n = basis.size();
  Tensor4<double> result(n, n, n, n);

  // Each (p,q) with p<=q owns a disjoint set of 8 output positions (the
  // canonical-representative selection below never revisits a symmetry
  // class), and twoElectronQuadruplet builds an entirely local libcint
  // system per call (no shared, mutably-touched state, and `opt` is
  // nullptr) -- so different threads handling different p's never race,
  // either on reads or on Tensor4 writes. schedule(dynamic) balances the
  // shrinking q-range (and, within it, the libcint cost, which grows with
  // angular momentum) across threads.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = r; s < n; ++s) {
          const std::size_t pq = p * n + q;
          const std::size_t rs = r * n + s;
          if (rs < pq) continue;  // (pq|rs) == (rs|pq); do the (pq)<=(rs) half.

          const double value = twoElectronQuadruplet(basis[p], basis[q], basis[r], basis[s]);

          result(p, q, r, s) = value;
          result(q, p, r, s) = value;
          result(p, q, s, r) = value;
          result(q, p, s, r) = value;
          result(r, s, p, q) = value;
          result(s, r, p, q) = value;
          result(r, s, q, p) = value;
          result(s, r, q, p) = value;
        }
      }
    }
  }
  return result;
}

PackedTwoElectronTensor twoElectronIntegralsPacked(const std::vector<BasisFunction>& basis) {
  const std::size_t n = basis.size();
  PackedTwoElectronTensor result(n);

  // Same reasoning as twoElectronIntegrals: each canonical (p,q,r,s) with
  // p<=q, r<=s, (pq)<=(rs) maps to exactly one triangular slot (no two
  // canonical representatives ever collide), so parallelizing over (p,q)
  // is safe -- and here there is no redundant write-out to 8 positions at
  // all, since PackedTwoElectronTensor::set already resolves any of the
  // 8 equivalent argument orders to that same slot.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = r; s < n; ++s) {
          const std::size_t pq = p * n + q;
          const std::size_t rs = r * n + s;
          if (rs < pq) continue;  // (pq|rs) == (rs|pq); do the (pq)<=(rs) half.

          const double value = twoElectronQuadruplet(basis[p], basis[q], basis[r], basis[s]);
          result.set(p, q, r, s, value);
        }
      }
    }
  }
  return result;
}

Tensor4<double> twoElectronIntegralsCross(const std::vector<BasisFunction>& basis_pq,
                                           const std::vector<BasisFunction>& basis_rs) {
  const std::size_t n_pq = basis_pq.size();
  const std::size_t n_rs = basis_rs.size();
  Tensor4<double> result(n_pq, n_pq, n_rs, n_rs);

  // Same reasoning as twoElectronIntegrals: each (p,q) with p<=q owns a
  // disjoint set of output positions, so parallelizing over it is safe.
#pragma omp parallel for schedule(dynamic)
  for (std::size_t p = 0; p < n_pq; ++p) {
    for (std::size_t q = p; q < n_pq; ++q) {
      for (std::size_t r = 0; r < n_rs; ++r) {
        for (std::size_t s = r; s < n_rs; ++s) {
          const double value =
              twoElectronQuadruplet(basis_pq[p], basis_pq[q], basis_rs[r], basis_rs[s]);

          result(p, q, r, s) = value;
          result(q, p, r, s) = value;
          result(p, q, s, r) = value;
          result(q, p, s, r) = value;
        }
      }
    }
  }
  return result;
}

}  // namespace rerdmft
