#include "CcWorkspace.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace rerdmft {

using cc::Tensor;
using cc::maxAbs;
using cc::permute;
using cplx = CcScalar;
using Idx = std::vector<std::size_t>;

CcWorkspace buildCcWorkspace(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const Idx& occ_frozen, const Idx& occ_active,
                             const Idx& vir, std::ostream* log) {
  CcWorkspace ws;
  const std::size_t o = occ_active.size(), v = vir.size(), na = o + v;
  ws.o = o; ws.v = v; ws.na = na; ws.O = occ_active; ws.V = vir;
  ws.act = occ_active;
  ws.act.insert(ws.act.end(), vir.begin(), vir.end());
  const Idx& act = ws.act;
  const Idx &O = ws.O, &V = ws.V;
  // Fock matrix over the active space (the frozen occupied spinors contribute to the mean field).
  Tensor& F = ws.F;
  F = Tensor({na, na});
  for (std::size_t p = 0; p < na; ++p)
    for (std::size_t q = 0; q < na; ++q) F.d[p * na + q] = h_mo(act[p], act[q]);
  {
    Idx occ_sum = occ_frozen;
    occ_sum.insert(occ_sum.end(), occ_active.begin(), occ_active.end());
    std::vector<cplx> jb(na * na), kb(na * na);
    for (std::size_t i : occ_sum) {
      ints.physicsBlock(act, {i}, act, {i}, jb.data());  // <p i | q i>
      ints.physicsBlock(act, {i}, {i}, act, kb.data());  // <p i | i q>
      for (std::size_t k = 0; k < na * na; ++k) F.d[k] += jb[k] - kb[k];
    }
  }
  for (std::size_t p = 0; p < na; ++p)
    for (std::size_t q = 0; q < na; ++q) ws.max_fock_asymmetry = std::max(ws.max_fock_asymmetry, std::abs(F.d[p * na + q] - std::conj(F.d[q * na + p])));
  const auto block = [&](std::size_t r0, std::size_t nr, std::size_t c0, std::size_t nc) {
    Tensor t({nr, nc});
    for (std::size_t i = 0; i < nr; ++i)
      for (std::size_t j = 0; j < nc; ++j) t.d[i * nc + j] = F.d[(r0 + i) * na + c0 + j];
    return t;
  };
  ws.f_oo = block(0, o, 0, o); ws.f_ov = block(0, o, o, v); ws.f_vo = block(o, v, 0, o); ws.f_vv = block(o, v, o, v);
  ws.max_fock_ov = maxAbs(ws.f_ov);
  if (log) *log << "  max |F - F^dagger| = " << std::scientific << std::setprecision(2) << ws.max_fock_asymmetry << ",  max |F_ov| = " << ws.max_fock_ov << "\n";

  // Orbital energies = real diagonal of the Fock matrix.
  ws.e_o.assign(o, 0.0); ws.e_v.assign(v, 0.0);
  for (std::size_t i = 0; i < o; ++i) ws.e_o[i] = F.d[i * na + i].real();
  for (std::size_t a = 0; a < v; ++a) ws.e_v[a] = F.d[(o + a) * na + o + a].real();

  // Antisymmetrized integral blocks that fit in memory (everything but ovvv and vvvv).
  const auto get = [&](const Idx& P, const Idx& Q, const Idx& R, const Idx& S) {
    Tensor t({P.size(), Q.size(), R.size(), S.size()});
    ints.physicsBlock(P, Q, R, S, t.d.data());
    return t;
  };
  {
    Tensor p = get(O, O, O, O);
    ws.g_oooo = p - permute(p, {0, 1, 3, 2});
    p = get(O, O, O, V);
    ws.g_ooov = p - permute(p, {1, 0, 2, 3});
    p = get(O, O, V, V);
    ws.g_oovv = p - permute(p, {0, 1, 3, 2});
    const Tensor povov = get(O, V, O, V), povvo = get(O, V, V, O);
    ws.g_ovov = povov - permute(povvo, {0, 1, 3, 2});
    ws.g_ovvo = povvo - permute(povov, {0, 1, 3, 2});
  }
  return ws;
}

}  // namespace rerdmft
