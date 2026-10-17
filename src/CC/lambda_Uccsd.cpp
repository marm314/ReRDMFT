#include "lambda_Uccsd.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include "CcTensor.h"
#include "CcWorkspace.h"
#include "DIIS.h"
#include "Progress.h"

namespace rerdmft {

namespace {

using cc::Tensor;
using cc::axpy;
using cc::conjugated;
using cc::einsum;
using cc::einsumN;
using cc::permute;
using cplx = CcScalar;
using Idx = std::vector<std::size_t>;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------------------------------------------------
// Equation data. Generated from the lists that were verified against PySCF (doc/uccsd.tex, Section 9): every term is
// coefficient * einsum(spec, operands); an operand name ending in '*' is complex conjugated.
// ---------------------------------------------------------------------------------------------------------------------
struct Term {
  double c;
  const char* spec;
  std::vector<const char*> ops;
};
struct EqDef {
  const char* name;
  std::vector<Term> terms;
};

static const std::vector<EqDef> kTIntermediates = {
  {"tau", {{1.0, "ijab->ijab", {"t2"}},
      {1.0, "ia,jb->ijab", {"t1", "t1"}},
      {-1.0, "ib,ja->ijab", {"t1", "t1"}}}},
  {"v1", {{1.0, "ba->ba", {"fvv"}},
      {-1.0, "ja,jb->ba", {"fov", "t1"}},
      {-1.0, "jbac,jc->ba", {"ovvv", "t1"}},
      {0.5, "jkca,jkbc->ba", {"oovv", "tau"}}}},
  {"v2", {{1.0, "ij->ij", {"foo"}},
      {1.0, "ib,jb->ij", {"fov", "t1"}},
      {-1.0, "kijb,kb->ij", {"ooov", "t1"}},
      {0.5, "ikbc,jkbc->ij", {"oovv", "tau"}}}},
  {"v3", {{1.0, "ijcd,klcd->ijkl", {"oovv", "tau"}}}},
  {"v4", {{1.0, "ljdb,klcd->jcbk", {"oovv", "t2"}},
      {1.0, "jcbk->jcbk", {"ovvo"}}}},
  {"Fov", {{1.0, "kc->kc", {"fov"}},
      {-1.0, "kldc,ld->kc", {"oovv", "t1"}}}},
  {"v5", {{1.0, "bj->bj", {"fvo"}},
      {1.0, "kc,jkbc->bj", {"fov", "t2"}},
      {1.0, "kc,kb,jc->bj", {"Fov", "t1", "t1"}},
      {-0.5, "kljc,klbc->bj", {"ooov", "t2"}},
      {0.5, "kbdc,jkcd->bj", {"ovvv", "t2"}}}},
  {"w3", {{1.0, "ck->ck", {"v5"}},
      {1.0, "jcbk,jb->ck", {"v4", "t1"}},
      {1.0, "cb,kb->ck", {"v1", "t1"}},
      {-1.0, "jk,jc->ck", {"v2", "t1"}}}},
  {"woooo", {{0.5, "ijkl->ijkl", {"oooo"}},
      {0.25, "ijkl->ijkl", {"v3"}},
      {1.0, "ijkc,lc->ijkl", {"ooov", "t1"}}}},
  {"wovvo", {{1.0, "jcbk->jcbk", {"v4"}},
      {-1.0, "ljdb,lc,kd->jcbk", {"oovv", "t1", "t1"}},
      {-1.0, "ljkb,lc->jcbk", {"ooov", "t1"}},
      {1.0, "jcbd,kd->jcbk", {"ovvv", "t1"}}}},
  {"wovoo", {{0.25, "icdb,jkdb->icjk", {"ovvv", "tau"}},
      {0.5, "jkic->icjk", {"ooov*"}},
      {1.0, "icbk,jb->icjk", {"v4", "t1"}},
      {-1.0, "lijb,klcb->icjk", {"ooov", "t2"}}}},
  {"wvvvo", {{1.0, "jcak,jb->bcak", {"v4", "t1"}},
      {0.25, "jlka,jlbc->bcak", {"ooov", "tau"}},
      {-0.5, "kacb->bcak", {"ovvv*"}},
      {1.0, "jbad,kjcd->bcak", {"ovvv", "t2"}}}},
};

static const std::vector<EqDef> kLambdaUpdate = {
  {"mba", {{0.5, "klca,klcb->ba", {"l2", "t2"}}}},
  {"mij", {{0.5, "kicd,kjcd->ij", {"l2", "t2"}}}},
  {"n1vv", {{1.0, "ba->ba", {"mba"}},
      {1.0, "ka,kb->ba", {"l1", "t1"}}}},
  {"n1oo", {{1.0, "ik->ik", {"mij"}},
      {1.0, "ic,kc->ik", {"l1", "t1"}}}},
  {"lt", {{1.0, "ijcd,klcd->ijkl", {"l2", "tau"}}}},
  {"lt1", {{1.0, "ijcd,kd->ijck", {"l2", "t1"}}}},
  {"m3", {{1.0, "klab,ijkl->ijab", {"l2", "woooo"}},
      {0.25, "klab,ijkl->ijab", {"oovv", "lt"}},
      {-1.0, "kcba,ijck->ijab", {"ovvv", "lt1"}}}},
  {"qjb", {{1.0, "jb->jb", {"t1"}},
      {1.0, "kc,kjcb->jb", {"l1", "t2"}},
      {-1.0, "bd,jd->jb", {"n1vv", "t1"}},
      {-1.0, "lj,lb->jb", {"mij", "t1"}}}},
};

static const std::vector<Term> kT1 = {{1.0, "ia,jb->ijab", {"l1", "Fov"}},
      {1.0, "kica,jcbk->ijab", {"l2", "wovvo"}}};

static const std::vector<Term> kT2 = {{1.0, "ka,ijkb->ijab", {"l1", "ooov"}},
      {1.0, "ijca,cb->ijab", {"l2", "v1m"}},
      {1.0, "ca,ijcb->ijab", {"n1vv", "oovv"}}};

static const std::vector<Term> kT3 = {{1.0, "jc,icab->ijab", {"l1", "ovvv"}},
      {1.0, "kiab,jk->ijab", {"l2", "v2m"}},
      {-1.0, "ik,kjab->ijab", {"n1oo", "oovv"}}};

static const std::vector<Term> kL2Direct = {{1.0, "ijab->ijab", {"oovv"}},
      {1.0, "ijab->ijab", {"m3"}}};

static const std::vector<Term> kL1 = {{1.0, "ia->ia", {"fov"}},
      {1.0, "jb,ibaj->ia", {"l1", "ovvo"}},
      {1.0, "ib,ba->ia", {"l1", "v1m"}},
      {-1.0, "ja,ij->ia", {"l1", "v2m"}},
      {-1.0, "kjca,icjk->ia", {"l2", "wovoo"}},
      {-1.0, "ikbc,bcak->ia", {"l2", "wvvvo"}},
      {1.0, "ijab,jb->ia", {"m3", "t1"}},
      {1.0, "jiba,bj->ia", {"l2", "w3"}},
      {1.0, "jiba,jb->ia", {"oovv", "qjb"}},
      {1.0, "icab,bc->ia", {"ovvv", "n1vv"}},
      {-1.0, "jika,kj->ia", {"ooov", "n1oo"}},
      {-1.0, "ik,ka->ia", {"mij", "Fov"}},
      {-1.0, "ca,ic->ia", {"mba", "Fov"}}};

static const std::vector<EqDef> kGamma1 = {
  {"doo", {{-1.0, "ie,je->ij", {"l1", "t1"}},
      {-0.5, "imef,jmef->ij", {"l2", "t2"}}}},
  {"dvv", {{1.0, "ma,mb->ab", {"t1", "l1"}},
      {0.5, "mnea,mneb->ab", {"t2", "l2"}}}},
  {"xt1", {{0.5, "mnef,inef->mi", {"l2", "t2"}}}},
  {"xt2", {{0.5, "mnfa,mnfe->ae", {"t2", "l2"}},
      {1.0, "ma,me->ae", {"t1", "l1"}}}},
  {"dvo", {{1.0, "imae,me->ai", {"t2", "l1"}},
      {-1.0, "mi,ma->ai", {"xt1", "t1"}},
      {-1.0, "ie,ae->ai", {"t1", "xt2"}},
      {1.0, "ia->ai", {"t1"}}}},
};

static const std::vector<EqDef> kGamma2A = {
  {"taub", {{1.0, "ijab->ijab", {"t2"}},
      {2.0, "ia,jb->ijab", {"t1", "t1"}}}},
  {"miajb", {{1.0, "ikac,kjcb->iajb", {"l2", "t2"}}}},
  {"g1", {{1.0, "kc,kica->ia", {"l1", "t2"}}}},
  {"g2", {{1.0, "kc,kb->cb", {"l1", "t1"}}}},
  {"g3", {{1.0, "kc,jc->kj", {"l1", "t1"}}}},
  {"g4", {{1.0, "ldjd->lj", {"miajb"}}}},
  {"g5", {{1.0, "ldlb->db", {"miajb"}}}},
  {"g6", {{0.0625, "klcd,ijcd->ijkl", {"l2", "taub"}}}},
  {"g7", {{0.25, "icjc->ij", {"miajb"}}}},
  {"g8", {{0.25, "kakb->ab", {"miajb"}}}},
};

static const std::vector<Term> kGoovv = {{0.25, "ijab->ijab", {"l2*"}},
      {0.25, "ijab->ijab", {"taub"}},
      {1.0, "ia,jb->ijab", {"g1", "t1"}},
      {0.5, "cb,ijca->ijab", {"g2", "t2"}},
      {0.5, "kiab,kj->ijab", {"taub", "g3"}},
      {-0.25, "lj,liba->ijab", {"g4", "taub"}},
      {-0.25, "db,jida->ijab", {"g5", "taub"}},
      {-0.5, "ldia,ljbd->ijab", {"miajb", "taub"}},
      {1.0, "ijkl,klab->ijab", {"g6", "taub"}}};

static const std::vector<Term> kGoooo = {{0.125, "klab,ijab->klij", {"l2", "taub"}}};

static const std::vector<Term> kGooov1 = {{-0.25, "jkba,ib->jkia", {"taub", "l1"}},
      {1.0, "iljk,la->jkia", {"goooo", "t1"}},
      {-1.0, "ij,ka->jkia", {"g7", "t1"}},
      {0.5, "icja,kc->jkia", {"miajb", "t1"}}};

static const std::vector<Term> kGooov2 = {{0.25, "jkab,ib->jkia", {"l2", "t1"}}};

static const std::vector<Term> kGovvo = {{1.0, "ia,jb->ibaj", {"l1", "t1"}},
      {1.0, "iajb->ibaj", {"miajb"}},
      {-1.0, "ikac,jc,kb->ibaj", {"l2", "t1", "t1"}}};

static const std::vector<Term> kGovvv1 = {{0.25, "ja,ijbc->iabc", {"l1", "taub"}},
      {0.125, "jkbc,jkad,id->iabc", {"taub", "l2", "t1"}},
      {1.0, "ac,ib->iabc", {"g8", "t1"}},
      {0.5, "kaib,kc->iabc", {"miajb", "t1"}}};

static const std::vector<Term> kGovvv2 = {{0.25, "ijbc,ja->iabc", {"l2", "t1"}}};

struct DBlockDef { const char* name; const char* src; std::vector<std::pair<int, std::array<int,4>>> terms; int post; const char* letters; };
static const std::vector<DBlockDef> kDBlocks = {
  {"dovov", "goovv", {{1, {0,2,1,3}}, {-1, {0,3,1,2}}}, 1, "iajb"},
  {"dvvvv", "gvvvv", {{1, {0,2,1,3}}, {-1, {0,3,1,2}}}, 2, "abcd"},
  {"doooo", "goooo", {{1, {0,2,1,3}}, {-1, {0,3,1,2}}}, 2, "ijkl"},
  {"dovvv", "govvv", {{1, {0,2,1,3}}, {-1, {0,3,1,2}}}, 0, "iabc"},
  {"dooov", "gooov", {{1, {0,2,1,3}}, {-1, {1,2,0,3}}}, 0, "ijka"},
  {"dovvo", "govvo", {{1, {0,2,1,3}}}, 3, "iabj"},
};

struct AsmRow { const char* dst; const char* src; bool has_axes; std::array<int,4> axes; bool conj; int sign; };
static const std::vector<AsmRow> kAssembly = {
  {"ovov", "dovov", false, {0,1,2,3}, false, 1},
  {"vovo", "dovov", true, {1,0,3,2}, true, 1},
  {"oovv", "dovvo", true, {0,3,2,1}, false, -1},
  {"vvoo", "dovvo", true, {2,1,0,3}, false, -1},
  {"ovvo", "dovvo", false, {0,1,2,3}, false, 1},
  {"voov", "dovvo", true, {1,0,3,2}, true, 1},
  {"vvvv", "dvvvv", false, {0,1,2,3}, false, 1},
  {"oooo", "doooo", false, {0,1,2,3}, false, 1},
  {"ovvv", "dovvv", false, {0,1,2,3}, false, 1},
  {"vvov", "dovvv", true, {2,3,0,1}, false, 1},
  {"vvvo", "dovvv", true, {3,2,1,0}, true, 1},
  {"vovv", "dovvv", true, {1,0,3,2}, true, 1},
  {"ooov", "dooov", false, {0,1,2,3}, false, 1},
  {"ovoo", "dooov", true, {2,3,0,1}, false, 1},
  {"oovo", "dooov", true, {1,0,3,2}, true, 1},
  {"vooo", "dooov", true, {3,2,1,0}, true, 1},
};

using Env = std::map<std::string, Tensor>;

// Sum of the terms of one equation (all terms share the output letters of their specs).
Tensor evalTerms(const std::vector<Term>& terms, const Env& env) {
  Tensor acc;
  bool first = true;
  for (const Term& t : terms) {
    std::vector<Tensor> conj_store;
    conj_store.reserve(t.ops.size());
    std::vector<const Tensor*> ptrs;
    for (const char* nm : t.ops) {
      std::string name(nm);
      const bool conj = !name.empty() && name.back() == '*';
      if (conj) name.pop_back();
      const auto it = env.find(name);
      if (it == env.end()) throw std::logic_error("UCCSD lambda: unknown operand " + name);
      if (conj) {
        conj_store.push_back(conjugated(it->second));
        ptrs.push_back(&conj_store.back());
      } else {
        ptrs.push_back(&it->second);
      }
    }
    Tensor r = einsumN(t.spec, ptrs, cplx(t.c));
    if (first) {
      acc = std::move(r);
      first = false;
    } else {
      axpy(acc, r, 1.0);
    }
  }
  return acc;
}

Tensor wrap(const std::vector<cplx>& flat, std::vector<std::size_t> dims) {
  Tensor t(std::move(dims));
  if (t.size() != flat.size()) throw std::logic_error("UCCSD lambda: amplitude size mismatch");
  t.d = flat;
  return t;
}

Tensor scaledCopy(const Tensor& a, cplx s) {
  Tensor out = a;
  for (auto& x : out.d) x *= s;
  return out;
}

// Little-endian stream of fixed-size records.
class RecordWriter {
 public:
  explicit RecordWriter(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {
    if (f_ == nullptr) throw std::runtime_error("CC_RDM: cannot open " + path + " for writing");
    std::setvbuf(f_, nullptr, _IOFBF, std::size_t(1) << 22);
  }
  ~RecordWriter() {
    if (f_ != nullptr) std::fclose(f_);
  }
  RecordWriter(const RecordWriter&) = delete;
  RecordWriter& operator=(const RecordWriter&) = delete;
  void put(const std::int32_t* idx, int n, double re, double im) {
    std::fwrite(idx, sizeof(std::int32_t), static_cast<std::size_t>(n), f_);
    std::fwrite(&re, sizeof(double), 1, f_);
    std::fwrite(&im, sizeof(double), 1, f_);
  }
  void close() {
    if (f_ != nullptr) std::fclose(f_);
    f_ = nullptr;
  }

 private:
  std::FILE* f_;
};

}  // namespace

std::vector<std::size_t> fileOrderGlobals(const CcOrbitalSpaces& spaces) {
  std::size_t n_out = 0;
  for (std::size_t fi : spaces.file_index) n_out = std::max(n_out, fi);
  std::vector<std::size_t> glob(n_out, std::size_t(-1));
  for (std::size_t g = 0; g < spaces.file_index.size(); ++g)
    if (spaces.file_index[g] > 0) glob[spaces.file_index[g] - 1] = g;
  return glob;
}

CcRdmResult runUccsdRdm(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const CcOrbitalSpaces& spaces,
                        const UccsdResult& amps, const CcRdmSettings& settings, std::ostream& log) {
  const auto t_start = Clock::now();
  const auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count(); };
  CcRdmResult result;
  CcWorkspace ws = buildCcWorkspace(h_mo, ints, spaces.occ_frozen, spaces.occ_active, spaces.vir, nullptr);
  const std::size_t o = ws.o, v = ws.v, na = ws.na;
  const Idx &V = ws.V;
  log << "\n  --- Lambda equations and density matrices (CC_RDM) ---\n";
  {
    const double gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
    log << "  ov^3 arrays held (g_ovvv, W_vvvo, ...): " << std::fixed << std::setprecision(3) << 16.0 * o * v * v * v * gib << " GiB each\n";
  }

  Env env;
  env["t1"] = wrap(amps.t1, {o, v});
  env["t2"] = wrap(amps.t2, {o, o, v, v});
  env["fov"] = ws.f_ov;
  env["foo"] = ws.f_oo;
  env["fvv"] = ws.f_vv;
  env["fvo"] = ws.f_vo;
  env["oooo"] = ws.g_oooo;
  env["ooov"] = ws.g_ooov;
  env["oovv"] = ws.g_oovv;
  env["ovvo"] = ws.g_ovvo;
  env["ovov"] = ws.g_ovov;

  // <ma||ef> (the one ov^3 integral block the Lambda/RDM stage keeps), from the slabs.
  {
    Tensor g({o, v, v, v});
    ints.prepareSlabs(V);
    Tensor raw({v, v, v});
    for (std::size_t m = 0; m < o; ++m) {
      ints.slab(ws.O[m], V, raw.d.data());
      cplx* dst = &g.d[m * v * v * v];
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t e = 0; e < v; ++e)
          for (std::size_t f = 0; f < v; ++f) dst[(a * v + e) * v + f] = raw.d[(a * v + e) * v + f] - raw.d[(a * v + f) * v + e];
    }
    env["ovvv"] = std::move(g);
  }

  // ---- T-dependent intermediates ----
  for (const EqDef& def : kTIntermediates) env[def.name] = evalTerms(def.terms, env);
  {
    Tensor v1m = env["v1"], v2m = env["v2"];
    for (std::size_t a = 0; a < v; ++a) v1m.d[a * v + a] -= ws.e_v[a];
    for (std::size_t i = 0; i < o; ++i) v2m.d[i * o + i] -= ws.e_o[i];
    env["v1m"] = std::move(v1m);
    env["v2m"] = std::move(v2m);
  }

  Tensor D1({o, v}), D2({o, o, v, v});
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) D1.d[i * v + a] = ws.e_o[i] - ws.e_v[a];
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) D2.d[((i * o + j) * v + a) * v + b] = ws.e_o[i] + ws.e_o[j] - ws.e_v[a] - ws.e_v[b];

  // ---- Lambda iterations ----
  const auto numerators = [&](const Tensor& l1, const Tensor& l2, Tensor& n1, Tensor& n2) {
    env["l1"] = l1;
    env["l2"] = l2;
    for (const EqDef& def : kLambdaUpdate) env[def.name] = evalTerms(def.terms, env);
    {
      // the vvvv ladder of M: 1/2 sum_cd lambda_ij^cd <cd||ab> = sum_cd lambda_ij^cd <cd|ab>, one slab <c d|a b> per c
      Tensor& m3 = env["m3"];
      Tensor P({v, v, v});
      for (std::size_t c = 0; c < v; ++c) {
        ints.slab(V[c], V, P.d.data());
        const Tensor lc = cc::takeAxis(l2, 2, c);  // [i][j][d]
        axpy(m3, einsum("ijd,dab->ijab", lc, P), 1.0);
      }
    }
    n2 = evalTerms(kL2Direct, env);
    axpy(n2, cc::antisymAB(cc::antisymIJ(evalTerms(kT1, env))), 1.0);
    axpy(n2, cc::antisymAB(evalTerms(kT2, env)), -1.0);
    axpy(n2, cc::antisymIJ(evalTerms(kT3, env)), 1.0);
    n1 = evalTerms(kL1, env);
  };

  std::vector<std::pair<std::size_t, std::size_t>> pij, pab;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = i + 1; j < o; ++j) pij.emplace_back(i, j);
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t b = a + 1; b < v; ++b) pab.emplace_back(a, b);
  const std::size_t n_vec = o * v + pij.size() * pab.size();
  const auto pack = [&](const Tensor& a1, const Tensor& a2) {
    Matrix<cplx> vec(1, n_vec);
    std::size_t k = 0;
    for (std::size_t x = 0; x < o * v; ++x) vec(0, k++) = a1.d[x];
    for (const auto& ij : pij)
      for (const auto& ab : pab) vec(0, k++) = a2.d[((ij.first * o + ij.second) * v + ab.first) * v + ab.second];
    return vec;
  };
  const auto unpack = [&](const Matrix<cplx>& vec, Tensor& a1, Tensor& a2) {
    std::size_t k = 0;
    for (std::size_t x = 0; x < o * v; ++x) a1.d[x] = vec(0, k++);
    for (const auto& ij : pij)
      for (const auto& ab : pab) {
        const cplx x = vec(0, k++);
        const std::size_t i = ij.first, j = ij.second, a = ab.first, b = ab.second;
        a2.d[((i * o + j) * v + a) * v + b] = x;
        a2.d[((j * o + i) * v + a) * v + b] = -x;
        a2.d[((i * o + j) * v + b) * v + a] = -x;
        a2.d[((j * o + i) * v + b) * v + a] = x;
      }
  };

  Tensor l1 = env["t1"], l2 = env["t2"];
  Diis<cplx> diis(static_cast<std::size_t>(std::max(settings.diis_size, 0)));
  const cplx shift(settings.level_shift, 0.0);
  log << "  Lambda iterations:\n    " << std::setw(4) << "iter" << std::setw(14) << "|dLambda|" << std::setw(12) << "time (s)" << "\n";
  for (int iter = 1; iter <= settings.max_iterations; ++iter) {
    Tensor n1, n2;
    numerators(l1, l2, n1, n2);
    Tensor l1n({o, v}), l2n({o, o, v, v});
    for (std::size_t k = 0; k < l1n.size(); ++k) l1n.d[k] = (n1.d[k] - shift * l1.d[k]) / (D1.d[k] - shift);
    for (std::size_t k = 0; k < l2n.size(); ++k) l2n.d[k] = (n2.d[k] - shift * l2.d[k]) / (D2.d[k] - shift);
    const Matrix<cplx> vec_new = pack(l1n, l2n), vec_old = pack(l1, l2);
    Matrix<cplx> err(1, n_vec);
    double dt2 = 0.0;
    for (std::size_t k = 0; k < n_vec; ++k) {
      err(0, k) = vec_new(0, k) - vec_old(0, k);
      dt2 += std::norm(err(0, k));
    }
    const double dt = std::sqrt(dt2);
    if (settings.diis_size >= 2) unpack(diis.extrapolate(err, vec_new), l1, l2);
    else unpack(vec_new, l1, l2);
    log << "    " << kTableRow << std::setw(4) << iter << std::scientific << std::setprecision(2) << std::setw(14) << dt << std::fixed
        << std::setprecision(1) << std::setw(12) << elapsed() << "\n";
    {
      std::ostringstream line;
      line << "(UCCSD Lambda) iteration " << iter << ": |dLambda| = " << std::scientific << std::setprecision(2) << dt;
      progress(line.str());
    }
    result.lambda_iterations = iter;
    if (dt < 100.0 * settings.convergence) {
      result.lambda_converged = true;
      break;
    }
  }
  if (!result.lambda_converged) log << "  WARNING: the Lambda equations did not converge in " << settings.max_iterations << " iterations\n";
  env["l1"] = l1;
  env["l2"] = l2;

  // =====================================================================================================================
  // Density matrices
  // =====================================================================================================================
  // ---- file index space: Kramers-pair order, frozen core included ----
  std::size_t n_out = 0;
  for (std::size_t g = 0; g < spaces.file_index.size(); ++g) n_out = std::max(n_out, spaces.file_index[g]);
  std::vector<std::size_t> glob(n_out, std::size_t(-1));
  std::vector<int> cls(n_out, -1);          // 0 virtual, 1 active occupied, 2 frozen occupied
  std::vector<std::size_t> loc(n_out, 0);   // index inside the virtual / active-occupied list
  const auto place = [&](const Idx& list, int c) {
    for (std::size_t k = 0; k < list.size(); ++k) {
      const std::size_t fi = spaces.file_index.at(list[k]);
      if (fi == 0) throw std::runtime_error("CC_RDM: a correlated spinor has no output index");
      glob[fi - 1] = list[k];
      cls[fi - 1] = c;
      loc[fi - 1] = k;
    }
  };
  place(spaces.occ_frozen, 2);
  place(ws.O, 1);
  place(spaces.vir, 0);
  for (std::size_t k = 0; k < n_out; ++k)
    if (cls[k] < 0) throw std::runtime_error("CC_RDM: output index space has a hole");
  const std::size_t n_occ_all = spaces.occ_frozen.size() + o;
  const auto isOcc = [&](std::size_t k) { return cls[k] != 0; };
  const auto actIndex = [&](std::size_t k) { return cls[k] == 1 ? loc[k] : o + loc[k]; };  // position in the active space

  // ---- 1-RDM ----
  for (const EqDef& def : kGamma1) env[def.name] = evalTerms(def.terms, env);
  std::vector<cplx> dcorr(na * na);  // PySCF's dm1 without the mean-field occupations: 1/2 (D + D^dagger), D = [[doo, l1], [dvo, dvv]]
  {
    std::vector<cplx> Dm(na * na);
    const Tensor &doo = env["doo"], &dvo = env["dvo"], &dvv = env["dvv"];
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j) Dm[i * na + j] = doo.d[i * o + j];
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) {
        Dm[i * na + o + a] = l1.d[i * v + a];
        Dm[(o + a) * na + i] = dvo.d[a * o + i];
      }
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b) Dm[(o + a) * na + o + b] = dvv.d[a * v + b];
    for (std::size_t p = 0; p < na; ++p)
      for (std::size_t q = 0; q < na; ++q) dcorr[p * na + q] = 0.5 * (Dm[p * na + q] + std::conj(Dm[q * na + p]));
  }
  const auto dFile = [&](std::size_t a, std::size_t b) -> cplx {  // correlation part of dm1[a][b], zero when a frozen spinor is involved
    if (cls[a] == 2 || cls[b] == 2) return cplx{};
    return dcorr[actIndex(a) * na + actIndex(b)];
  };
  const auto gammaBar = [&](std::size_t p, std::size_t q) -> cplx {  // <a_p^+ a_q>, Hermitian part
    cplx val = dFile(q, p);
    if (p == q && isOcc(p)) val += 1.0;
    return val;
  };

  const double thr = settings.threshold;
  cplx e1{};
  {
    RecordWriter w1(settings.dm1_path);
    for (std::size_t p = 0; p < n_out; ++p)
      for (std::size_t q = 0; q < n_out; ++q) {
        const cplx val = gammaBar(p, q);
        if (p == q) result.trace_dm1 += val.real();
        if (std::abs(val) < thr) continue;
        const std::int32_t idx[2] = {static_cast<std::int32_t>(p + 1), static_cast<std::int32_t>(q + 1)};
        w1.put(idx, 2, val.real(), val.imag());
        ++result.n_dm1;
        e1 += h_mo(glob[p], glob[q]) * val;
      }
    const std::int32_t zero[2] = {0, 0};
    w1.put(zero, 2, 0.0, 0.0);
    w1.close();
  }
  result.e_onebody = e1.real();
  result.n_electrons = static_cast<double>(n_occ_all);
  log << "  1-RDM written to " << settings.dm1_path << " (" << result.n_dm1 << " elements), trace = " << std::fixed << std::setprecision(10)
      << result.trace_dm1 << "\n";

  // ---- 2-RDM blocks ----
  for (const EqDef& def : kGamma2A) env[def.name] = evalTerms(def.terms, env);
  env["goovv"] = conjugated(evalTerms(kGoovv, env));
  env["goooo"] = evalTerms(kGoooo, env);
  {
    Tensor x = conjugated(evalTerms(kGooov1, env));
    axpy(x, evalTerms(kGooov2, env), 1.0);
    env["gooov"] = std::move(x);
  }
  env["govvo"] = evalTerms(kGovvo, env);
  {
    Tensor x = conjugated(evalTerms(kGovvv1, env));
    axpy(x, evalTerms(kGovvv2, env), 1.0);
    env["govvv"] = std::move(x);
  }
  for (const char* big : {"wvvvo", "wovoo", "wovvo", "woooo", "ovvv"}) env.erase(big);

  std::map<std::string, Tensor> dblock;
  for (const DBlockDef& def : kDBlocks) {
    if (std::string(def.name) == "dvvvv") continue;  // streamed below
    Tensor x;
    bool first = true;
    for (const auto& term : def.terms) {
      std::vector<int> axes(term.second.begin(), term.second.end());
      Tensor pt = permute(env.at(def.src), axes);
      if (first) {
        x = scaledCopy(pt, cplx(term.first));
        first = false;
      } else {
        axpy(x, pt, cplx(term.first));
      }
    }
    if (def.post == 1) {
      Tensor t = permute(x, {2, 3, 0, 1});
      axpy(x, t, 1.0);
      x = scaledCopy(x, 0.5);
    } else if (def.post == 2) {
      axpy(x, conjugated(permute(x, {1, 0, 3, 2})), 1.0);
    } else if (def.post == 3) {
      axpy(x, conjugated(permute(x, {3, 2, 1, 0})), 1.0);
      x = scaledCopy(x, 0.5);
    }
    dblock[def.name] = std::move(x);
  }
  for (const char* g : {"goovv", "gooov", "goooo", "govvo", "govvv"}) env.erase(g);

  // R-block lookup by the (virtual=0 / occupied=1) character of its four indices
  const auto typeCode = [](const char* s) {
    int code = 0;
    for (int k = 0; k < 4; ++k) code = code * 2 + (s[k] == 'o' ? 1 : 0);
    return code;
  };
  std::array<int, 16> row_for;
  row_for.fill(-1);
  for (std::size_t k = 0; k < kAssembly.size(); ++k) row_for[typeCode(kAssembly[k].dst)] = static_cast<int>(k);
  std::vector<const Tensor*> row_block(kAssembly.size(), nullptr);
  for (std::size_t k = 0; k < kAssembly.size(); ++k) {
    const auto it = dblock.find(kAssembly[k].src);
    if (it != dblock.end()) row_block[k] = &it->second;
  }

  // Gamma_{pqrs} = R_{rpsq}; R = d-block assembly + the mean-field/one-body parts (doc/uccsd.tex, Section 9.4).
  const auto gammaElement = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> cplx {
    const std::size_t x = r, y = p, z = s, w = q;
    cplx val{};
    if (cls[x] != 2 && cls[y] != 2 && cls[z] != 2 && cls[w] != 2) {
      const int code = (cls[x] << 3) | (cls[y] << 2) | (cls[z] << 1) | cls[w];
      const int row = row_for[code];
      if (row >= 0 && row_block[row] != nullptr) {
        const AsmRow& ar = kAssembly[row];
        const std::size_t out_idx[4] = {loc[x], loc[y], loc[z], loc[w]};
        std::size_t in_idx[4];
        for (int k = 0; k < 4; ++k) in_idx[ar.has_axes ? ar.axes[k] : k] = out_idx[k];
        const Tensor& t = *row_block[row];
        cplx dv = t.d[((in_idx[0] * t.dims[1] + in_idx[1]) * t.dims[2] + in_idx[2]) * t.dims[3] + in_idx[3]];
        if (ar.conj) dv = std::conj(dv);
        val = static_cast<double>(ar.sign) * dv;
      }
    }
    const bool xy = (x == y) && isOcc(x), zw = (z == w) && isOcc(z), yz = (y == z) && isOcc(y), xw = (x == w) && isOcc(x);
    if (xy) val += dFile(z, w);
    if (zw) val += dFile(x, y);
    if (yz) val -= dFile(x, w);
    if (xw) val -= dFile(z, y);
    if (xy && zw) val += 1.0;
    if (xw && yz) val -= 1.0;
    return val;
  };

  // ---- sizes for the integral-slab energy check ----
  Idx glob_list(glob.begin(), glob.end());
  ints.prepareSlabs(glob_list);
  std::vector<cplx> islab(n_out * n_out * n_out);
  cplx e2{};
  cplx e2cat[3][3] = {};
  double tr2 = 0.0;

  // ---- all-virtual block: Gamma_{pqrs} = conj d'^{vvvv}_{p r q s}, d' streamed one first index at a time ----
  const Tensor& taub = env.at("taub");
  Tensor dslab({v, v, v});
  const auto buildDvvvvSlab = [&](std::size_t a) {
    const Tensor tau_a = cc::takeAxis(taub, 2, a);  // [i][j][c]
    const Tensor M1 = einsum("ijc,ijbd->cbd", tau_a, env.at("l2"), 0.125);
    const Tensor M2 = einsum("ijbd,ijc->bdc", taub, cc::takeAxis(env.at("l2"), 2, a), 0.125);
    const Tensor M3 = einsum("ijbd,ijc->bdc", taub, cc::takeAxis(env.at("l2"), 3, a), 0.125);
    for (std::size_t b = 0; b < v; ++b)
      for (std::size_t c = 0; c < v; ++c)
        for (std::size_t d = 0; d < v; ++d)
          dslab.d[(b * v + c) * v + d] = M1.d[(c * v + b) * v + d] - M1.d[(c * v + d) * v + b] +
                                         std::conj(M2.d[(b * v + d) * v + c] - M3.d[(b * v + d) * v + c]);
  };

  RecordWriter w2(settings.dm2_path);
  struct Rec {
    std::int32_t idx[4];
    double re, im;
  };
  for (std::size_t p = 0; p < n_out; ++p) {
    ints.slab(glob[p], glob_list, islab.data());  // <p q | r s> over the output spinors
    if (cls[p] == 0) buildDvvvvSlab(loc[p]);
    std::vector<Rec> recs;
    cplx e2_local{};
    // Per-thread results with a STATIC schedule, merged in thread order after the region (an `omp critical` merge in arrival order would
    // change the record order of the density-matrix file and the last bits of the energy sums from run to run).
    struct Part {
      std::vector<Rec> mine;
      cplx e2_mine{};
      cplx cat_mine[3][3] = {};
      double tr2_mine = 0.0;
    };
    std::vector<Part> parts(static_cast<std::size_t>(omp_get_max_threads()));
#pragma omp parallel
    {
      std::vector<Rec>& mine = parts[static_cast<std::size_t>(omp_get_thread_num())].mine;
      cplx& e2_mine = parts[static_cast<std::size_t>(omp_get_thread_num())].e2_mine;
      auto& cat_mine = parts[static_cast<std::size_t>(omp_get_thread_num())].cat_mine;
      double& tr2_mine = parts[static_cast<std::size_t>(omp_get_thread_num())].tr2_mine;
#pragma omp for schedule(static, 4) nowait
      for (std::ptrdiff_t qq = static_cast<std::ptrdiff_t>(p) + 1; qq < static_cast<std::ptrdiff_t>(n_out); ++qq) {
        const std::size_t q = static_cast<std::size_t>(qq);
        for (std::size_t r = 0; r < n_out; ++r)
          for (std::size_t s = r + 1; s < n_out; ++s) {
            cplx val;
            if (cls[p] == 0 && cls[q] == 0 && cls[r] == 0 && cls[s] == 0) val = std::conj(dslab.d[(loc[r] * v + loc[q]) * v + loc[s]]);
            else val = gammaElement(p, q, r, s);
            if (p == r && q == s) tr2_mine += 2.0 * val.real();  // sum_pq Gamma_pqpq = 2 sum_{p<q} Gamma_pqpq
            if (std::abs(val) < thr) continue;
            mine.push_back({{static_cast<std::int32_t>(p + 1), static_cast<std::int32_t>(q + 1), static_cast<std::int32_t>(r + 1),
                             static_cast<std::int32_t>(s + 1)},
                            val.real(), val.imag()});
            const cplx anti = islab[(q * n_out + r) * n_out + s] - islab[(q * n_out + s) * n_out + r];  // <pq||rs>
            e2_mine += anti * val;
            cat_mine[(cls[p] != 0) + (cls[q] != 0)][(cls[r] != 0) + (cls[s] != 0)] += anti * val;
          }
      }
    }
    for (const Part& pt : parts) {
      recs.insert(recs.end(), pt.mine.begin(), pt.mine.end());
      e2_local += pt.e2_mine;
      tr2 += pt.tr2_mine;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) e2cat[a][b] += pt.cat_mine[a][b];
    }
    for (const Rec& rc : recs) w2.put(rc.idx, 4, rc.re, rc.im);
    result.n_dm2 += recs.size();
    e2 += e2_local;
  }
  {
    const std::int32_t zero[4] = {0, 0, 0, 0};
    w2.put(zero, 4, 0.0, 0.0);
    w2.close();
  }
  result.e_elec_rdm = (e1 + e2).real();
  result.trace_dm2 = tr2;
  if (std::getenv("RERDMFT_CC_RDM_DEBUG") != nullptr) {
    log << "  [debug] E2 by pair character (rows: (pq) = vv, ov, oo; columns: (rs)):\n";
    for (int a = 0; a < 3; ++a) {
      log << "   ";
      for (int b = 0; b < 3; ++b) log << std::setw(18) << std::fixed << std::setprecision(10) << e2cat[a][b].real();
      log << "\n";
    }
  }

  // reference electronic energy of the converged amplitudes
  double e0_ref = 0.0;
  {
    Idx occ_all = spaces.occ_frozen;
    occ_all.insert(occ_all.end(), spaces.occ_active.begin(), spaces.occ_active.end());
    const std::size_t n = occ_all.size();
    std::vector<cplx> blk(n * n * n * n);
    ints.physicsBlock(occ_all, occ_all, occ_all, occ_all, blk.data());
    cplx e0{};
    for (std::size_t i = 0; i < n; ++i) {
      e0 += h_mo(occ_all[i], occ_all[i]);
      for (std::size_t j = 0; j < n; ++j) e0 += 0.5 * (blk[((i * n + j) * n + i) * n + j] - blk[((i * n + j) * n + j) * n + i]);
    }
    result.e_elec_cc = e0.real() + amps.e_corr;
    e0_ref = e0.real();
  }
  log << "  2-RDM written to " << settings.dm2_path << " (" << result.n_dm2 << " elements with p<q, r<s)\n";
  printEnergyLine(log, 2, "E_0 (reference determinant)", e0_ref);
  printEnergyLine(log, 2, "E_c (correlation)", amps.e_corr);
  printEnergyLine(log, 2, "One-body part", e1.real());
  printEnergyLine(log, 2, "Two-body part", e2.real());
  result.nuclear_repulsion = settings.reference_total_energy - e0_ref;
  result.e_total_rdm = result.e_elec_rdm + result.nuclear_repulsion;
  printEnergyLine(log, 2, "Trace of the 2-RDM", result.trace_dm2, "");
  printEnergyLine(log, 2, "N(N-1)", static_cast<double>(n_occ_all) * (static_cast<double>(n_occ_all) - 1.0), "");
  printEnergyLine(log, 2, "Total energy (density matrices)", result.e_total_rdm);
  log << "  Energy from the density matrices (elements written, threshold " << std::scientific << std::setprecision(1) << thr << "):\n"
      << std::defaultfloat << std::setprecision(6);
  printEnergyLine(log, 4, "Electronic energy", result.e_elec_rdm);
  printEnergyLine(log, 4, "E_0 + E_c", result.e_elec_cc);
  printScientificLine(log, 4, "Difference", result.e_elec_rdm - result.e_elec_cc, " Hartree");
  log << "  CC_RDM (molecular basis) total time " << std::fixed << std::setprecision(1) << elapsed() << " s\n";

  if (settings.on_gamma1) {
    Matrix<cplx> g1(n_out, n_out);
    for (std::size_t p = 0; p < n_out; ++p)
      for (std::size_t q = 0; q < n_out; ++q) g1(p, q) = gammaBar(p, q);
    try {
      settings.on_gamma1(g1, settings.reference_total_energy + amps.e_corr, log);
    } catch (const std::exception& ex) {
      log << "  AIMPAC WFN of the CC density FAILED: " << ex.what() << "\n";
    }
  }

  // ---- the same densities in the basis of the Gaussian primitives (RHO2_OPS format) ----
  if (settings.gauss != nullptr) {
    Matrix<cplx> g1(n_out, n_out);
    for (std::size_t p = 0; p < n_out; ++p)
      for (std::size_t q = 0; q < n_out; ++q) g1(p, q) = gammaBar(p, q);
    const std::size_t n3 = n_out * n_out * n_out;
    const GammaSlabFn fill_slab = [&](std::size_t p, cplx* out) {
      std::fill(out, out + n3, cplx{});
      if (cls[p] == 0) buildDvvvvSlab(loc[p]);
#pragma omp parallel for schedule(dynamic, 2) if (n3 > (std::size_t(1) << 18))
      for (std::ptrdiff_t qq = 0; qq < static_cast<std::ptrdiff_t>(n_out); ++qq) {
        const std::size_t q = static_cast<std::size_t>(qq);
        if (q == p) continue;
        for (std::size_t r = 0; r < n_out; ++r)
          for (std::size_t s = r + 1; s < n_out; ++s) {
            cplx val;
            if (cls[p] == 0 && cls[q] == 0 && cls[r] == 0 && cls[s] == 0) val = std::conj(dslab.d[(loc[r] * v + loc[q]) * v + loc[s]]);
            else val = gammaElement(p, q, r, s);
            out[(q * n_out + r) * n_out + s] = val;
            out[(q * n_out + s) * n_out + r] = -val;
          }
      }
    };
    try {
      const GaussRdmResult gr = writeGaussRdm(*settings.gauss, g1, n_out, fill_slab, settings.gauss_settings, log);
      log << "  Gaussian primitives: " << gr.n_primitives << " (Large " << gr.n_primitives_large << ", Small " << gr.n_primitives - gr.n_primitives_large << ")\n";
      log << "  basis written to " << settings.gauss_settings.basis_path << "\n";
      log << "  1-RDM written to " << settings.gauss_settings.dm1_path << " (" << gr.n_dm1 << " elements)\n";
      log << "  2-RDM written to " << settings.gauss_settings.dm2_path << " (" << gr.n_dm2 << " elements, " << gr.passes << " sweep(s) over the MO-basis 2-RDM)\n";
      log << "  Gaussian-basis checks: sum gamma S = " << std::fixed << std::setprecision(10) << gr.trace_dm1 << " (N = " << n_occ_all
          << "),  sum D S S = " << gr.trace_dm2 << " (N(N-1) = " << static_cast<double>(n_occ_all) * (static_cast<double>(n_occ_all) - 1.0)
          << "),  max |Im D_sym| = " << std::scientific << std::setprecision(2) << gr.max_imag << "\n";
      if (!std::isnan(gr.vee_primitives))
      {
        log << "  [check] V_ee from the primitive-basis 2-RDM and primitive integrals:\n";
        printEnergyLine(log, 4, "V_ee (primitive basis)", gr.vee_primitives);
        printEnergyLine(log, 4, "V_ee (MO basis)", e2.real());
        printScientificLine(log, 4, "Difference", gr.vee_primitives - e2.real(), " Hartree");
      }
      log << "  CC_RDM (Gaussian basis) total time " << std::fixed << std::setprecision(1) << gr.seconds << " s\n";
    } catch (const std::exception& ex) {
      log << "  CC_RDM (Gaussian basis) FAILED: " << ex.what() << "\n";
    }
  }
  return result;
}

}  // namespace rerdmft
