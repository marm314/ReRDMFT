#include "Uccsd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "CcTensor.h"
#include "CcWorkspace.h"
#include "DIIS.h"
#include "Progress.h"

namespace rerdmft {

namespace {

using cplx = CcScalar;
using Idx = std::vector<std::size_t>;
using cc::Tensor;
using cc::permute;
using cc::conjugated;
using cc::axpy;
using cc::dotNoConj;
using cc::einsum;
using cc::antisymIJ;
using cc::antisymAB;
using cc::maxAbs;

// OpenMP regions are entered only above this many elements: OpenMP and OpenBLAS each keep a spinning thread pool, and
// alternating between them for small loops oversubscribes the machine (a 52-spinor CCSD ran 7x slower, a 22-spinor
// one 100x); the permutation/gather loops are memory-bound, so the large ones are the only ones worth a region.
constexpr std::size_t kOmpMinWork = std::size_t(1) << 22;

// ---------------------------------------------------------------------------------------------------------
// Pair factors
// ---------------------------------------------------------------------------------------------------------

// factor_P(p,q) = data[P*n*n + p*n + q] (RiMoEri's B tensor, row-major (n_aux, n*n)).
class DensePairFactor : public CcPairFactor {
 public:
  DensePairFactor(const cplx* data, std::size_t n_aux, std::size_t n) : data_(data), n_aux_(n_aux), n_(n) {}
  std::size_t nAux() const override { return n_aux_; }
  void gather(const Idx& I, const Idx& J, cplx* out) const override {
    const std::size_t ni = I.size(), nj = J.size();
#pragma omp parallel for schedule(static) if (n_aux_ * ni * nj > kOmpMinWork)
    for (std::ptrdiff_t P = 0; P < static_cast<std::ptrdiff_t>(n_aux_); ++P) {
      const cplx* row = data_ + static_cast<std::size_t>(P) * n_ * n_;
      cplx* o = out + static_cast<std::size_t>(P) * ni * nj;
      for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < nj; ++j) o[i * nj + j] = row[I[i] * n_ + J[j]];
    }
  }

 private:
  const cplx* data_;
  std::size_t n_aux_, n_;
};

// NON_REL spin-orbitals: factor_P(p,q) = B(P, spatial(p), spatial(q)) when p and q have the same spin, else 0
// (RiNonRelSpinMoEri::operator()'s own selection rule), block layout [alpha; beta].
class SpinPairFactor : public CcPairFactor {
 public:
  SpinPairFactor(const Matrix<double>& b, std::size_t n_spatial) : b_(b), ns_(n_spatial) {}
  std::size_t nAux() const override { return b_.rows(); }
  void gather(const Idx& I, const Idx& J, cplx* out) const override {
    const std::size_t ni = I.size(), nj = J.size(), n_aux = b_.rows();
#pragma omp parallel for schedule(static) if (n_aux * ni * nj > kOmpMinWork)
    for (std::ptrdiff_t P = 0; P < static_cast<std::ptrdiff_t>(n_aux); ++P) {
      cplx* o = out + static_cast<std::size_t>(P) * ni * nj;
      for (std::size_t i = 0; i < ni; ++i)
        for (std::size_t j = 0; j < nj; ++j)
          o[i * nj + j] = (I[i] / ns_ == J[j] / ns_) ? cplx(b_(static_cast<std::size_t>(P), (I[i] % ns_) * ns_ + (J[j] % ns_)))
                                                       : cplx{};
    }
  }

 private:
  const Matrix<double>& b_;
  std::size_t ns_;
};

inline double conjOf(double x) { return x; }
inline cplx conjOf(cplx x) { return std::conj(x); }

// CholeskyEri: eri(a,b,c,d) = sum_L V_L(c,a) conj(V_L(b,d)) = (ac|bd): bra_L(a,c) = V_L(c,a), ket_L(b,d) = conj V_L(b,d).
template <typename T>
class CholeskyPairFactor : public CcPairFactor {
 public:
  CholeskyPairFactor(const CholeskyEri<T>& eri, bool bra) : w_(eri.rawVectors()), n_(eri.dim0()), nchol_(eri.nVectors()), bra_(bra) {}
  std::size_t nAux() const override { return nchol_; }
  void gather(const Idx& I, const Idx& J, cplx* out) const override {
    const std::size_t ni = I.size(), nj = J.size();
#pragma omp parallel for schedule(static) if (ni * nj * nchol_ > kOmpMinWork)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(ni); ++i)
      for (std::size_t j = 0; j < nj; ++j) {
        const std::size_t x = bra_ ? J[j] : I[static_cast<std::size_t>(i)];
        const std::size_t y = bra_ ? I[static_cast<std::size_t>(i)] : J[j];
        const T* v = &w_[(x * n_ + y) * nchol_];
        for (std::size_t L = 0; L < nchol_; ++L)
          out[L * ni * nj + static_cast<std::size_t>(i) * nj + j] = bra_ ? cplx(v[L]) : cplx(conjOf(v[L]));
      }
  }

 private:
  const std::vector<T>& w_;
  std::size_t n_, nchol_;
  bool bra_;
};

Idx shifted(const Idx& l, std::size_t shift) {
  Idx out(l);
  for (auto& x : out) x -= shift;
  return out;
}

void zgemmTN(std::size_t M, std::size_t N, std::size_t K, const cplx* A, const cplx* B, cplx* C) {
  // C (M x N) = A^T B with A stored K x M, B stored K x N, all row-major.
  const cplx one(1.0, 0.0), zero(0.0, 0.0);
  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), &one, A,
              static_cast<int>(M), B, static_cast<int>(N), &zero, C, static_cast<int>(N));
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Factorized integral source
// ---------------------------------------------------------------------------------------------------------

void FactoredCcIntegrals::physicsBlock(const Idx& P, const Idx& Q, const Idx& R, const Idx& S, cplx* out) const {
  const std::size_t np = P.size(), nq = Q.size(), nr = R.size(), ns = S.size(), n_aux = bra_->nAux();
  const std::size_t M = np * nr, N = nq * ns;
  if (M == 0 || N == 0) return;
  const Idx Ps = shifted(P, shift_), Qs = shifted(Q, shift_), Rs = shifted(R, shift_), Ss = shifted(S, shift_);
  std::vector<cplx> a(n_aux * M), x(M * N);
  bra_->gather(Ps, Rs, a.data());
  if (bra_ == ket_ && Ps == Qs && Rs == Ss) {
    zgemmTN(M, N, n_aux, a.data(), a.data(), x.data());
  } else {
    std::vector<cplx> b(n_aux * N);
    ket_->gather(Qs, Ss, b.data());
    zgemmTN(M, N, n_aux, a.data(), b.data(), x.data());
  }
  // x[(p r), (q s)] -> out[p][q][r][s]
#pragma omp parallel for schedule(static) if (M * N > kOmpMinWork)
  for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(np); ++p)
    for (std::size_t q = 0; q < nq; ++q)
      for (std::size_t r = 0; r < nr; ++r) {
        const cplx* src = &x[(static_cast<std::size_t>(p) * nr + r) * N + q * ns];
        cplx* dst = out + ((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns;
        for (std::size_t s = 0; s < ns; ++s) dst[s] = src[s];
      }
}

void FactoredCcIntegrals::prepareSlabs(const Idx& list) const {
  if (slab_list_ == list && !slab_ket_.empty()) return;
  const std::size_t n_aux = ket_->nAux(), L = list.size();
  const Idx ls = shifted(list, shift_);
  slab_ket_.assign(n_aux * L * L, cplx{});
  ket_->gather(ls, ls, slab_ket_.data());
  slab_list_ = list;
}

void FactoredCcIntegrals::slab(std::size_t p, const Idx& list, cplx* out) const {
  prepareSlabs(list);
  const std::size_t L = list.size(), n_aux = bra_->nAux();
  std::vector<cplx> a(n_aux * L), x(L * L * L);
  bra_->gather(Idx{p - shift_}, shifted(list, shift_), a.data());
  // X[r][(q s)] = sum_P bra_P(p, list[r]) ket_P(list[q], list[s])
  zgemmTN(L, L * L, n_aux, a.data(), slab_ket_.data(), x.data());
#pragma omp parallel for schedule(static) if (L * L * L > kOmpMinWork)
  for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(L); ++q)
    for (std::size_t r = 0; r < L; ++r) {
      const cplx* src = &x[(r * L + static_cast<std::size_t>(q)) * L];
      cplx* dst = out + (static_cast<std::size_t>(q) * L + r) * L;
      for (std::size_t s = 0; s < L; ++s) dst[s] = src[s];
    }
}

std::unique_ptr<CcIntegrals> makeCcIntegrals(const RiMoEri& eri, std::size_t shift) {
  const auto f = std::make_shared<DensePairFactor>(eri.b().data(), eri.b().rows(), eri.dim0());
  return std::make_unique<FactoredCcIntegrals>(f, f, shift);
}

std::unique_ptr<CcIntegrals> makeCcIntegrals(const RiNonRelSpinMoEri& eri, std::size_t shift) {
  const auto f = std::make_shared<SpinPairFactor>(eri.b(), eri.nSpatial());
  return std::make_unique<FactoredCcIntegrals>(f, f, shift);
}

std::unique_ptr<CcIntegrals> makeCcIntegrals(const CholeskyEri<double>& eri, std::size_t shift) {
  return std::make_unique<FactoredCcIntegrals>(std::make_shared<CholeskyPairFactor<double>>(eri, true),
                                               std::make_shared<CholeskyPairFactor<double>>(eri, false), shift);
}

std::unique_ptr<CcIntegrals> makeCcIntegrals(const CholeskyEri<cplx>& eri, std::size_t shift) {
  return std::make_unique<FactoredCcIntegrals>(std::make_shared<CholeskyPairFactor<cplx>>(eri, true),
                                               std::make_shared<CholeskyPairFactor<cplx>>(eri, false), shift);
}

// ---------------------------------------------------------------------------------------------------------
// Orbital spaces
// ---------------------------------------------------------------------------------------------------------

CcOrbitalSpaces contiguousCcSpaces(std::size_t first, std::size_t dim, std::size_t n_electrons, std::size_t n_frozen) {
  if (n_frozen > n_electrons) throw std::runtime_error("UCCSD: CC_FROZEN_CORE exceeds the number of electrons");
  if (first + n_electrons > dim) throw std::runtime_error("UCCSD: more electrons than spinors");
  CcOrbitalSpaces s;
  for (std::size_t p = 0; p < n_frozen; ++p) s.occ_frozen.push_back(first + p);
  for (std::size_t p = n_frozen; p < n_electrons; ++p) s.occ_active.push_back(first + p);
  for (std::size_t p = n_electrons; p < dim - first; ++p) s.vir.push_back(first + p);
  s.file_index.assign(dim, 0);
  for (std::size_t g = first; g < dim; ++g) s.file_index[g] = g - first + 1;
  return s;
}

CcOrbitalSpaces nonRelCcSpaces(std::size_t n_spatial, std::size_t n_electrons, std::size_t n_frozen) {
  if (n_electrons % 2 != 0) throw std::runtime_error("UCCSD (NON_REL): closed-shell reference needs an even electron count");
  const std::size_t n_occ = n_electrons / 2, n_fr = n_frozen / 2;
  if (n_fr > n_occ) throw std::runtime_error("UCCSD: CC_FROZEN_CORE exceeds the number of electrons");
  CcOrbitalSpaces s;
  for (std::size_t spin = 0; spin < 2; ++spin) {
    for (std::size_t p = 0; p < n_fr; ++p) s.occ_frozen.push_back(spin * n_spatial + p);
    for (std::size_t p = n_fr; p < n_occ; ++p) s.occ_active.push_back(spin * n_spatial + p);
    for (std::size_t p = n_occ; p < n_spatial; ++p) s.vir.push_back(spin * n_spatial + p);
  }
  // alpha_k -> 2k+1, beta_k -> 2k+2 (Kramers-pair order, like X2C/C4_SPINOR)
  s.file_index.assign(2 * n_spatial, 0);
  for (std::size_t k = 0; k < n_spatial; ++k) {
    s.file_index[k] = 2 * k + 1;
    s.file_index[n_spatial + k] = 2 * k + 2;
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------
// CCSD
// ---------------------------------------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

struct Amplitudes {
  Tensor t1, t2;
};

}  // namespace

UccsdResult runUccsd(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const Idx& occ_frozen, const Idx& occ_active,
                     const Idx& vir, const UccsdSettings& settings, std::ostream& log) {
  const std::size_t o = occ_active.size(), v = vir.size();
  if (o < 2 || v < 2) throw std::runtime_error("UCCSD: needs at least two active occupied and two virtual spinors");
  UccsdResult result;
  const auto t_start = Clock::now();
  const auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count(); };
  const double gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
  log << "  active occupied spinors: " << o << " (frozen: " << occ_frozen.size() << "), virtual spinors: " << v << "\n";
  log << "  amplitude storage (T2): " << std::fixed << std::setprecision(3) << 16.0 * o * o * v * v * gib << " GiB per copy, ~10 copies live\n";

  CcWorkspace ws = buildCcWorkspace(h_mo, ints, occ_frozen, occ_active, vir, &log);
  result.max_fock_asymmetry = ws.max_fock_asymmetry;
  result.max_fock_ov = ws.max_fock_ov;
  const Idx &O = ws.O, &V = ws.V;
  const Tensor &f_oo = ws.f_oo, &f_ov = ws.f_ov, &f_vo = ws.f_vo, &f_vv = ws.f_vv;
  const std::vector<double> &e_o = ws.e_o, &e_v = ws.e_v;
  const Tensor &g_oooo = ws.g_oooo, &g_ooov = ws.g_ooov, &g_oovv = ws.g_oovv, &g_ovov = ws.g_ovov, &g_ovvo = ws.g_ovvo;
  ints.prepareSlabs(V);
  const Tensor g_ovoo_c = conjugated(permute(g_ooov, {2, 3, 0, 1}));  // <mb||ij> = conj <ij||mb>, [m][b][i][j]
  const Tensor g_vvoo_c = conjugated(g_oovv);                         // <ab||ij> = conj <ij||ab>, stored [i][j][a][b]
  ints.prepareSlabs(V);

  // Denominators.
  Tensor D1({o, v}), D2({o, o, v, v});
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) D1.d[i * v + a] = e_o[i] - e_v[a];
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) D2.d[((i * o + j) * v + a) * v + b] = e_o[i] + e_o[j] - e_v[a] - e_v[b];
  const Tensor f_vo_T = permute(f_vo, {1, 0});  // f_ai as [i][a]

  const auto energy = [&](const Tensor& t1, const Tensor& t2) {
    cplx e = dotNoConj(f_ov, t1) + 0.25 * dotNoConj(g_oovv, t2);
    e += 0.5 * dotNoConj(einsum("ijab,jb->ia", g_oovv, t1), t1);
    return e;
  };

  const std::size_t n_electrons_active = o;
  // One Jacobi sweep numerators: returns (n1, n2) with t_new = n / D.
  const auto residualNumerators = [&](const Tensor& t1, const Tensor& t2, Tensor& n1, Tensor& n2) {
    const Tensor outer = einsum("ia,jb->ijab", t1, t1);
    const Tensor outer_a = outer - permute(outer, {1, 0, 2, 3});  // t_i^a t_j^b - t_j^a t_i^b ... antisym in (i,j) with (a,b) fixed
    // tau_ij^ab = t2 + t_i^a t_j^b - t_i^b t_j^a = t2 + outer[i,j,a,b] - outer[i,j,b,a] = t2 + (outer - outer^{ij swapped})
    const Tensor tau = t2 + outer_a;
    Tensor tl = t2;
    axpy(tl, outer_a, 0.5);

    Tensor Fae = einsum("me,ma->ae", f_ov, t1, -0.5);
    axpy(Fae, einsum("mnaf,mnef->ae", tl, g_oovv), -0.5);
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t e = 0; e < v; ++e)
        if (a != e) Fae.d[a * v + e] += f_vv.d[a * v + e];

    Tensor W3 = g_ovvo;  // [m][b][e][j]
    Tensor Z({o, o, o, v});
    Tensor w({o, o, v, v});
    n1 = f_vo_T;
    n2 = g_vvoo_c;

    // ---- Streams over the ovvv slabs <m a||e f> (m occupied) ----
    {
      std::vector<cplx> raw(v * v * v);
      Tensor S({v, v, v});
      for (std::size_t m = 0; m < o; ++m) {
        ints.slab(O[m], V, raw.data());
#pragma omp parallel for schedule(static) if (v * v * v > kOmpMinWork)
        for (std::ptrdiff_t a = 0; a < static_cast<std::ptrdiff_t>(v); ++a)
          for (std::size_t e = 0; e < v; ++e)
            for (std::size_t f = 0; f < v; ++f) {
              const std::size_t au = static_cast<std::size_t>(a);
              S.d[(au * v + e) * v + f] = raw[(au * v + e) * v + f] - raw[(au * v + f) * v + e];
            }
        Tensor t1m({v});
        for (std::size_t f = 0; f < v; ++f) t1m.d[f] = t1.d[m * v + f];
        axpy(Fae, einsum("afe,f->ae", S, t1m), 1.0);  // + sum_f t_m^f <ma||fe>
        Tensor t2m({o, v, v});
        for (std::size_t i = 0; i < o; ++i)
          std::copy_n(&t2.d[((i * o + m) * v) * v], v * v, &t2m.d[i * v * v]);
        axpy(n1, einsum("ief,aef->ia", t2m, S), -0.5);  // -1/2 sum t_im^ef <ma||ef>
        const Tensor c = einsum("bef,jf->bej", S, t1);   // + sum_f t_j^f <mb||ef>
        for (std::size_t k = 0; k < c.size(); ++k) W3.d[m * c.size() + k] += c.d[k];
        const Tensor z = einsum("ijef,aef->ija", tau, S, -0.5);  // 1/2 sum tau_ij^ef <am||ef>
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t j = 0; j < o; ++j)
            std::copy_n(&z.d[(i * o + j) * v], v, &Z.d[((i * o + j) * o + m) * v]);
        const Tensor wm = einsum("ie,eba->iab", t1, conjugated(S));  // [i][a][b] = sum_e t_i^e conj <me||ba>
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t ab = 0; ab < v * v; ++ab) w.d[(i * o + m) * v * v + ab] = wm.d[i * v * v + ab];
      }
    }

    // ---- vvvv particle-particle ladder, one a at a time: n2[i,j,a,b] += sum_ef tau_ij^ef <ab|ef> ----
    {
      Tensor Pa({v, v, v});
      for (std::size_t a = 0; a < v; ++a) {
        ints.slab(V[a], V, Pa.d.data());
        const Tensor c = einsum("ijef,bef->ijb", tau, Pa);
#pragma omp parallel for schedule(static) if (o * o * v > kOmpMinWork)
        for (std::ptrdiff_t ij = 0; ij < static_cast<std::ptrdiff_t>(o * o); ++ij)
          for (std::size_t b = 0; b < v; ++b) n2.d[(static_cast<std::size_t>(ij) * v + a) * v + b] += c.d[static_cast<std::size_t>(ij) * v + b];
      }
    }

    // ---- Remaining intermediates ----
    Tensor Fmi = einsum("ie,me->mi", t1, f_ov, 0.5);
    axpy(Fmi, einsum("ne,mnie->mi", t1, g_ooov), 1.0);
    axpy(Fmi, einsum("inef,mnef->mi", tl, g_oovv), 0.5);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t i = 0; i < o; ++i)
        if (m != i) Fmi.d[m * o + i] += f_oo.d[m * o + i];
    Tensor Fme = f_ov;
    axpy(Fme, einsum("nf,mnef->me", t1, g_oovv), 1.0);

    // Woooo, with the full tau tau term (the vvvv ladder above carries none of it).
    const Tensor X = einsum("je,mnie->mnij", t1, g_ooov);
    Tensor W1 = g_oooo + (X - permute(X, {0, 1, 3, 2}));
    axpy(W1, einsum("ijef,mnef->mnij", tau, g_oovv), 0.5);

    // W_mbej remaining terms.
    axpy(W3, einsum("nb,mnje->mbej", t1, g_ooov), 1.0);
    axpy(W3, einsum("jnfb,mnef->mbej", t2, g_oovv), -0.5);
    {
      const Tensor Y = einsum("jf,mnef->mnej", t1, g_oovv);
      axpy(W3, einsum("nb,mnej->mbej", t1, Y), -1.0);
    }

    // ---- T1 ----
    axpy(n1, einsum("ie,ae->ia", t1, Fae), 1.0);
    axpy(n1, einsum("ma,mi->ia", t1, Fmi), -1.0);
    axpy(n1, einsum("imae,me->ia", t2, Fme), 1.0);
    axpy(n1, einsum("nf,naif->ia", t1, g_ovov), -1.0);
    axpy(n1, einsum("mnae,nmie->ia", t2, g_ooov), 0.5);

    // ---- T2 ----
    {
      Tensor A = Fae;
      axpy(A, einsum("mb,me->be", t1, Fme), -0.5);
      axpy(n2, antisymAB(einsum("ijae,be->ijab", t2, A)), 1.0);
      Tensor B = Fmi;
      axpy(B, einsum("je,me->mj", t1, Fme), 0.5);
      axpy(n2, antisymIJ(einsum("imab,mj->ijab", t2, B)), -1.0);
    }
    axpy(n2, einsum("mnab,mnij->ijab", tau, W1), 0.5);
    {
      Tensor zz = einsum("imae,mbej->ijab", t2, W3);
      const Tensor U = einsum("ie,mbej->mbij", t1, g_ovvo);
      axpy(zz, einsum("ma,mbij->ijab", t1, U), -1.0);
      axpy(n2, antisymAB(antisymIJ(zz)), 1.0);
    }
    axpy(n2, antisymIJ(w), 1.0);                                                 // + P(ij) t_i^e <ab||ej>
    axpy(n2, antisymAB(einsum("ma,mbij->ijab", t1, g_ovoo_c)), -1.0);            // - P(ab) t_m^a <mb||ij>
    {
      const Tensor Y = einsum("mb,ijma->ijab", t1, Z);                           // -P(ab) t_m^b Z_ijm^a
      axpy(n2, antisymAB(Y), -1.0);
    }
  };

  // ---- Initial guess (MP2-like) ----
  Tensor t1({o, v}), t2({o, o, v, v});
  for (std::size_t k = 0; k < t1.size(); ++k) t1.d[k] = f_vo_T.d[k] / D1.d[k];
  for (std::size_t k = 0; k < t2.size(); ++k) t2.d[k] = g_vvoo_c.d[k] / D2.d[k];
  result.e_mp2 = energy(Tensor({o, v}), t2).real();

  // DIIS on the antisymmetry-unique amplitudes.
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
  Diis<cplx> diis(static_cast<std::size_t>(std::max(settings.diis_size, 0)));
  const cplx shift(settings.level_shift, 0.0);

  log << "  iter      E_corr (Eh)        dE            |dT|        time (s)\n";
  cplx e_old = energy(t1, t2);
  for (int iter = 1; iter <= settings.max_iterations; ++iter) {
    Tensor n1, n2;
    residualNumerators(t1, t2, n1, n2);
    Tensor t1n({o, v}), t2n({o, o, v, v});
    for (std::size_t k = 0; k < t1n.size(); ++k) t1n.d[k] = (n1.d[k] - shift * t1.d[k]) / (D1.d[k] - shift);
    for (std::size_t k = 0; k < t2n.size(); ++k) t2n.d[k] = (n2.d[k] - shift * t2.d[k]) / (D2.d[k] - shift);
    // Antisymmetric projection (round-off): the update is built from antisymmetrized pieces already.
    const Matrix<cplx> vec_new = pack(t1n, t2n), vec_old = pack(t1, t2);
    Matrix<cplx> err(1, n_vec);
    double dt2 = 0.0;
    for (std::size_t k = 0; k < n_vec; ++k) {
      err(0, k) = vec_new(0, k) - vec_old(0, k);
      dt2 += std::norm(err(0, k));
    }
    const double dt = std::sqrt(dt2);
    if (settings.diis_size >= 2) {
      const Matrix<cplx> extrapolated = diis.extrapolate(err, vec_new);
      unpack(extrapolated, t1, t2);
    } else {
      unpack(vec_new, t1, t2);
    }
    const cplx e_new = energy(t1, t2);
    const double de = e_new.real() - e_old.real();
    log << std::setw(6) << iter << std::fixed << std::setprecision(10) << std::setw(20) << e_new.real() << std::scientific << std::setprecision(2)
        << std::setw(12) << de << std::setw(12) << dt << std::fixed << std::setprecision(1) << std::setw(12) << elapsed() << "\n";
    {
      std::ostringstream line;
      line << "(UCCSD) iteration " << iter << ": E_corr = " << std::fixed << std::setprecision(10) << e_new.real() << ", dE = " << std::scientific
           << std::setprecision(2) << de << ", |dT| = " << dt;
      progress(line.str());
    }
    e_old = e_new;
    result.iterations = iter;
    if (std::abs(de) < settings.convergence && dt < 100.0 * settings.convergence) {
      result.converged = true;
      break;
    }
  }
  result.e_corr = e_old.real();
  result.e_corr_imag = e_old.imag();
  result.t1 = t1.d;
  result.t2 = t2.d;
  double t1n2 = 0.0;
  for (const auto& x : t1.d) t1n2 += std::norm(x);
  result.t1_diagnostic = std::sqrt(t1n2 / static_cast<double>(n_electrons_active));
  log << std::scientific << std::setprecision(2) << "  Im(E_corr) = " << result.e_corr_imag << ",  T1 diagnostic = " << result.t1_diagnostic
      << ",  " << std::fixed << std::setprecision(1) << elapsed() << " s\n";
  return result;
}

}  // namespace rerdmft
