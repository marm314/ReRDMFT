#include "gauss_rdm.h"

#include <algorithm>
#include <array>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "ElectronRepulsion.h"
#include "Element.h"
#include "Tensor4.h"
#include "Progress.h"

namespace rerdmft {

namespace {

using cplx = GaussScalar;
using Clock = std::chrono::steady_clock;

// libcint's kernels evaluate coefficient * fac(l) * x^lx y^ly z^lz exp(-a r^2) (see Utils/AIMPAC.cpp): fac(l) is folded into the weights.
double libcintCommonFac(int l) { return l == 0 ? 0.282094791773878143 : l == 1 ? 0.488602511902919921 : 1.0; }

double doubleFactorial(int n) {  // n!! with (-1)!! = 0!! = 1
  double r = 1.0;
  for (int k = n; k > 1; k -= 2) r *= k;
  return r;
}

// Normalization of the primitive x^lx y^ly z^lz exp(-a r^2) (the factor RHO2_OPS multiplies by).
double primitiveNorm(double a, int lx, int ly, int lz) {
  const double pi = 3.14159265358979323846;
  const int l = lx + ly + lz;
  return std::pow(2.0 * a / pi, 0.75) * std::pow(4.0 * a, 0.5 * l) /
         std::sqrt(doubleFactorial(2 * lx - 1) * doubleFactorial(2 * ly - 1) * doubleFactorial(2 * lz - 1));
}

struct Prim {
  int z = 0;
  double x = 0, y = 0, zc = 0, alpha = 0;
  int lx = 0, ly = 0, lz = 0;
  double norm = 1.0;
};

double binom(int n, int k) {
  double r = 1.0;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

// 1-D overlap of x_A^la e^{-a (x-A)^2} and x_B^lb e^{-b (x-B)^2}
double overlap1d(int la, int lb, double A, double B, double a, double b) {
  const double pi = 3.14159265358979323846;
  const double g = a + b, P = (a * A + b * B) / g, pa = P - A, pb = P - B;
  double s = 0.0;
  for (int i = 0; i <= la; ++i)
    for (int j = 0; j <= lb; ++j) {
      const int m = i + j;
      if (m % 2 != 0) continue;
      s += binom(la, i) * binom(lb, j) * std::pow(pa, la - i) * std::pow(pb, lb - j) * doubleFactorial(m - 1) / std::pow(2.0 * g, 0.5 * m);
    }
  return s * std::sqrt(pi / g) * std::exp(-a * b / g * (A - B) * (A - B));
}

double primitiveOverlap(const Prim& p, const Prim& q) {
  return overlap1d(p.lx, q.lx, p.x, q.x, p.alpha, q.alpha) * overlap1d(p.ly, q.ly, p.y, q.y, p.alpha, q.alpha) *
         overlap1d(p.lz, q.lz, p.zc, q.zc, p.alpha, q.alpha);
}

// One component sector (Large or Small): its distinct primitives and the AO -> primitive weights.
struct Sector {
  std::vector<Prim> prims;
  Matrix<double> weights;  // (n_ao x P): AO i = sum_mu weights(i,mu) * (plain primitive mu)
};

Sector buildSector(const std::vector<BasisFunction>& basis) {
  Sector sec;
  std::map<std::tuple<double, double, double, double, int, int, int>, std::size_t> index;
  struct Entry { std::size_t ao, mu; double w; };
  std::vector<Entry> entries;
  for (std::size_t i = 0; i < basis.size(); ++i) {
    const BasisFunction& fn = basis[i];
    for (std::size_t k = 0; k < fn.exponents.size(); ++k) {
      const auto key = std::make_tuple(fn.x, fn.y, fn.z, fn.exponents[k], fn.cartesian.lx, fn.cartesian.ly, fn.cartesian.lz);
      auto it = index.find(key);
      if (it == index.end()) {
        Prim p;
        p.z = atomicNumber(fn.element);
        p.x = fn.x; p.y = fn.y; p.zc = fn.z;
        p.alpha = fn.exponents[k];
        p.lx = fn.cartesian.lx; p.ly = fn.cartesian.ly; p.lz = fn.cartesian.lz;
        p.norm = primitiveNorm(p.alpha, p.lx, p.ly, p.lz);
        it = index.emplace(key, sec.prims.size()).first;
        sec.prims.push_back(p);
      }
      entries.push_back({i, it->second, libcintCommonFac(fn.l) * fn.coefficients[k]});
    }
  }
  sec.weights = Matrix<double>(basis.size(), sec.prims.size(), 0.0);
  for (const Entry& e : entries) sec.weights(e.ao, e.mu) += e.w;
  return sec;
}

// One spinor component: coefficients of the (normalized, unless CNORM1) primitives of its sector, and their complex conjugate.
struct Component {
  int sector = 0;
  std::size_t P = 0;
  std::vector<cplx> C, Cc;  // (P x n) row-major; Cc = conj(C)
};

Component makeComponent(int sector, const Sector& sec, const Matrix<cplx>& ao_coef, std::size_t n, bool normalize) {
  Component c;
  c.sector = sector;
  c.P = sec.prims.size();
  c.C.assign(c.P * n, cplx{});
  if (ao_coef.rows() != sec.weights.rows() || ao_coef.cols() != n)
    throw std::runtime_error("gauss_rdm: spinor coefficient matrix has the wrong shape");
  for (std::size_t i = 0; i < ao_coef.rows(); ++i)
    for (std::size_t mu = 0; mu < c.P; ++mu) {
      const double w = sec.weights(i, mu);
      if (w == 0.0) continue;
      for (std::size_t p = 0; p < n; ++p) c.C[mu * n + p] += w * ao_coef(i, p);
    }
  if (normalize)
    for (std::size_t mu = 0; mu < c.P; ++mu)
      for (std::size_t p = 0; p < n; ++p) c.C[mu * n + p] /= sec.prims[mu].norm;
  c.Cc.resize(c.C.size());
  for (std::size_t k = 0; k < c.C.size(); ++k) c.Cc[k] = std::conj(c.C[k]);
  return c;
}

void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, std::size_t M, std::size_t N, std::size_t K, cplx alpha, const cplx* A,
          std::size_t lda, const cplx* B, std::size_t ldb, cplx beta, cplx* C, std::size_t ldc) {
  if (M == 0 || N == 0) return;
  cblas_zgemm(CblasRowMajor, ta, tb, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), &alpha, A, static_cast<int>(lda), B,
              static_cast<int>(ldb), &beta, C, static_cast<int>(ldc));
}

// Fortran-unformatted-style records: int32 length, payload, int32 length.
class FortranRecords {
 public:
  explicit FortranRecords(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {
    if (f_ == nullptr) throw std::runtime_error("gauss_rdm: cannot open " + path + " for writing");
    std::setvbuf(f_, nullptr, _IOFBF, std::size_t(1) << 22);
  }
  ~FortranRecords() {
    if (f_ != nullptr) std::fclose(f_);
  }
  FortranRecords(const FortranRecords&) = delete;
  FortranRecords& operator=(const FortranRecords&) = delete;
  void put(const std::int32_t* idx, int n, double value) {
    const std::int32_t len = static_cast<std::int32_t>(n * sizeof(std::int32_t) + sizeof(double));
    std::fwrite(&len, sizeof(len), 1, f_);
    std::fwrite(idx, sizeof(std::int32_t), static_cast<std::size_t>(n), f_);
    std::fwrite(&value, sizeof(double), 1, f_);
    std::fwrite(&len, sizeof(len), 1, f_);
  }
  void close() {
    if (f_ != nullptr) std::fclose(f_);
    f_ = nullptr;
  }

 private:
  std::FILE* f_;
};

}  // namespace

GaussRdmResult writeGaussRdm(const CartesianSpinors& spinors, const Matrix<GaussScalar>& gamma1, std::size_t n,
                             const GammaSlabFn& slab, const GaussRdmSettings& settings, std::ostream& log) {
  const auto t_start = Clock::now();
  const auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t_start).count(); };
  GaussRdmResult result;
  const bool normalize = !settings.unnormalized_primitives;
  const bool has_small = !spinors.small_basis.empty();

  // ---- primitives ----
  std::vector<Sector> sectors;
  sectors.push_back(buildSector(spinors.large_basis));
  if (has_small) sectors.push_back(buildSector(spinors.small_basis));
  const std::size_t n_sec = sectors.size();
  std::vector<std::size_t> P(n_sec), first(n_sec);  // primitives per sector, global index of the first one
  std::size_t P_total = 0;
  for (std::size_t s = 0; s < n_sec; ++s) {
    P[s] = sectors[s].prims.size();
    first[s] = P_total;
    P_total += P[s];
  }
  result.n_primitives = P_total;
  result.n_primitives_large = P[0];

  std::vector<std::vector<Component>> comps(n_sec);  // components of each sector
  comps[0].push_back(makeComponent(0, sectors[0], spinors.large_alpha, n, normalize));
  comps[0].push_back(makeComponent(0, sectors[0], spinors.large_beta, n, normalize));
  if (has_small) {
    comps[1].push_back(makeComponent(1, sectors[1], spinors.small_alpha, n, normalize));
    comps[1].push_back(makeComponent(1, sectors[1], spinors.small_beta, n, normalize));
  }

  // ---- .basis ----
  {
    std::FILE* fb = std::fopen(settings.basis_path.c_str(), "w");
    if (fb == nullptr) throw std::runtime_error("gauss_rdm: cannot open " + settings.basis_path + " for writing");
    std::fprintf(fb, "# Z  \t X_A \t Y_A \t Z_A \t Exp \t Quant(nx,ny,nz)\n");
    for (std::size_t s = 0; s < n_sec; ++s)
      for (const Prim& p : sectors[s].prims)
        std::fprintf(fb, "%3d\t%.15e\t%.15e\t%.15e\t%.15e\t%d\t%d\t%d\n", p.z, p.x, p.y, p.zc, p.alpha, p.lx, p.ly, p.lz);
    std::fclose(fb);
  }

  // ---- primitive overlap (diagnostics) ----
  std::vector<std::vector<double>> S(n_sec);
  for (std::size_t s = 0; s < n_sec; ++s) {
    S[s].assign(P[s] * P[s], 0.0);
    for (std::size_t a = 0; a < P[s]; ++a)
      for (std::size_t b = 0; b <= a; ++b) {
        const Prim &pa = sectors[s].prims[a], &pb = sectors[s].prims[b];
        double v = primitiveOverlap(pa, pb);
        if (normalize) v *= pa.norm * pb.norm;
        S[s][a * P[s] + b] = S[s][b * P[s] + a] = v;
      }
  }

  // ---- 1-RDM ----
  {
    FortranRecords w1(settings.dm1_path);
    for (std::size_t s = 0; s < n_sec; ++s) {
      std::vector<cplx> M(P[s] * P[s], cplx{});
      for (const Component& c : comps[s]) {
        std::vector<cplx> Z(P[s] * n);
        gemm(CblasNoTrans, CblasNoTrans, P[s], n, n, 1.0, c.Cc.data(), n, gamma1.data(), n, 0.0, Z.data(), n);   // conj(C) gamma
        gemm(CblasNoTrans, CblasTrans, P[s], P[s], n, 1.0, Z.data(), n, c.C.data(), n, 1.0, M.data(), P[s]);     // ... C^T
      }
      for (std::size_t mu = 0; mu < P[s]; ++mu)
        for (std::size_t la = 0; la <= mu; ++la) {
          const double v = 0.5 * (M[mu * P[s] + la] + M[la * P[s] + mu]).real();
          result.trace_dm1 += (mu == la ? 1.0 : 2.0) * v * S[s][mu * P[s] + la];
          if (std::abs(v) < settings.threshold) continue;
          const std::int32_t idx[2] = {static_cast<std::int32_t>(first[s] + mu + 1), static_cast<std::int32_t>(first[s] + la + 1)};
          w1.put(idx, 2, v);
          ++result.n_dm1;
        }
    }
    const std::int32_t zero[2] = {0, 0};
    w1.put(zero, 2, 0.0);
    w1.close();
  }

  // ---- optional check: V_ee = 1/2 sum D (mu lambda|nu sigma) with the integrals of the primitives themselves ----
  const bool check_vee = std::getenv("RERDMFT_CC_RDM_GAU_CHECK") != nullptr;
  Tensor4<double> prim_eri;
  double vee = 0.0;
  if (check_vee) {
    std::vector<BasisFunction> prim_ao;
    for (std::size_t s = 0; s < n_sec; ++s)
      for (const Prim& p : sectors[s].prims) {
        BasisFunction fn;
        fn.element = "X";
        fn.x = p.x; fn.y = p.y; fn.z = p.zc;
        fn.l = p.lx + p.ly + p.lz;
        fn.cartesian.lx = p.lx; fn.cartesian.ly = p.ly; fn.cartesian.lz = p.lz;
        fn.exponents = {p.alpha};
        fn.coefficients = {(normalize ? p.norm : 1.0) / libcintCommonFac(fn.l)};
        prim_ao.push_back(fn);
      }
    prim_eri = twoElectronIntegrals(prim_ao);
  }

  // ---- 2-RDM ----
  const std::size_t n2 = n * n, n3 = n2 * n;
  const double budget = settings.max_memory_gb * 1024.0 * 1024.0 * 1024.0;
  FortranRecords w2(settings.dm2_path);
  std::vector<cplx> gbatch, gperm(n3);
  const std::size_t bp_max = std::max<std::size_t>(1, std::min<std::size_t>(n, 8));
  gbatch.resize(bp_max * n3);

  for (std::size_t s1 = 0; s1 < n_sec; ++s1) {
    const std::size_t P1 = P[s1];
    const std::size_t nc1 = comps[s1].size();
    // block of mu rows: R and W for every component of the sector (2 * nc1 * Bm * n^3 complex numbers)
    const double per_mu = 16.0 * 2.0 * static_cast<double>(nc1) * static_cast<double>(n3);
    const std::size_t Bm_max = std::max<std::size_t>(1, std::min<std::size_t>(P1, static_cast<std::size_t>(budget / per_mu)));
    log << "  Gaussian 2-RDM, sector " << (s1 == 0 ? "Large" : "Small") << ": " << P1 << " primitives, block of " << Bm_max
        << " rows, " << (P1 + Bm_max - 1) / Bm_max << " sweep(s) over the MO-basis 2-RDM\n";
    std::vector<std::vector<cplx>> R(nc1), W(nc1);
    for (std::size_t mu0 = 0; mu0 < P1; mu0 += Bm_max) {
      const std::size_t Bm = std::min(Bm_max, P1 - mu0);
      for (std::size_t c = 0; c < nc1; ++c) {
        R[c].assign(Bm * n3, cplx{});
        W[c].assign(Bm * n * n2, cplx{});
      }
      // ---- stage 1: sweep over p ----
      for (std::size_t p0 = 0; p0 < n; p0 += bp_max) {
        const std::size_t bp = std::min(bp_max, n - p0);
        for (std::size_t k = 0; k < bp; ++k) slab(p0 + k, &gbatch[k * n3]);
        for (std::size_t c = 0; c < nc1; ++c) {
          const Component& cm = comps[s1][c];
          // R[b][(q r s)] += sum_p conj(C[mu0+b][p]) G_p[(q r s)]
          gemm(CblasNoTrans, CblasNoTrans, Bm, n3, bp, 1.0, &cm.Cc[mu0 * n + p0], n, gbatch.data(), n3, 1.0, R[c].data(), n3);
        }
        for (std::size_t k = 0; k < bp; ++k) {
          const cplx* g = &gbatch[k * n3];
          for (std::size_t q = 0; q < n; ++q)
            for (std::size_t r = 0; r < n; ++r)
              for (std::size_t s = 0; s < n; ++s) gperm[(r * n + q) * n + s] = g[(q * n + r) * n + s];
          for (std::size_t c = 0; c < nc1; ++c) {
            const Component& cm = comps[s1][c];
            // W[b][p][(q s)] = sum_r C[mu0+b][r] G_p[r][(q s)]
            gemm(CblasNoTrans, CblasNoTrans, Bm, n2, n, 1.0, &cm.C[mu0 * n], n, gperm.data(), n2, 0.0, &W[c][(p0 + k) * n2], n * n2);
          }
        }
      }
      ++result.passes;

      // ---- stage 2 and 3, one mu at a time ----
      for (std::size_t b = 0; b < Bm; ++b) {
        const std::size_t mu = mu0 + b;
        const std::size_t Lr = mu + 1;  // lambda <= mu
        std::vector<cplx> Y(Lr * n2, cplx{}), Rperm(n3);
        for (std::size_t c = 0; c < nc1; ++c) {
          const Component& cm = comps[s1][c];
          const cplx* Rb = &R[c][b * n3];
          for (std::size_t q = 0; q < n; ++q)
            for (std::size_t r = 0; r < n; ++r)
              for (std::size_t s = 0; s < n; ++s) Rperm[(r * n + q) * n + s] = Rb[(q * n + r) * n + s];
          gemm(CblasNoTrans, CblasNoTrans, Lr, n2, n, 0.5, cm.C.data(), n, Rperm.data(), n2, 1.0, Y.data(), n2);                   // 1/2 T1
          gemm(CblasNoTrans, CblasNoTrans, Lr, n2, n, 0.5, cm.Cc.data(), n, &W[c][b * n * n2], n2, 1.0, Y.data(), n2);            // 1/2 T2
        }
        std::vector<cplx> Yq(n * Lr * n);
        for (std::size_t la = 0; la < Lr; ++la)
          for (std::size_t q = 0; q < n; ++q)
            for (std::size_t s = 0; s < n; ++s) Yq[(q * Lr + la) * n + s] = Y[(la * n + q) * n + s];

        for (std::size_t s2 = 0; s2 < n_sec; ++s2) {
          if (s1 == 0 && s2 == 1) continue;  // nu in Small has a larger global index than mu in Large: these elements are the partners
          const std::size_t nu_total = (s2 == s1) ? mu + 1 : P[s2];
          const std::size_t P2 = P[s2];
          const std::size_t chunk_budget = std::max<std::size_t>(1, static_cast<std::size_t>((budget / 4.0) / (16.0 * static_cast<double>(Lr) * static_cast<double>(n + P2))));
          for (std::size_t nu0 = 0; nu0 < nu_total; nu0 += chunk_budget) {
            const std::size_t nuc = std::min(chunk_budget, nu_total - nu0);
            std::vector<cplx> E(nuc * Lr * P2, cplx{}), Bq(nuc * Lr * n);
            for (const Component& c2 : comps[s2]) {
              gemm(CblasNoTrans, CblasNoTrans, nuc, Lr * n, n, 1.0, &c2.Cc[nu0 * n], n, Yq.data(), Lr * n, 0.0, Bq.data(), Lr * n);
              gemm(CblasNoTrans, CblasTrans, nuc * Lr, P2, n, 1.0, Bq.data(), n, c2.C.data(), n, 1.0, E.data(), P2);
            }
            // ---- emit ----
            for (std::size_t nn = 0; nn < nuc; ++nn) {
              const std::size_t nu = nu0 + nn;
              for (std::size_t la = 0; la < Lr; ++la) {
                const double m_pr = (la == mu) ? 1.0 : 2.0;
                const std::int32_t gp = static_cast<std::int32_t>(first[s1] + mu + 1), gr = static_cast<std::int32_t>(first[s1] + la + 1);
                const std::int32_t gq = static_cast<std::int32_t>(first[s2] + nu + 1);
                const double s_pr = S[s1][mu * P1 + la];
                for (std::size_t sg = 0; sg <= nu; ++sg) {
                  // Re E[nu][la][sg] equals the real part of the fully symmetrized element 1/2 (E[nu][la][sg] + E[sg][la][nu]);
                  // the imaginary part of that symmetrized element vanishes (diagnostic, when row sg is in the chunk)
                  const cplx e_nu = E[(nn * Lr + la) * P2 + sg];
                  if (sg >= nu0) result.max_imag = std::max(result.max_imag, std::abs(0.5 * (e_nu.imag() + E[((sg - nu0) * Lr + la) * P2 + nu].imag())));
                  const cplx val_c = e_nu;
                  const double m_qs = (sg == nu) ? 1.0 : 2.0;
                  const double value = val_c.real() * m_pr * m_qs;
                  const double s_qs = S[s2][nu * P2 + sg];
                  const std::int32_t gs = static_cast<std::int32_t>(first[s2] + sg + 1);
                  const bool partner = !(s1 == s2 && nu == mu);
                  result.trace_dm2 += value * s_pr * s_qs * (partner ? 2.0 : 1.0);
                  if (check_vee) vee += 0.5 * value * prim_eri(gp - 1, gr - 1, gq - 1, gs - 1) * (partner ? 2.0 : 1.0);
                  if (std::abs(value) < settings.threshold) continue;
                  const std::int32_t idx[4] = {gp, gq, gr, gs};
                  w2.put(idx, 4, value);
                  ++result.n_dm2;
                  if (partner) {
                    const std::int32_t idx2[4] = {gq, gp, gs, gr};
                    w2.put(idx2, 4, value);
                    ++result.n_dm2;
                  }
                }
              }
            }
          }
        }
      }
      {
        std::ostringstream line;
        line << "(CC_RDM Gaussian) " << (s1 == 0 ? "Large" : "Small") << " rows " << mu0 + 1 << "-" << mu0 + Bm << " of " << P1 << " done";
        progress(line.str());
      }
    }
  }
  {
    const std::int32_t zero[4] = {0, 0, 0, 0};
    w2.put(zero, 4, 0.0);
    w2.close();
  }
  if (check_vee) result.vee_primitives = vee;
  result.seconds = elapsed();
  return result;
}

}  // namespace rerdmft
