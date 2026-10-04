#include "RkbCholesky.h"

#include <cblas.h>

#include <omp.h>

#include <iomanip>
#include <stdexcept>

#include "BlasThreads.h"
#include "Progress.h"
#include "ElectronRepulsion.h"
#include "LinearAlgebra.h"
#include "RkbTwoElectron.h"
#include "SphericalTransform.h"

namespace rerdmft {

namespace {

using C = std::complex<double>;

// The real AO Coulomb matrix over the pair index {LL pairs (nl^2)} u {SS pairs (ns^2)}, blocks
// (LL|LL), (LL|SS), (SS|SS), presented to the pivoted decomposition without ever being assembled
// into the UNION (there is no dense/packed n2 x n2 array either way) -- but (LL|LL)/(LL|SS)/(SS|SS)
// THEMSELVES are each pre-built once, up front, in packed form. Fast (every quadruplet evaluated
// exactly once, however many times the decomposition later reads it) but needs O(n_small^4/8)
// memory for (SS|SS) -- use ONLY when kMaxFastBytes (RkbCholesky::build) says that fits; otherwise
// OnDemandUnionCoulombPairs below.
class UnionCoulombPairs : public PairMatrixSource<double> {
 public:
  // (LL|LL) and (SS|SS) are the 8-fold-packed real AO tensors; the cross block (LL|SS) is the
  // 4-fold-packed CrossPackedTwoElectronTensor (p<->q and r<->s symmetry, no (pq)<->(rs) swap --
  // see its own comment) -- none of the three is ever the dense n^4 array.
  UnionCoulombPairs(const PackedTwoElectronTensor& ll_ll, const CrossPackedTwoElectronTensor& ll_ss,
                     const PackedTwoElectronTensor& ss_ss)
      : ll_ll_(ll_ll), ll_ss_(ll_ss), ss_ss_(ss_ss), nl_(ll_ll.dim()), ns_(ss_ss.dim()) {}

  std::size_t size() const override { return nl_ * nl_ + ns_ * ns_; }
  double diagonal(std::size_t i) const override { return at(i, i); }
  double at(std::size_t i, std::size_t j) const override {
    const std::size_t nl2 = nl_ * nl_;
    if (i < nl2) {
      if (j < nl2) return ll_ll_(i / nl_, i % nl_, j / nl_, j % nl_);
      const std::size_t jj = j - nl2;
      return ll_ss_(i / nl_, i % nl_, jj / ns_, jj % ns_);
    }
    const std::size_t ii = i - nl2;
    if (j < nl2) return ll_ss_(j / nl_, j % nl_, ii / ns_, ii % ns_);
    const std::size_t jj = j - nl2;
    return ss_ss_(ii / ns_, ii % ns_, jj / ns_, jj % ns_);
  }
  void row(std::size_t i, double* out) const override {
    const std::size_t nl2 = nl_ * nl_, ns2 = ns_ * ns_, total = nl2 + ns2;
    for (std::size_t j = 0; j < total; ++j) out[j] = at(i, j);
    (void)ns2;
  }

 private:
  const PackedTwoElectronTensor& ll_ll_;
  const CrossPackedTwoElectronTensor& ll_ss_;
  const PackedTwoElectronTensor& ss_ss_;
  std::size_t nl_, ns_;
};

// Same pair matrix as UnionCoulombPairs above, but with NEITHER the dense nor the packed
// representation of (LL|LL)/(LL|SS)/(SS|SS) EVER materialized: every (pq|rs) quadruplet is instead
// evaluated lazily, via twoElectronQuadruplet (ElectronRepulsion.h), each time the decomposition
// asks for it. Needed because the raw analytic RKB small basis (RkbDerivativeTerms.h) is NOT
// deduplicated across large functions the way the old unrestricted-kinetic-balance basis was, so
// n_small can run into the hundreds even for a single heavy atom -- (SS|SS)'s packed storage alone
// would already be tens of GB there (confirmed: 540 small functions for Xe/dyall.v2z, ~85 GB),
// long before the O(n_small) Cholesky-vector storage the decomposition itself actually needs.
// Schwarz screening still applies (same sqrt-diagonal bound as before), precomputed once up front
// (O(n^2), trivial memory either way). The real cost of avoiding the upfront build: nothing here is
// cached between calls, so a quadruplet needed by several different `row()` calls (one per pivot)
// is re-evaluated from scratch every time -- measurably slower than UnionCoulombPairs for any case
// that could have afforded the fast path instead, which is exactly why RkbCholesky::build only
// reaches for this when the fast path's own memory estimate says it would not fit.
class OnDemandUnionCoulombPairs : public PairMatrixSource<double> {
 public:
  OnDemandUnionCoulombPairs(const std::vector<BasisFunction>& large_basis,
                            const std::vector<BasisFunction>& small_basis, double threshold)
      : large_(large_basis),
        small_(small_basis),
        nl_(large_basis.size()),
        ns_(small_basis.size()),
        threshold_(threshold),
        sqrt_diag_ll_(threshold > 0.0 ? sqrtPairDiagonal(large_basis) : std::vector<double>()),
        sqrt_diag_ss_(threshold > 0.0 ? sqrtPairDiagonal(small_basis) : std::vector<double>()) {}

  std::size_t size() const override { return nl_ * nl_ + ns_ * ns_; }
  double diagonal(std::size_t i) const override { return at(i, i); }

  double at(std::size_t i, std::size_t j) const override {
    const std::size_t nl2 = nl_ * nl_;
    const bool i_large = i < nl2;
    const bool j_large = j < nl2;
    const std::size_t pq = i_large ? i : i - nl2;
    const std::size_t rs = j_large ? j : j - nl2;
    const std::size_t p = pq / (i_large ? nl_ : ns_), q = pq % (i_large ? nl_ : ns_);
    const std::size_t r = rs / (j_large ? nl_ : ns_), s = rs % (j_large ? nl_ : ns_);
    if (threshold_ > 0.0 &&
        sqrtDiag(i_large, p, q) * sqrtDiag(j_large, r, s) < threshold_) {
      return 0.0;
    }
    return twoElectronQuadruplet(basisFn(i_large, p), basisFn(i_large, q), basisFn(j_large, r),
                                  basisFn(j_large, s));
  }

  void row(std::size_t i, double* out) const override {
    const std::size_t nl2 = nl_ * nl_, total = nl2 + ns_ * ns_;
    const bool i_large = i < nl2;
    const std::size_t pq = i_large ? i : i - nl2;
    const std::size_t p = pq / (i_large ? nl_ : ns_), q = pq % (i_large ? nl_ : ns_);
    const BasisFunction& fp = basisFn(i_large, p);
    const BasisFunction& fq = basisFn(i_large, q);
    const double sqrt_diag_i = sqrtDiag(i_large, p, q);
#pragma omp parallel for schedule(dynamic)
    for (std::size_t j = 0; j < total; ++j) {
      const bool j_large = j < nl2;
      const std::size_t rs = j_large ? j : j - nl2;
      const std::size_t r = rs / (j_large ? nl_ : ns_), s = rs % (j_large ? nl_ : ns_);
      if (threshold_ > 0.0 && sqrt_diag_i * sqrtDiag(j_large, r, s) < threshold_) {
        out[j] = 0.0;
        continue;
      }
      out[j] = twoElectronQuadruplet(fp, fq, basisFn(j_large, r), basisFn(j_large, s));
    }
  }

 private:
  const BasisFunction& basisFn(bool is_large, std::size_t idx) const {
    return is_large ? large_[idx] : small_[idx];
  }
  double sqrtDiag(bool is_large, std::size_t p, std::size_t q) const {
    if (threshold_ <= 0.0) return 0.0;
    const std::size_t lo = p < q ? p : q, hi = p < q ? q : p;
    const std::vector<double>& table = is_large ? sqrt_diag_ll_ : sqrt_diag_ss_;
    return table[hi * (hi + 1) / 2 + lo];
  }

  const std::vector<BasisFunction>& large_;
  const std::vector<BasisFunction>& small_;
  std::size_t nl_, ns_;
  double threshold_;
  std::vector<double> sqrt_diag_ll_, sqrt_diag_ss_;
};

void zgemm(bool conj_trans_a, int m, int n, int k, const C* a, int lda, const C* b, int ldb, double beta, C* c,
           int ldc) {
  const C alpha(1.0, 0.0), bt(beta, 0.0);
  cblas_zgemm(CblasRowMajor, conj_trans_a ? CblasConjTrans : CblasNoTrans, CblasNoTrans, m, n, k, &alpha, a, lda, b,
              ldb, &bt, c, ldc);
}

Matrix<C> promote(const Matrix<double>& m) {
  Matrix<C> out(m.rows(), m.cols());
  for (std::size_t i = 0; i < m.rows() * m.cols(); ++i) out.data()[i] = C(m.data()[i], 0.0);
  return out;
}

}  // namespace

RkbCholesky RkbCholesky::build(const std::vector<BasisFunction>& large_basis,
                               const std::vector<BasisFunction>& small_basis,
                               const Matrix<C>& rkb_coefficients, const Matrix<double>& large_transform,
                               double threshold, bool on_demand, CholeskyCheckReport* report) {
  const std::size_t nl = large_basis.size(), ns = small_basis.size();
  const std::size_t nl_sph = large_transform.cols();
  FlatVectors<double> flat;
  {
    // ON_DEMAND_ERI (Input.h) selects between the fast path (UnionCoulombPairs: every quadruplet
    // evaluated exactly once, into packed (LL|LL)/(LL|SS)/(SS|SS) tensors, but O(n_small^4/8) memory
    // for (SS|SS)) and the on-demand one (OnDemandUnionCoulombPairs: no AO tensor ever materialized,
    // but slower -- see its own comment). Default TRUE (on-demand): makes the std::bad_alloc this
    // basis once hit (Xe/dyall.v2z's (SS|SS) alone needs ~85 GB) structurally impossible by default.
    if (!on_demand) {
      // Transient: released as soon as the decomposition is done. About a 4-8x reduction over the
      // dense nl^2 x ns^2 array either way (PackedTwoElectronTensor/CrossPackedTwoElectronTensor),
      // since the decomposition itself only ever reads individual elements/rows (PairMatrixSource),
      // never needs the dense layout.
      progress("RKB Cholesky: real AO integrals (LL|LL), (LL|SS), (SS|SS)");
      // Schwarz-prescreened at the SAME threshold the decomposition itself targets: an element
      // this build drops was already going to contribute at most `threshold` of error to the
      // reconstruction, which is exactly the accuracy the pivoted decomposition's own stopping
      // rule (dmax < threshold) already budgets for -- see Cholesky_Decomposition.h's own comment
      // on why the largest diagonal residual bounds every off-diagonal one.
      const PackedTwoElectronTensor ll_ll = twoElectronIntegralsPacked(large_basis, threshold);
      const CrossPackedTwoElectronTensor ll_ss =
          twoElectronIntegralsCrossPacked(large_basis, small_basis, threshold);
      const PackedTwoElectronTensor ss_ss = twoElectronIntegralsPacked(small_basis, threshold);
      ProgressLine() << "RKB Cholesky: AO integrals done; decomposing the {LL} u {SS} pair matrix (dimension "
                     << nl * nl + ns * ns << ")";
      const UnionCoulombPairs pairs(ll_ll, ll_ss, ss_ss);
      flat = choleskyDecomposePairsChecked<double>(pairs, threshold, report);
    } else {
      ProgressLine() << "RKB Cholesky: ON_DEMAND_ERI TRUE -- decomposing the {LL} u {SS} pair matrix "
                        "(dimension "
                     << nl * nl + ns * ns << ") on demand, with no AO tensor ever built";
      const OnDemandUnionCoulombPairs pairs(large_basis, small_basis, threshold);
      flat = choleskyDecomposePairsChecked<double>(pairs, threshold, report);
    }
  }
  ProgressLine() << "RKB Cholesky: " << flat.count << " vectors; projecting them into the RKB basis";
  RkbCholesky out;
  out.n_large = nl_sph;
  out.large.resize(flat.count);
  out.small[0].resize(flat.count);
  out.small[1].resize(flat.count);
#pragma omp parallel for schedule(dynamic)
  for (std::size_t l = 0; l < flat.count; ++l) {
    const double* row = flat.data.data() + l * flat.pair_dim;
    Matrix<double> b(nl, nl), s(ns, ns);
    std::copy(row, row + nl * nl, b.data());
    std::copy(row + nl * nl, row + nl * nl + ns * ns, s.data());
    // Large-component Cartesian-to-spherical reduction, same transform/reasoning as the one-electron
    // H_RKB side (main.cpp's X2C/C4_SPINOR construction block) -- B'_L = T^T B_L T.
    out.large[l] = transformToSpherical(b, large_transform);
    for (std::size_t y = 0; y < 2; ++y)
      out.small[y][l] = rkbProjectSmallVector(s, rkb_coefficients, y, nl_sph, ns);
  }
  return out;
}

Matrix<C> rkbFockMatrix(const Matrix<C>& h_rkb, const RkbCholesky& ao, const Matrix<C>& p) {
  const std::size_t n = h_rkb.rows();
  if (n % 4 != 0) throw std::runtime_error("rkbFockMatrix: RKB basis dimension must be a multiple of 4");
  const std::size_t nl = n / 4;
  if (ao.n_large != nl) throw std::runtime_error("rkbFockMatrix: RKB Cholesky dimension does not match H_RKB");
  if (p.rows() != n || p.cols() != n) throw std::runtime_error("rkbFockMatrix: density dimensions inconsistent");
  Matrix<C> pb[4][4];
  for (int f = 0; f < 4; ++f)
    for (int g = 0; g < 4; ++g) {
      pb[f][g] = Matrix<C>(nl, nl);
      for (std::size_t a = 0; a < nl; ++a)
        for (std::size_t b = 0; b < nl; ++b) pb[f][g](a, b) = p(f * nl + a, g * nl + b);
    }
  Matrix<C> jm[4], km[4][4];
  for (int f = 0; f < 4; ++f) {
    jm[f] = Matrix<C>(nl, nl, C{});
    for (int g = 0; g < 4; ++g) km[f][g] = Matrix<C>(nl, nl, C{});
  }
  struct Part {
    Matrix<C> j[4];
    Matrix<C> k[4][4];
  };
  const int max_threads = omp_get_max_threads();
  std::vector<Part> parts(static_cast<std::size_t>(max_threads));  // per-thread partial sums, merged in thread order
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    Part& part = parts[static_cast<std::size_t>(omp_get_thread_num())];
    Matrix<C> bf[4], tmp(nl, nl);
    for (int f = 0; f < 4; ++f) {
      part.j[f] = Matrix<C>(nl, nl, C{});
      for (int g = 0; g < 4; ++g) part.k[f][g] = Matrix<C>(nl, nl, C{});
    }
#pragma omp for schedule(static)
    for (std::size_t l = 0; l < ao.nVectors(); ++l) {
      bf[0] = promote(ao.large[l]);
      bf[1] = bf[0];
      bf[2] = ao.small[0][l];
      bf[3] = ao.small[1][l];
      C t{};  // tr( sum_f B^f P_ff )
      for (int f = 0; f < 4; ++f)
        for (std::size_t q = 0; q < nl; ++q)
          for (std::size_t s = 0; s < nl; ++s) t += bf[f](q, s) * pb[f][f](s, q);
      for (int f = 0; f < 4; ++f)
        for (std::size_t i = 0; i < nl * nl; ++i) part.j[f].data()[i] += t * bf[f].data()[i];
      for (int f = 0; f < 4; ++f)
        for (int g = 0; g < 4; ++g) {
          zgemm(false, static_cast<int>(nl), static_cast<int>(nl), static_cast<int>(nl), pb[f][g].data(),
                static_cast<int>(nl), bf[g].data(), static_cast<int>(nl), 0.0, tmp.data(), static_cast<int>(nl));  // P_fg B^g
          zgemm(false, static_cast<int>(nl), static_cast<int>(nl), static_cast<int>(nl), bf[f].data(),
                static_cast<int>(nl), tmp.data(), static_cast<int>(nl), 1.0, part.k[f][g].data(), static_cast<int>(nl));  // += B^f P_fg B^g
        }
    }
  }
  for (const Part& part : parts) {
    if (part.j[0].rows() == 0) continue;
    for (int f = 0; f < 4; ++f) {
      for (std::size_t i = 0; i < nl * nl; ++i) jm[f].data()[i] += part.j[f].data()[i];
      for (int g = 0; g < 4; ++g)
        for (std::size_t i = 0; i < nl * nl; ++i) km[f][g].data()[i] += part.k[f][g].data()[i];
    }
  }
  Matrix<C> fock = h_rkb;
  for (int f = 0; f < 4; ++f)
    for (std::size_t a = 0; a < nl; ++a)
      for (std::size_t b = 0; b < nl; ++b) {
        fock(f * nl + a, f * nl + b) += jm[f](a, b);
        for (int g = 0; g < 4; ++g) fock(f * nl + a, g * nl + b) -= km[f][g](a, b);
      }
  return fock;
}

CholeskyEri<C> rkbCholeskyToMo(const RkbCholesky& ao, const Matrix<C>& c, std::size_t n_negative, double threshold) {
  const std::size_t nl = ao.n_large, nmo = c.cols();
  if (c.rows() != 4 * nl) throw std::runtime_error("rkbCholeskyToMo: C must have 4*n_large rows");
  // Transform directly into the (nn:nmo, nn:nmo) positive-energy block instead of building the full nmo x nmo
  // W_L and slicing afterward: W_L(p,q) = sum_f C_f(:,p)^+ B^f_L C_f(:,q) depends only on columns p and q of C,
  // never on any other column, so restricting every C_f used below to its columns [nn, nmo) (via a pointer
  // offset + the UNCHANGED leading dimension nmo, i.e. a strided view, no copy) gives bit-identical results to
  // computing the full matrix first -- just without the n_negative rows/columns that no-pair discards right
  // after. With nn = nmo/2 (the default no-pair case) this is a 4x reduction in both the per-vector GEMM cost
  // and the O(N_chol * nmo^2) `out` storage below, which otherwise dwarfs everything else in memory for a
  // heavy element's large RKB dimension. nn = 0 (the untrimmed FULL_OPTIMIZATION_4C_NEG factory call) makes
  // m = nmo, reducing exactly to the old unrestricted transform.
  const std::size_t nn = n_negative, m = nmo - nn;
  std::vector<Matrix<C>> out(ao.nVectors());
  const SerialBlasScope serial_blas_guard;
#pragma omp parallel
  {
    Matrix<C> bc, tmp(nl, m), bp(m, m);
#pragma omp for schedule(dynamic)
    for (std::size_t l = 0; l < ao.nVectors(); ++l) {
      for (int f = 0; f < 4; ++f) {
        const Matrix<C>* bf;
        Matrix<C> large_c;
        if (f < 2) {
          large_c = promote(ao.large[l]);
          bf = &large_c;
        } else {
          bf = &ao.small[static_cast<std::size_t>(f - 2)][l];
        }
        const C* cf = c.data() + static_cast<std::size_t>(f) * nl * nmo + nn;  // flavor block, columns [nn, nmo), ld=nmo
        zgemm(false, static_cast<int>(nl), static_cast<int>(m), static_cast<int>(nl), bf->data(), static_cast<int>(nl),
              cf, static_cast<int>(nmo), 0.0, tmp.data(), static_cast<int>(m));                          // B^f C_f(:,nn:)
        zgemm(true, static_cast<int>(m), static_cast<int>(m), static_cast<int>(nl), cf, static_cast<int>(nmo),
              tmp.data(), static_cast<int>(m), f == 0 ? 0.0 : 1.0, bp.data(), static_cast<int>(m));   // += C_f(:,nn:)^+ B^f C_f(:,nn:)
      }
      Matrix<C> w(m, m);
      for (std::size_t x = 0; x < m; ++x)
        for (std::size_t y = 0; y < m; ++y) w(x, y) = bp(y, x);  // W = B'^T
      out[l] = std::move(w);
    }
  }
  if (nn == 0) return CholeskyEri<C>::fromVectors(out);

  // Recompress: rows A_L = the m^2 entries of W_L (already the positive-energy block, nothing left to slice);
  // M = A^T conj(A) is unchanged by any unitary mixing of the rows, so diagonalize the Gram matrix
  // G = A A^dagger (G u_k = lambda_k u_k) and keep A'_k = sum_L conj(u_k(L)) A_L for lambda_k > tau:
  // dropping the others changes every element of M by at most the sum of their eigenvalues.
  const std::size_t nchol = out.size();
  Matrix<C> a(nchol, m * m);
  for (std::size_t l = 0; l < nchol; ++l)
    for (std::size_t x = 0; x < m; ++x)
      for (std::size_t y = 0; y < m; ++y) a(l, x * m + y) = out[l](x, y);
  Matrix<C> gram(nchol, nchol);
  {
    const C alpha(1.0, 0.0), beta(0.0, 0.0);
    // G = A A^dagger via zgemm (A: nchol x m^2)
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, static_cast<int>(nchol), static_cast<int>(nchol),
                static_cast<int>(m * m), &alpha, a.data(), static_cast<int>(m * m), a.data(), static_cast<int>(m * m),
                &beta, gram.data(), static_cast<int>(nchol));
  }
  const HermitianEigenResult eig = diagonalizeHermitian(gram);
  const double tau = threshold / static_cast<double>(std::max<std::size_t>(1, nchol));
  std::vector<std::size_t> keep;
  for (std::size_t k = 0; k < nchol; ++k)
    if (eig.eigenvalues[k] > tau) keep.push_back(k);
  if (keep.empty()) throw std::runtime_error("rkbCholeskyToMo: no Cholesky direction survives the compression");
  std::vector<Matrix<C>> compressed(keep.size());
  const int nk = static_cast<int>(keep.size());
  // A'(k,:) = sum_L conj(u_k(L)) A(L,:)  ==  (conj(U)^T A) with U = eigenvectors (columns u_k)
  Matrix<C> uc(nchol, keep.size());
  for (std::size_t l = 0; l < nchol; ++l)
    for (std::size_t k = 0; k < keep.size(); ++k) uc(l, k) = std::conj(eig.eigenvectors(l, keep[k]));
  Matrix<C> ap(keep.size(), m * m);
  {
    const C alpha(1.0, 0.0), beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nk, static_cast<int>(m * m), static_cast<int>(nchol), &alpha,
                uc.data(), nk, a.data(), static_cast<int>(m * m), &beta, ap.data(), static_cast<int>(m * m));
  }
  for (std::size_t k = 0; k < keep.size(); ++k) {
    Matrix<C> w(nmo, nmo, C{});  // zero-padded: the negative branch keeps its (zero) rows/columns
    for (std::size_t x = 0; x < m; ++x)
      for (std::size_t y = 0; y < m; ++y) w(nn + x, nn + y) = ap(k, x * m + y);
    compressed[k] = std::move(w);
  }
  return CholeskyEri<C>::fromVectors(compressed);
}

}  // namespace rerdmft
