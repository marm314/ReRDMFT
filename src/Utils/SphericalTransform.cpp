#include "SphericalTransform.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "Integrals.h"
#include "LinearAlgebra.h"
#include "Shell.h"

namespace rerdmft {

namespace {

// shellSelfOverlap(l, {1.0}, {1.0}), per-component normalized exactly as normalizeCartesianBasis
// does it (divide row/col k by sqrt(diagonal_k)) -- see this file's header comment for why the
// reference coefficient/exponent choice does not matter: any uniform rescaling cancels out of this
// per-component normalization exactly.
Matrix<double> referenceNormalizedOverlap(int l) {
  const Matrix<double> raw = shellSelfOverlap(l, {1.0}, {1.0});
  const std::size_t nf = raw.rows();
  std::vector<double> inv_sqrt_diag(nf);
  for (std::size_t k = 0; k < nf; ++k) inv_sqrt_diag[k] = 1.0 / std::sqrt(raw(k, k));
  Matrix<double> normalized(nf, nf);
  for (std::size_t i = 0; i < nf; ++i)
    for (std::size_t j = 0; j < nf; ++j) normalized(i, j) = raw(i, j) * inv_sqrt_diag[i] * inv_sqrt_diag[j];
  return normalized;
}

// The genuine, standard criterion for "real solid harmonic expressed in Cartesian monomials" is
// NOT a property of the overlap matrix (a Cartesian d shell's own 6x6 overlap is perfectly
// well-conditioned -- there is no near-zero eigenvalue to find there at all) -- it is that the
// polynomial is annihilated by the Laplacian (harmonic). d^2/dx^2 (x^a y^b z^c) = a(a-1) x^(a-2) y^b
// z^c (zero if a<2), similarly for y,z; summing the three gives the Laplacian's action on one
// Cartesian monomial, an exact, sparse, small-integer linear map from degree-l monomials
// (dimension (l+1)(l+2)/2) to degree-(l-2) monomials (dimension (l-1)l/2). Its null space has
// EXACTLY dimension 2l+1 (a standard fact: the Laplacian map degree-l -> degree-(l-2) is onto, so
// dim(kernel) = (l+1)(l+2)/2 - (l-1)l/2 = 2l+1), and null(L) = null(L^T L) (real L), so the null
// space is just the zero-eigenvalue eigenspace of the symmetric, positive-semidefinite L^T L --
// reusing diagonalizeSymmetric, no separate SVD code needed, and the zero eigenvalues are EXACT
// (machine precision), not approximate, since L itself has only small-integer entries.
Matrix<double> harmonicNullSpaceBasis(int l) {
  const std::vector<CartesianExponents> src = cartesianComponents(l);
  const std::vector<CartesianExponents> dst = cartesianComponents(l - 2);
  const std::size_t n_src = src.size(), n_dst = dst.size();
  Matrix<double> laplacian(n_dst, n_src, 0.0);
  for (std::size_t s = 0; s < n_src; ++s) {
    const CartesianExponents& c = src[s];
    const auto add = [&](int lx, int ly, int lz, double coeff) {
      if (coeff == 0.0) return;
      const int idx = cartesianComponentIndex(l - 2, CartesianExponents{lx, ly, lz});
      laplacian(static_cast<std::size_t>(idx), s) += coeff;
    };
    if (c.lx >= 2) add(c.lx - 2, c.ly, c.lz, static_cast<double>(c.lx * (c.lx - 1)));
    if (c.ly >= 2) add(c.lx, c.ly - 2, c.lz, static_cast<double>(c.ly * (c.ly - 1)));
    if (c.lz >= 2) add(c.lx, c.ly, c.lz - 2, static_cast<double>(c.lz * (c.lz - 1)));
  }

  Matrix<double> gram(n_src, n_src, 0.0);  // L^T L
  for (std::size_t i = 0; i < n_src; ++i)
    for (std::size_t j = 0; j < n_src; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n_dst; ++k) sum += laplacian(k, i) * laplacian(k, j);
      gram(i, j) = sum;
    }

  const SymmetricEigenResult eig = diagonalizeSymmetric(gram);
  const std::size_t n_harmonic = static_cast<std::size_t>(2 * l + 1);
  // Ascending order: the null space is exactly the first n_harmonic eigenvalues (should be ~0;
  // the rest are O(1) or larger -- an exact, not approximate, gap).
  Matrix<double> basis(n_src, n_harmonic);
  for (std::size_t j = 0; j < n_harmonic; ++j)
    for (std::size_t i = 0; i < n_src; ++i) basis(i, j) = eig.eigenvectors(i, j);
  return basis;
}

}  // namespace

Matrix<double> cartesianToSphericalBlock(int l) {
  static std::map<int, Matrix<double>> cache;
  const auto it = cache.find(l);
  if (it != cache.end()) return it->second;

  const std::size_t nf = cartesianComponents(l).size();
  const std::size_t n_sph = static_cast<std::size_t>(2 * l + 1);
  if (nf == n_sph) {
    // l=0,1: already exactly spherical, no redundancy to remove -- skip the reference
    // overlap/canonicalOrthogonalize round trip entirely (and avoid any machine-precision noise
    // it would otherwise introduce for a case that needs none).
    Matrix<double> identity(nf, nf, 0.0);
    for (std::size_t i = 0; i < nf; ++i) identity(i, i) = 1.0;
    cache.emplace(l, identity);
    return identity;
  }

  // N: (nf x n_sph), an orthogonal (Euclidean-coefficient sense) basis of the degree-l harmonic
  // subspace -- genuine angular-momentum purity, not yet normalized under the physical metric.
  const Matrix<double> n_basis = harmonicNullSpaceBasis(l);
  const Matrix<double> reference = referenceNormalizedOverlap(l);

  // Restrict the physical overlap metric to this subspace: metric_sub = N^T S N (n_sph x n_sph).
  // This submatrix is expected to be full rank (genuine spherical harmonics have no redundancy among
  // themselves) -- canonicalOrthogonalize is used anyway as a safety net, with a loose threshold
  // (this is a well-conditioned, small, physically-meaningful metric, nothing to legitimately drop).
  Matrix<double> metric_sub(n_sph, n_sph, 0.0);
  for (std::size_t i = 0; i < n_sph; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      double sum = 0.0;
      for (std::size_t a = 0; a < nf; ++a)
        for (std::size_t b = 0; b < nf; ++b) sum += n_basis(a, i) * reference(a, b) * n_basis(b, j);
      metric_sub(i, j) = sum;
    }
  RankReductionReport report;
  const Matrix<double> lowdin = canonicalOrthogonalize(metric_sub, 1e-8, &report);
  if (report.n_kept != n_sph) {
    throw std::runtime_error("cartesianToSphericalBlock: l=" + std::to_string(l) +
                              ": the physical metric restricted to the harmonic subspace is rank-"
                              "deficient (" +
                              std::to_string(report.n_kept) + " of " + std::to_string(n_sph) +
                              ") -- this should never happen for genuine spherical harmonics, "
                              "indicates a bug");
  }

  // Final transform: T = N * lowdin (nf x n_sph), i.e. express each Loewdin-orthonormalized
  // combination (lowdin's columns, in the N-basis) back in the original Cartesian monomial basis.
  Matrix<double> block(nf, n_sph, 0.0);
  for (std::size_t i = 0; i < nf; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n_sph; ++k) sum += n_basis(i, k) * lowdin(k, j);
      block(i, j) = sum;
    }
  cache.emplace(l, block);
  return block;
}

SphericalTransformResult buildSphericalTransform(const std::vector<BasisFunction>& cartesian_basis) {
  SphericalTransformResult result;
  std::size_t i = 0;
  // First pass: figure out total spherical dimension and per-shell layout.
  while (i < cartesian_basis.size()) {
    const int l = cartesian_basis[i].l;
    const std::size_t nf = cartesianComponents(l).size();
    SphericalShellLayout layout;
    layout.l = l;
    layout.cartesian_offset = i;
    layout.spherical_offset = result.n_spherical;
    result.shells.push_back(layout);
    result.n_spherical += static_cast<std::size_t>(2 * l + 1);
    i += nf;
  }

  result.transform = Matrix<double>(cartesian_basis.size(), result.n_spherical, 0.0);
  for (const SphericalShellLayout& shell : result.shells) {
    const Matrix<double> block = cartesianToSphericalBlock(shell.l);
    for (std::size_t a = 0; a < block.rows(); ++a)
      for (std::size_t b = 0; b < block.cols(); ++b)
        result.transform(shell.cartesian_offset + a, shell.spherical_offset + b) = block(a, b);
  }
  return result;
}

Matrix<double> transformToSpherical(const Matrix<double>& m, const Matrix<double>& transform) {
  const std::size_t n_cart = transform.rows(), n_sph = transform.cols();
  if (m.rows() != n_cart || m.cols() != n_cart) {
    throw std::runtime_error("transformToSpherical: matrix dimension does not match the transform's cartesian side");
  }
  // mt = M * T (n_cart x n_sph), then T^T * mt (n_sph x n_sph).
  Matrix<double> mt(n_cart, n_sph, 0.0);
  for (std::size_t i = 0; i < n_cart; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n_cart; ++k) sum += m(i, k) * transform(k, j);
      mt(i, j) = sum;
    }
  Matrix<double> result(n_sph, n_sph, 0.0);
  for (std::size_t i = 0; i < n_sph; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n_cart; ++k) sum += transform(k, i) * mt(k, j);
      result(i, j) = sum;
    }
  return result;
}

Matrix<std::complex<double>> transformToSpherical(const Matrix<std::complex<double>>& m,
                                                   const Matrix<double>& transform) {
  const std::size_t n_cart = transform.rows(), n_sph = transform.cols();
  if (m.rows() != n_cart || m.cols() != n_cart) {
    throw std::runtime_error("transformToSpherical: matrix dimension does not match the transform's cartesian side");
  }
  Matrix<std::complex<double>> mt(n_cart, n_sph, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_cart; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < n_cart; ++k) sum += m(i, k) * transform(k, j);
      mt(i, j) = sum;
    }
  Matrix<std::complex<double>> result(n_sph, n_sph, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_sph; ++i)
    for (std::size_t j = 0; j < n_sph; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (std::size_t k = 0; k < n_cart; ++k) sum += transform(k, i) * mt(k, j);
      result(i, j) = sum;
    }
  return result;
}

Matrix<std::complex<double>> spinDuplicateComplex(const Matrix<double>& transform) {
  const std::size_t n_cart = transform.rows(), n_sph = transform.cols();
  Matrix<std::complex<double>> out(2 * n_cart, 2 * n_sph, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_cart; ++i) {
    for (std::size_t j = 0; j < n_sph; ++j) {
      const std::complex<double> value(transform(i, j), 0.0);
      out(i, j) = value;
      out(n_cart + i, n_sph + j) = value;
    }
  }
  return out;
}

Matrix<std::complex<double>> sphericalLargeEmbedding(const Matrix<double>& transform,
                                                      std::size_t n_small2) {
  const Matrix<std::complex<double>> t_spin = spinDuplicateComplex(transform);
  const std::size_t n_large2_cart = t_spin.rows(), n_large2_sph = t_spin.cols();
  Matrix<std::complex<double>> v(n_large2_cart + n_small2, n_large2_sph + n_small2,
                                 std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < n_large2_cart; ++i)
    for (std::size_t j = 0; j < n_large2_sph; ++j) v(i, j) = t_spin(i, j);
  for (std::size_t i = 0; i < n_small2; ++i) v(n_large2_cart + i, n_large2_sph + i) = std::complex<double>(1.0, 0.0);
  return v;
}

}  // namespace rerdmft
