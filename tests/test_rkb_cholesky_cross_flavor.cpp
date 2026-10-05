// Validates the cross-RKB-flavor J fix applied to RkbCholesky.cpp (RkbCholesky::build's new
// small_cross vectors, rkbFockMatrix(h_rkb,RkbCholesky,p)'s new jm_cross output, and
// rkbCholeskyToMo's new cross-flavor MO contraction) against the already-fixed, already-validated
// dense RkbTwoElectronTensor-based path (rkbTwoElectronIntegrals + rkbFockMatrix(h_rkb,eri,p)), on
// LiH/6-31G -- the same system that exposed the original missing-cross-term bug.
// Build/run: make test_rkb_cholesky_cross_flavor
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "BasisSet.h"
#include "Input.h"
#include "Integrals.h"
#include "KramersPairing.h"
#include "LinearAlgebra.h"
#include "Matrix.h"
#include "MolecularBasis.h"
#include "RkbCholesky.h"
#include "RkbFockMatrix.h"
#include "RkbHamiltonian.h"
#include "RkbMoTransform.h"
#include "RkbOverlap.h"
#include "RkbTransformation.h"
#include "RkbTwoElectron.h"
#include "SmallComponentBasis.h"
#include "SphericalTransform.h"
#include "UkbHamiltonian.h"

using namespace rerdmft;

namespace {

int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

constexpr double kAngstromToBohr = 1.0 / 0.52917721067;
using C = std::complex<double>;

void runSystem(const std::string& name, std::vector<BasisFunction> large_cart, const std::vector<Atom>& geometry,
               std::size_t n_electrons) {
  normalizeCartesianBasis(large_cart);

  SmallComponentBasis small_basis;
  small_basis.build(large_cart);
  const std::vector<BasisFunction>& small_cart = small_basis.functions();

  const Matrix<C> rkb_coeff_cart = rkbCoefficients(large_cart, small_basis.termIndex());
  const SphericalTransformResult large_spherical = buildSphericalTransform(large_cart);
  const Matrix<double>& large_transform_final = large_spherical.transform;
  const Matrix<C> rkb_coeff_final = dagger(spinDuplicateComplex(large_transform_final)) * rkb_coeff_cart;

  const Matrix<C> h_ukb_cart = ukbHamiltonianMatrix(large_cart, small_cart, geometry);
  const Matrix<C> v_sph = sphericalLargeEmbedding(large_transform_final, 2 * small_cart.size());
  const Matrix<C> h_ukb_final = dagger(v_sph) * (h_ukb_cart * v_sph);
  const Matrix<C> h_rkb = rkbHamiltonianMatrix(h_ukb_final, rkb_coeff_final);
  const std::size_t n_rkb = h_rkb.rows();
  const std::size_t nl_final = large_transform_final.cols();

  // Dense reference path (already fixed and validated against ukbFockTwoElectronDirect).
  const RkbTwoElectronTensor eri =
      rkbTwoElectronIntegrals(large_cart, small_cart, rkb_coeff_final, large_transform_final, /*use_cholesky=*/false);

  // RkbCholesky path under test (use_cholesky semantics mirrored via the dedicated RkbCholesky::build).
  const RkbCholesky ao_cholesky =
      RkbCholesky::build(large_cart, small_cart, rkb_coeff_final, large_transform_final, /*threshold=*/1e-10,
                          /*on_demand=*/false);
  std::printf("[%s] n_large_final=%zu n_rkb=%zu n_cholesky_vectors=%zu\n", name.c_str(), nl_final, n_rkb,
              ao_cholesky.nVectors());

  // Genuinely Kramers-paired, natural-orbital density from H_RKB's own eigenvectors (same construction
  // as tests/test_ukb_fock_direct.cpp) -- idempotent (HF-like) case.
  const Matrix<double> s_large = transformToSpherical(overlapMatrix(large_cart), large_transform_final);
  const Matrix<C> s_small = rkbSmallOverlapMatrix(small_cart, rkb_coeff_final);
  Matrix<C> s_rkb(n_rkb, n_rkb, C{});
  for (std::size_t i = 0; i < nl_final; ++i)
    for (std::size_t j = 0; j < nl_final; ++j) {
      s_rkb(i, j) = C(s_large(i, j), 0.0);
      s_rkb(nl_final + i, nl_final + j) = C(s_large(i, j), 0.0);
    }
  for (std::size_t i = 0; i < 2 * nl_final; ++i)
    for (std::size_t j = 0; j < 2 * nl_final; ++j) s_rkb(2 * nl_final + i, 2 * nl_final + j) = s_small(i, j);

  const Matrix<C> x = canonicalOrthogonalizeHermitian(s_rkb, 1e-10);
  const Matrix<C> h_ortho = dagger(x) * (h_rkb * x);
  const HermitianEigenResult eig = diagonalizeHermitian(h_ortho);
  Matrix<C> c_mo(x.rows(), x.cols(), C{});
  for (std::size_t a = 0; a < x.cols(); ++a)
    for (std::size_t i = 0; i < x.rows(); ++i) {
      C val{};
      for (std::size_t k = 0; k < x.cols(); ++k) val += x(i, k) * eig.eigenvectors(k, a);
      c_mo(i, a) = val;
    }
  const std::size_t n_neg = n_rkb / 2;
  std::vector<double> occ(n_rkb, 0.0);
  for (std::size_t k = 0; k < n_electrons / 2; ++k) occ[n_neg + 2 * k] = occ[n_neg + 2 * k + 1] = 1.0;
  Matrix<C> p_rkb(n_rkb, n_rkb, C{});
  for (std::size_t i = 0; i < n_rkb; ++i)
    for (std::size_t j = 0; j < n_rkb; ++j) {
      C s{};
      for (std::size_t a = 0; a < n_rkb; ++a)
        if (occ[a] != 0.0) s += occ[a] * c_mo(i, a) * std::conj(c_mo(j, a));
      p_rkb(i, j) = s;
    }

  const Matrix<C> f_dense = rkbFockMatrix(h_rkb, eri, p_rkb);
  const Matrix<C> f_cholesky = rkbFockMatrix(h_rkb, ao_cholesky, p_rkb);
  double worst_fock = 0.0;
  std::size_t worst_i = 0, worst_j = 0;
  for (std::size_t i = 0; i < f_dense.rows(); ++i)
    for (std::size_t j = 0; j < f_dense.cols(); ++j) {
      const double d = std::abs(f_dense(i, j) - f_cholesky(i, j));
      if (d > worst_fock) {
        worst_fock = d;
        worst_i = i;
        worst_j = j;
      }
    }
  const std::size_t block_size = n_rkb / 4;
  std::printf("[%s] idempotent density: max|F_dense-F_cholesky|=%.3e at (%zu,%zu) [block (%zu,%zu) of 4], "
              "f_dense=%.6e%+.6ei f_cholesky=%.6e%+.6ei\n",
              name.c_str(), worst_fock, worst_i, worst_j, worst_i / block_size, worst_j / block_size,
              f_dense(worst_i, worst_j).real(), f_dense(worst_i, worst_j).imag(), f_cholesky(worst_i, worst_j).real(),
              f_cholesky(worst_i, worst_j).imag());
  // Noise-floor threshold, not 1e-8: h_rkb's own ~1e5 dynamic range (the negative-energy branch's
  // -2c^2-scale diagonal) propagates into the raw AO-basis Fock regardless of which two-electron path
  // is used -- same reasoning as tests/test_ukb_fock_direct.cpp's kNoiseFloorThreshold. The MO-basis
  // comparison below (built in the orthonormal eigenbasis, no such dynamic range) is the tight check.
  check(worst_fock < 1e-4, name + ": RkbCholesky-based Fock matches the dense RkbTwoElectronTensor-based Fock");

  // rkbCholeskyToMo's own cross-flavor fix: build the MO-basis Cholesky vectors (n_negative=0, i.e.
  // untrimmed), densify them, and compare DIRECTLY (element-by-element) against
  // rkbMoTwoElectronTransformPhysics's dense MO tensor built from the (already-fixed) dense AO eri --
  // an entirely independent transform path, so this cross-checks BOTH fixes against each other, not
  // just against the dense AO-basis Fock comparison above.
  const CholeskyEri<C> mo_cholesky = rkbCholeskyToMo(ao_cholesky, c_mo, /*n_negative=*/0, 1e-10);
  const Tensor4<C> mo_cholesky_dense = mo_cholesky.toDense();
  const Tensor4<C> mo_dense_reference = rkbMoTwoElectronTransformPhysics(eri, c_mo);
  double worst_mo = 0.0;
  for (std::size_t a = 0; a < n_rkb; ++a)
    for (std::size_t b = 0; b < n_rkb; ++b)
      for (std::size_t cc = 0; cc < n_rkb; ++cc)
        for (std::size_t d = 0; d < n_rkb; ++d)
          worst_mo = std::max(worst_mo, std::abs(mo_cholesky_dense(a, b, cc, d) - mo_dense_reference(a, b, cc, d)));
  std::printf("[%s] max|MO tensor from rkbCholeskyToMo - MO tensor from dense path|=%.3e\n", name.c_str(), worst_mo);
  check(worst_mo < 1e-8, name + ": rkbCholeskyToMo's cross-flavor fix matches the dense path too");
}

}  // namespace

int main() {
  {
    BasisSet basis_set;
    basis_set.read("examples/lih-6-31g.gbs");
    const std::vector<Atom> geometry = {{"Li", 0.0, 0.0, 0.0}, {"H", 0.0, 0.0, 1.5949 * kAngstromToBohr}};
    MolecularBasis mb;
    mb.build(geometry, basis_set);
    runSystem("LiH/6-31G", mb.functions(), geometry, /*n_electrons=*/4);
  }

  std::printf("\n%d / %d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
