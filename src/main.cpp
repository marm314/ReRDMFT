#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "BasisSet.h"
#include "DiracKinetic.h"
#include "Input.h"
#include "Integrals.h"
#include "MolecularBasis.h"
#include "Shell.h"
#include "SmallComponentBasis.h"
#include "SpinorBasis.h"

namespace {

void printAoList(const std::string& label,
                  const std::vector<rerdmft::BasisFunction>& functions,
                  const std::vector<rerdmft::NormalizationCheck>& normalization) {
  std::cout << "\n" << label << " (" << functions.size()
             << " total), normalized via libcint overlap integrals:\n";
  for (std::size_t i = 0; i < functions.size(); ++i) {
    const auto& fn = functions[i];
    const auto& check = normalization[i];
    std::cout << "  " << std::setw(2) << fn.element << "  "
               << rerdmft::angularMomentumLabel(fn.l) << "(" << fn.cartesian.lx
               << fn.cartesian.ly << fn.cartesian.lz << ")  "
               << "center=(" << fn.x << ", " << fn.y << ", " << fn.z << ")  "
               << "nprim=" << fn.exponents.size() << "  "
               << "S_self(before)=" << check.self_overlap_before << "  "
               << (check.was_renormalized ? "renormalized" : "already normalized")
               << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input file>\n";
    return 1;
  }

  rerdmft::Input input;
  rerdmft::BasisSet basis_set;
  rerdmft::MolecularBasis large_basis;
  rerdmft::SmallComponentBasis small_basis;
  rerdmft::SpinorBasis spinor_basis;
  std::vector<rerdmft::NormalizationCheck> large_normalization;
  std::vector<rerdmft::NormalizationCheck> small_normalization;
  rerdmft::Matrix<std::complex<double>> dirac_kinetic;
  try {
    input.read(argv[1]);
    basis_set.read(input.basis_file());

    large_basis.build(input.geometry(), basis_set);
    large_normalization = rerdmft::normalizeCartesianBasis(large_basis.functions());

    small_basis.build(input.geometry(), basis_set);
    small_normalization = rerdmft::normalizeCartesianBasis(small_basis.functions());

    spinor_basis.build(large_basis.functions(), small_basis.functions());

    dirac_kinetic = rerdmft::diracKineticMatrix(large_basis.functions(), small_basis.functions());
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Number of electrons: " << input.n_electrons() << "\n";
  std::cout << "Basis set file:      " << input.basis_file() << "\n";
  std::cout << "Geometry (" << input.geometry().size() << " atoms):\n";
  std::cout << std::fixed << std::setprecision(6);
  for (const auto& atom : input.geometry()) {
    std::cout << "  " << std::setw(2) << atom.symbol << "  " << std::setw(12)
               << atom.x << "  " << std::setw(12) << atom.y << "  "
               << std::setw(12) << atom.z << "\n";
  }

  printAoList("Large-component cartesian atomic orbitals", large_basis.functions(),
              large_normalization);
  printAoList("Small-component cartesian atomic orbitals (unrestricted kinetic balance)",
              small_basis.functions(), small_normalization);

  std::cout << "\nFour-component spinor basis:\n";
  std::cout << "  Large-component AOs per spin block: " << spinor_basis.nLarge() << "\n";
  std::cout << "  Small-component AOs per spin block: " << spinor_basis.nSmall() << "\n";
  std::cout << "  Total spinor basis functions: " << spinor_basis.size()
             << " (Large-alpha, Large-beta, Small-alpha, Small-beta)\n";
  const auto describeSpinor = [&](std::size_t index) {
    const auto& ao = spinor_basis.ao(index);
    std::cout << "    index " << index << "  "
               << rerdmft::spinBlockLabel(spinor_basis.block(index)) << "  ao="
               << spinor_basis.aoIndex(index) << "  " << ao.element << "  "
               << rerdmft::angularMomentumLabel(ao.l) << "(" << ao.cartesian.lx
               << ao.cartesian.ly << ao.cartesian.lz << ")\n";
  };
  const std::size_t n_large = spinor_basis.nLarge();
  const std::size_t n_small = spinor_basis.nSmall();
  if (n_large > 0) {
    describeSpinor(0);
    describeSpinor(n_large - 1);
    describeSpinor(n_large);
    describeSpinor(2 * n_large - 1);
  }
  if (n_small > 0) {
    describeSpinor(2 * n_large);
    describeSpinor(2 * n_large + n_small - 1);
    describeSpinor(2 * n_large + n_small);
    describeSpinor(spinor_basis.size() - 1);
  }

  std::cout << "\nDirac kinetic energy matrix T = -i c (alpha . grad_r):\n";
  std::cout << "  Dimensions: " << dirac_kinetic.rows() << " x " << dirac_kinetic.cols()
             << "\n";

  double max_hermiticity_error = 0.0;
  double max_large_large = 0.0;
  double max_small_small = 0.0;
  for (std::size_t p = 0; p < dirac_kinetic.rows(); ++p) {
    for (std::size_t q = 0; q < dirac_kinetic.cols(); ++q) {
      const double herm_err = std::abs(dirac_kinetic(p, q) - std::conj(dirac_kinetic(q, p)));
      max_hermiticity_error = std::max(max_hermiticity_error, herm_err);
      const bool p_large = p < 2 * n_large;
      const bool q_large = q < 2 * n_large;
      if (p_large && q_large) {
        max_large_large = std::max(max_large_large, std::abs(dirac_kinetic(p, q)));
      } else if (!p_large && !q_large) {
        max_small_small = std::max(max_small_small, std::abs(dirac_kinetic(p, q)));
      }
    }
  }
  std::cout << "  Max |T - T^dagger| (Hermiticity check): " << max_hermiticity_error << "\n";
  std::cout << "  Max |T| within Large-Large block (expect 0): " << max_large_large << "\n";
  std::cout << "  Max |T| within Small-Small block (expect 0): " << max_small_small << "\n";
  if (n_large > 0 && n_small > 0) {
    // Large-alpha[0] paired with Small-beta[0], i.e. the alpha-beta spin
    // block -c*i*(Dx - i*Dy): typically nonzero and illustrates that the
    // matrix is genuinely complex (from the sigma_y contribution), unlike
    // e.g. the alpha-alpha block for an S/p_z pair on the same center,
    // which vanishes exactly by parity.
    const std::size_t col = 2 * n_large + n_small;
    const auto& sample = dirac_kinetic(0, col);
    std::cout << "  T[0, " << col << "] (Large-alpha[0], Small-beta[0]) = " << sample.real()
               << (sample.imag() >= 0 ? " + " : " - ") << std::abs(sample.imag()) << "i\n";
  }

  return 0;
}
