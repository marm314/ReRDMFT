#include <iomanip>
#include <iostream>
#include <vector>

#include "BasisSet.h"
#include "Input.h"
#include "Integrals.h"
#include "MolecularBasis.h"
#include "Shell.h"
#include "SpinorBasis.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input file>\n";
    return 1;
  }

  rerdmft::Input input;
  rerdmft::BasisSet basis_set;
  rerdmft::MolecularBasis molecular_basis;
  rerdmft::SpinorBasis spinor_basis;
  std::vector<rerdmft::NormalizationCheck> normalization;
  try {
    input.read(argv[1]);
    basis_set.read(input.basis_file());
    molecular_basis.build(input.geometry(), basis_set);
    normalization = rerdmft::normalizeCartesianBasis(molecular_basis.functions());
    spinor_basis.build(molecular_basis.functions());
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

  std::cout << "\nCartesian atomic orbitals (" << molecular_basis.functions().size()
             << " total), normalized via libcint overlap integrals:\n";
  for (std::size_t i = 0; i < molecular_basis.functions().size(); ++i) {
    const auto& fn = molecular_basis.functions()[i];
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

  std::cout << "\nFour-component spinor basis (Large component only):\n";
  std::cout << "  Cartesian AOs per spin block: " << spinor_basis.nao() << "\n";
  std::cout << "  Total spinor basis functions: " << spinor_basis.size()
             << " (all Large-alpha first, then all Large-beta)\n";
  const auto describeSpinor = [&](std::size_t index) {
    const auto& ao = spinor_basis.ao(index);
    std::cout << "    index " << index << "  "
               << rerdmft::spinBlockLabel(spinor_basis.block(index)) << "  ao="
               << spinor_basis.aoIndex(index) << "  " << ao.element << "  "
               << rerdmft::angularMomentumLabel(ao.l) << "(" << ao.cartesian.lx
               << ao.cartesian.ly << ao.cartesian.lz << ")\n";
  };
  if (spinor_basis.size() > 0) {
    describeSpinor(0);
    describeSpinor(spinor_basis.nao() - 1);
    describeSpinor(spinor_basis.nao());
    describeSpinor(spinor_basis.size() - 1);
  }

  return 0;
}
