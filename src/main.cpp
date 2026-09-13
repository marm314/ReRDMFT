#include <iomanip>
#include <iostream>

#include "BasisSet.h"
#include "Input.h"
#include "MolecularBasis.h"

namespace {

char angularMomentumLabel(int l) {
  static const char labels[] = {'S', 'P', 'D', 'F', 'G', 'H', 'I'};
  if (l < 0 || l >= static_cast<int>(sizeof(labels))) return '?';
  return labels[l];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input file>\n";
    return 1;
  }

  rerdmft::Input input;
  rerdmft::BasisSet basis_set;
  rerdmft::MolecularBasis molecular_basis;
  try {
    input.read(argv[1]);
    basis_set.read(input.basis_file());
    molecular_basis.build(input.geometry(), basis_set);
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
             << " total):\n";
  for (const auto& fn : molecular_basis.functions()) {
    std::cout << "  " << std::setw(2) << fn.element << "  "
               << angularMomentumLabel(fn.l) << "(" << fn.cartesian.lx
               << fn.cartesian.ly << fn.cartesian.lz << ")  "
               << "center=(" << fn.x << ", " << fn.y << ", " << fn.z << ")  "
               << "nprim=" << fn.exponents.size() << "\n";
  }

  return 0;
}
