// check_wfn: reads an AIMPAC WFN file and, with libcint, builds the overlap of its molecular
// orbitals S_mo = C^T P C (P: overlap of the file's normalized Cartesian primitives). For a valid
// natural-orbital WFN S_mo = I; S_mo * occupancies then reproduces the density, i.e.
// Tr(S_mo diag(n)) = integral of rho = number of electrons.
// Usage: check_wfn file.wfn
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "AIMPAC.h"
#include "Integrals.h"
#include "Matrix.h"
#include "MolecularBasis.h"

using namespace rerdmft;

namespace {

double parseFortranDouble(std::string tok) {
  for (char& c : tok) {
    if (c == 'D' || c == 'd') c = 'E';
  }
  return std::stod(tok);
}

struct Wfn {
  std::vector<double> centre_xyz;  // 3 per nucleus
  std::vector<int> centre, type;
  std::vector<double> exponent;
  std::vector<double> occupation;
  std::vector<std::vector<double>> mo;  // [mo][primitive]
};

Wfn readWfn(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  Wfn w;
  std::string line;
  std::getline(in, line);  // title
  std::getline(in, line);
  std::size_t n_mo = 0, n_prim = 0, n_nuc = 0;
  {
    std::istringstream ss(line);
    std::string g, m, o, p, q, nuc;
    ss >> g >> n_mo >> m >> o >> n_prim >> p >> n_nuc;
    if (g != "GAUSSIAN" && g != "GTO") throw std::runtime_error("only GAUSSIAN-type WFN files are supported");
  }
  for (std::size_t a = 0; a < n_nuc; ++a) {
    std::getline(in, line);
    const auto close = line.find(')');
    if (close == std::string::npos) throw std::runtime_error("bad nucleus line: " + line);
    double x, y, z;
    if (std::sscanf(line.c_str() + close + 1, "%lf %lf %lf", &x, &y, &z) != 3) {
      throw std::runtime_error("bad nucleus coordinates: " + line);
    }
    w.centre_xyz.insert(w.centre_xyz.end(), {x, y, z});
  }
  const auto readInts = [&](std::vector<int>& dst, const std::string& label) {
    while (dst.size() < n_prim) {
      std::getline(in, line);
      if (line.compare(0, label.size(), label) != 0) throw std::runtime_error("expected " + label + ", got: " + line);
      std::istringstream ss(line.substr(label.size()));
      int v;
      while (ss >> v) dst.push_back(v);
    }
  };
  readInts(w.centre, "CENTRE ASSIGNMENTS");
  readInts(w.type, "TYPE ASSIGNMENTS");
  while (w.exponent.size() < n_prim) {
    std::getline(in, line);
    if (line.compare(0, 9, "EXPONENTS") != 0) throw std::runtime_error("expected EXPONENTS, got: " + line);
    std::istringstream ss(line.substr(9));
    std::string tok;
    while (ss >> tok) w.exponent.push_back(parseFortranDouble(tok));
  }
  for (std::size_t m = 0; m < n_mo; ++m) {
    std::getline(in, line);
    const auto pos = line.find("OCC NO =");
    if (line.compare(0, 2, "MO") != 0 || pos == std::string::npos) {
      throw std::runtime_error("expected MO header, got: " + line);
    }
    w.occupation.push_back(parseFortranDouble(line.substr(pos + 8, 13)));
    std::vector<double> c;
    while (c.size() < n_prim) {
      std::getline(in, line);
      std::istringstream ss(line);
      std::string tok;
      while (ss >> tok) c.push_back(parseFortranDouble(tok));
    }
    w.mo.push_back(std::move(c));
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: check_wfn file.wfn\n";
    return 1;
  }
  try {
    const Wfn w = readWfn(argv[1]);
    const std::size_t n_prim = w.exponent.size(), n_mo = w.mo.size();

    // One BasisFunction per WFN primitive. Convention confirmed against a real AIMPAC-written WFN
    // (ESI N2_1.0.wfn, S_mo = I to 1e-8): the file's coefficients multiply PLAIN, unnormalized Gaussians
    // x^lx y^ly z^lz exp(-a r^2) -- the primitive normalization is already folded into them. libcint's
    // cint1e_ovlp_cart, however, evaluates coefficient * fac * (plain Gaussian) with
    // fac = 1/sqrt(4 pi) (s), sqrt(3/(4 pi)) (p), 1 (l>=2) (CINTcommon_fac_sp), so fac is divided out here.
    std::vector<BasisFunction> prims(n_prim);
    for (std::size_t p = 0; p < n_prim; ++p) {
      BasisFunction& f = prims[p];
      const std::size_t c = static_cast<std::size_t>(w.centre[p]) - 1;
      f.element = "X";
      f.x = w.centre_xyz[3 * c];
      f.y = w.centre_xyz[3 * c + 1];
      f.z = w.centre_xyz[3 * c + 2];
      f.cartesian = aimpacTypeExponents(w.type[p]);
      f.l = f.cartesian.lx + f.cartesian.ly + f.cartesian.lz;
      const double fac = f.l == 0 ? 0.282094791773878143 : f.l == 1 ? 0.488602511902919921 : 1.0;
      f.exponents = {w.exponent[p]};
      f.coefficients = {1.0 / fac};
    }

    const Matrix<double> prim_overlap = overlapMatrix(prims);
    Matrix<double> c(n_prim, n_mo);
    for (std::size_t m = 0; m < n_mo; ++m) {
      for (std::size_t p = 0; p < n_prim; ++p) c(p, m) = w.mo[m][p];
    }
    const Matrix<double> s_mo = transpose(c) * prim_overlap * c;

    double max_err = 0.0, max_diag = 0.0, max_off = 0.0, frob = 0.0;
    for (std::size_t i = 0; i < n_mo; ++i) {
      for (std::size_t j = 0; j < n_mo; ++j) {
        const double e = s_mo(i, j) - (i == j ? 1.0 : 0.0);
        max_err = std::max(max_err, std::abs(e));
        (i == j ? max_diag : max_off) = std::max(i == j ? max_diag : max_off, std::abs(e));
        frob += e * e;
      }
    }
    double n_sum = 0.0, n_dens = 0.0;
    for (std::size_t i = 0; i < n_mo; ++i) {
      n_sum += w.occupation[i];
      n_dens += s_mo(i, i) * w.occupation[i];  // Tr(S_mo diag(n)) = integral of the density
    }
    std::printf("WFN: %zu MOs, %zu primitives, %zu nuclei\n", n_mo, n_prim, w.centre_xyz.size() / 3);
    std::printf("S_mo - I:  max|diag| = %.3e   max|offdiag| = %.3e   max = %.3e   Frobenius = %.3e\n", max_diag,
                max_off, max_err, std::sqrt(frob));
    std::printf("Sum of occupancies               = %.10f\n", n_sum);
    std::printf("Tr(S_mo * diag(occupancies))     = %.10f   (integrated density)\n", n_dens);
    std::printf("Integrated density - sum(occ)    = %.3e\n", n_dens - n_sum);
  } catch (const std::exception& e) {
    std::cerr << "check_wfn: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
