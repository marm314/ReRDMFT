#include "AIMPAC.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "Element.h"
#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

// AIMPAC's fixed "TYPE ASSIGNMENTS" code for a Cartesian primitive (lx, ly, lz), S..G
// (codes 1..35) -- this table is the WFN FORMAT ITSELF (fixed by the AIMPAC/AIMAll readers,
// completely independent of any internal basis-function ordering this project happens to use;
// matches the table used by, e.g., PySCF's tools/wfn_format.py).
int aimpacTypeCode(int lx, int ly, int lz) {
  static const std::array<std::array<int, 3>, 35> table = {{
      {{0, 0, 0}},                                                    // 1: S
      {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},                          // 2-4: P
      {{2, 0, 0}}, {{0, 2, 0}}, {{0, 0, 2}}, {{1, 1, 0}}, {{1, 0, 1}}, {{0, 1, 1}},  // 5-10: D
      {{3, 0, 0}}, {{0, 3, 0}}, {{0, 0, 3}}, {{1, 2, 0}}, {{2, 1, 0}}, {{2, 0, 1}},
      {{1, 0, 2}}, {{0, 1, 2}}, {{0, 2, 1}}, {{1, 1, 1}},              // 11-20: F
      {{4, 0, 0}}, {{0, 4, 0}}, {{0, 0, 4}}, {{3, 1, 0}}, {{3, 0, 1}}, {{1, 3, 0}},
      {{0, 3, 1}}, {{1, 0, 3}}, {{0, 1, 3}}, {{2, 2, 0}}, {{2, 0, 2}}, {{0, 2, 2}},
      {{2, 1, 1}}, {{1, 2, 1}}, {{1, 1, 2}},                          // 21-35: G
  }};
  for (std::size_t i = 0; i < table.size(); ++i) {
    if (table[i][0] == lx && table[i][1] == ly && table[i][2] == lz) return static_cast<int>(i) + 1;
  }
  throw std::runtime_error(
      "writeAimpacWfn: angular momentum (lx=" + std::to_string(lx) + ", ly=" + std::to_string(ly) +
      ", lz=" + std::to_string(lz) + ") has no AIMPAC WFN TYPE code -- only S..G (l<=4) are defined");
}

// Formats `value` the way AIMPAC WFN files do: a mantissa normalized to [0.1, 1.0) (or exactly 0)
// with `mantissa_digits` decimal digits, times 10^exponent (2-digit signed exponent) -- e.g.
// 0.29207543E+00, NOT the usual d.ddddE+dd scientific form. Right-justified in `field_width`.
std::string sciField(double value, int mantissa_digits, int field_width) {
  std::string sign;
  double av = value;
  if (av < 0.0) {
    sign = "-";
    av = -av;
  }
  const double scale = std::pow(10.0, mantissa_digits);
  int exponent = 0;
  long long rounded = 0;
  if (av > 0.0) {
    exponent = static_cast<int>(std::floor(std::log10(av))) + 1;
    double mantissa = av / std::pow(10.0, exponent);
    rounded = static_cast<long long>(std::llround(mantissa * scale));
    if (rounded >= static_cast<long long>(scale)) {
      // Rounding carried the mantissa to 1.0 exactly (e.g. 0.999999996 at 8 digits) -- bump the
      // exponent and rescale rather than emit a mantissa with one digit too many.
      exponent += 1;
      rounded /= 10;
    }
  }
  std::ostringstream mantissa_ss;
  mantissa_ss << std::setw(mantissa_digits) << std::setfill('0') << rounded;
  std::ostringstream exp_ss;
  exp_ss << (exponent < 0 ? '-' : '+') << std::setw(2) << std::setfill('0') << std::abs(exponent);
  std::ostringstream out;
  out << sign << "0." << mantissa_ss.str() << "E" << exp_ss.str();
  std::string s = out.str();
  if (static_cast<int>(s.size()) < field_width) s = std::string(field_width - s.size(), ' ') + s;
  return s;
}

std::string padRight(const std::string& s, std::size_t width) {
  if (s.size() >= width) return s;
  return s + std::string(width - s.size(), ' ');
}

// One Cartesian AO's worth of unrolled WFN primitives: every (exponent, contraction coefficient)
// pair of that AO, each becoming its own TYPE/EXPONENT/CENTRE entry (WFN does not support
// contraction -- a contracted shell's primitives are all listed individually, repeated across its
// Cartesian components, with the per-primitive MO coefficient carrying the contraction weight).
struct PrimitiveSlot {
  int centre = 0;      // 1-indexed nucleus
  int type_code = 0;    // AIMPAC TYPE ASSIGNMENTS code
  double exponent = 0.0;
};

int centreIndexOf(const BasisFunction& fn, const std::vector<Atom>& geometry) {
  constexpr double kTol = 1e-8;
  for (std::size_t a = 0; a < geometry.size(); ++a) {
    const Atom& atom = geometry[a];
    if (std::abs(atom.x - fn.x) < kTol && std::abs(atom.y - fn.y) < kTol && std::abs(atom.z - fn.z) < kTol) {
      return static_cast<int>(a) + 1;
    }
  }
  throw std::runtime_error("writeAimpacWfn: basis function center does not match any atom in the geometry");
}

}  // namespace

std::vector<NaturalOrbital> naturalOrbitalsFromDensity(const Matrix<double>& s,
                                                        const Matrix<std::complex<double>>& d_ao,
                                                        double lin_dep_threshold) {
  const std::size_t n = s.rows();
  if (s.cols() != n) throw std::runtime_error("naturalOrbitalsFromDensity: s is not square");
  if (d_ao.rows() != n || d_ao.cols() != n) {
    throw std::runtime_error("naturalOrbitalsFromDensity: d_ao size does not match s");
  }

  RankReductionReport report;
  const Matrix<double> x = canonicalOrthogonalize(s, lin_dep_threshold, &report);
  const std::size_t n_mo = x.cols();

  Matrix<double> d_real(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) d_real(i, j) = d_ao(i, j).real();
  }

  // See this function's own header comment for the derivation: diagonalize X^T (S D S) X, not
  // X^T D X.
  const Matrix<double> s_d_s = s * d_real * s;
  Matrix<double> d_ortho = transpose(x) * s_d_s * x;
  // Symmetrize defensively: `d_ao` need not be exactly Hermitian to machine precision (it is the
  // caller's own sum of possibly several blocks), and diagonalizeSymmetric assumes symmetry.
  for (std::size_t i = 0; i < n_mo; ++i) {
    for (std::size_t j = i + 1; j < n_mo; ++j) {
      const double avg = 0.5 * (d_ortho(i, j) + d_ortho(j, i));
      d_ortho(i, j) = avg;
      d_ortho(j, i) = avg;
    }
  }

  const SymmetricEigenResult eig = diagonalizeSymmetric(d_ortho);
  const Matrix<double> c_no = x * eig.eigenvectors;  // (n x n_mo), in the ORIGINAL (non-orthogonal) AO basis

  std::vector<std::size_t> order(n_mo);
  for (std::size_t i = 0; i < n_mo; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return eig.eigenvalues[a] > eig.eigenvalues[b]; });

  std::vector<NaturalOrbital> orbitals(n_mo);
  for (std::size_t rank = 0; rank < n_mo; ++rank) {
    const std::size_t mo = order[rank];
    orbitals[rank].occupation = eig.eigenvalues[mo];
    orbitals[rank].coefficients.resize(n);
    for (std::size_t i = 0; i < n; ++i) orbitals[rank].coefficients[i] = c_no(i, mo);
  }
  return orbitals;
}

void writeAimpacWfn(const std::string& path, const std::string& title, const std::vector<Atom>& geometry,
                     const std::vector<BasisFunction>& basis, const std::vector<NaturalOrbital>& orbitals,
                     double total_energy, double virial_ratio, double occ_print_threshold) {
  const std::size_t n_cart = basis.size();
  if (geometry.empty()) throw std::runtime_error("writeAimpacWfn: empty geometry");
  for (const auto& orb : orbitals) {
    if (orb.coefficients.size() != n_cart) {
      throw std::runtime_error("writeAimpacWfn: an orbital's coefficients size does not match basis.size()");
    }
  }

  // Unroll every Cartesian AO's primitives into the flat WFN primitive list, and remember each
  // AO's own [first, count) slice of it so the per-MO coefficient expansion below can reuse it.
  std::vector<PrimitiveSlot> primitives;
  std::vector<std::pair<std::size_t, std::size_t>> ao_primitive_range(n_cart);
  for (std::size_t i = 0; i < n_cart; ++i) {
    const BasisFunction& fn = basis[i];
    const int centre = centreIndexOf(fn, geometry);
    const int type_code = aimpacTypeCode(fn.cartesian.lx, fn.cartesian.ly, fn.cartesian.lz);
    const std::size_t first = primitives.size();
    for (double exponent : fn.exponents) {
      primitives.push_back({centre, type_code, exponent});
    }
    ao_primitive_range[i] = {first, fn.exponents.size()};
  }
  const std::size_t n_primitives = primitives.size();

  // Drop orbitals that round to zero at the printed precision (OCC NO has 7 decimals -- AIMPAC
  // gains nothing from them, they only bloat the file), then re-sort by decreasing occupation
  // (the caller may have merged several independently-diagonalized basis blocks, e.g. C4_DHF's
  // Large and Small components, each in its own internal order).
  std::vector<const NaturalOrbital*> kept;
  for (const auto& orb : orbitals) {
    if (std::abs(orb.occupation) >= occ_print_threshold) kept.push_back(&orb);
  }
  std::sort(kept.begin(), kept.end(),
            [](const NaturalOrbital* a, const NaturalOrbital* b) { return a->occupation > b->occupation; });
  const std::size_t n_mo = kept.size();

  std::ofstream out(path);
  if (!out) throw std::runtime_error("writeAimpacWfn: could not open '" + path + "' for writing");

  out << " " << title << "\n";

  char buf[256];
  std::snprintf(buf, sizeof(buf), "GAUSSIAN%15lld MOL ORBITALS%7lld PRIMITIVES%9lld NUCLEI\n",
                static_cast<long long>(n_mo), static_cast<long long>(n_primitives),
                static_cast<long long>(geometry.size()));
  out << buf;

  for (std::size_t a = 0; a < geometry.size(); ++a) {
    const Atom& atom = geometry[a];
    const double charge = static_cast<double>(atomicNumber(atom.symbol));
    std::snprintf(buf, sizeof(buf), "%3s%5d    (CENTRE%3d) %12.8f%12.8f%12.8f  CHARGE =%5.1f\n",
                  atom.symbol.c_str(), static_cast<int>(a) + 1, static_cast<int>(a) + 1, atom.x, atom.y,
                  atom.z, charge);
    out << buf;
  }

  const auto writeIntBlock = [&](const std::string& label, const std::vector<int>& values) {
    for (std::size_t i = 0; i < values.size(); i += 20) {
      out << padRight(label, 20);
      for (std::size_t j = i; j < std::min(i + 20, values.size()); ++j) out << std::setw(3) << values[j];
      out << "\n";
    }
  };
  std::vector<int> centres(n_primitives), types(n_primitives);
  std::vector<double> exponents(n_primitives);
  for (std::size_t p = 0; p < n_primitives; ++p) {
    centres[p] = primitives[p].centre;
    types[p] = primitives[p].type_code;
    exponents[p] = primitives[p].exponent;
  }
  writeIntBlock("CENTRE ASSIGNMENTS", centres);
  writeIntBlock("TYPE ASSIGNMENTS", types);
  for (std::size_t i = 0; i < n_primitives; i += 5) {
    // Label field width 10 here (NOT 20, unlike CENTRE/TYPE ASSIGNMENTS above -- verified against
    // a real AIMPAC-written WFN file column-for-column): "EXPONENTS" is 9 characters, so this is a
    // single trailing pad space.
    out << padRight("EXPONENTS", 10);
    // Field width 14, not the 13 characters "0.NNNNNNNE+EE" actually needs: EXPONENTS are always
    // positive, so every field is exactly 13 characters with zero slack -- without at least one
    // guaranteed padding space, consecutive fields run together with no separator at all (unlike
    // the MO coefficient fields below, whose optional sign already guarantees padding on the
    // common positive case).
    for (std::size_t j = i; j < std::min(i + 5, n_primitives); ++j) out << sciField(exponents[j], 7, 14);
    out << "\n";
  }

  std::vector<double> primitive_coeffs(n_primitives);
  for (std::size_t rank = 0; rank < n_mo; ++rank) {
    const NaturalOrbital& orb = *kept[rank];
    for (std::size_t i = 0; i < n_cart; ++i) {
      const auto [first, count] = ao_primitive_range[i];
      const double c_ao = orb.coefficients[i];
      for (std::size_t p = 0; p < count; ++p) {
        primitive_coeffs[first + p] = c_ao * basis[i].coefficients[p];
      }
    }
    std::snprintf(buf, sizeof(buf), "MO%5d     MO 0.0        OCC NO =%13.7f  ORB. ENERGY =%12.6f\n",
                  static_cast<int>(rank) + 1, orb.occupation, 0.0);
    out << buf;
    for (std::size_t i = 0; i < n_primitives; i += 5) {
      for (std::size_t j = i; j < std::min(i + 5, n_primitives); ++j) out << sciField(primitive_coeffs[j], 8, 16);
      out << "\n";
    }
  }

  out << "END DATA\n";
  std::ostringstream eline;
  eline << " TOTAL ENERGY =" << std::fixed << std::setw(22) << std::setprecision(12) << total_energy
        << " THE VIRIAL(-V/T)=" << std::fixed << std::setw(13) << std::setprecision(8) << virial_ratio << "\n";
  out << eline.str();
}

}  // namespace rerdmft
