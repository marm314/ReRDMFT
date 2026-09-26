// Unit test of Utils/Restart.h (binary RESTART file) and of the gamma <-> occupation maps of
// Occ_opt/PNOFs.h that fill its GAMMAS records.
// Build/run: make test_restart LIBCINT=/path/to/libcint.a
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Matrix.h"
#include "PNOFs.h"
#include "Restart.h"

using namespace rerdmft;
using C = std::complex<double>;

namespace {

int g_failures = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) { ++g_failures; std::cout << "  FAIL: " << what << "\n"; }
}

struct Rng {
  std::uint64_t s = 246813579ULL;
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
};

RestartData sample(bool complex_orbitals, Rng& rng) {
  RestartData d;
  d.method = complex_orbitals ? "X2C_HF" : "NON_REL";
  d.functional = "GNOF";
  d.kind = "GAMMAS";
  d.basis_fingerprint = 0xDEADBEEF12345678ULL;
  d.n_electrons = 4;
  d.pnof_subspaces = 2;
  d.pnof_coupling = 3;
  d.n_core = 1;
  d.total_energy = -7.99760894712345;
  d.orbitals_optimized = true;
  d.converged = true;
  for (int i = 0; i < 12; ++i) d.occupations.push_back(rng.next());
  for (int i = 0; i < 4; ++i) d.gammas.push_back(rng.next());
  if (complex_orbitals) {
    Matrix<C> c(6, 5);
    for (std::size_t i = 0; i < 6; ++i)
      for (std::size_t j = 0; j < 5; ++j) c(i, j) = C(rng.next() - 0.5, rng.next() - 0.5);
    d.setCoefficients(c);
  } else {
    Matrix<double> c(6, 5);
    for (std::size_t i = 0; i < 6; ++i)
      for (std::size_t j = 0; j < 5; ++j) c(i, j) = rng.next() - 0.5;
    d.setCoefficients(c);
  }
  return d;
}

bool sameData(const RestartData& a, const RestartData& b) {
  return a.method == b.method && a.functional == b.functional && a.kind == b.kind &&
         a.basis_fingerprint == b.basis_fingerprint && a.n_electrons == b.n_electrons &&
         a.pnof_subspaces == b.pnof_subspaces && a.pnof_coupling == b.pnof_coupling &&
         a.n_core == b.n_core && a.jk_frozen_pairs == b.jk_frozen_pairs && a.jk_active_pairs == b.jk_active_pairs &&
         a.total_energy == b.total_energy &&
         a.orbitals_optimized == b.orbitals_optimized && a.converged == b.converged &&
         a.occupations == b.occupations && a.gammas == b.gammas &&
         a.complex_coefficients == b.complex_coefficients && a.rows == b.rows && a.cols == b.cols &&
         a.coefficients == b.coefficients;
}

template <typename F>
bool throws(F&& f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}

}  // namespace

int main() {
  Rng rng;
  const std::string path = "test_restart_tmp.bin";

  std::cout << "Restart file round trip\n";
  for (bool cplx : {false, true}) {
    const RestartData d = sample(cplx, rng);
    writeRestart(path, d);
    const RestartData back = readRestart(path);
    check(sameData(d, back), std::string("bit-identical round trip, ") + (cplx ? "complex" : "real") + " orbitals");
    const auto cm = back.coefficientsComplex();
    check(cm.rows() == 6 && cm.cols() == 5, "coefficient matrix shape");
    check(cplx ? (cm(2, 3) == C(d.coefficients[2 * (2 * 5 + 3)], d.coefficients[2 * (2 * 5 + 3) + 1]))
               : (cm(2, 3) == C(d.coefficients[2 * 5 + 3], 0.0)),
          "row-major coefficient layout");
  }

  std::cout << "JK_only style file (no gammas)\n";
  {
    RestartData d = sample(false, rng);
    d.kind = "OCCUPATIONS";
    d.gammas.clear();
    d.pnof_subspaces = d.pnof_coupling = d.n_core = 0;
    d.jk_frozen_pairs = 1;
    d.jk_active_pairs = 4;
    writeRestart(path, d);
    check(sameData(d, readRestart(path)), "OCCUPATIONS file round trip");
  }

  std::cout << "Rejection of bad input\n";
  {
    RestartData d = sample(false, rng);
    RestartData bad = d;
    bad.kind = "OTHER";
    check(throws([&] { writeRestart(path, bad); }), "unknown kind refused");
    bad = d;
    bad.occupations.clear();
    check(throws([&] { writeRestart(path, bad); }), "empty occupations refused");
    bad = d;
    bad.coefficients.pop_back();
    check(throws([&] { writeRestart(path, bad); }), "inconsistent coefficient size refused");
    bad = d;
    bad.gammas.clear();
    check(throws([&] { writeRestart(path, bad); }), "GAMMAS kind without gammas refused");
    check(throws([&] { readRestart("this_file_does_not_exist.bin"); }), "missing file refused");

    writeRestart(path, d);
    std::string bytes;
    {
      std::ifstream in(path, std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const auto rewrite = [&](const std::string& content) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out.write(content.data(), static_cast<std::streamsize>(content.size()));
    };
    rewrite(bytes.substr(0, bytes.size() - 8));
    check(throws([&] { readRestart(path); }), "truncated file refused");
    std::string wrong_magic = bytes;
    wrong_magic[0] = 'X';
    rewrite(wrong_magic);
    check(throws([&] { readRestart(path); }), "wrong magic refused");
    std::string wrong_version = bytes;
    wrong_version[8] = static_cast<char>(kRestartVersion + 1);
    rewrite(wrong_version);
    check(throws([&] { readRestart(path); }), "wrong version refused");
    std::string wrong_order = bytes;
    wrong_order[12] = 0x01;
    rewrite(wrong_order);
    check(throws([&] { readRestart(path); }), "wrong byte-order marker refused");
  }
  std::remove(path.c_str());

  std::cout << "restartCoefficients\n";
  {
    Matrix<C> c(4, 3), u(3, 3);
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = 0; j < 3; ++j) c(i, j) = C(rng.next(), rng.next());
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j) u(i, j) = C(rng.next(), rng.next());
    const auto same = restartCoefficients(c, Matrix<C>());
    const auto rot = restartCoefficients(c, u);
    bool same_ok = true, rot_ok = true;
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = 0; j < 3; ++j) {
        same_ok = same_ok && same(i, j) == c(i, j);
        C ref = 0.0;
        for (std::size_t k = 0; k < 3; ++k) ref += c(i, k) * u(k, j);
        rot_ok = rot_ok && std::abs(rot(i, j) - ref) < 1e-14;
      }
    check(same_ok, "empty rotation = identity");
    check(rot_ok, "C U for a general rotation");
    check(throws([&] { restartCoefficients(c, Matrix<C>(2, 2)); }), "rotation of the wrong size refused");
  }

  std::cout << "PNOF gammas <-> occupations\n";
  for (int coupling = 2; coupling <= 6; ++coupling) {
    double worst_occ = 0.0, worst_gamma = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
      std::vector<double> gammas(static_cast<std::size_t>(coupling - 1));
      for (double& g : gammas) g = 1e-3 + (M_PI / 2 - 2e-3) * rng.next();  // interior of [0, pi/2]
      const auto occ = pnofSubspaceOccupationsFromGammas(coupling, gammas);
      const auto back = pnofSubspaceGammasFromOccupations(coupling, occ);
      const auto occ2 = pnofSubspaceOccupationsFromGammas(coupling, back);
      for (std::size_t i = 0; i < occ.size(); ++i) worst_occ = std::max(worst_occ, std::abs(occ[i] - occ2[i]));
      for (std::size_t i = 0; i < gammas.size(); ++i) worst_gamma = std::max(worst_gamma, std::abs(gammas[i] - back[i]));
    }
    check(worst_occ < 1e-12, "occupations -> gammas -> occupations, coupling " + std::to_string(coupling));
    check(worst_gamma < 1e-6, "gammas -> occupations -> gammas recovers the angles, coupling " + std::to_string(coupling));
  }
  {
    // Edge cases: a fully occupied principal geminal (remaining hole 0) and a converged SQP
    // result at the box bounds still round-trip.
    const std::vector<double> occ_full = {1.0, 0.0, 0.0};
    const auto g = pnofSubspaceGammasFromOccupations(3, occ_full);
    const auto occ2 = pnofSubspaceOccupationsFromGammas(3, g);
    double dev = 0.0;
    for (std::size_t i = 0; i < 3; ++i) dev = std::max(dev, std::abs(occ_full[i] - occ2[i]));
    check(dev < 1e-12, "n_principal = 1 round trip");
    const std::vector<double> occ_bound = {1.0 - 1e-6, 1.0 - 2e-6 + 1e-12, 1e-6 - 1e-12 + 1e-6};
    // sum is not exactly 1 here on purpose: the map must not throw or return NaN.
    const auto g2 = pnofSubspaceGammasFromOccupations(3, occ_bound);
    check(std::isfinite(g2[0]) && std::isfinite(g2[1]), "near-bound occupations give finite angles");
    check(throws([&] { pnofSubspaceGammasFromOccupations(3, {0.6, 0.4}); }), "wrong occupation count refused");
    check(throws([&] { pnofSubspaceGammasFromOccupations(1, {1.0}); }), "coupling < 2 refused");
  }

  std::cout << (g_failures == 0 ? "ALL PASSED" : "FAILURES") << " (" << g_checks - g_failures << "/" << g_checks << ")\n";
  return g_failures == 0 ? 0 : 1;
}
