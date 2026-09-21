#include "Restart.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rerdmft {

namespace {

constexpr char kMagic[8] = {'R', 'E', 'R', 'D', 'M', 'F', 'T', '\0'};
constexpr std::uint32_t kByteOrderMarker = 0x01020304u;

bool hostIsLittleEndian() {
  const std::uint32_t probe = kByteOrderMarker;
  unsigned char bytes[4];
  std::memcpy(bytes, &probe, 4);
  return bytes[0] == 0x04;
}

template <typename V>
void put(std::ostream& out, const V& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(V));
}

void putString(std::ostream& out, const std::string& s) {
  put<std::uint64_t>(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void putVector(std::ostream& out, const std::vector<double>& v) {
  put<std::uint64_t>(out, v.size());
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
}

template <typename V>
V get(std::istream& in) {
  V value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(V));
  if (!in) throw std::runtime_error("readRestart: the file is truncated");
  return value;
}

// Bound on any count read from a file, so a corrupt length cannot trigger a huge allocation.
constexpr std::uint64_t kMaxCount = std::uint64_t{1} << 40;

std::string getString(std::istream& in) {
  const auto n = get<std::uint64_t>(in);
  if (n > 4096) throw std::runtime_error("readRestart: implausible string length");
  std::string s(n, '\0');
  in.read(s.data(), static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("readRestart: the file is truncated");
  return s;
}

std::vector<double> getVector(std::istream& in) {
  const auto n = get<std::uint64_t>(in);
  if (n > kMaxCount) throw std::runtime_error("readRestart: implausible vector length");
  std::vector<double> v(n);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
  if (!in) throw std::runtime_error("readRestart: the file is truncated");
  return v;
}

}  // namespace

void RestartData::setCoefficients(const Matrix<double>& c) {
  complex_coefficients = false;
  rows = c.rows();
  cols = c.cols();
  coefficients.assign(c.data(), c.data() + c.rows() * c.cols());
}

void RestartData::setCoefficients(const Matrix<std::complex<double>>& c) {
  complex_coefficients = true;
  rows = c.rows();
  cols = c.cols();
  coefficients.resize(2 * c.rows() * c.cols());
  for (std::size_t i = 0; i < c.rows() * c.cols(); ++i) {
    coefficients[2 * i] = c.data()[i].real();
    coefficients[2 * i + 1] = c.data()[i].imag();
  }
}

Matrix<std::complex<double>> RestartData::coefficientsComplex() const {
  Matrix<std::complex<double>> c(rows, cols);
  for (std::size_t i = 0; i < rows * cols; ++i) {
    c.data()[i] = complex_coefficients
                      ? std::complex<double>(coefficients[2 * i], coefficients[2 * i + 1])
                      : std::complex<double>(coefficients[i], 0.0);
  }
  return c;
}

void writeRestart(const std::string& path, const RestartData& data) {
  if (!hostIsLittleEndian()) throw std::runtime_error("writeRestart: big-endian hosts are not supported");
  if (data.kind != "OCCUPATIONS" && data.kind != "GAMMAS") {
    throw std::runtime_error("writeRestart: kind must be OCCUPATIONS or GAMMAS");
  }
  if (data.occupations.empty()) throw std::runtime_error("writeRestart: no occupation numbers");
  if (data.kind == "GAMMAS" && data.gammas.empty()) {
    throw std::runtime_error("writeRestart: kind GAMMAS without gamma angles");
  }
  const std::size_t entries = data.rows * data.cols * (data.complex_coefficients ? 2 : 1);
  if (data.rows == 0 || data.cols == 0 || data.coefficients.size() != entries) {
    throw std::runtime_error("writeRestart: the coefficient matrix is missing or inconsistent");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("writeRestart: cannot open " + path + " for writing");
  out.write(kMagic, sizeof(kMagic));
  put<std::uint32_t>(out, kRestartVersion);
  put<std::uint32_t>(out, kByteOrderMarker);
  putString(out, data.method);
  putString(out, data.functional);
  putString(out, data.kind);
  put<std::uint64_t>(out, data.basis_fingerprint);
  put<std::int64_t>(out, data.n_electrons);
  put<std::int64_t>(out, data.pnof_subspaces);
  put<std::int64_t>(out, data.pnof_coupling);
  put<std::int64_t>(out, data.n_core);
  put<double>(out, data.total_energy);
  put<std::uint8_t>(out, data.orbitals_optimized ? 1 : 0);
  put<std::uint8_t>(out, data.converged ? 1 : 0);
  putVector(out, data.occupations);
  putVector(out, data.gammas);
  put<std::uint8_t>(out, data.complex_coefficients ? 1 : 0);
  put<std::uint64_t>(out, data.rows);
  put<std::uint64_t>(out, data.cols);
  out.write(reinterpret_cast<const char*>(data.coefficients.data()),
            static_cast<std::streamsize>(data.coefficients.size() * sizeof(double)));
  out.flush();
  if (!out) throw std::runtime_error("writeRestart: error while writing " + path);
}

RestartData readRestart(const std::string& path) {
  if (!hostIsLittleEndian()) throw std::runtime_error("readRestart: big-endian hosts are not supported");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("readRestart: cannot open " + path);
  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("readRestart: " + path + " is not a RERDMFT restart file");
  }
  if (get<std::uint32_t>(in) != kRestartVersion) {
    throw std::runtime_error("readRestart: unsupported restart-file version");
  }
  if (get<std::uint32_t>(in) != kByteOrderMarker) {
    throw std::runtime_error("readRestart: the file has a different byte order");
  }
  RestartData d;
  d.method = getString(in);
  d.functional = getString(in);
  d.kind = getString(in);
  d.basis_fingerprint = get<std::uint64_t>(in);
  d.n_electrons = get<std::int64_t>(in);
  d.pnof_subspaces = get<std::int64_t>(in);
  d.pnof_coupling = get<std::int64_t>(in);
  d.n_core = get<std::int64_t>(in);
  d.total_energy = get<double>(in);
  d.orbitals_optimized = get<std::uint8_t>(in) != 0;
  d.converged = get<std::uint8_t>(in) != 0;
  d.occupations = getVector(in);
  d.gammas = getVector(in);
  d.complex_coefficients = get<std::uint8_t>(in) != 0;
  d.rows = get<std::uint64_t>(in);
  d.cols = get<std::uint64_t>(in);
  if (d.rows > kMaxCount || d.cols > kMaxCount || d.rows * d.cols > kMaxCount) {
    throw std::runtime_error("readRestart: implausible coefficient-matrix size");
  }
  d.coefficients.resize(d.rows * d.cols * (d.complex_coefficients ? 2 : 1));
  in.read(reinterpret_cast<char*>(d.coefficients.data()),
          static_cast<std::streamsize>(d.coefficients.size() * sizeof(double)));
  if (!in) throw std::runtime_error("readRestart: the file is truncated");
  return d;
}

Matrix<std::complex<double>> restartCoefficients(const Matrix<std::complex<double>>& c_scf,
                                                 const Matrix<std::complex<double>>& u) {
  if (u.rows() == 0) return c_scf;
  if (u.rows() != c_scf.cols() || u.cols() != c_scf.cols()) {
    throw std::runtime_error("restartCoefficients: rotation size differs from the number of MOs");
  }
  return c_scf * u;
}

}  // namespace rerdmft
