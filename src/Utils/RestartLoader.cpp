#include "RestartLoader.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "LinearAlgebra.h"

namespace rerdmft {

namespace {

double maxDeviationFromIdentity(const Matrix<std::complex<double>>& m) {
  double dev = 0.0;
  for (std::size_t i = 0; i < m.rows(); ++i)
    for (std::size_t j = 0; j < m.cols(); ++j) dev = std::max(dev, std::abs(m(i, j) - (i == j ? 1.0 : 0.0)));
  return dev;
}

}  // namespace

Matrix<std::complex<double>> lowdinOrthonormalize(const Matrix<std::complex<double>>& c,
                                                  const Matrix<std::complex<double>>& s, double* deviation_before,
                                                  double* deviation_after, double* min_eigenvalue,
                                                  double min_eigenvalue_floor) {
  if (s.rows() != c.rows() || s.cols() != c.rows()) {
    throw std::runtime_error("lowdinOrthonormalize: the overlap does not match the coefficient rows");
  }
  const auto s_check = dagger(c) * (s * c);
  if (deviation_before != nullptr) *deviation_before = maxDeviationFromIdentity(s_check);
  const HermitianEigenResult eig = diagonalizeHermitian(s_check);
  const double lowest = eig.eigenvalues.front();
  if (min_eigenvalue != nullptr) *min_eigenvalue = lowest;
  if (!(lowest > min_eigenvalue_floor)) {
    throw std::runtime_error("lowdinOrthonormalize: C^dagger S C has an eigenvalue " + std::to_string(lowest) +
                             " (linearly dependent orbitals) -- cannot orthonormalize");
  }
  // S_check^(-1/2) = V diag(w^(-1/2)) V^dagger.
  const std::size_t n = s_check.rows();
  Matrix<std::complex<double>> scaled(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) scaled(i, j) = eig.eigenvectors(i, j) / std::sqrt(eig.eigenvalues[j]);
  const auto inverse_sqrt = scaled * dagger(eig.eigenvectors);
  const auto c_new = c * inverse_sqrt;
  if (deviation_after != nullptr) *deviation_after = maxDeviationFromIdentity(dagger(c_new) * (s * c_new));
  return c_new;
}

RestartOrbitals readRestartOrbitals(const std::string& path, const std::string& method, long long n_electrons,
                                    std::size_t expected_rows, std::size_t expected_cols, bool expected_complex,
                                    const Matrix<std::complex<double>>& s_ao, std::ostream& log, double tolerance) {
  RestartOrbitals out;
  out.data = readRestart(path);
  const RestartData& d = out.data;
  if (d.method != method) {
    throw std::runtime_error("READ_RESTART: " + path + " was written by method " + d.method + ", not " + method);
  }
  if (d.n_electrons != n_electrons) {
    throw std::runtime_error("READ_RESTART: " + path + " is for " + std::to_string(d.n_electrons) +
                             " electrons but NELEC is " + std::to_string(n_electrons));
  }
  if (d.rows != expected_rows || d.cols != expected_cols) {
    throw std::runtime_error("READ_RESTART: " + path + " holds " + std::to_string(d.rows) + " x " +
                             std::to_string(d.cols) + " coefficients but this basis needs " +
                             std::to_string(expected_rows) + " x " + std::to_string(expected_cols) +
                             " (a different basis set?)");
  }
  if (d.complex_coefficients != expected_complex) {
    throw std::runtime_error("READ_RESTART: " + path + " stores " +
                             std::string(d.complex_coefficients ? "complex" : "real") + " coefficients, not " +
                             (expected_complex ? "complex" : "real"));
  }
  if (d.occupations.size() != expected_cols) {
    throw std::runtime_error("READ_RESTART: " + path + " has " + std::to_string(d.occupations.size()) +
                             " occupation numbers for " + std::to_string(expected_cols) + " orbitals");
  }
  const Matrix<std::complex<double>> c_read = d.coefficientsComplex();
  double dev_after = 0.0;
  const auto s_check = dagger(c_read) * (s_ao * c_read);
  out.overlap_deviation_read = maxDeviationFromIdentity(s_check);
  if (out.overlap_deviation_read > tolerance) {
    out.c = lowdinOrthonormalize(c_read, s_ao, nullptr, &dev_after, &out.min_overlap_eigenvalue);
    out.lowdin_applied = true;
    out.overlap_deviation_final = dev_after;
  } else {
    out.c = c_read;
    out.overlap_deviation_final = out.overlap_deviation_read;
  }
  log << std::scientific << std::setprecision(2);
  log << "  READ_RESTART (" << method << "): " << path << ": " << d.functional << " (" << d.kind << "), "
      << d.occupations.size() << " orbitals, " << (d.orbitals_optimized ? "FULL_OPTIMIZATION" : "SCF")
      << " orbitals, energy written " << std::fixed << std::setprecision(10) << d.total_energy << std::scientific
      << std::setprecision(2) << " Hartree" << (d.converged ? "" : " (that run had not converged)") << "\n";
  log << "    S_check = C_read^dagger S C_read: max |S_check - 1| = " << out.overlap_deviation_read << "  ";
  if (out.lowdin_applied) {
    log << "-> Loewdin orthonormalization C = C_read S_check^(-1/2) (S_check eigenvalues down to " << out.min_overlap_eigenvalue
        << "): max |C^dagger S C - 1| = " << out.overlap_deviation_final << "\n";
  } else {
    log << "(orthonormal already, kept as read)\n";
  }
  log << std::defaultfloat << std::setprecision(6);
  return out;
}

}  // namespace rerdmft
