#include "KramersRestriction.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rerdmft {

namespace {

inline double spinorSign(std::size_t i) { return (i % 2 == 0) ? 1.0 : -1.0; }
inline std::size_t partner(std::size_t i) { return i ^ std::size_t{1}; }

}  // namespace

KramersRestriction::KramersRestriction(
    std::size_t n_spinors, const std::vector<std::pair<std::size_t, std::size_t>>& pairs)
    : n_spinors_(n_spinors), n_pairs_(pairs.size()) {
  if (n_spinors % 2 != 0) {
    throw std::runtime_error("KramersRestriction: the number of spinors must be even");
  }
  std::vector<long> index(n_spinors * n_spinors, -1);
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const auto [p, q] = pairs[i];
    if (p >= n_spinors || q >= p) {
      throw std::runtime_error("KramersRestriction: pairs must satisfy n_spinors > p > q");
    }
    if (index[p * n_spinors + q] >= 0) {
      throw std::runtime_error("KramersRestriction: repeated pair");
    }
    index[p * n_spinors + q] = static_cast<long>(i);
  }

  // Time-reversal image of every pair: (p,q) -> (P p, P q) with
  // t' = a t, y' = b y (see the header comment).
  std::vector<std::size_t> image(pairs.size());
  std::vector<double> sign_t(pairs.size()), sign_y(pairs.size());
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const auto [p, q] = pairs[i];
    const std::size_t pp = partner(p), qq = partner(q);
    const double ss = spinorSign(p) * spinorSign(q);
    long j;
    if (pp > qq) {
      j = index[pp * n_spinors + qq];
      sign_t[i] = ss;
      sign_y[i] = -ss;
    } else {
      j = index[qq * n_spinors + pp];
      sign_t[i] = -ss;
      sign_y[i] = -ss;
    }
    if (j < 0) {
      throw std::runtime_error(
          "KramersRestriction: the pair list is not closed under Kramers partners");
    }
    image[i] = static_cast<std::size_t>(j);
  }
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    if (image[image[i]] != i || sign_t[image[i]] * sign_t[i] != 1.0 ||
        sign_y[image[i]] * sign_y[i] != 1.0) {
      throw std::runtime_error("KramersRestriction: internal error, time reversal is not an involution");
    }
  }

  for (std::size_t i = 0; i < pairs.size(); ++i) {
    if (image[i] < i) continue;  // already listed with its lower-index partner
    orbits_.push_back({i, image[i], sign_t[i], sign_y[i]});
    if (image[i] == i) {
      ++n_fixed_;
      if (sign_t[i] != 1.0 || sign_y[i] != 1.0) {
        throw std::runtime_error("KramersRestriction: internal error, within-pair rotation not invariant");
      }
    }
  }
}

std::vector<double> KramersRestriction::expand(const std::vector<double>& reduced) const {
  if (reduced.size() != reducedSize()) {
    throw std::runtime_error("KramersRestriction::expand: wrong reduced size");
  }
  const std::size_t n_orb = orbits_.size();
  std::vector<double> full(fullSize(), 0.0);
  for (std::size_t j = 0; j < n_orb; ++j) {
    const Orbit& o = orbits_[j];
    if (o.first == o.second) {
      full[o.first] = reduced[j];
      full[n_pairs_ + o.first] = reduced[n_orb + j];
    } else {
      const double c = 1.0 / std::sqrt(2.0);
      full[o.first] = c * reduced[j];
      full[o.second] = c * o.sign_t * reduced[j];
      full[n_pairs_ + o.first] = c * reduced[n_orb + j];
      full[n_pairs_ + o.second] = c * o.sign_y * reduced[n_orb + j];
    }
  }
  return full;
}

std::vector<double> KramersRestriction::contract(const std::vector<double>& full) const {
  if (full.size() != fullSize()) {
    throw std::runtime_error("KramersRestriction::contract: wrong full size");
  }
  const std::size_t n_orb = orbits_.size();
  std::vector<double> reduced(reducedSize(), 0.0);
  for (std::size_t j = 0; j < n_orb; ++j) {
    const Orbit& o = orbits_[j];
    if (o.first == o.second) {
      reduced[j] = full[o.first];
      reduced[n_orb + j] = full[n_pairs_ + o.first];
    } else {
      const double c = 1.0 / std::sqrt(2.0);
      reduced[j] = c * (full[o.first] + o.sign_t * full[o.second]);
      reduced[n_orb + j] = c * (full[n_pairs_ + o.first] + o.sign_y * full[n_pairs_ + o.second]);
    }
  }
  return reduced;
}

std::vector<double> KramersRestriction::expandRepresentative(const std::vector<double>& reduced) const {
  if (reduced.size() != reducedSize()) {
    throw std::runtime_error("KramersRestriction::expandRepresentative: wrong reduced size");
  }
  const std::size_t n_orb = orbits_.size();
  std::vector<double> full(fullSize(), 0.0);
  for (std::size_t j = 0; j < n_orb; ++j) {
    const Orbit& o = orbits_[j];
    full[o.first] = reduced[j];
    full[n_pairs_ + o.first] = reduced[n_orb + j];
    if (o.first != o.second) {
      full[o.second] = o.sign_t * reduced[j];
      full[n_pairs_ + o.second] = o.sign_y * reduced[n_orb + j];
    }
  }
  return full;
}

std::vector<double> KramersRestriction::contractRepresentative(const std::vector<double>& full) const {
  if (full.size() != fullSize()) {
    throw std::runtime_error("KramersRestriction::contractRepresentative: wrong full size");
  }
  const std::size_t n_orb = orbits_.size();
  std::vector<double> reduced(reducedSize(), 0.0);
  for (std::size_t j = 0; j < n_orb; ++j) {
    const Orbit& o = orbits_[j];
    reduced[j] = full[o.first];
    reduced[n_orb + j] = full[n_pairs_ + o.first];
    if (o.first != o.second) {
      reduced[j] += o.sign_t * full[o.second];
      reduced[n_orb + j] += o.sign_y * full[n_pairs_ + o.second];
    }
  }
  return reduced;
}

std::vector<double> KramersRestriction::timeReversed(const std::vector<double>& full) const {
  if (full.size() != fullSize()) {
    throw std::runtime_error("KramersRestriction::timeReversed: wrong full size");
  }
  std::vector<double> out(fullSize(), 0.0);
  for (const Orbit& o : orbits_) {
    // R maps first -> second and second -> first with the same signs.
    out[o.second] += o.sign_t * full[o.first];
    out[n_pairs_ + o.second] += o.sign_y * full[n_pairs_ + o.first];
    if (o.first != o.second) {
      out[o.first] += o.sign_t * full[o.second];
      out[n_pairs_ + o.first] += o.sign_y * full[n_pairs_ + o.second];
    }
  }
  return out;
}

std::vector<double> KramersRestriction::project(const std::vector<double>& full) const {
  std::vector<double> r = timeReversed(full);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = 0.5 * (full[i] + r[i]);
  return r;
}

double KramersRestriction::asymmetry(const std::vector<double>& full) const {
  const std::vector<double> r = timeReversed(full);
  double m = 0.0;
  for (std::size_t i = 0; i < r.size(); ++i) m = std::max(m, std::abs(full[i] - r[i]));
  return m;
}

Matrix<double> KramersRestriction::contractMatrix(const Matrix<double>& full) const {
  const std::size_t n = fullSize();
  if (full.rows() != n || full.cols() != n) {
    throw std::runtime_error("KramersRestriction::contractMatrix: wrong matrix size");
  }
  const std::size_t r = reducedSize();
  // Q^T M Q column by column: expand each reduced unit vector.
  Matrix<double> mq(n, r, 0.0);
  for (std::size_t c = 0; c < r; ++c) {
    std::vector<double> e(r, 0.0);
    e[c] = 1.0;
    const std::vector<double> q = expand(e);
    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (std::size_t k = 0; k < n; ++k) sum += full(i, k) * q[k];
      mq(i, c) = sum;
    }
  }
  Matrix<double> out(r, r, 0.0);
  for (std::size_t c = 0; c < r; ++c) {
    std::vector<double> col(n);
    for (std::size_t i = 0; i < n; ++i) col[i] = mq(i, c);
    const std::vector<double> reduced = contract(col);
    for (std::size_t i = 0; i < r; ++i) out(i, c) = reduced[i];
  }
  return out;
}

std::vector<double> KramersRestriction::contractDiagonal(const std::vector<double>& d) const {
  if (d.size() != fullSize()) {
    throw std::runtime_error("KramersRestriction::contractDiagonal: wrong full size");
  }
  const std::size_t n_orb = orbits_.size();
  std::vector<double> out(reducedSize(), 0.0);
  for (std::size_t j = 0; j < n_orb; ++j) {
    const Orbit& o = orbits_[j];
    if (o.first == o.second) {
      out[j] = d[o.first];
      out[n_orb + j] = d[n_pairs_ + o.first];
    } else {
      out[j] = 0.5 * (d[o.first] + d[o.second]);
      out[n_orb + j] = 0.5 * (d[n_pairs_ + o.first] + d[n_pairs_ + o.second]);
    }
  }
  return out;
}

std::vector<double> KramersRestriction::jointFromComplex(const std::vector<std::complex<double>>& c) {
  std::vector<double> joint(2 * c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    joint[i] = c[i].real();
    joint[c.size() + i] = c[i].imag();
  }
  return joint;
}

std::vector<std::complex<double>> KramersRestriction::complexFromJoint(const std::vector<double>& joint) {
  if (joint.size() % 2 != 0) {
    throw std::runtime_error("KramersRestriction::complexFromJoint: odd joint size");
  }
  const std::size_t m = joint.size() / 2;
  std::vector<std::complex<double>> c(m);
  for (std::size_t i = 0; i < m; ++i) c[i] = {joint[i], joint[m + i]};
  return c;
}

double kramersOccupationDeviation(const std::vector<double>& occupations) {
  double dev = 0.0;
  for (std::size_t k = 0; k + 1 < occupations.size(); k += 2) {
    dev = std::max(dev, std::abs(occupations[k] - occupations[k + 1]));
  }
  return dev;
}

// ---------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------

KramersNeoProblem::KramersNeoProblem(NeoProblem<double>& full, KramersRestriction restriction)
    : full_(full), restriction_(std::move(restriction)) {
  if (full_.dimension() != restriction_.fullSize()) {
    throw std::runtime_error("KramersNeoProblem: problem dimension differs from the restriction's full size");
  }
}

std::vector<double> KramersNeoProblem::gradient() { return restriction_.contract(full_.gradient()); }

std::vector<double> KramersNeoProblem::hessianVector(const std::vector<double>& v) {
  return restriction_.contract(full_.hessianVector(restriction_.expand(v)));
}

std::vector<double> KramersNeoProblem::hessianDiagonal() {
  const std::vector<double> d = full_.hessianDiagonal();
  if (d.empty()) return {};
  return restriction_.contractDiagonal(d);
}

double KramersNeoProblem::trialEnergy(const std::vector<double>& d) {
  return full_.trialEnergy(restriction_.expand(d));
}

void KramersNeoProblem::accept(const std::vector<double>& d) { full_.accept(restriction_.expand(d)); }

KramersAdamProblem::KramersAdamProblem(AdamProblem<std::complex<double>>& full,
                                       KramersRestriction restriction)
    : full_(full), restriction_(std::move(restriction)) {
  if (full_.dimension() != restriction_.nPairs()) {
    throw std::runtime_error("KramersAdamProblem: problem dimension differs from the restriction's pair count");
  }
}

std::vector<std::complex<double>> KramersAdamProblem::gradient() {
  return KramersRestriction::complexFromJoint(restriction_.contractRepresentative(
      KramersRestriction::jointFromComplex(full_.gradient())));
}

void KramersAdamProblem::rotate(const std::vector<std::complex<double>>& step) {
  full_.rotate(KramersRestriction::complexFromJoint(restriction_.expandRepresentative(
      KramersRestriction::jointFromComplex(step))));
}

}  // namespace rerdmft
