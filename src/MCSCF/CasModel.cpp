#include "CasModel.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "CasFock.h"
#include "LinearAlgebra.h"
#include "OrbitalGradient.h"
#include "SymmetricEri.h"

namespace rerdmft {

namespace {

using Cx = std::complex<double>;

std::vector<double> naturalOccupations(const Matrix<double>& rdm1) {
  std::vector<double> w = diagonalizeSymmetric(rdm1).eigenvalues;
  std::reverse(w.begin(), w.end());
  return w;
}
std::vector<double> naturalOccupations(const Matrix<Cx>& rdm1) {
  std::vector<double> w = diagonalizeHermitian(rdm1).eigenvalues;
  std::reverse(w.begin(), w.end());
  return w;
}

template <typename T>
bool sameMatrix(const Matrix<T>& a, const Matrix<T>& b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i)
    if (a.data()[i] != b.data()[i]) return false;
  return true;
}

}  // namespace

template <typename T, typename Eri>
RdmftModel<T, Eri> makeCasModel(std::vector<std::size_t> core, std::vector<std::size_t> active, std::size_t n_active_electrons,
                                std::vector<int> active_spin, std::size_t n_total, CasciOptions ci_options) {
  const std::size_t a = active.size();
  if (a == 0) throw std::runtime_error("makeCasModel: empty active space");
  if (!active_spin.empty() && active_spin.size() != a) throw std::runtime_error("makeCasModel: active_spin must have one entry per active index");

  // orbital class of every index: 0 core, 1 active, 2 virtual
  std::vector<int> kind(n_total, 2);
  for (std::size_t i : core) kind[i] = 0;
  for (std::size_t t : active) {
    if (kind[t] == 0) throw std::runtime_error("makeCasModel: an index is both core and active");
    kind[t] = 1;
  }

  struct Cache {
    std::shared_ptr<Casci<T>> ci;
    std::vector<T> ci_vector;
    Matrix<T> rdm1;
    std::vector<T> rdm2;
    std::vector<double> natural;
    double ci_energy = 0.0;
    bool valid = false;
    std::uint64_t version = 0;
    // casFock result at the last (h) asked for
    Matrix<T> h_key;
    CasFockResult<T> fock;
    std::uint64_t fock_version = std::numeric_limits<std::uint64_t>::max();
  };
  const auto cache = std::make_shared<Cache>();
  cache->ci = std::make_shared<Casci<T>>(a, n_active_electrons);

  RdmftModel<T, Eri> model;

  const auto fockAt = [=](const Matrix<T>& h, const Eri& eri) -> const CasFockResult<T>& {
    if (!cache->valid) throw std::runtime_error("makeCasModel: energy/gradient called before the CI was ever solved");
    if (cache->fock_version != cache->version || !sameMatrix(cache->h_key, h)) {
      cache->fock = casFock(h, eri, core, active, cache->rdm1, cache->rdm2);
      cache->h_key = h;
      cache->fock_version = cache->version;
    }
    return cache->fock;
  };

  model.energy = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>&) { return fockAt(h, eri).energy; };

  model.gradient = [=](const Matrix<T>& h, const Eri& eri, const std::vector<double>&) {
    Matrix<T> g = orbitalGradient(fockAt(h, eri).fock);
    for (std::size_t p = 0; p < g.rows(); ++p)
      for (std::size_t q = 0; q <= p; ++q)
        if (kind[p] == kind[q]) g(p, q) = T{};
    return g;
  };

  model.optimize_occupations = [=](const Matrix<T>& h, const Eri& eri, std::vector<double>& state) {
    const CasciHamiltonian<T> ham = casciHamiltonian(h, eri, core, active, n_active_electrons);
    const Casci<T>& ci = *cache->ci;
    const std::size_t dim = ci.dimension();

    std::vector<T> start;
    const bool is_complex = !std::is_same_v<T, double>;
    if (state.size() == (is_complex ? 2 * dim : dim)) {
      start.resize(dim);
      for (std::size_t i = 0; i < dim; ++i) {
        if constexpr (std::is_same_v<T, double>) start[i] = state[i];
        else start[i] = Cx(state[i], state[dim + i]);
      }
    } else {
      // first call: the lowest-diagonal determinant (restricted to Ms = 0 when the active spin-orbitals' spins are known)
      const std::vector<double> hd = ci.diagonal(ham);
      std::size_t best = dim;
      for (std::size_t i = 0; i < dim; ++i) {
        if (!active_spin.empty()) {  // Ms = 0 always (the spin-restricted rotations assume it)
          int diff = 0;
          for (std::size_t t = 0; t < a; ++t)
            if ((ci.determinants()[i] >> t) & 1u) diff += active_spin[t] == 0 ? 1 : -1;
          if (diff != 0) continue;
          if (ci_options.symmetric_start) {
          // also spin-symmetric (alpha and beta occupy the same spatial orbitals): the starting vector is then invariant under
          // the alpha<->beta exchange, which the spin-restricted rotations assume (a bare lowest-diagonal Ms = 0 determinant
          // can be open-shell, e.g. at the core-Hamiltonian guess of a stretched molecule). Active layout: [alpha m | beta m].
          const std::size_t m = a / 2;
          bool sym = (a % 2 == 0);
          for (std::size_t t = 0; t < m && sym; ++t)
            sym = ((ci.determinants()[i] >> t) & 1u) == ((ci.determinants()[i] >> (t + m)) & 1u);
          if (!sym) continue;
          }
        } else if (is_complex && ci_options.symmetric_start) {
          // spinors (X2C/4C): active spinors come in adjacent Kramers pairs (2k, 2k+1); start from a pair-closed
          // determinant so the starting vector is time-reversal even and the Davidson space stays so (a bare
          // lowest-diagonal determinant can be open-shell in a degenerate shell, e.g. at the core-Hamiltonian guess)
          bool closed = true;
          for (std::size_t t = 0; t + 1 < a && closed; t += 2)
            closed = ((ci.determinants()[i] >> t) & 1u) == ((ci.determinants()[i] >> (t + 1)) & 1u);
          if (!closed) continue;
        }
        if (best == dim || hd[i] < hd[best]) best = i;
      }
      if (best == dim) throw std::runtime_error("makeCasModel: no Ms = 0 determinant exists for this active space");
      start.assign(dim, T{});
      start[best] = T(1.0);
    }

    const CasciResult<T> res = ci.solve(ham, ci_options, &start);
    cache->ci_vector = res.ci;
    ci.densities(res.ci, cache->rdm1, cache->rdm2);
    cache->natural = naturalOccupations(cache->rdm1);
    cache->ci_energy = res.energy;
    cache->valid = true;
    ++cache->version;

    state.assign(is_complex ? 2 * dim : dim, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
      state[i] = std::real(res.ci[i]);
      if (is_complex) state[dim + i] = std::imag(Cx(res.ci[i]));
    }

    std::vector<double> occ(n_total, 0.0);
    for (std::size_t i : core) occ[i] = 1.0;
    for (std::size_t t = 0; t < a; ++t) occ[active[t]] = std::real(cache->rdm1(t, t));

    std::string warning;
    if (!res.converged) {
      std::ostringstream w;
      w << "WARNING: the CASSCF CI solve did not converge (residual " << res.residual_norm << " after " << res.iterations << " iterations)";
      warning = w.str();
    }
    return RdmftOccupationResult{occ, res.energy, res.converged, res.iterations, warning};
  };

  model.print_occupations = [=](const std::vector<double>&, std::ostream& out) {
    if (!cache->valid) return;
    const Casci<T>& ci = *cache->ci;
    out << "    CASSCF (" << n_active_electrons << " electrons in " << a << " active spinors, " << core.size() << " core spinors): CI dimension "
        << ci.dimension() << ", CI energy " << std::fixed << std::setprecision(10) << cache->ci_energy << " (electronic)\n";
    if (cache->h_key.rows() == n_total && cache->fock_version == cache->version) {
      // energy components at the last evaluated point: h-onebody = sum_core h_ii + sum_tu h(a_t,a_u) D_tu, Vee = E - h-onebody
      T h1{};
      for (std::size_t i : core) h1 += cache->h_key(i, i);
      for (std::size_t t = 0; t < a; ++t)
        for (std::size_t u = 0; u < a; ++u) h1 += cache->h_key(active[t], active[u]) * cache->rdm1(t, u);
      const double h_onebody = std::real(Cx(h1));
      out << "    Electronic energy:  " << std::setprecision(10) << cache->fock.energy << " Hartree\n";
      out << "    h-onebody:          " << h_onebody << " Hartree\n";
      out << "    Vee:                " << cache->fock.energy - h_onebody << " Hartree\n";
    }
    out << "    Natural occupation numbers of the active space:\n";
    out << std::setprecision(6);
    for (std::size_t k = 0; k < cache->natural.size(); ++k) {
      out << "      " << std::setw(4) << k + 1 << "  " << cache->natural[k] << "\n";
    }
    // dominant determinants: bit t of the mask = occupation of active index active[t]
    std::vector<std::size_t> order(ci.dimension());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    const std::size_t top = std::min<std::size_t>(5, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top), order.end(), [&](std::size_t x, std::size_t y) {
      return std::norm(Cx(cache->ci_vector[x])) > std::norm(Cx(cache->ci_vector[y]));
    });
    out << "    Leading determinants (weight, occupied active spinors by MO index):\n";
    for (std::size_t k = 0; k < top; ++k) {
      const std::size_t i = order[k];
      out << "      " << std::setprecision(5) << std::norm(Cx(cache->ci_vector[i])) << "  [";
      bool first = true;
      for (std::size_t t = 0; t < a; ++t)
        if ((ci.determinants()[i] >> t) & 1u) { out << (first ? "" : " ") << active[t]; first = false; }
      out << "]\n";
    }
    out << std::defaultfloat << std::setprecision(6);
  };
  return model;
}

template RdmftModel<double, Tensor4<double>> makeCasModel<double, Tensor4<double>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<double, CholeskyEri<double>> makeCasModel<double, CholeskyEri<double>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<double, SymmetricEri<double>> makeCasModel<double, SymmetricEri<double>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<Cx, Tensor4<Cx>> makeCasModel<Cx, Tensor4<Cx>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<Cx, CholeskyEri<Cx>> makeCasModel<Cx, CholeskyEri<Cx>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<Cx, SymmetricEri<Cx>> makeCasModel<Cx, SymmetricEri<Cx>>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<double, RiNonRelSpinMoEri> makeCasModel<double, RiNonRelSpinMoEri>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);
template RdmftModel<Cx, RiMoEri> makeCasModel<Cx, RiMoEri>(std::vector<std::size_t>, std::vector<std::size_t>, std::size_t, std::vector<int>, std::size_t, CasciOptions);

}  // namespace rerdmft
