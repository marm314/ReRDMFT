#ifndef RERDMFT_INPUT_H
#define RERDMFT_INPUT_H

#include <string>
#include <vector>

#include "PhysicalConstants.h"

namespace rerdmft {

// Coordinates are in Bohr (atomic units), regardless of the input file's
// units (see Input::read).
struct Atom {
  std::string symbol;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

// Parses the ReRDMFT input file and stores the run parameters: number of
// electrons, gaussian basis set file name, and molecular geometry. The
// input file's geometry is in Angstrom; it is converted to Bohr on read.
class Input {
 public:
  void read(const std::string& filename);

  int n_electrons() const { return n_electrons_; }
  const std::string& basis_file() const { return basis_file_; }
  const std::vector<Atom>& geometry() const { return geometry_; }
  // Optional; defaults to false when the DEBUG keyword is absent from the
  // input file. When true, the program prints detailed basis and matrix
  // diagnostics; otherwise it only prints a concise summary.
  bool debug() const { return debug_; }
  // Optional; defaults to 0 when the VERBOSE keyword is absent. Only
  // meaningful alongside DEBUG TRUE: controls which of DEBUG's own
  // cross-checks run, from cheapest to most expensive, so a routine DEBUG
  // run does not always pay for the priciest ones. At the default
  // verbose == 0, the O(n^5) dense-2-RDM cross-check against
  // Hessian_opt/GeneralizedFock.h's fully general generalizedFockMatrix
  // (both the aggregate gradient-norm summary and the element-wise
  // comparison in the finite-difference report) is SKIPPED; at
  // verbose > 0 it runs. Every other DEBUG cross-check (the O(n^4)
  // "efficient" gradient, the finite-difference test itself) is cheap and
  // always runs whenever DEBUG is on, regardless of this setting.
  int verbose() const { return verbose_; }
  // Optional; defaults to the standard CODATA value (PhysicalConstants.h)
  // when the SPEED_OF_LIGHT keyword is absent. Overriding it (e.g. to a
  // very large number) lets you probe the nonrelativistic limit
  // (c -> infinity) or otherwise vary relativistic effects.
  double speed_of_light() const { return speed_of_light_; }
  // Optional; defaults to false when the NON_RELATIVISTIC keyword is
  // absent. When true, additionally builds and diagonalizes the
  // nonrelativistic Schrodinger core Hamiltonian in the Large-component AO
  // basis (SchrodingerKinetic.h), printing its eigenvalues for comparison
  // against the relativistic ones.
  bool non_relativistic() const { return non_relativistic_; }
  // Optional; defaults to false when the C4_SPINOR keyword is absent.
  // When true, additionally builds the full 4-component Dirac-Hartree-Fock
  // two-electron Coulomb repulsion tensor in the RKB spinor basis
  // (C4_DHF/RkbTwoElectron.h) -- opt-in since both its time and memory
  // cost scale steeply with basis size.
  bool c4_spinor() const { return c4_spinor_; }
  // Optional; defaults to false when the HESSIAN_NON_REL keyword is
  // absent. When true, builds the FULL cheap (Hartree/exchange-ansatz,
  // Hessian_opt/HartreeExchangeHessian.h) orbital-rotation Hessian over
  // the whole spin-orbital space for the converged NON_REL solution and
  // diagonalizes it, reporting whether it is a genuine minimum (no
  // negative eigenvalues). Independent of DEBUG/VERBOSE -- its own
  // opt-in diagnostic, since the full Hessian/diagonalization cost is
  // O(n^5)/O(n^6) and not needed for a routine DEBUG run.
  bool hessian_non_rel() const { return hessian_non_rel_; }
  // Optional; defaults to false when the HESSIAN_4C keyword is absent.
  // Same as hessian_non_rel(), but for the converged C4_DHF solution --
  // the Hessian spans the FULL RKB spinor space (including the
  // negative-energy branch), expected to show negative eigenvalues
  // (a saddle point) rather than a minimum.
  bool hessian_4c() const { return hessian_4c_; }
  // Optional; defaults to 0.4 when the MIXING keyword is absent. Linear
  // density-matrix mixing weight for the C4_DHF SCF loop (C4_DHF/C4_DHF.h):
  // the density fed into the next iteration's Fock build is
  // mixing*P_new + (1-mixing)*P_current. Must be in (0, 1].
  double mixing() const { return mixing_; }
  // Optional; defaults to 100 when MAX_ITERATIONS is absent. Maximum
  // number of C4_DHF SCF cycles (C4_DHF/C4_DHF.h) before giving up
  // (converged is false in that case, but the last cycle's results are
  // still returned). Must be positive.
  int max_iterations() const { return max_iterations_; }
  // Optional; defaults to 1e-8 Hartree when ENERGY_TOLERANCE is absent.
  // The C4_DHF SCF loop's energy-change convergence threshold. Combined
  // with density_tolerance() by OR (see its note): converged as soon as
  // EITHER one is satisfied, not only when both are. Must be positive.
  double energy_tolerance() const { return energy_tolerance_; }
  // Optional; defaults to 1e-6 when DENSITY_TOLERANCE is absent. The
  // C4_DHF SCF loop's density-change convergence threshold. Combined with
  // energy_tolerance() by OR: the SCF loop calls itself converged (after
  // at least one iteration) as soon as EITHER the density change or the
  // energy change falls below its own tolerance. Must be positive.
  double density_tolerance() const { return density_tolerance_; }
  // Optional; defaults to false when CACHE_INTEGRALS is absent. When
  // true, the two-electron integral tensors (C4_DHF's RKB spinor tensor
  // and/or NON_REL's Large-basis tensor, whichever apply) are cached to
  // disk under cache_dir() and reused on a later run with the same
  // geometry+basis (regardless of SPEED_OF_LIGHT, which these tensors do
  // not depend on -- see IntegralCache.h) instead of being recomputed
  // via libcint from scratch. Off by default since it writes files to
  // disk and a cache built by a different code version is only detected
  // (and safely ignored) via an embedded format-version tag, not a
  // content check.
  bool cache_integrals() const { return cache_integrals_; }
  // Optional; defaults to ".rerdmft_cache" when CACHE_DIR is absent.
  // Directory (created if missing) that cache_integrals() cache files
  // are written to/read from.
  const std::string& cache_dir() const { return cache_dir_; }
  // Optional; defaults to "SD" when FUNCTIONAL is absent. Selects which
  // JK-only density matrix functional approximation (Occ_opt/JK_only.h,
  // Table 1 of Rodriguez-Mayorga et al., PCCP (2017)) main.cpp's
  // buildFunctionalReport evaluates on the converged HF/DHF orbitals --
  // one of SD, MBB (or MULLER), BBC2, CA, CGA, ML, MLSIC, GU, POWER
  // (case-insensitive; validated against this exact list, throws
  // otherwise). Stored as a plain string rather than Occ_opt's own
  // JkFunctional enum so that Input.h stays independent of Occ_opt.
  // The default here is never actually USED for that evaluation --
  // see has_functional() below, which gates the whole step on the
  // keyword being explicitly present.
  const std::string& functional() const { return functional_; }
  // True iff the FUNCTIONAL keyword was explicitly present in the input
  // file (as opposed to functional() just reading its default "SD").
  // main.cpp's fractional-occupation RDMFT functional evaluation
  // (Fermi-Dirac smearing + Occ_opt/JK_only.h, printed right after each
  // "Total ... energy" line) runs ONLY when this is true -- with no
  // FUNCTIONAL keyword given, there is no RDMFT functional to evaluate,
  // so the whole step (and its extra cost) is skipped entirely.
  bool has_functional() const { return has_functional_; }
  // Optional; defaults to 1000.0 (Kelvin) when TEMPERATURE is absent.
  // The ELECTRONIC temperature used to smear the converged HF/DHF
  // orbital energies into fractional Fermi-Dirac occupations
  // (Occ_opt/FermiDirac.h) -- only used when occupation_init() selects
  // "FERMI_DIRAC" (see below); ignored (but still validated as
  // positive) otherwise. Must be positive.
  double temperature() const { return temperature_; }
  // Optional; defaults to "PROPORTIONAL" when OCCUPATION_INIT is
  // absent. Selects how main.cpp's buildFunctionalReport generates the
  // initial fractional occupation numbers functional() is evaluated at
  // / an SQP occupation-number optimization (Occ_opt/SQP.h) starts
  // from -- one of PROPORTIONAL (Occ_opt/OccupationInit.h's aufbau
  // reference, redistributed proportionally into the interior box;
  // temperature-independent, the DEFAULT) or FERMI_DIRAC (smeared at
  // temperature() instead). Case-insensitive, validated against this
  // exact list, throws otherwise. Stored as a plain string rather than
  // Occ_opt's own OccupationInitMethod enum, for the same reason
  // functional() is (Input.h stays independent of Occ_opt). Only
  // meaningful when has_functional() is true.
  const std::string& occupation_init() const { return occupation_init_; }

 private:
  int n_electrons_ = 0;
  std::string basis_file_;
  std::vector<Atom> geometry_;
  bool debug_ = false;
  int verbose_ = 0;
  double speed_of_light_ = kSpeedOfLight;
  bool non_relativistic_ = false;
  bool c4_spinor_ = false;
  bool hessian_non_rel_ = false;
  bool hessian_4c_ = false;
  double mixing_ = 0.4;
  int max_iterations_ = 100;
  double energy_tolerance_ = 1e-8;
  double density_tolerance_ = 1e-6;
  bool cache_integrals_ = false;
  std::string cache_dir_ = ".rerdmft_cache";
  std::string functional_ = "SD";
  bool has_functional_ = false;
  double temperature_ = 1000.0;
  std::string occupation_init_ = "PROPORTIONAL";
};

}  // namespace rerdmft

#endif  // RERDMFT_INPUT_H
