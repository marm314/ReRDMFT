#ifndef RERDMFT_INPUT_H
#define RERDMFT_INPUT_H

#include <iosfwd>
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
  // At verbose > 1 (C4_SPINOR only), an additional, less commonly
  // needed check ALSO runs: the MIXED real/imaginary orbital-rotation
  // Hessian block (Hessian_opt/HartreeExchangeHessian.h's
  // hartreeExchangeHessianElementMixed) is validated against a genuine
  // mixed-direction (real step on one pair, imaginary step on another)
  // 2D finite difference -- a strictly higher bar than the dense-2-RDM
  // check's own verbose > 0, since it is newer and not needed for a
  // routine DHF DEBUG run.
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
  // Optional; defaults to false when the X2C keyword is absent. When
  // true:
  //   1. Prints the one-electron X2C decoupling report
  //      (X2C_DHF/X2C_decoupling.h): the eigenvalues of the
  //      orthonormalized bare RKB Dirac Hamiltonian H_RKB_ortho, in two
  //      columns (Kramers pairs side by side), plus the Kramers-pair
  //      splitting check, and the exact vs. approximate X2C Hamiltonian
  //      eigenvalue comparison (X2C_DHF/X2C_hamiltonian.h).
  //   2. Runs an approximate X2C Hartree-Fock SCF (X2C_DHF/X2C_HF.h):
  //      the one-electron core Hamiltonian is the EXACT X2C Hamiltonian
  //      (X2C_DHF/X2C_hamiltonian.h's h_x2c, built ONCE from the bare
  //      one-electron RKB Hamiltonian -- no picture-change correction
  //      of any kind, one-electron or two-electron), the two-electron
  //      part is the ORDINARY (real, non-relativistic) Large-component
  //      Coulomb integrals, and the Fock matrix is orthogonalized at
  //      EVERY iteration using only the plain large-component overlap
  //      (X_Large), never the exact renormalization metric Lambda --
  //      so this is a genuinely approximate treatment, not exact
  //      X2C-DHF. The density matrix's coefficients are C = X_Large * U
  //      (U = the eigenvectors diagonalizing Fock_ortho).
  //   3. After the SCF converges, transforms h_x2c and the Large-
  //      component spin-orbital two-electron integrals into the
  //      converged X2C-HF MO basis (X2C_DHF/X2C_MoTransform.h) and runs
  //      the SAME Hessian_opt gradient/Hessian test suite as C4_SPINOR
  //      does for DHF (RDMFT-ansatz gradient always on; under DEBUG,
  //      the X2C-HF-specific efficient gradient
  //      (X2C_DHF/X2C_OrbitalGradient.h), the general dense-2-RDM cross-
  //      check at VERBOSE > 0, a finite-difference gradient/Hessian
  //      check, and the mixed real/imaginary Hessian block at
  //      VERBOSE > 1) -- see hessian_x2c() below for the corresponding
  //      full-Hessian-diagonalization diagnostic.
  // This whole report/SCF is printed between the NON_RELATIVISTIC and
  // C4_SPINOR (4-component DHF) final reports, regardless of whether
  // either of those keywords is itself on. With DEBUG also true, extra
  // detail is added throughout: the decoupling step's own Kramers
  // eigenvector-partner-deviation and generalized-eigenproblem-residual
  // checks, per-iteration orbital energies within the X2C-HF SCF history,
  // a residual check confirming C = X_Large * U genuinely solves
  // F C = S_Large C E (validating the coefficient formula above), and
  // everything listed in point 3 above.
  // H_RKB_ortho/h_x2c themselves are always built regardless (H_RKB_
  // ortho is needed as the C4_DHF SCF's own initial guess whenever
  // C4_SPINOR is on); this keyword only gates running/printing the
  // above. Independent of C4_SPINOR.
  bool x2c() const { return x2c_; }
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
  // Optional; defaults to false when the HESSIAN_X2C keyword is
  // absent. Same as hessian_non_rel()/hessian_4c(), but for the
  // converged X2C-HF solution (X2C_DHF/X2C_HF.h) -- the Hessian spans
  // the FULL X2C-HF spinor space (2*nLarge, no negative-energy branch
  // at all, unlike C4_DHF: X2C's own decoupling already eliminated
  // it), so this is expected to show a genuine MINIMUM (no negative
  // eigenvalues), same as hessian_non_rel(), NOT a saddle point.
  bool hessian_x2c() const { return hessian_x2c_; }
  // Optional; defaults to false when the HESSIAN_FUNCTIONAL keyword is
  // absent. Only meaningful together with FUNCTIONAL (JK-only or PNOF,
  // Occ_opt/): after the occupation-number optimization at fixed
  // orbitals, builds the FULL real-step orbital-rotation Hessian of
  // THAT functional (Hessian_opt/JkOnlyHessian.h's jkOnlyHessianMatrix
  // for JK-only, PnofHessian.h's pnofHessianMatrix for PNOF) at the
  // optimized occupations, symmetrizes it, diagonalizes it, and reports
  // the number of negative/near-zero/positive eigenvalues -- the
  // fractional-occupation analogue of hessian_non_rel()/hessian_x2c()/
  // hessian_4c(), applied to whichever SCF paths (NON_REL, X2C,
  // C4_DHF) run the FUNCTIONAL step. The point is stationary w.r.t. the
  // occupations only (orbitals stay the converged HF/DHF ones), so the
  // report also prints the orbital gradient norm and the Hessian's
  // asymmetry. Expensive (O(n^5) build + O(n^6) diagonalization), hence
  // its own opt-in keyword.
  bool hessian_functional() const { return hessian_functional_; }
  // Optional; defaults to 0.4 when the MIXING keyword is absent. Linear
  // density-matrix mixing weight for the C4_DHF SCF loop (C4_DHF/C4_DHF.h):
  // the density fed into the next iteration's Fock build is
  // mixing*P_new + (1-mixing)*P_current. Must be in (0, 1].
  double mixing() const { return mixing_; }
  // Optional; defaults to TRUE when the DIIS keyword is absent. TRUE: the three Hartree-Fock SCF
  // loops (NON_REL, X2C, C4_SPINOR) use Pulay's DIIS (Utils/DIIS.h) on the Fock matrix -- error
  // F P S - S P F in the AO basis -- instead of the linear density mixing (MIXING is then
  // ignored). DIIS FALSE selects the linear density mixing with weight MIXING.
  bool diis() const { return diis_; }
  // Optional; defaults to 5 when DIIS_SIZE is absent. Number of (error, Fock) pairs DIIS keeps.
  // Must be at least 2. Only used with DIIS TRUE.
  int diis_size() const { return diis_size_; }
  // The history length handed to the SCF loops: diis_size() with DIIS TRUE, else 0 (linear
  // mixing, DIIS FALSE).
  int scf_diis_size() const { return diis_ ? diis_size_ : 0; }
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
  // Optional; defaults to "RESTART" when RESTART_FILE is absent. Base name of the binary restart
  // files (Utils/Restart.h) written at the end of a NON_REL / X2C RDMFT run with a FUNCTIONAL:
  // "<base>.NON_REL" and "<base>.X2C_HF". The word NONE disables the files.
  const std::string& restart_file() const { return restart_file_; }
  // Optional; defaults to FALSE when the CHOLESKY keyword is absent --
  // the original direct 4-leg transform is the default everywhere. When
  // TRUE, every two-electron integral basis TRANSFORMATION in this
  // project (NON_REL's AO->MO transform, C4_DHF's RKB-spinor AO->MO
  // transform, X2C_DHF's Large-spin-orbital AO->MO transform, and
  // C4_DHF/RkbTwoElectron.cpp's own unrestricted- to restricted-kinetic-
  // balance Small-basis projection) instead goes through a pivoted
  // Cholesky decomposition of the SOURCE tensor first (Utils/
  // Cholesky_Decomposition.h), transforming only the resulting (often
  // far fewer) Cholesky VECTORS instead of the full O(n^4) tensor, then
  // reconstructing in the new basis. This helps when the source tensor
  // actually compresses (e.g. the UKB->RKB Small-basis projection); the
  // AO/spinor->MO transforms often show little to no rank reduction (the
  // complex spinor pair-space lacks the symmetry real AO integrals have),
  // so CHOLESKY TRUE is opt-in rather than default. Numerically
  // negligible accuracy cost when enabled, controlled by the
  // decomposition's own internal threshold (tight enough that CHOLESKY
  // TRUE reproduces CHOLESKY FALSE's converged energies to the full
  // displayed precision, not merely approximately).
  bool cholesky() const { return cholesky_; }
  // Residual-diagonal cutoff for the pivoted Cholesky decomposition above
  // (Utils/Cholesky_Decomposition.h's own `threshold` parameter) -- below
  // this, a pivot is considered numerical noise and decomposition stops.
  // Defaults to 1e-10, tight enough in practice that CHOLESKY TRUE
  // reproduces CHOLESKY FALSE's converged energies to full displayed
  // precision on every tested example; exposed as a keyword so the user
  // can loosen it (fewer Cholesky vectors, faster, less accurate) or
  // tighten it (more vectors, slower, closer to exact) without a rebuild.
  // Only meaningful when CHOLESKY is TRUE.
  double cholesky_threshold() const { return cholesky_threshold_; }
  // Optional; defaults to "SD" when FUNCTIONAL is absent. Selects which
  // JK-only density matrix functional approximation (Occ_opt/JK_only.h,
  // Table 1 of Rodriguez-Mayorga et al., PCCP (2017)) main.cpp's
  // buildFunctionalReport evaluates on the converged HF/DHF orbitals --
  // one of SD, MBB (or MULLER), BBC2, CA, CGA, ML, MLSIC, GU, POWER,
  // MULLER_AS (case-insensitive; validated against this exact list, throws
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
  // / an SQP occupation-number optimization (Utils/SQP.h) starts
  // from -- one of PROPORTIONAL (Occ_opt/OccupationInit.h's aufbau
  // reference, redistributed proportionally into the interior box;
  // temperature-independent, the DEFAULT) or FERMI_DIRAC (smeared at
  // temperature() instead). Case-insensitive, validated against this
  // exact list, throws otherwise. Stored as a plain string rather than
  // Occ_opt's own OccupationInitMethod enum, for the same reason
  // functional() is (Input.h stays independent of Occ_opt). Only
  // meaningful when has_functional() is true.
  const std::string& occupation_init() const { return occupation_init_; }
  // Optional; defaults to 1 when PNOF_SUBSPACES is absent. How many
  // independent PNOF-style (Piris) coupling subspaces to build
  // (Occ_opt/Orb_subspaces.h), one per occupied orbital PAIR outward
  // from HOMO: PNOF_SUBSPACES 1 (the default) builds just the HOMO
  // subspace, PNOF_SUBSPACES 2 additionally builds a separate HOMO-1
  // subspace, and so on -- where "pair" means a spin-orbital/spinor's
  // own degenerate partner (NON_REL's alpha/beta partner, or X2C/
  // C4_DHF's Kramers partner). Each subspace's own SIZE (how many
  // unoccupied pairs it couples its occupied pair to) is set by
  // pnof_coupling() below. The default PNOF_SUBSPACES=1 combined with
  // pnof_coupling()'s own default of 2 is a genuine, meaningful
  // configuration (plain HOMO/LUMO perfect pairing), not a "disabled"
  // sentinel -- building the subspace table is currently the only thing
  // this keyword feeds; it is not yet wired into the FUNCTIONAL/SQP
  // occupation-optimization step itself, so nothing actually USES this
  // value yet regardless of what it is set to. Must be at least 1; if
  // combined with pnof_coupling() into a combination that asks for more
  // occupied or unoccupied orbitals than the basis actually has,
  // Occ_opt/Orb_subspaces.h's buildOrbitalSubspaces throws a
  // std::runtime_error explaining exactly how many were requested vs.
  // how many are available.
  int pnof_subspaces() const { return pnof_subspaces_; }
  // Optional; defaults to 2 when PNOF_COUPLING is absent. The SIZE of
  // each PNOF subspace above, counted in orbital PAIRS: 1 occupied pair
  // + (PNOF_COUPLING - 1) unoccupied pairs. PNOF_COUPLING 2 (the
  // default) is plain perfect pairing (each subspace's occupied pair
  // coupled to exactly one unoccupied pair, e.g. HOMO with LUMO alone);
  // PNOF_COUPLING 3 couples each occupied pair to its TWO closest
  // unoccupied pairs instead (e.g. HOMO with BOTH LUMO and LUMO+1), and
  // so on. When pnof_subspaces() > 1, every subspace's unoccupied pairs
  // are a DISJOINT block (see Occ_opt/Orb_subspaces.h) -- e.g.
  // PNOF_SUBSPACES 2 + PNOF_COUPLING 3 couples HOMO with {LUMO,LUMO+1}
  // and, separately, HOMO-1 with {LUMO+2,LUMO+3}. Must be at least 2;
  // see pnof_subspaces()'s own comment for what happens if the
  // requested combination exceeds the basis's available orbitals.
  int pnof_coupling() const { return pnof_coupling_; }
  // Optional; defaults to FALSE. PNOF occupation-number optimization
  // (main.cpp's buildPnofFunctionalReport) defaults to L-BFGS over the
  // UNCONSTRAINED gamma angles (Utils/LBFGS.h + Occ_opt/PNOFs.h's
  // trigonometric parameterization, standalone_donof's own approach) --
  // SQP_PNOF_OCC TRUE switches to the box+equality-constrained SQP
  // solver (Utils/SQP.h) over the occupations directly instead. The two
  // methods solve the SAME problem and agree to full displayed precision
  // whenever both converge cleanly (see README.md's own PNOF section);
  // gamma/LBFGS is the default because it needs no explicit constraint
  // handling at all (gamma already guarantees sum(n)=1 and 0<n<1 for any
  // real angle) and has empirically converged at least as reliably.
  bool sqp_pnof_occ() const { return sqp_pnof_occ_; }

  // FULL_OPTIMIZATION: after the occupation-number optimization at the
  // HF/X2C orbitals, validate the ADAM/Kramers-restriction machinery and
  // then macro-iterate ADAM orbital rotations and occupation re-
  // optimization to convergence (Full_opt/FullOptimization.h). NON_REL and
  // X2C only (not the 4-component path). MAX_MACRO_ITERATIONS,
  // MACRO_ENERGY_TOLERANCE (the Fortran's tolE) and
  // ORBITAL_GRADIENT_TOLERANCE (ADAM's 10**-itolLambda) control it.
  bool full_optimization() const { return full_optimization_; }
  int max_macro_iterations() const { return max_macro_iterations_; }
  double macro_energy_tolerance() const { return macro_energy_tolerance_; }
  double orbital_gradient_tolerance() const { return orbital_gradient_tolerance_; }

  // Prints the current value of EVERY input variable (one per line, after
  // the geometry), so a run's output records exactly what was in effect.
  // MAINTENANCE: whenever a new input keyword/member is added to Input,
  // add it to Input::print in Input.cpp too.
  void print(std::ostream& out) const;

 private:
  int n_electrons_ = 0;
  std::string basis_file_;
  std::vector<Atom> geometry_;
  bool debug_ = false;
  int verbose_ = 0;
  double speed_of_light_ = kSpeedOfLight;
  bool non_relativistic_ = false;
  bool c4_spinor_ = false;
  bool x2c_ = false;
  bool hessian_non_rel_ = false;
  bool hessian_4c_ = false;
  bool hessian_x2c_ = false;
  bool hessian_functional_ = false;
  double mixing_ = 0.4;
  bool diis_ = true;
  int diis_size_ = 5;
  int max_iterations_ = 100;
  double energy_tolerance_ = 1e-8;
  double density_tolerance_ = 1e-6;
  bool cache_integrals_ = false;
  bool cholesky_ = false;
  double cholesky_threshold_ = 1e-10;
  std::string cache_dir_ = ".rerdmft_cache";
  std::string restart_file_ = "RESTART";
  std::string functional_ = "SD";
  bool has_functional_ = false;
  double temperature_ = 1000.0;
  std::string occupation_init_ = "PROPORTIONAL";
  int pnof_subspaces_ = 1;
  int pnof_coupling_ = 2;
  bool sqp_pnof_occ_ = false;
  bool full_optimization_ = false;
  int max_macro_iterations_ = 1000;
  double macro_energy_tolerance_ = 1e-9;
  double orbital_gradient_tolerance_ = 1e-5;
};

}  // namespace rerdmft

#endif  // RERDMFT_INPUT_H
