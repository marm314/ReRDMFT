#ifndef RERDMFT_UTILS_AIMPAC_H
#define RERDMFT_UTILS_AIMPAC_H

#include <complex>
#include <string>
#include <vector>

#include "Input.h"
#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// One natural orbital, ready to be written to a WFN file (or merged with orbitals from a second,
// independent basis block -- see writeAimpacWfn below): `coefficients` is in terms of whichever
// basis the orbital was diagonalized against, in that basis's own BasisFunction order.
struct NaturalOrbital {
  double occupation = 0.0;
  std::vector<double> coefficients;
};

// Diagonalizes a 1-RDM `d_ao` given in a non-orthogonal Cartesian AO basis (overlap `s`, real,
// e.g. overlapMatrix(basis)) into natural orbitals, the way AIMPAC itself needs them. Only the
// real part of `d_ao` is used -- the imaginary part cannot contribute to a density expressed over
// real Gaussian functions (X2C/C4_DHF orbitals are complex; NON_REL's are trivially already real).
//
// Natural orbitals of a NON-orthogonal-basis density matrix D (the standard AO convention, D = C
// diag(occ) C^T with C^T S C = I) solve D S c = n c, NOT D c = n S c -- verified numerically: for
// a single normalized orbital c with occupation 1 (D = cc^T), D S c = c exactly (eigenvalue 1),
// while D c = n S c would force n = (c^T c)/(c^T S c) != 1 whenever S != I. Substituting c = X y
// (X = canonicalOrthogonalize(s, ...), spanning exactly the kept, non-linearly-dependent AO
// directions, X^T S X = I) turns D S c = n c into the ordinary symmetric eigenproblem
// X^T (S D S) X y = n y (left-multiplying by (S X)^T and using X^T S X = I) -- NOT X^T D X, which
// silently gives wrong eigenvalues whenever S != I. Recovered coefficients (in the ORIGINAL,
// non-orthogonalized, normalized AO basis `s`/`d_ao` are indexed by) are then c = X y -- same as
// the textbook S^{1/2} D S^{1/2} / C_no = S^{-1/2} V recipe, without ever forming S^{1/2}/S^{-1/2}
// explicitly. The canonical orthogonalization itself is purely an internal numerical device (AIMPAC
// never diagonalizes/orthogonalizes the Gaussians -- see writeAimpacWfn's own comment), not a
// property of the returned orbitals, which stay expressed over the plain, non-orthogonal `s`'s own
// AO indexing.
//
// Returned in DECREASING occupation order. `lin_dep_threshold`: passed straight to
// canonicalOrthogonalize (e.g. input.x_lin_dep_thrs_l() for a Large AO basis,
// input.x_lin_dep_thrs_s() for a Small one).
std::vector<NaturalOrbital> naturalOrbitalsFromDensity(const Matrix<double>& s,
                                                        const Matrix<std::complex<double>>& d_ao,
                                                        double lin_dep_threshold);

// Writes an AIMPAC/AIMAll-style WFN file from an ALREADY-COMPUTED list of natural orbitals (e.g.
// naturalOrbitalsFromDensity's own output -- see its comment for the AIMPAC-specific
// diagonalization; AIMPAC itself just reads the plain, already-normalized Cartesian Gaussian
// primitives and these orbitals' coefficients over them, no orthogonalization on its end either).
// `basis` is the FULL list of Cartesian AOs every orbital's `coefficients` is indexed against; for
// a single physical component (NON_REL, X2C) this is just that method's own Large AO basis, but
// C4_DHF's Large and Small components are genuinely different Gaussians with NO cross density term
// (rho(r) = rho_Large(r) + rho_Small(r) exactly, no interference cross term -- they live in
// disjoint 4-spinor sectors) -- so main.cpp calls naturalOrbitalsFromDensity ONCE per component
// (Large against its own overlap, Small against ITS OWN, via the RKB small AO basis and
// rkb_coefficients' closed-form relation back to it), zero-pads each orbital's coefficients over
// the OTHER component's primitives, and passes the concatenated `basis` (Large ++ Small) and
// merged orbital list here. Orbitals are re-sorted by decreasing occupation and renumbered
// (MO 1, 2, ...) regardless of the input list's own order, and any orbital whose |occupation| is
// below `occ_print_threshold` (default: half the last printed decimal, i.e. "rounds to 0 at the
// printed precision") is dropped entirely rather than written.
// `total_energy`/`virial_ratio`: written verbatim on the trailing "TOTAL ENERGY" line (the virial
// ratio -V/T has no clean meaning for a relativistic density; callers for whom it is not available
// may pass 0.0 -- AIMPAC itself does not use either value for its QTAIM integration).
void writeAimpacWfn(const std::string& path, const std::string& title, const std::vector<Atom>& geometry,
                     const std::vector<BasisFunction>& basis, const std::vector<NaturalOrbital>& orbitals,
                     double total_energy, double virial_ratio, double occ_print_threshold = 5e-8);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_AIMPAC_H
