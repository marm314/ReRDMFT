#ifndef RERDMFT_INTEGRALS_H
#define RERDMFT_INTEGRALS_H

#include <string>
#include <vector>

#include "Matrix.h"
#include "MolecularBasis.h"

namespace rerdmft {

// Diagnostic record for one cartesian atomic orbital's self-overlap check.
struct NormalizationCheck {
  std::string element;
  int l = 0;
  CartesianExponents cartesian;
  double self_overlap_before = 0.0;  // after primitive normalization, before the fix
  bool was_renormalized = false;
};

// Applies primitive Gaussian normalization (via libcint's CINTgto_norm) to
// every basis function, then uses libcint's cartesian overlap integral to
// verify each cartesian AO has unit self-overlap, rescaling its contraction
// coefficients whenever it does not. This is what happens for cartesian
// components that mix axes (e.g. d_xy, f_xyz): primitive normalization
// alone is not enough for them, since it does not depend on how the l
// quanta are split between lx, ly and lz.
//
// Returns a diagnostic report, one entry per basis function, in the same
// order as `functions`.
std::vector<NormalizationCheck> normalizeCartesianBasis(
    std::vector<BasisFunction>& functions, double tolerance = 1e-8);

// Computes the full (real, symmetric) AO overlap matrix <AO_i|AO_j> for an
// already-normalized cartesian AO basis, via libcint. Diagonal entries
// should come out as 1 (up to floating-point round-off), which also serves
// as an ongoing check on normalizeCartesianBasis's result.
Matrix<double> overlapMatrix(const std::vector<BasisFunction>& basis);

// The nf x nf (nf = cartesianComponents(l).size()) self-overlap of ONE contracted shell (all its
// cartesian components at once), placed at an arbitrary center -- a Gaussian shell's self overlap
// does not depend on where it is centered. `exponents`/`coefficients` are raw (pre-primitive-
// normalization) values, exactly as normalizeCartesianBasis itself expects them -- it is in fact
// built from this same routine (Integrals.cpp). Exposed for Utils/SphericalTransform.h, which needs
// this exact overlap convention (same primitive normalization, before normalizeCartesianBasis's own
// subsequent per-component rescaling) to derive the universal Cartesian-to-real-solid-harmonic
// transformation for a given l: that transformation only depends on l, not on any particular shell's
// actual exponents/contraction (the shell self-overlap is always a single overall scalar times a
// universal, l-dependent matrix -- see SphericalTransform.cpp's own derivation).
Matrix<double> shellSelfOverlap(int l, const std::vector<double>& exponents,
                                 const std::vector<double>& coefficients);

}  // namespace rerdmft

#endif  // RERDMFT_INTEGRALS_H
