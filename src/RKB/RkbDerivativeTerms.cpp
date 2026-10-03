#include "RkbDerivativeTerms.h"

#include <cmath>

#include "Integrals.h"

namespace rerdmft {

namespace {

// 1D Gaussian moment: integral x^{2n} exp(-2*a*x^2) dx = (2n-1)!! / (4a)^n * sqrt(pi/(2a)).
double oneDimensionalGaussianMoment(int n, double a) {
  double double_factorial = 1.0;
  for (int k = 2 * n - 1; k > 0; k -= 2) double_factorial *= static_cast<double>(k);
  return double_factorial / std::pow(4.0 * a, n) * std::sqrt(M_PI / (2.0 * a));
}

// The TRUE (libcint-convention-independent) self-overlap of a single, unit-coefficient cartesian
// primitive x^lx y^ly z^lz exp(-a r^2) against itself -- pure calculus, no normalization convention
// attached. Used (via libcintConventionScale below) to work out how libcint's own internal
// convention rescales a primitive when its cartesian type (hence `l`) changes, which plain
// differentiation of the literal polynomial does not know about.
double trueCartesianSelfOverlap(const CartesianExponents& c, double a) {
  return oneDimensionalGaussianMoment(c.lx, a) * oneDimensionalGaussianMoment(c.ly, a) *
         oneDimensionalGaussianMoment(c.lz, a);
}

// libcint's cartesian integral routines do not treat a shell's contraction coefficient as a literal
// multiplier of the bare monomial*Gaussian -- there is a hidden, (l, component)-dependent internal
// scale (confirmed empirically: a unit-coefficient s-type primitive's raw libcint self-overlap
// differs from the true mathematical self-overlap of the same bare function by a factor of 4*pi, a
// p-type one by 4*pi/3). normalizeCartesianBasis's CINTgto_norm step absorbs exactly this for a
// function's OWN (l, component), but buildRkbSmallBasis changes a term's cartesian type (and often
// `l`) relative to its parent, so the same numeric coefficient means a different physical magnitude
// under libcint's convention before and after. This returns the ratio (libcint-convention
// coefficient) / (literal, convention-free coefficient) for one primitive of a given (l, component,
// exponent) -- computed directly from shellSelfOverlap (the same, already-validated machinery
// normalizeCartesianBasis itself uses), not from any assumption about libcint's internal formula.
double libcintConventionScale(int l, const CartesianExponents& c, double a) {
  const Matrix<double> raw = shellSelfOverlap(l, {a}, {1.0});
  const int idx = cartesianComponentIndex(l, c);
  const double raw_self_overlap = raw(static_cast<std::size_t>(idx), static_cast<std::size_t>(idx));
  return std::sqrt(trueCartesianSelfOverlap(c, a) / raw_self_overlap);
}

// Rescales `coefficient` (already in libcint convention for (l_old, comp_old)) by `weight` (the
// pure-calculus differentiation weight) and corrects for libcint's own (l, component)-dependent
// internal scale changing between (l_old, comp_old) and (l_new, comp_new) at the same exponent `a`.
double convertDerivativeCoefficient(double coefficient, double weight, int l_old,
                                     const CartesianExponents& comp_old, int l_new,
                                     const CartesianExponents& comp_new, double a) {
  const double scale_old = libcintConventionScale(l_old, comp_old, a);
  const double scale_new = libcintConventionScale(l_new, comp_new, a);
  return weight * coefficient * (scale_new / scale_old);
}

// Appends d/d(axis) of `fn` (lowering piece only, when present) to `basis`, returning its index.
std::size_t appendLowering(std::vector<BasisFunction>& basis, const BasisFunction& fn, int axis,
                            int exponent_along_axis) {
  if (exponent_along_axis <= 0) return RkbDerivativeTerms::kNone;
  BasisFunction term = fn;
  CartesianExponents comp_new = fn.cartesian;
  if (axis == 0) {
    term.cartesian.lx -= 1;
    comp_new.lx -= 1;
  } else if (axis == 1) {
    term.cartesian.ly -= 1;
    comp_new.ly -= 1;
  } else {
    term.cartesian.lz -= 1;
    comp_new.lz -= 1;
  }
  term.l = fn.l - 1;
  for (std::size_t k = 0; k < term.coefficients.size(); ++k) {
    term.coefficients[k] =
        convertDerivativeCoefficient(fn.coefficients[k], static_cast<double>(exponent_along_axis),
                                      fn.l, fn.cartesian, term.l, comp_new, fn.exponents[k]);
  }
  const std::size_t index = basis.size();
  basis.push_back(std::move(term));
  return index;
}

// Appends d/d(axis) of `fn` (raising piece, always present) to `basis`, returning its index.
std::size_t appendRaising(std::vector<BasisFunction>& basis, const BasisFunction& fn, int axis) {
  BasisFunction term = fn;
  CartesianExponents comp_new = fn.cartesian;
  if (axis == 0) {
    term.cartesian.lx += 1;
    comp_new.lx += 1;
  } else if (axis == 1) {
    term.cartesian.ly += 1;
    comp_new.ly += 1;
  } else {
    term.cartesian.lz += 1;
    comp_new.lz += 1;
  }
  term.l = fn.l + 1;
  for (std::size_t k = 0; k < term.coefficients.size(); ++k) {
    term.coefficients[k] = convertDerivativeCoefficient(
        fn.coefficients[k], -2.0 * fn.exponents[k], fn.l, fn.cartesian, term.l, comp_new,
        fn.exponents[k]);
  }
  const std::size_t index = basis.size();
  basis.push_back(std::move(term));
  return index;
}

}  // namespace

std::vector<BasisFunction> buildRkbSmallBasis(const std::vector<BasisFunction>& large_basis,
                                               std::vector<RkbDerivativeTerms>* terms_out) {
  std::vector<BasisFunction> small_basis;
  small_basis.reserve(large_basis.size() * 6);
  if (terms_out) {
    terms_out->clear();
    terms_out->reserve(large_basis.size());
  }

  for (const BasisFunction& fn : large_basis) {
    RkbDerivativeTerms terms;
    terms.lower_x = appendLowering(small_basis, fn, 0, fn.cartesian.lx);
    terms.raise_x = appendRaising(small_basis, fn, 0);
    terms.lower_y = appendLowering(small_basis, fn, 1, fn.cartesian.ly);
    terms.raise_y = appendRaising(small_basis, fn, 1);
    terms.lower_z = appendLowering(small_basis, fn, 2, fn.cartesian.lz);
    terms.raise_z = appendRaising(small_basis, fn, 2);
    if (terms_out) terms_out->push_back(terms);
  }

  return small_basis;
}

}  // namespace rerdmft
