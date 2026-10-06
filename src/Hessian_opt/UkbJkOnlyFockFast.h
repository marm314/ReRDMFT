#ifndef RERDMFT_UKBJKONLYFOCKFAST_H
#define RERDMFT_UKBJKONLYFOCKFAST_H

#include <complex>
#include <cstddef>
#include <vector>

#include "JK_only.h"
#include "Matrix.h"
#include "UkbFockMatrixDirect.h"

namespace rerdmft {

// Cost-optimized counterpart to UkbJkOnlyFock.h's ukbJkOnlyFockMatrix: for the 6 functionals
// whose two_rdm_h/x have NO diagonal (i==j) special case in JK_only.cpp's own
// jkHartreeFunction/jkExchangeFunction (verified directly against that file, not assumed) --
// SD, MBB, CA, CGA, POWER, MULLER_AS -- two_rdm_h(s,q) and two_rdm_x(q,s) are both genuinely
// separable as (function of s) * (function of q), needing at most 2 shared UKB-AO densities
// total (reused, scaled per output row q), instead of the generic path's 2*n_mo. For the other 4
// (BBC2, GU: a diagonal override on an otherwise rank-1-3 off-diagonal form; ML, MLSIC: the
// Marques-Lathiotakis rational form itself has no finite separable sum) this falls through to the
// existing, already-validated generic ukbJkOnlyFockMatrix unchanged -- those are genuinely
// O(n_mo) by this scheme regardless, so there is no decomposition to hand-code for them without
// just reimplementing the generic path under another name.
Matrix<std::complex<double>> ukbJkOnlyFockMatrixFast(const Matrix<std::complex<double>>& h_rkb,
                                                      const UkbDirectEriSource& eri,
                                                      const Matrix<std::complex<double>>& c_rkb,
                                                      const std::vector<double>& occupations,
                                                      JkFunctional functional, std::size_t f_l = 0,
                                                      double power_alpha = 1.0);

}  // namespace rerdmft

#endif  // RERDMFT_UKBJKONLYFOCKFAST_H
