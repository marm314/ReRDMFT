#ifndef RERDMFT_RIGEMMFOCK_H
#define RERDMFT_RIGEMMFOCK_H

#include <complex>
#include <cstddef>
#include <vector>

#include "Matrix.h"
#include "NonRelSpinRiMoEri.h"
#include "RiMoEri.h"

namespace rerdmft {

// GEMM-based generalized Fock for the RI MO-ERI classes. Same formula, same result as the generic
// hartreeExchangeFockMatrix/jkOnlyFockMatrix (which evaluate eri(a,b,c,d) = sum_P B(P,a,c) B(P,b,d)
// one element at a time, an O(n_aux) strided dot product per call): here the s- and P-sums are
// folded into a single (n x n_aux*n) @ (n_aux*n x n) GEMM over the B tensor, O(n_aux n^3) as
// before but at BLAS speed and with B read contiguously. Non-template overloads, so they win over
// the generic templates for these two types with no change at any call site.
Matrix<std::complex<double>> hartreeExchangeFockMatrix(
    const Matrix<std::complex<double>>& h, const RiMoEri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(), const Matrix<double>& two_rdm_l2 = Matrix<double>());

Matrix<double> hartreeExchangeFockMatrix(
    const Matrix<double>& h, const RiNonRelSpinMoEri& eri, const std::vector<double>& occupations,
    const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x, const std::vector<std::size_t>& pair_of = {},
    const Matrix<double>& two_rdm_l1 = Matrix<double>(), const Matrix<double>& two_rdm_l2 = Matrix<double>());

Matrix<std::complex<double>> jkOnlyFockMatrix(const Matrix<std::complex<double>>& h, const RiMoEri& eri,
                                               const std::vector<double>& occupations,
                                               const Matrix<double>& two_rdm_h, const Matrix<double>& two_rdm_x);

Matrix<double> jkOnlyFockMatrix(const Matrix<double>& h, const RiNonRelSpinMoEri& eri,
                                 const std::vector<double>& occupations, const Matrix<double>& two_rdm_h,
                                 const Matrix<double>& two_rdm_x);

}  // namespace rerdmft

#endif  // RERDMFT_RIGEMMFOCK_H
