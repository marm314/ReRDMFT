#ifndef RERDMFT_UTILS_RDMFILES_H
#define RERDMFT_UTILS_RDMFILES_H

#include <complex>
#include <cstddef>
#include <functional>
#include <ostream>
#include <string>

#include "Matrix.h"

namespace rerdmft {

// Molecular-basis density matrix files of a correlated state (CASSCF_RDM / FUNCTIONAL_RDM), in the record layout of the UCCSD
// CC_RDM files (CC/lambda_Uccsd.h): over n output spinors, 1-based indices,
//   .dm1: records (int32 p, q, float64 re, im) of gamma_pq = <a_p^+ a_q>, every p, q with |value| >= threshold,
//   .dm2: records (int32 p, q, r, s, float64 re, im) of Gamma_pqrs = <a_p^+ a_q^+ a_s a_r>,
// each file ended by an all-zero record. A true (antisymmetric) 2-RDM is written for p<q, r<s only; `antisymmetric = false`
// (the RDMFT functionals' ansatz 2-RDM is not antisymmetric) writes every element with |value| >= threshold instead.
// E = sum h_pq gamma_pq + 1/2 sum <pq|rs> Gamma_pqrs  (<pq|rs> in physics notation, over all p, q, r, s).
struct RdmSource {
  std::size_t n = 0;                           // number of output spinors
  Matrix<std::complex<double>> gamma1;         // <a_p^+ a_q>, n x n
  // out[(q*n + r)*n + s] = Gamma_pqrs for all q, r, s (one slab per p).
  std::function<void(std::size_t p, std::complex<double>* out)> slab;
  bool antisymmetric = true;
  Matrix<std::complex<double>> h;              // optional one-electron matrix over the output spinors (for the h-onebody report)
};

struct RdmFilesResult {
  std::size_t n_dm1 = 0, n_dm2 = 0;            // records written (without the terminating zero record)
  double trace_dm1 = 0.0;                      // sum_p gamma_pp (= N)
  double trace_dm2 = 0.0;                      // sum_pq Gamma_pqpq (= N(N-1) for a normalized 2-RDM)
  double e_onebody = 0.0;                      // Re sum h_pq gamma_pq (0 without `h`)
};

RdmFilesResult writeMoRdmFiles(const RdmSource& source, const std::string& dm1_path, const std::string& dm2_path,
                               double threshold, std::ostream& log);

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_RDMFILES_H
