#ifndef RERDMFT_CC_CCWORKSPACE_H
#define RERDMFT_CC_CCWORKSPACE_H

#include <cstddef>
#include <ostream>
#include <vector>

#include "CcTensor.h"
#include "Matrix.h"
#include "Uccsd.h"

namespace rerdmft {

// The reference-dependent data shared by the amplitude (Uccsd.cpp) and the Lambda/RDM (lambda_Uccsd.cpp) solvers: the Fock
// matrix over the active space, its orbital energies and every antisymmetrized integral block that fits in memory (all but
// ovvv and vvvv, which are streamed). Indices o / v run over the active occupied / virtual spinors.
struct CcWorkspace {
  std::size_t o = 0, v = 0, na = 0;
  std::vector<std::size_t> O, V, act;
  cc::Tensor F, f_oo, f_ov, f_vo, f_vv;
  std::vector<double> e_o, e_v;
  cc::Tensor g_oooo, g_ooov, g_oovv, g_ovov, g_ovvo;  // <mn||ij>, <mn||ie>, <ij||ab>, <ma||if>, <mb||ej>
  double max_fock_asymmetry = 0.0, max_fock_ov = 0.0;
};

// Builds the workspace; the frozen occupied spinors enter the Fock matrix only. Prints a summary line to `log` if non-null.
CcWorkspace buildCcWorkspace(const Matrix<CcScalar>& h_mo, const CcIntegrals& ints, const std::vector<std::size_t>& occ_frozen,
                             const std::vector<std::size_t>& occ_active, const std::vector<std::size_t>& vir, std::ostream* log);

}  // namespace rerdmft

#endif  // RERDMFT_CC_CCWORKSPACE_H
