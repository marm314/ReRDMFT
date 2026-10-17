#ifndef RERDMFT_MCSCF_CASMODEL_H
#define RERDMFT_MCSCF_CASMODEL_H

#include <cstddef>
#include <vector>

#include "Casci.h"
#include "FullOptimization.h"

namespace rerdmft {

// CASSCF as an RdmftModel for Full_opt/FullOptimization.h's macro loop (ADAM orbital steps at a FIXED CI vector, then a CI
// re-solve at the new orbitals -- the role the occupation re-optimization plays for the functionals):
//   * `optimize_occupations` solves the lowest CI root of the core-dressed active-space Hamiltonian (MCSCF/Casci.h),
//     warm-started from the previous root (`state` = [Re c; Im c]), and caches the active 1-/2-RDM;
//   * `energy` / `gradient` are MCSCF/CasFock.h's casFock at those cached RDMs; the gradient is zeroed for pairs inside one
//     orbital class (core-core, active-active, virtual-virtual): the first and last leave the energy invariant, the active-active
//     ones are redundant once the CI is re-optimized, so only core-active, core-virtual and active-virtual rotations are optimized;
//   * the "occupation" vector is the diagonal of the total 1-RDM (core 1, active <a+_t a_t>, virtual 0).
// `core`/`active` are index lists (not necessarily contiguous) into the MO basis of the (h, eri) the loop works on;
// `n_active_electrons` electrons in the |active| spinors. `active_spin` (optional, one 0/1 per active index = alpha/beta,
// NON_REL spin-orbitals only): restricts the STARTING determinant to Ms = 0 (the Hamiltonian then keeps the CI there).
// No Hessian-vector product exists: ORBITAL_OPTIMIZER NEO falls back to ADAM with the usual printed note.
template <typename T, typename Eri = Tensor4<T>>
RdmftModel<T, Eri> makeCasModel(std::vector<std::size_t> core, std::vector<std::size_t> active, std::size_t n_active_electrons,
                                std::vector<int> active_spin, std::size_t n_total, CasciOptions ci_options = {});

}  // namespace rerdmft

#endif
