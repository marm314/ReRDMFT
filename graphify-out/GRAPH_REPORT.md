# Graph Report - ReRDMFT  (2026-09-29)

## Corpus Check
- 176 files · ~185,905 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 81 file(s) not represented in the graph (top: .inp 58, .gbs 7, .aux 4)

## Summary
- 2173 nodes · 5631 edges · 131 communities (113 shown, 18 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 548 edges (avg confidence: 0.84)
- Token cost: 182,273 input · 0 output

## Community Hubs (Navigation)
- NEO Problem Interface
- X2C Decoupling & Kramers SCF
- JK-only Hessian Model
- RKB Hamiltonian Construction
- NEO Algorithm & Pair-Term Hessian
- Occupation Init & RKB Utilities
- Cholesky Decomposition Checks
- PNOF Geminal & 2-RDM Builders
- Input Keyword Accessors
- 4-Component DHF & Full Optimization Core
- RKB Density & Fock Matrices
- PNOF Hessian Model
- NEO Davidson Ritz Set
- NEO Step Solver
- Tensor4 Template Storage
- Spin/Kramers Rotation Helpers
- NEO Orbital Problem
- Kramers-Restricted ADAM/NEO Problems
- Non-Relativistic HF & Gradient
- Hartree-Exchange Hessian Header
- Basis Set Construction
- Dirac-Hartree-Fock Result
- ADAM Optimizer Core
- Functional Report Builders
- Cholesky ERI Unit Test
- Hartree-Exchange Energy & Fock
- Kramers Restriction Utility
- NEO Eigenpair Result
- Symmetric ERI Storage
- Hartree-Exchange Hessian Checks
- Matrix Block-Norm Diagnostics
- Restart File I/O
- Hessian Finite-Difference Validation
- CholeskyEri Class
- Restart Data Schema
- RDMFT Core Headers
- X2C Hartree-Fock Result
- Kramers Restriction Unit Test
- NEO & JK-only Functional Concepts
- Integral Representation Helpers
- RKB Two-Electron Tensor
- RKB Transformation Utilities
- Spin Block Labeling
- AO Cholesky Vectors
- Basis Function Data
- Cartesian Angular Momentum Labels
- Full Optimization Result
- Non-Relativistic HF Result
- NEO Test Bowl Model
- ADAM Unit Test
- Restart Binary I/O
- Restart Capture State
- ADAM Learning-Rate State
- X2C MO Transform
- NEO Orbital Toy Model
- NEO Quadratic Test Model
- Symmetric Transform Unit Test
- NEO Step Options
- Symmetric ERI Unit Test
- LBFGS Optimizer
- SQP Optimizer
- RKB MO Transform
- NEO Orbital Toy Variant
- Live Progress Reporting
- DIIS & Non-Relativistic SCF
- Full Optimization Settings
- Packed Two-Electron Tensor
- AO Normalization Checks
- CholeskyEri Methods
- DIIS Extrapolation
- NEO Iteration Record
- NEO Options
- Kramers Pairing & 4C SCF
- Generalized Fock Matrix
- SQP Result Types
- Orbital Subspace Table
- ADAM Options
- ADAM Result
- Relativistic PNOF Functionals
- Generalized Orbital Hessian
- ADAM Optimizer Class
- X2C-HF SCF Loop
- Integral Rotation via Cholesky
- Serial BLAS Thread Guard
- RKB Cholesky Build
- Packed Coulomb Pair Indexing
- RkbCholesky Class
- AO Cholesky Unit Test
- LBFGS Options
- NEO Result
- NEO Step Solver Methods
- NEO Trust-Region Options
- Orbital Hessian Derivation (paper)
- PNOF L1/L2 Hessian Derivation
- Nuclear Attraction Integrals
- Molecular Geometry
- ADAM Problem Interface
- Basis Fingerprint Hash
- Loewdin Orthonormalization
- GEMM-Accelerated Transform
- Kinetic Energy Integrals
- PNOF Subspace Keywords
- RKB Overlap & Vext
- Closed-Shell Spin-Orbital Expansion
- Orbital Subspace Builder
- Kramers ADAM Problem Methods
- LBFGS Result
- NEO Eigenoptions
- Restart Orbital Loading
- Packed Tensor Unit Test
- SQP Options
- Kramers Orbit Structure
- Test RNG Helper
- JK-only Functional Test Protocol
- Davidson Iteration (paper)
- Extended Hessian Matrix (paper)
- Fletcher Trust-Radius Update (paper)
- JDSF96 Citation
- JJ84 Citation
- Level Shifting (paper)
- Matrix-Free Hessian-Vector Product
- NGL21 Citation
- TFJ08 Citation
- JK Frozen/Active Pairs Keyword
- LIBCINT Dependency
- NON_REL Keyword
- OCCUPATION_INIT Keyword
- ReRDMFT Project Root
- SPEED_OF_LIGHT Keyword

## God Nodes (most connected - your core abstractions)
1. `Matrix` - 387 edges
2. `Tensor4` - 131 edges
3. `main()` - 68 edges
4. `Input` - 55 edges
5. `SymmetricEri` - 46 edges
6. `BasisFunction` - 41 edges
7. `h` - 38 edges
8. `NeoStepSolver` - 37 edges
9. `CholeskyEri` - 35 edges
10. `KramersRestriction` - 33 edges

## Surprising Connections (you probably didn't know these)
- `Complex rotation parameters / doubled trial vectors` --semantically_similar_to--> `Kramers-restricted SCF (TR-even density projection)`  [INFERRED] [semantically similar]
  doc/NEO.pdf → README.md
- `HESSIAN_MEAN_FIELD keyword` --conceptually_related_to--> `Orbital rotation Hessian G_pq,rs (Fock-matrix form)`  [INFERRED]
  README.md → doc/orbital_hessian.pdf
- `main()` --calls--> `second`  [INFERRED]
  tests/test_diis.cpp → src/Utils/KramersRestriction.h
- `No-pair Dirac-Hartree-Fock limit` --conceptually_related_to--> `4-Component Dirac-Hartree-Fock SCF (C4_DHF)`  [INFERRED]
  doc/rel_pnofs.pdf → README.md
- `General JK-only two-electron energy functional f(na,nb)` --conceptually_related_to--> `JK-only functionals (SD, MULLER, BBC2, CA, CGA, ML, MLSIC, GU, POWER)`  [INFERRED]
  doc/orbital_hessian_jk_only.pdf → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PNOF5/PNOF7/PNOF7s/GNOF all collapse to the no-pair DHF limit** — doc_rel_pnofs_pnof5, doc_rel_pnofs_pnof7, doc_rel_pnofs_pnof7s, doc_rel_pnofs_gnof, doc_rel_pnofs_dhf_limit [EXTRACTED 1.00]
- **NEO algorithm implemented as the ORBITAL_OPTIMIZER NEO second-order optimizer** — doc_neo_neo_algorithm, src_utils_neo_h_src_utils_neo, readme_orbital_optimizer_neo [INFERRED 0.85]
- **Cross-document PNOF Hessian L1/L2 derivation chain (rel_pnofs Eq.98 to orbital_hessian ansatz to PnofFock.h)** — doc_orbital_hessian_pnof_ansatz, doc_orbital_hessian_l1_l2_extension, doc_rel_pnofs_2rdm_reconstruction, src_hessian_opt_pnoffock_h_src_hessian_opt_pnoffock [EXTRACTED 1.00]

## Communities (131 total, 18 thin omitted)

### Community 0 - "NEO Problem Interface"
Cohesion: 0.06
Nodes (51): neo, h, NeoProblem, accept, dimension, energy, gradient, hessianVector (+43 more)

### Community 1 - "X2C Decoupling & Kramers SCF"
Cohesion: 0.06
Nodes (60): Complex rotation parameters / doubled trial vectors, lapacke, linearalgebra, Kramers-restricted SCF (TR-even density projection), X2C decoupling and X2C-HF, complex, diagonalizeHermitian(), diagonalizeSymmetric() (+52 more)

### Community 2 - "JK-only Hessian Model"
Cohesion: 0.11
Nodes (58): EriT, JkFunctional, jkOnlyHessianDiagonalImpl(), jkOnlyHessianMatrixImpl(), jkOnlyHessianVectorImpl(), makeJkOnlyModel(), checkJkOnlyArgs(), complex (+50 more)

### Community 3 - "RKB Hamiltonian Construction"
Cohesion: 0.08
Nodes (56): rkbhamiltonian, complex, rkbEmbeddingMatrix(), rkbHamiltonianMatrix(), columnOf(), C, size_t, vector (+48 more)

### Community 4 - "NEO Algorithm & Pair-Term Hessian"
Cohesion: 0.08
Nodes (16): cstddef, cstdint, Norm-Extended Optimization (NEO) Algorithm, Hartree/exchange contribution G^HX_pq,rs, Pair (L1/L2) contribution G^L_pq,rs, molecularbasis, CHOLESKY keyword (Cholesky-vector ERI storage), DIIS keyword (Pulay DIIS) (+8 more)

### Community 5 - "Occupation Init & RKB Utilities"
Cohesion: 0.06
Nodes (46): algorithm, cctype, cmath, fstream, istringstream, OccupationInitMethod, rkbdensitymatrix, rkborthogonalization (+38 more)

### Community 6 - "Cholesky Decomposition Checks"
Cohesion: 0.09
Nodes (50): checkedFlat(), checkedImpl(), CholeskyCheckReport, batch_used, max_error, n_vectors, retried, tolerance (+42 more)

### Community 7 - "PNOF Geminal & 2-RDM Builders"
Cohesion: 0.12
Nodes (45): buildPnofGeminals(), buildPnofTwoRdm(), checkNegligibleImag(), complex, Eri, PnofFunctional, size_t, string (+37 more)

### Community 8 - "Input Keyword Accessors"
Cohesion: 0.10
Nodes (12): vector, Input, basis_file_, functional_, geometry_, occupation_init_, orbital_optimizer_, buildC4SpinorEri() (+4 more)

### Community 9 - "4-Component DHF & Full Optimization Core"
Cohesion: 0.05
Nodes (39): basisfingerprint, basisset, c4_dhf, closedshellspinorbitals, ctime, dirackinetic, fermidirac, fulloptimization (+31 more)

### Community 10 - "RKB Density & Fock Matrices"
Cohesion: 0.09
Nodes (25): complex, rkbCoefficientMatrix(), rkbDensityMatrix(), complex, RkbTwoElectronTensor, dhfOrbitalGradientEfficient(), dagger(), complex (+17 more)

### Community 11 - "PNOF Hessian Model"
Cohesion: 0.17
Nodes (36): choleskyeri, PnofFunctional, makePnofModel(), pnofHessianDiagonalImpl(), pnofHessianMatrixImpl(), pnofHessianVectorImpl(), buildPnofFullTwoRdm(), buildPnofPairOf() (+28 more)

### Community 12 - "NEO Davidson Ritz Set"
Cohesion: 0.14
Nodes (29): numeric, RitzSet, conjugate(), complex, NeoHessianVectorFn, size_t, T, vector (+21 more)

### Community 13 - "NEO Step Solver"
Cohesion: 0.07
Nodes (30): NeoHessianVectorFn, size_t, NeoStepSolver, a_, basis_, c_, davidson, diag_ (+22 more)

### Community 14 - "Tensor4 Template Storage"
Cohesion: 0.11
Nodes (21): size_t, T, vector, Tensor4, d0_, d1_, d2_, d3_ (+13 more)

### Community 15 - "Spin/Kramers Rotation Helpers"
Cohesion: 0.17
Nodes (15): contractSpinRestricted(), Pair, size_t, vector, excludeNegative(), generatorMatrix(), lowerPairs(), runFullOptimization() (+7 more)

### Community 16 - "NEO Orbital Problem"
Cohesion: 0.10
Nodes (17): Scalar, Eri, NeoOrbitalProblem, g0_, g0_valid_, occ_, RotationProblem, eri_b_ (+9 more)

### Community 17 - "Kramers-Restricted ADAM/NEO Problems"
Cohesion: 0.18
Nodes (20): size_t, vector, KramersAdamProblem::KramersAdamProblem(), KramersNeoProblem::KramersNeoProblem(), KramersRestriction, contract, contractDiagonal, contractMatrix (+12 more)

### Community 18 - "Non-Relativistic HF & Gradient"
Cohesion: 0.10
Nodes (12): aocholesky, electronrepulsion, complex, x2cDensityMatrix(), complex, PackedTwoElectronTensor, size_t, flavorOf() (+4 more)

### Community 19 - "Hartree-Exchange Hessian Header"
Cohesion: 0.09
Nodes (24): function, hartreeexchangehessian, integralrotation, jk_only, limits, ostream, T, vector (+16 more)

### Community 20 - "Basis Set Construction"
Cohesion: 0.11
Nodes (16): iosfwd, map, set, BasisSet, shells_by_element_, string, vector, vector (+8 more)

### Community 21 - "Dirac-Hartree-Fock Result"
Cohesion: 0.08
Nodes (25): DiracHartreeFockResult, c_dhf, converged, density_matrix, electronic_energy, fock_matrix, fock_ortho_eigenvectors, history (+17 more)

### Community 22 - "ADAM Optimizer Core"
Cohesion: 0.12
Nodes (19): T, spinorRotationMatrix(), Adam<double>, Adam<std::complex<double>>, Adam<T>::Adam(), Adam<T>::step(), adamGradientVector(), adamKappaMatrix() (+11 more)

### Community 23 - "Functional Report Builders"
Cohesion: 0.18
Nodes (24): blockSpinPartner(), buildFullHessianReport(), buildFunctionalReport(), buildPnofFunctionalReport(), Eri, size_t, T, time_point (+16 more)

### Community 24 - "Cholesky ERI Unit Test"
Cohesion: 0.17
Nodes (20): M, absOf(), check(), conjOf(), C, size_t, string, T (+12 more)

### Community 25 - "Hartree-Exchange Energy & Fock"
Cohesion: 0.20
Nodes (22): Eri, size_t, T, vector, hartreeExchangeEnergy(), hartreeExchangeFockMatrix(), eri, conjugateValue() (+14 more)

### Community 26 - "Kramers Restriction Utility"
Cohesion: 0.21
Nodes (22): complex, pair, size_t, vector, gradient, rotate, KramersNeoProblem, accept (+14 more)

### Community 27 - "NEO Eigenpair Result"
Cohesion: 0.09
Nodes (22): T, vector, NeoEigenResult, converged, eigenvalues, eigenvectors, hessian_products, iterations (+14 more)

### Community 28 - "Symmetric ERI Storage"
Cohesion: 0.17
Nodes (11): size_t, T, vector, Loc, conj, slot, SymmetricEri, data_ (+3 more)

### Community 29 - "Hartree-Exchange Hessian Checks"
Cohesion: 0.41
Nodes (21): checkHartreeExchangeHessianDimensions(), checkPairTerms(), conjugate(), complex, Eri, pair, size_t, T (+13 more)

### Community 30 - "Matrix Block-Norm Diagnostics"
Cohesion: 0.13
Nodes (22): analyzeBlocks(), blockDiagTwice(), BlockNorms, max_hermiticity_error, max_large_large, max_large_small, max_small_small, complex (+14 more)

### Community 31 - "Restart File I/O"
Cohesion: 0.14
Nodes (19): cstdio, F, iostream, restart, restartloader, complex, restartCoefficients(), coefficientsComplex (+11 more)

### Community 32 - "Hessian Finite-Difference Validation"
Cohesion: 0.16
Nodes (18): Validation against GeneralizedHessian.h and e^kappa finite difference, RdmftEnergyFn, RdmftGradientFn, DEBUG keyword, conjugate(), complex, size_t, T (+10 more)

### Community 33 - "CholeskyEri Class"
Cohesion: 0.14
Nodes (14): CholeskyEri, fromDense, fromVectors, n_, nchol_, restricted, rotated, toDense (+6 more)

### Community 34 - "Restart Data Schema"
Cohesion: 0.10
Nodes (21): uint64_t, RestartData, basis_fingerprint, coefficients, cols, complex_coefficients, converged, functional (+13 more)

### Community 35 - "RDMFT Core Headers"
Cohesion: 0.15
Nodes (19): hartreeexchangegradient, jkonlyfock, jkonlyhessian, kramersrestriction, lbfgs, memory, occupationenergy, optional (+11 more)

### Community 36 - "X2C Hartree-Fock Result"
Cohesion: 0.11
Nodes (20): complex, vector, X2CHartreeFockResult, c_matrix, converged, electronic_energy, fock_matrix, fock_ortho_eigenvectors (+12 more)

### Community 37 - "Kramers Restriction Unit Test"
Cohesion: 0.21
Nodes (17): check(), AdamPair, size_t, string, uint64_t, kappaFromJoint(), lowerPairs(), main() (+9 more)

### Community 38 - "NEO & JK-only Functional Concepts"
Cohesion: 0.11
Nodes (19): Hessian index / prescribed-order saddle targeting, Trust-region restricted step, Antisymmetric Muller variant, General JK-only two-electron energy functional f(na,nb), Muller functional f(na,nb)=sqrt(na*nb), No-pair Dirac-Hartree-Fock limit, np-ReNOFT energy expression Eq. (64), SciPost Chem. 1, 004 (2022) (+11 more)

### Community 39 - "Integral Representation Helpers"
Cohesion: 0.18
Nodes (15): E, conjugate(), complex, string, T, eriKind(), oneElectronRotated(), rotateEri() (+7 more)

### Community 40 - "RKB Two-Electron Tensor"
Cohesion: 0.29
Nodes (18): addInPlace(), conjMatrix(), complex, PackedTwoElectronTensor, RkbTwoElectronTensor, size_t, vector, outerSumTensor() (+10 more)

### Community 41 - "RKB Transformation Utilities"
Cohesion: 0.15
Nodes (15): integrals, nablaintegrals, invert(), complex, vector, rkbCoefficients(), toComplex(), complex (+7 more)

### Community 42 - "Spin Block Labeling"
Cohesion: 0.18
Nodes (13): pair, size_t, SpinBlock, vector, size_t, SpinBlock, vector, spinBlockLabel() (+5 more)

### Community 43 - "AO Cholesky Vectors"
Cohesion: 0.21
Nodes (16): cblas, omp, AoCholesky, n, vectors, aoCholeskyToMoSpinor(), aoCholeskyToMoSpinOrbital(), checkSquare() (+8 more)

### Community 44 - "Basis Function Data"
Cohesion: 0.18
Nodes (16): BasisFunction, cartesian, coefficients, element, exponents, l, x, y (+8 more)

### Community 45 - "Cartesian Angular Momentum Labels"
Cohesion: 0.14
Nodes (14): cartesianComponentIndex(), cartesianComponents(), CartesianExponents, lx, ly, lz, vector, vector (+6 more)

### Community 46 - "Full Optimization Result"
Cohesion: 0.12
Nodes (17): FullOptResult, checks_passed, converged, electronic_energy, gradient_max, iterations, occupation_state, occupations (+9 more)

### Community 47 - "Non-Relativistic HF Result"
Cohesion: 0.12
Nodes (17): vector, NonRelHartreeFockResult, c_matrix, converged, electronic_energy, fock_matrix, history, iterations (+9 more)

### Community 48 - "NEO Test Bowl Model"
Cohesion: 0.17
Nodes (10): Bowl, a_, best_, c_, restores, saves, trial_, T (+2 more)

### Community 49 - "ADAM Unit Test"
Cohesion: 0.28
Nodes (13): adam, spinorrotation, check(), C, string, main(), near(), testComplexStep() (+5 more)

### Community 50 - "Restart Binary I/O"
Cohesion: 0.25
Nodes (15): cstring, istream, ostream, string, vector, get(), getString(), getVector() (+7 more)

### Community 51 - "Restart Capture State"
Cohesion: 0.12
Nodes (16): int64_t, complex, string, vector, RestartCapture, converged, electronic_energy, gammas (+8 more)

### Community 52 - "ADAM Learning-Rate State"
Cohesion: 0.12
Nodes (12): Adam, clean, lr_, m1_, m2_, m2max_, opt_, step (+4 more)

### Community 53 - "X2C MO Transform"
Cohesion: 0.25
Nodes (14): complex, PackedTwoElectronTensor, size_t, SpinBlockSource, n, toComplex(), transformLeg1(), transformLeg2() (+6 more)

### Community 54 - "NEO Orbital Toy Model"
Cohesion: 0.17
Nodes (9): AdamPair, size_t, OrbitalToy, a_, best_, d_, n_, pairs_ (+1 more)

### Community 55 - "NEO Quadratic Test Model"
Cohesion: 0.24
Nodes (7): vector, dot(), matvec(), Quadratic, g_, h_, x_

### Community 56 - "Symmetric Transform Unit Test"
Cohesion: 0.22
Nodes (12): S, check(), cj(), C, size_t, string, uint64_t, rnd() (+4 more)

### Community 57 - "NEO Step Options"
Cohesion: 0.13
Nodes (15): NeoStepOptions, decoupled_z0, guess_from_diagonal, linear_dependence, max_alpha, max_alpha_iterations, max_micro_iterations, max_subspace (+7 more)

### Community 58 - "Symmetric ERI Unit Test"
Cohesion: 0.21
Nodes (12): absOf(), check(), C, size_t, string, T, uint64_t, rnd() (+4 more)

### Community 59 - "LBFGS Optimizer"
Cohesion: 0.26
Nodes (13): LbfgsGradientFn, LbfgsValueFn, axpy(), deque, vector, CurvaturePair, rho, s (+5 more)

### Community 60 - "SQP Optimizer"
Cohesion: 0.27
Nodes (13): SqpGradientFn, SqpHessianFn, SqpValueFn, vector, dot(), EqualityQpSolution, d, lambda (+5 more)

### Community 61 - "RKB MO Transform"
Cohesion: 0.34
Nodes (13): complex, RkbTwoElectronTensor, size_t, densify(), occupiedPositiveEnergyDensity(), rkbMoOneElectronTransform(), rkbMoTwoElectronSymmetric(), rkbMoTwoElectronTransformPhysics() (+5 more)

### Community 62 - "NEO Orbital Toy Variant"
Cohesion: 0.21
Nodes (8): C, OrbitalToy, a_, best_, d_, n_, pairs_, trial_

### Community 63 - "Live Progress Reporting"
Cohesion: 0.19
Nodes (11): chrono, iomanip, mutex, ostringstream, string, time_point, progress(), ProgressLine (+3 more)

### Community 64 - "DIIS & Non-Relativistic SCF"
Cohesion: 0.24
Nodes (12): diis, nuclearrepulsion, Eri, PackedTwoElectronTensor, vector, maxAbsDifference(), nonRelDensityMatrix(), nonRelFockMatrix() (+4 more)

### Community 65 - "Full Optimization Settings"
Cohesion: 0.15
Nodes (13): OrbitalOptimizer, FullOptSettings, cholesky, cholesky_threshold, debug, enabled, energy_tolerance, gradient_tolerance (+5 more)

### Community 66 - "Packed Two-Electron Tensor"
Cohesion: 0.29
Nodes (6): size_t, vector, PackedTwoElectronTensor, data_, n_, pairs_

### Community 67 - "AO Normalization Checks"
Cohesion: 0.21
Nodes (12): vector, string, NormalizationCheck, cartesian, element, l, self_overlap_before, was_renormalized (+4 more)

### Community 68 - "CholeskyEri Methods"
Cohesion: 0.27
Nodes (12): CholeskyEri<std::complex<double>>, CholeskyEri<T>::fromDense(), CholeskyEri<T>::fromVectors(), CholeskyEri<T>::operator()(), CholeskyEri<T>::restricted(), CholeskyEri<T>::rotated(), CholeskyEri<T>::toDense(), conjugate() (+4 more)

### Community 69 - "DIIS Extrapolation"
Cohesion: 0.18
Nodes (11): Diis, clear, errors_, extrapolate, last_weights_, max_vectors_, values_, deque (+3 more)

### Community 70 - "NEO Iteration Record"
Cohesion: 0.15
Nodes (13): NeoIteration, actual_change, alpha, energy, gradient_max, hessian_products, micro_iterations, predicted_change (+5 more)

### Community 71 - "NEO Options"
Cohesion: 0.15
Nodes (13): NeoOptions, gradient_tolerance, initial_radius, max_iterations, max_rejections, min_radius, progress, ratio_floor (+5 more)

### Community 72 - "Kramers Pairing & 4C SCF"
Cohesion: 0.27
Nodes (11): kramerspairing, complex, Eri, RkbTwoElectronTensor, vector, dhfScfImpl(), maxAbsDifference(), runDiracHartreeFockScf() (+3 more)

### Community 73 - "Generalized Fock Matrix"
Cohesion: 0.18
Nodes (7): T, generalizedFockMatrix(), conjugate(), complex, T, orbitalGradient(), stdexcept

### Community 74 - "SQP Result Types"
Cohesion: 0.17
Nodes (12): BoxEqualityQpResult, bound_multipliers, converged, d, eq_multipliers, iterations, vector, SqpResult (+4 more)

### Community 75 - "Orbital Subspace Table"
Cohesion: 0.27
Nodes (10): array, size_t, vector, OrbitalSubspace, occupied, unoccupied, OrbitalSubspaceTable, frozen_occupied (+2 more)

### Community 76 - "ADAM Options"
Cohesion: 0.18
Nodes (11): AdamOptions, base_iterations, beta1, beta2, energy_tolerance, epsilon, gradient_tolerance, iterations_increment (+3 more)

### Community 77 - "ADAM Result"
Cohesion: 0.18
Nodes (11): AdamResult, energy_best, energy_converged, energy_difference, energy_start, gradient_converged, improved, iterations (+3 more)

### Community 78 - "Relativistic PNOF Functionals"
Cohesion: 0.27
Nodes (9): Relativistic GNOF functional, Master formula Eq. (54), Relativistic PNOF5 functional, Relativistic PNOF7 functional, Relativistic PNOF7s functional, vector, PnofSubspaceOccupationsWithGradient, docc_dgamma (+1 more)

### Community 79 - "Generalized Orbital Hessian"
Cohesion: 0.44
Nodes (8): checkHessianDimensions(), conjugate(), complex, size_t, T, generalizedOrbitalHessianElement(), generalizedOrbitalHessianElementImag(), rawHessianTerm()

### Community 80 - "ADAM Optimizer Class"
Cohesion: 0.20
Nodes (7): AdamOptimizer, extra_iterations_, learning_rate_, opt_, reset, restart_, run

### Community 81 - "X2C-HF SCF Loop"
Cohesion: 0.36
Nodes (9): kramersSymmetrizeAo(), complex, Eri, PackedTwoElectronTensor, vector, maxAbsDifference(), runX2CHartreeFockScf(), traceOfProduct() (+1 more)

### Community 82 - "Integral Rotation via Cholesky"
Cohesion: 0.28
Nodes (7): cholesky_decomposition, conjugate(), complex, T, T, RotatedIntegrals, rotateIntegrals()

### Community 83 - "Serial BLAS Thread Guard"
Cohesion: 0.28
Nodes (7): dlfcn, GetFn, SetFn, SerialBlasScope, get_, saved_, set_

### Community 84 - "RKB Cholesky Build"
Cohesion: 0.47
Nodes (7): C, vector, promote(), build, rkbCholeskyToMo(), rkbFockMatrix(), zgemm()

### Community 85 - "Packed Coulomb Pair Indexing"
Cohesion: 0.39
Nodes (5): PackedTwoElectronTensor, size_t, UnionCoulombPairs, nl_, ns_

### Community 86 - "RkbCholesky Class"
Cohesion: 0.28
Nodes (7): complex, size_t, vector, RkbCholesky, large, n_large, small

### Community 87 - "AO Cholesky Unit Test"
Cohesion: 0.22
Nodes (8): fromPacked, PackedTwoElectronTensor, size_t, check(), C, string, main(), maxDiff()

### Community 88 - "LBFGS Options"
Cohesion: 0.22
Nodes (9): size_t, LbfgsOptions, armijo_c1, backtrack_factor, curvature_skip_tolerance, gradient_tolerance, history_size, max_iterations (+1 more)

### Community 89 - "NEO Result"
Cohesion: 0.22
Nodes (9): NeoResult, converged, energy, gradient_max, hessian_products, history, iterations, lowest_hessian_eigenvalues (+1 more)

### Community 90 - "NEO Step Solver Methods"
Cohesion: 0.25
Nodes (9): addTrial, addUnitVector, addVector, collapse, effectiveTolerance, NeoStepSolver<T>::addUnitVector(), NeoStepSolver<T>::davidson(), NeoStepSolver<T>::initialize() (+1 more)

### Community 91 - "NEO Trust-Region Options"
Cohesion: 0.22
Nodes (9): NeoTrustOptions, grow_factor, max_radius, ratio_high, ratio_low, saddle_grow, saddle_ratio_good, saddle_ratio_min (+1 more)

### Community 92 - "Orbital Hessian Derivation (paper)"
Cohesion: 0.29
Nodes (8): Generalized Fock matrix F_qp, Orbital rotation gradient g_pq, Orbital rotation Hessian G_pq,rs (Fock-matrix form), JK-only orbital gradient g_pq, JK-only orbital Hessian G_pq,rs, Real/imaginary directional Hessian blocks (Hess^tt, Hess^yy, Hess^ty), main.cpp HESSIAN_4C diagnostic, HESSIAN_MEAN_FIELD keyword

### Community 93 - "PNOF L1/L2 Hessian Derivation"
Cohesion: 0.39
Nodes (5): L1/L2 pair-term extension fixing the PNOF Hessian, Cheap diagonal-D 2-RDM ansatz for PNOF (H, X, L1, L2), General no-pair relativistic 2-RDM reconstruction (Eqs. 97-98), orb_subspaces, pnofs

### Community 94 - "Nuclear Attraction Integrals"
Cohesion: 0.32
Nodes (7): element, vector, nuclearAttractionMatrix(), nuclearAttractionPair(), complex, vector, vextMatrix()

### Community 95 - "Molecular Geometry"
Cohesion: 0.25
Nodes (8): Atom, symbol, x, y, z, string, vector, nuclearRepulsionEnergy()

### Community 96 - "ADAM Problem Interface"
Cohesion: 0.25
Nodes (7): AdamProblem, dimension, energy, gradient, restoreBest, rotate, saveBest

### Community 97 - "Basis Fingerprint Hash"
Cohesion: 0.43
Nodes (7): basisFingerprint(), size_t, T, uint64_t, vector, fnv1aUpdateBytes(), fnv1aUpdateValue()

### Community 98 - "Loewdin Orthonormalization"
Cohesion: 0.43
Nodes (7): complex, ostream, size_t, string, lowdinOrthonormalize(), maxDeviationFromIdentity(), readRestartOrbitals()

### Community 99 - "GEMM-Accelerated Transform"
Cohesion: 0.32
Nodes (8): conjugate(), gemm(), gemmAcc(), complex, size_t, Src, T, transformToSymmetric()

### Community 100 - "Kinetic Energy Integrals"
Cohesion: 0.33
Nodes (4): cint, vector, kineticPair(), schrodingerKineticMatrix()

### Community 101 - "PNOF Subspace Keywords"
Cohesion: 0.33
Nodes (5): Subspace (geminal) partition, functional, PNOF functionals (PNOF5, PNOF7, PNOF7S, GNOF), PNOF_SUBSPACES / PNOF_COUPLING keywords, SQP_PNOF_OCC keyword

### Community 102 - "RKB Overlap & Vext"
Cohesion: 0.52
Nodes (6): nuclearattraction, complex, vector, rkbSmallOverlapMatrix(), rkbSmallVextMatrix(), rkbTransformSmallScalarMatrix()

### Community 103 - "Closed-Shell Spin-Orbital Expansion"
Cohesion: 0.71
Nodes (6): closedShellSpinOrbitalDensity(), closedShellSpinOrbitalOneElectron(), closedShellSpinOrbitalTwoElectron(), size_t, spatialIndex(), spinOf()

### Community 104 - "Orbital Subspace Builder"
Cohesion: 0.57
Nodes (6): buildOrbitalSubspaces(), array, size_t, vector, enumeratePairs(), validatePairOf()

### Community 105 - "Kramers ADAM Problem Methods"
Cohesion: 0.29
Nodes (3): complex, KramersAdamProblem, restriction_

### Community 106 - "LBFGS Result"
Cohesion: 0.29
Nodes (7): vector, LbfgsResult, converged, gradient, iterations, objective_value, x

### Community 107 - "NEO Eigenoptions"
Cohesion: 0.29
Nodes (7): NeoEigenOptions, linear_dependence, max_iterations, max_subspace, n_random_guesses, preconditioner_floor, residual_tolerance

### Community 108 - "Restart Orbital Loading"
Cohesion: 0.29
Nodes (7): complex, RestartOrbitals, data, lowdin_applied, min_overlap_eigenvalue, overlap_deviation_final, overlap_deviation_read

### Community 109 - "Packed Tensor Unit Test"
Cohesion: 0.29
Nodes (6): PackedTwoElectronTensor, size_t, uint64_t, makeAo(), Rng, s

### Community 110 - "SQP Options"
Cohesion: 0.33
Nodes (6): SqpOptions, armijo_c1, backtrack_factor, max_iterations, max_line_search_steps, step_tolerance

### Community 111 - "Kramers Orbit Structure"
Cohesion: 0.40
Nodes (5): Orbit, first, second, sign_t, sign_y

### Community 112 - "Test RNG Helper"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

## Knowledge Gaps
- **512 isolated node(s):** `shells_by_element_`, `element`, `x`, `y`, `z` (+507 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 762 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `RKB Density & Fock Matrices` to `NEO Problem Interface`, `X2C Decoupling & Kramers SCF`, `JK-only Hessian Model`, `RKB Hamiltonian Construction`, `NEO Algorithm & Pair-Term Hessian`, `Occupation Init & RKB Utilities`, `Cholesky Decomposition Checks`, `PNOF Geminal & 2-RDM Builders`, `Input Keyword Accessors`, `PNOF Hessian Model`, `NEO Davidson Ritz Set`, `Tensor4 Template Storage`, `Spin/Kramers Rotation Helpers`, `NEO Orbital Problem`, `Kramers-Restricted ADAM/NEO Problems`, `Non-Relativistic HF & Gradient`, `Hartree-Exchange Hessian Header`, `Dirac-Hartree-Fock Result`, `ADAM Optimizer Core`, `Functional Report Builders`, `Cholesky ERI Unit Test`, `Hartree-Exchange Energy & Fock`, `Hartree-Exchange Hessian Checks`, `Matrix Block-Norm Diagnostics`, `Restart File I/O`, `Hessian Finite-Difference Validation`, `RDMFT Core Headers`, `X2C Hartree-Fock Result`, `Kramers Restriction Unit Test`, `Integral Representation Helpers`, `RKB Two-Electron Tensor`, `RKB Transformation Utilities`, `AO Cholesky Vectors`, `Full Optimization Result`, `Non-Relativistic HF Result`, `NEO Test Bowl Model`, `ADAM Unit Test`, `Restart Capture State`, `X2C MO Transform`, `NEO Orbital Toy Model`, `NEO Quadratic Test Model`, `Symmetric Transform Unit Test`, `SQP Optimizer`, `RKB MO Transform`, `NEO Orbital Toy Variant`, `DIIS & Non-Relativistic SCF`, `AO Normalization Checks`, `CholeskyEri Methods`, `DIIS Extrapolation`, `Kramers Pairing & 4C SCF`, `Generalized Fock Matrix`, `Relativistic PNOF Functionals`, `Generalized Orbital Hessian`, `X2C-HF SCF Loop`, `Integral Rotation via Cholesky`, `RKB Cholesky Build`, `RkbCholesky Class`, `AO Cholesky Unit Test`, `PNOF L1/L2 Hessian Derivation`, `Nuclear Attraction Integrals`, `Loewdin Orthonormalization`, `GEMM-Accelerated Transform`, `Kinetic Energy Integrals`, `PNOF Subspace Keywords`, `RKB Overlap & Vext`, `Closed-Shell Spin-Orbital Expansion`, `Restart Orbital Loading`?**
  _High betweenness centrality (0.617) - this node is a cross-community bridge._
- **Why does `AdamProblem` connect `ADAM Problem Interface` to `NEO Algorithm & Pair-Term Hessian`, `Kramers ADAM Problem Methods`, `NEO Orbital Problem`, `Kramers-Restricted ADAM/NEO Problems`, `NEO Test Bowl Model`, `ADAM Optimizer Core`, `NEO Orbital Toy Model`, `NEO Orbital Toy Variant`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `Input` connect `Input Keyword Accessors` to `NEO Algorithm & Pair-Term Hessian`, `Occupation Init & RKB Utilities`, `4-Component DHF & Full Optimization Core`, `Non-Relativistic HF & Gradient`, `Basis Set Construction`, `Matrix Block-Norm Diagnostics`, `Molecular Geometry`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **What connects `shells_by_element_`, `element`, `x` to the rest of the system?**
  _512 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `NEO Problem Interface` be split into smaller, more focused modules?**
  _Cohesion score 0.05510388437217705 - nodes in this community are weakly interconnected._
- **Should `X2C Decoupling & Kramers SCF` be split into smaller, more focused modules?**
  _Cohesion score 0.05625 - nodes in this community are weakly interconnected._
- **Should `JK-only Hessian Model` be split into smaller, more focused modules?**
  _Cohesion score 0.10546448087431694 - nodes in this community are weakly interconnected._