# Graph Report - ReRDMFT  (2026-11-03)

## Corpus Check
- 276 files · ~369,756 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 138 file(s) not represented in the graph (top: .inp 99, .gbs 11, .tex 8)

## Summary
- 3764 nodes · 10350 edges · 207 communities (184 shown, 23 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 1185 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `61b3ee63`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_neo.cpp
- test_diis.cpp
- JK_only.cpp
- KramersPairing.cpp
- vector
- AuxiliaryBasis.cpp
- CholeskyCheckReport
- PNOFs.cpp
- Input
- main.cpp
- Matrix
- PnofGeminal
- NEO.cpp
- NeoStepSolver
- Tensor4
- vector
- RotationProblem
- KramersRestriction
- RiNonRelSpinMoEri
- RdmftModel
- runSystem
- DiracHartreeFockResult
- ADAM.cpp
- vector
- test_cholesky_eri.cpp
- jkOnlyOrbitalGradient
- KramersRestriction.cpp
- NeoStep
- SymmetricEri
- HartreeExchangeHessian.cpp
- string
- main
- OrbitalRotationGradientCheck
- CholeskyEri
- RestartData
- iomanip
- X2CHartreeFockResult
- test_kramers_restriction.cpp
- PNOFs.h
- FullOptimization.cpp
- RkbTwoElectron.cpp
- ukbHamiltonianMatrix
- SpinorBasis
- AoCholesky
- BasisFunction
- BasisSet
- FullOptResult
- nonRelScfImpl
- Bowl
- test_adam.cpp
- Restart.cpp
- RestartCapture
- Adam
- X2C_MoTransform.cpp
- OrbitalToy
- vector
- test_symmetric_transform.cpp
- NeoStepOptions
- test_symmetric_eri.cpp
- Diis
- solveSqp
- RkbMoTransform.cpp
- OrbitalToy
- stdexcept
- test_ukb_fock_direct.cpp
- FullOptSettings
- CrossPackedTwoElectronTensor
- CartesianExponents
- CholeskyEri.cpp
- run_examples.py
- NeoIteration
- NeoOptions
- dhfScfImpl
- makePccdModel
- Tensor
- run
- AdamOptions
- AdamResult
- PnofSubspaceOccupationsWithGradient
- rawHessianTerm
- AdamOptimizer
- C
- rotateIntegrals
- SerialBlasScope
- OnDemandUnionCoulombPairs
- size_t
- rkbCholeskyToMoFused
- Uccsd.cpp
- CcRdmSettings
- FactoredCcIntegrals
- t
- NeoTrustOptions
- Orbital rotation Hessian G_pq,rs (Fock-matrix form)
- complex
- SectionIndentBuf
- diagonalizeHermitian
- AdamProblem
- basisFingerprint
- readRestartOrbitals
- transformToSymmetric
- CartesianSpinors
- functional
- Atom
- ClosedShellSpinOrbitals.cpp
- test_casfock.cpp
- Casci
- RiDirectEriSource
- NeoEigenOptions
- RestartOrbitals
- makeAo
- RdmSource
- Casci.cpp
- Rng
- Test protocol: e^-kappa rotation, central differences t=1e-4
- Davidson iteration for the extended eigenproblem
- Extended (augmented Hessian) matrix L(alpha)
- Fletcher's trust radius update algorithm
- Jensen, Dyall, Saue & Faegri, J. Chem. Phys. 104, 4083 (1996)
- Jensen & Jorgensen, J. Chem. Phys. 80, 1204 (1984)
- Level shifting via scalar alpha
- Matrix-free Hessian-vector product (finite difference)
- Nottoli, Gauss & Lipparini, J. Chem. Theory Comput. (2021)
- Thyssen, Fleig & Jensen, J. Chem. Phys. 129, 034109 (2008)
- JK_FROZEN_PAIRS / JK_ACTIVE_PAIRS occupation window
- LIBCINT integral library dependency
- Nonrelativistic Hartree-Fock SCF (NON_REL)
- OCCUPATION_INIT keyword (PROPORTIONAL / FERMI_DIRAC)
- ReRDMFT
- SPEED_OF_LIGHT keyword
- UccsdResult
- fock
- h
- fock
- solvePccdZAmplitudes
- Cholesky_Decomposition.cpp
- cartesianComponents
- CcWorkspace
- JkOnlySigmaHessianVector.cpp
- test_casci.cpp
- main
- eri
- main
- UkbDirectEriSource
- T
- X2CDecouplingResult
- KramersSymmetry.cpp
- ThreeCenterIntegrals.cpp
- DBlockDef
- MoIntegralTransform.cpp
- PccdRdm
- test_pccd_amplitudes.cpp
- PairOp
- RiMoEri
- writeGaussRdm
- GaussRdmResult
- Term
- RiNonRelFullOptFock.cpp
- RiSigmaHessianToolkit
- CasciHamiltonian
- PccdSettings
- pairContract
- OnDemandPairs
- Wfn
- DIIS.cpp
- Prim
- makeJkOnlyModel
- Component
- IssueTee
- RkbDerivativeTerms
- KramersPairingReport
- SphericalTransformResult
- X2C_hamiltonian.cpp
- ShellQuartetBlock
- GaussRdmSettings
- RecordWriter
- buildRiSigmaHessianToolkit
- Reference
- buildSector
- hartreeExchangeEnergy
- overlap1d
- jointOrbitalGradient
- RiNonRelDirectSource
- binomial
- Rng
- lowerPairsLocal
- main
- lowerPairsLocal
- main
- PrimitiveSlot
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- Rng
- basisset
- optional
- ostream
- set
- unique_ptr

## God Nodes (most connected - your core abstractions)
1. `Matrix` - 673 edges
2. `Tensor4` - 161 edges
3. `Input` - 138 edges
4. `main()` - 121 edges
5. `C` - 116 edges
6. `BasisFunction` - 87 edges
7. `SymmetricEri` - 54 edges
8. `RotationProblem` - 44 edges
9. `runFullOptimization()` - 44 edges
10. `dagger()` - 43 edges

## Surprising Connections (you probably didn't know these)
- `Complex rotation parameters / doubled trial vectors` --semantically_similar_to--> `Kramers-restricted SCF (TR-even density projection)`  [INFERRED] [semantically similar]
  doc/NEO.pdf → README.md
- `HESSIAN_MEAN_FIELD keyword` --conceptually_related_to--> `Orbital rotation Hessian G_pq,rs (Fock-matrix form)`  [INFERRED]
  README.md → doc/orbital_hessian.pdf
- `main()` --calls--> `second`  [INFERRED]
  tests/test_diis.cpp → src/Utils/KramersRestriction.h
- `bruteForceH()` --calls--> `index`  [INFERRED]
  tests/test_casci.cpp → src/main.cpp
- `Subspace (geminal) partition` --conceptually_related_to--> `PNOF_SUBSPACES / PNOF_COUPLING keywords`  [INFERRED]
  doc/rel_pnofs.pdf → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PNOF5/PNOF7/PNOF7s/GNOF all collapse to the no-pair DHF limit** — doc_rel_pnofs_pnof5, doc_rel_pnofs_pnof7, doc_rel_pnofs_pnof7s, doc_rel_pnofs_gnof, doc_rel_pnofs_dhf_limit [EXTRACTED 1.00]
- **Cross-document PNOF Hessian L1/L2 derivation chain (rel_pnofs Eq.98 to orbital_hessian ansatz to PnofFock.h)** — doc_orbital_hessian_pnof_ansatz, doc_orbital_hessian_l1_l2_extension, doc_rel_pnofs_2rdm_reconstruction, src_hessian_opt_pnoffock_h_src_hessian_opt_pnoffock [EXTRACTED 1.00]
- **NEO algorithm implemented as the ORBITAL_OPTIMIZER NEO second-order optimizer** — doc_neo_neo_algorithm, src_utils_neo_h_src_utils_neo, readme_orbital_optimizer_neo [INFERRED 0.85]

## Communities (207 total, 23 thin omitted)

### Community 0 - "test_neo.cpp"
Cohesion: 0.05
Nodes (49): neo, NeoProblem, accept, dimension, energy, gradient, hessianVector, trialEnergy (+41 more)

### Community 1 - "test_diis.cpp"
Cohesion: 0.24
Nodes (18): adjoint(), check(), cj(), size_t, string, T, uint64_t, eigh() (+10 more)

### Community 2 - "JK_only.cpp"
Cohesion: 0.16
Nodes (34): JkFunctional, size_t, string, csanyiAriasDerivatives(), csanyiGoedeckerAriasDerivatives(), jkExchangeFunction(), jkExchangeFunctionD1(), jkExchangeFunctionD11() (+26 more)

### Community 3 - "KramersPairing.cpp"
Cohesion: 0.30
Nodes (15): columnOf(), size_t, vector, fixKramersPairingLarge(), fixKramersPairingOrthonormal(), kramersRkbAoDeviation(), kramersSymmetrizeRkbAo(), metricInner() (+7 more)

### Community 4 - "vector"
Cohesion: 0.06
Nodes (14): cint, cstddef, dlfcn, Norm-Extended Optimization (NEO) Algorithm, element, fulloptimization, molecularbasis, CHOLESKY keyword (Cholesky-vector ERI storage) (+6 more)

### Community 5 - "AuxiliaryBasis.cpp"
Cohesion: 0.05
Nodes (60): AuxBasisType, cctype, istringstream, OccupationInitMethod, autoCandidates(), buildAutoAuxiliaryBasis(), Candidate, am (+52 more)

### Community 6 - "CholeskyCheckReport"
Cohesion: 0.11
Nodes (20): checkedFlat(), CholeskyCheckReport, batch_used, max_error, n_vectors, retried, tolerance, choleskyDecomposePairsChecked() (+12 more)

### Community 7 - "PNOFs.cpp"
Cohesion: 0.14
Nodes (40): buildPnofTwoRdm(), checkNegligibleImag(), complex, Eri, PnofFunctional, size_t, string, T (+32 more)

### Community 8 - "Input"
Cohesion: 0.04
Nodes (29): ostream, vector, Input, aux_basis_type_, basis_file_, casscf_ci_start_, casscf_init_, casscf_rdm_options_set_ (+21 more)

### Community 9 - "main.cpp"
Cohesion: 0.03
Nodes (65): aimpac, auxiliarybasis, basisfingerprint, c4_dhf, closedshellspinorbitals, ctime, dirackinetic, fermidirac (+57 more)

### Community 10 - "Matrix"
Cohesion: 0.12
Nodes (22): canonicalOrthogonalize(), complex, size_t, T, vector, Matrix, operator+(), transpose() (+14 more)

### Community 11 - "PnofGeminal"
Cohesion: 0.14
Nodes (43): PnofFunctional, makePnofModel(), pnofHessianDiagonalImpl(), pnofHessianMatrixImpl(), pnofHessianVectorImpl(), buildPnofFullTwoRdm(), buildPnofPairOf(), Eri (+35 more)

### Community 12 - "NEO.cpp"
Cohesion: 0.14
Nodes (30): numeric, RitzSet, conjugate(), complex, NeoHessianVectorFn, size_t, T, vector (+22 more)

### Community 13 - "NeoStepSolver"
Cohesion: 0.07
Nodes (37): NeoHessianVectorFn, NeoStepSolver, a_, addTrial, addUnitVector, addVector, basis_, c_ (+29 more)

### Community 14 - "Tensor4"
Cohesion: 0.06
Nodes (23): aocholesky, diis, Validation against GeneralizedHessian.h and e^kappa finite difference, electronrepulsion, mutex, nuclearrepulsion, DEBUG keyword, size_t (+15 more)

### Community 15 - "vector"
Cohesion: 0.10
Nodes (19): Scalar, contractSpinRestricted(), Pair, vector, excludeNegative(), lowerPairs(), NeoOrbitalProblem, energy_floor_ (+11 more)

### Community 16 - "RotationProblem"
Cohesion: 0.09
Nodes (23): Eri, RotationProblem, eri_b_, eri_t_, h_b_, h_t_, n_, occ_ (+15 more)

### Community 17 - "KramersRestriction"
Cohesion: 0.15
Nodes (23): size_t, vector, KramersRestriction, asymmetry, contract, contractDiagonal, contractMatrix, expand (+15 more)

### Community 18 - "RiNonRelSpinMoEri"
Cohesion: 0.06
Nodes (67): Op, checkJkOnlyArgs(), complex, Eri, pair, size_t, T, vector (+59 more)

### Community 19 - "RdmftModel"
Cohesion: 0.08
Nodes (25): function, string, T, vector, RdmftModel, energy, gradient, hessian_diagonal (+17 more)

### Community 20 - "runSystem"
Cohesion: 0.10
Nodes (33): vector, vector, SmallComponentBasis, build, functions_, term_index_, overlapMatrix(), complex (+25 more)

### Community 21 - "DiracHartreeFockResult"
Cohesion: 0.07
Nodes (28): DiracHartreeFockResult, converged, density_matrix, electronic_energy, fock_matrix, fock_ortho_eigenvectors, history, iterations (+20 more)

### Community 22 - "ADAM.cpp"
Cohesion: 0.15
Nodes (17): Adam<double>, Adam<std::complex<double>>, Adam<T>::Adam(), Adam<T>::step(), adamGradientVector(), adamKappaMatrix(), AdamOptimizer<double>, AdamOptimizer<std::complex<double>> (+9 more)

### Community 23 - "vector"
Cohesion: 0.14
Nodes (40): blockSpinPartner(), buildFullHessianReport(), buildFunctionalReport(), buildNonRelCartesianSpinors(), buildPccdFunctionalReport(), buildPnofFunctionalReport(), Eri, JkFunctional (+32 more)

### Community 24 - "test_cholesky_eri.cpp"
Cohesion: 0.14
Nodes (23): jkonlyfock, occupationenergy, pnoffock, v, absOf(), check(), conjOf(), M (+15 more)

### Community 25 - "jkOnlyOrbitalGradient"
Cohesion: 0.29
Nodes (10): conjugateValue(), complex, Eri, size_t, T, vector, jkOnlyDenseTwoRdm(), jkOnlyEnergy() (+2 more)

### Community 26 - "KramersRestriction.cpp"
Cohesion: 0.19
Nodes (22): complex, pair, size_t, vector, gradient, rotate, KramersNeoProblem, accept (+14 more)

### Community 27 - "NeoStep"
Cohesion: 0.09
Nodes (22): T, vector, NeoEigenResult, converged, eigenvalues, eigenvectors, hessian_products, iterations (+14 more)

### Community 28 - "SymmetricEri"
Cohesion: 0.17
Nodes (11): size_t, T, vector, Loc, conj, slot, SymmetricEri, data_ (+3 more)

### Community 29 - "HartreeExchangeHessian.cpp"
Cohesion: 0.41
Nodes (21): checkHartreeExchangeHessianDimensions(), checkPairTerms(), conjugate(), complex, Eri, pair, size_t, T (+13 more)

### Community 30 - "string"
Cohesion: 0.12
Nodes (29): functional_, geometry_, blockDiagTwice(), casWfnOutputPath(), ccWfnOutputPath(), complexTranspose(), complex, string (+21 more)

### Community 31 - "main"
Cohesion: 0.24
Nodes (11): F, complex, restartCoefficients(), coefficientsComplex, setCoefficients, check(), string, main() (+3 more)

### Community 32 - "OrbitalRotationGradientCheck"
Cohesion: 0.18
Nodes (16): RdmftEnergyFn, RdmftGradientFn, conjugate(), complex, size_t, T, OrbitalRotationGradientCheck, abs_diff (+8 more)

### Community 33 - "CholeskyEri"
Cohesion: 0.14
Nodes (14): CholeskyEri, fromDense, fromVectors, n_, nchol_, restricted, rotated, toDense (+6 more)

### Community 34 - "RestartData"
Cohesion: 0.07
Nodes (28): int64_t, string, uint64_t, vector, RestartData, amplitudes, basis_fingerprint, cas_electrons (+20 more)

### Community 35 - "iomanip"
Cohesion: 0.17
Nodes (12): cstdint, hartreeexchangegradient, hartreeexchangehessian, hartreeexchangesigmahessianvector, hartreeexchangesigmahessianvectorri, iomanip, iostream, jkonlyhessian (+4 more)

### Community 36 - "X2CHartreeFockResult"
Cohesion: 0.07
Nodes (31): complex, PackedTwoElectronTensor, size_t, flavorOf(), x2cFockMatrix(), complex, X2CHamiltonianResult, h_x2c (+23 more)

### Community 37 - "test_kramers_restriction.cpp"
Cohesion: 0.21
Nodes (17): check(), AdamPair, size_t, string, uint64_t, kappaFromJoint(), lowerPairs(), main() (+9 more)

### Community 38 - "PNOFs.h"
Cohesion: 0.06
Nodes (30): Complex rotation parameters / doubled trial vectors, Hessian index / prescribed-order saddle targeting, Trust-region restricted step, Antisymmetric Muller variant, General JK-only two-electron energy functional f(na,nb), Muller functional f(na,nb)=sqrt(na*nb), No-pair Dirac-Hartree-Fock limit, np-ReNOFT energy expression Eq. (64) (+22 more)

### Community 39 - "FullOptimization.cpp"
Cohesion: 0.15
Nodes (37): casmodel, E, pccdhessian, rijkonlyfock, ripccdfock, ripnoffock, conjugate(), complex (+29 more)

### Community 40 - "RkbTwoElectron.cpp"
Cohesion: 0.26
Nodes (23): build, addInPlace(), conjMatrix(), complex, PackedTwoElectronTensor, RkbTwoElectronTensor, size_t, vector (+15 more)

### Community 41 - "ukbHamiltonianMatrix"
Cohesion: 0.19
Nodes (12): nablaintegrals, array, nablaIntegral(), complex, vector, ukbHamiltonianMatrix(), complex, vector (+4 more)

### Community 42 - "SpinorBasis"
Cohesion: 0.19
Nodes (12): pair, size_t, SpinBlock, vector, size_t, vector, spinBlockLabel(), SpinorBasis (+4 more)

### Community 43 - "AoCholesky"
Cohesion: 0.18
Nodes (21): bp, AoCholesky, fromOnDemand, fromPacked, n, vectors, aoCholeskyToMoSpinor(), aoCholeskyToMoSpinOrbital() (+13 more)

### Community 44 - "BasisFunction"
Cohesion: 0.10
Nodes (37): BasisFunction, cartesian, coefficients, element, exponents, l, x, y (+29 more)

### Community 45 - "BasisSet"
Cohesion: 0.19
Nodes (8): map, BasisSet, shells_by_element_, vector, Shell, coefficients, exponents, l

### Community 46 - "FullOptResult"
Cohesion: 0.08
Nodes (25): FullOptResult, checks_passed, converged, electronic_energy, gradient_max, iterations, occupation_state, occupations (+17 more)

### Community 47 - "nonRelScfImpl"
Cohesion: 0.07
Nodes (32): size_t, RiNonRelEriSource, eri3_L, nl, overlap, x_large, Eri, PackedTwoElectronTensor (+24 more)

### Community 48 - "Bowl"
Cohesion: 0.16
Nodes (10): Bowl, a_, best_, c_, restores, saves, trial_, T (+2 more)

### Community 49 - "test_adam.cpp"
Cohesion: 0.32
Nodes (13): adam, spinorrotation, check(), string, main(), near(), testComplexStep(), testExternalMove() (+5 more)

### Community 50 - "Restart.cpp"
Cohesion: 0.25
Nodes (15): cstring, istream, ostream, string, vector, get(), getString(), getVector() (+7 more)

### Community 51 - "RestartCapture"
Cohesion: 0.06
Nodes (35): aimpacView(), buildNonRelEri(), PackedTwoElectronTensor, hartreeFockAimpacCapture(), nonrelSpatialRotation(), writeFunctionalRdm(), writeNonRelFcidump(), complex (+27 more)

### Community 52 - "Adam"
Cohesion: 0.12
Nodes (12): Adam, clean, lr_, m1_, m2_, m2max_, opt_, step (+4 more)

### Community 53 - "X2C_MoTransform.cpp"
Cohesion: 0.17
Nodes (19): complex, x2cDensityMatrix(), c_matrix, complex, PackedTwoElectronTensor, size_t, SpinBlockSource, n (+11 more)

### Community 54 - "OrbitalToy"
Cohesion: 0.17
Nodes (9): AdamPair, size_t, OrbitalToy, a_, best_, d_, n_, pairs_ (+1 more)

### Community 55 - "vector"
Cohesion: 0.22
Nodes (8): vector, dot(), matvec(), maxDiff(), Quadratic, g_, h_, x_

### Community 56 - "test_symmetric_transform.cpp"
Cohesion: 0.23
Nodes (11): S, check(), cj(), size_t, string, uint64_t, rnd(), Rng (+3 more)

### Community 57 - "NeoStepOptions"
Cohesion: 0.11
Nodes (18): NeoStepOptions, decoupled_z0, guess_from_diagonal, linear_dependence, max_alpha, max_alpha_iterations, max_micro_iterations, max_subspace (+10 more)

### Community 58 - "test_symmetric_eri.cpp"
Cohesion: 0.22
Nodes (11): absOf(), check(), size_t, string, T, uint64_t, rnd(), Rng (+3 more)

### Community 59 - "Diis"
Cohesion: 0.06
Nodes (39): LbfgsGradientFn, LbfgsValueFn, Diis, clear, errors_, extrapolate, last_weights_, max_vectors_ (+31 more)

### Community 60 - "solveSqp"
Cohesion: 0.09
Nodes (31): SqpGradientFn, SqpHessianFn, SqpValueFn, BoxEqualityQpResult, bound_multipliers, converged, d, eq_multipliers (+23 more)

### Community 61 - "RkbMoTransform.cpp"
Cohesion: 0.34
Nodes (13): complex, RkbTwoElectronTensor, size_t, densify(), occupiedPositiveEnergyDensity(), rkbMoOneElectronTransform(), rkbMoTwoElectronSymmetric(), rkbMoTwoElectronTransformPhysics() (+5 more)

### Community 62 - "OrbitalToy"
Cohesion: 0.14
Nodes (12): OrbitalToy, a_, best_, d_, n_, pairs_, trial_, complex (+4 more)

### Community 63 - "stdexcept"
Cohesion: 0.10
Nodes (17): algorithm, cblas, chrono, cmath, cstdio, cstdlib, fstream, integrals (+9 more)

### Community 64 - "test_ukb_fock_direct.cpp"
Cohesion: 0.10
Nodes (50): kramerspairing, kramerssymmetry, random, rkbcholesky, rkbfockmatrix, rkbhamiltonian, rkbmotransform, rkboverlap (+42 more)

### Community 65 - "FullOptSettings"
Cohesion: 0.06
Nodes (31): OrbitalOptimizer, FullOptSettings, adam_neo_initial_adam_iterations, adam_neo_min_adam_iterations, adam_neo_switch_tolerance, check_hess_neo, checkpoint, checkpoint_interval (+23 more)

### Community 66 - "CrossPackedTwoElectronTensor"
Cohesion: 0.15
Nodes (11): CrossPackedTwoElectronTensor, data_, m_pq_, m_rs_, n_pq_, n_rs_, size_t, PackedTwoElectronTensor (+3 more)

### Community 67 - "CartesianExponents"
Cohesion: 0.14
Nodes (21): CartesianExponents, lx, ly, lz, string, NormalizationCheck, cartesian, element (+13 more)

### Community 68 - "CholeskyEri.cpp"
Cohesion: 0.24
Nodes (13): CholeskyEri<double>, CholeskyEri<std::complex<double>>, CholeskyEri<T>::fromDense(), CholeskyEri<T>::fromVectors(), CholeskyEri<T>::operator()(), CholeskyEri<T>::restricted(), CholeskyEri<T>::rotated(), CholeskyEri<T>::toDense() (+5 more)

### Community 69 - "run_examples.py"
Cohesion: 0.06
Nodes (39): argparse, Reader for the molecular-basis CC_RDM files <name>_MO.dm1 / <name>_MO.dm2…, check(), consistency(), extract_energies(), key_energies(), main(), occupation_stats() (+31 more)

### Community 70 - "NeoIteration"
Cohesion: 0.08
Nodes (23): size_t, NeoIteration, actual_change, alpha, energy, gradient_max, hessian_products, micro_iterations (+15 more)

### Community 71 - "NeoOptions"
Cohesion: 0.14
Nodes (14): NeoOptions, energy_offset, gradient_tolerance, initial_radius, max_iterations, max_rejections, min_radius, progress (+6 more)

### Community 72 - "dhfScfImpl"
Cohesion: 0.09
Nodes (30): complex, Eri, RkbTwoElectronTensor, vector, dhfScfImpl(), maxAbsDifference(), runDiracHartreeFockScf(), traceOfProduct() (+22 more)

### Community 73 - "makePccdModel"
Cohesion: 0.13
Nodes (26): makePccdModel(), pccdHessianDiagonalImpl(), pccdHessianMatrixImpl(), pccdHessianVectorImpl(), conjugate(), complex, orbitalGradient(), buildPccdFullTwoRdm() (+18 more)

### Community 74 - "Tensor"
Cohesion: 0.12
Nodes (36): Env, addIntoAxis(), antisymAB(), antisymIJ(), axpy(), conjugated(), cplx, size_t (+28 more)

### Community 75 - "run"
Cohesion: 0.14
Nodes (20): buildOrbitalSubspaces(), array, size_t, vector, enumeratePairs(), size_t, vector, OrbitalSubspace (+12 more)

### Community 76 - "AdamOptions"
Cohesion: 0.18
Nodes (11): AdamOptions, base_iterations, beta1, beta2, energy_tolerance, epsilon, gradient_tolerance, iterations_increment (+3 more)

### Community 77 - "AdamResult"
Cohesion: 0.18
Nodes (11): AdamResult, energy_best, energy_converged, energy_difference, energy_start, gradient_converged, improved, iterations (+3 more)

### Community 78 - "PnofSubspaceOccupationsWithGradient"
Cohesion: 0.50
Nodes (4): vector, PnofSubspaceOccupationsWithGradient, docc_dgamma, occ

### Community 79 - "rawHessianTerm"
Cohesion: 0.53
Nodes (8): checkHessianDimensions(), conjugate(), complex, size_t, T, generalizedOrbitalHessianElement(), generalizedOrbitalHessianElementImag(), rawHessianTerm()

### Community 80 - "AdamOptimizer"
Cohesion: 0.20
Nodes (7): AdamOptimizer, extra_iterations_, learning_rate_, opt_, reset, restart_, run

### Community 81 - "C"
Cohesion: 0.14
Nodes (24): C, dst, size_t, T, vector, gemmNN(), gemmTN(), hartreeExchangeFockMatrix() (+16 more)

### Community 82 - "rotateIntegrals"
Cohesion: 0.28
Nodes (7): cholesky_decomposition, conjugate(), complex, T, T, RotatedIntegrals, rotateIntegrals()

### Community 83 - "SerialBlasScope"
Cohesion: 0.38
Nodes (6): GetFn, SetFn, SerialBlasScope, get_, saved_, set_

### Community 84 - "OnDemandUnionCoulombPairs"
Cohesion: 0.20
Nodes (10): vector, OnDemandUnionCoulombPairs, large_shells_, local_of_large_, nl_, ns_, shell_of_large_, sqrt_diag_ll_ (+2 more)

### Community 85 - "size_t"
Cohesion: 0.25
Nodes (5): PackedTwoElectronTensor, size_t, UnionCoulombPairs, nl_, ns_

### Community 86 - "rkbCholeskyToMoFused"
Cohesion: 0.18
Nodes (14): complex, size_t, vector, promote(), recompressMoVectors(), RkbCholesky, large, n_large (+6 more)

### Community 87 - "Uccsd.cpp"
Cohesion: 0.11
Nodes (27): Amplitudes, t1, t2, CholeskyPairFactor, bra_, n_, nchol_, conjOf() (+19 more)

### Community 88 - "CcRdmSettings"
Cohesion: 0.07
Nodes (33): CcRdmResult, e_elec_cc, e_elec_rdm, e_onebody, e_total_rdm, lambda_converged, lambda_iterations, n_dm1 (+25 more)

### Community 89 - "FactoredCcIntegrals"
Cohesion: 0.09
Nodes (25): shared_ptr, CcIntegrals, physicsBlock, CcOrbitalSpaces, file_index, occ_active, occ_frozen, vir (+17 more)

### Community 90 - "t"
Cohesion: 0.13
Nodes (30): SpinBlock, activeEps(), buildBlocks(), buildDelta(), buildSigmas(), size_t, size_t, ooBlock() (+22 more)

### Community 91 - "NeoTrustOptions"
Cohesion: 0.17
Nodes (12): NeoTrustDecision, accept, radius, NeoTrustOptions, grow_factor, max_radius, ratio_high, ratio_low (+4 more)

### Community 92 - "Orbital rotation Hessian G_pq,rs (Fock-matrix form)"
Cohesion: 0.29
Nodes (8): Generalized Fock matrix F_qp, Orbital rotation gradient g_pq, Orbital rotation Hessian G_pq,rs (Fock-matrix form), JK-only orbital gradient g_pq, JK-only orbital Hessian G_pq,rs, Real/imaginary directional Hessian blocks (Hess^tt, Hess^yy, Hess^ty), main.cpp HESSIAN_4C diagnostic, HESSIAN_MEAN_FIELD keyword

### Community 93 - "complex"
Cohesion: 0.09
Nodes (17): choleskyeri, Hartree/exchange contribution G^HX_pq,rs, L1/L2 pair-term extension fixing the PNOF Hessian, Pair (L1/L2) contribution G^L_pq,rs, Cheap diagonal-D 2-RDM ansatz for PNOF (H, X, L1, L2), General no-pair relativistic 2-RDM reconstruction (Eqs. 97-98), jk_only, memory (+9 more)

### Community 94 - "SectionIndentBuf"
Cohesion: 0.12
Nodes (20): int_type, ostream, size_t, streambuf, string, LineWrapBuf, line_, sink_ (+12 more)

### Community 95 - "diagonalizeHermitian"
Cohesion: 0.12
Nodes (25): lapacke, k, canonicalOrthogonalizeHermitian(), complex, diagonalizeHermitian(), diagonalizeSymmetric(), complex, size_t (+17 more)

### Community 96 - "AdamProblem"
Cohesion: 0.12
Nodes (11): AdamProblem, dimension, energy, gradient, restoreBest, rotate, saveBest, complex (+3 more)

### Community 97 - "basisFingerprint"
Cohesion: 0.43
Nodes (7): basisFingerprint(), size_t, T, uint64_t, vector, fnv1aUpdateBytes(), fnv1aUpdateValue()

### Community 98 - "readRestartOrbitals"
Cohesion: 0.43
Nodes (7): complex, ostream, size_t, string, lowdinOrthonormalize(), maxDeviationFromIdentity(), readRestartOrbitals()

### Community 99 - "transformToSymmetric"
Cohesion: 0.32
Nodes (8): conjugate(), gemm(), gemmAcc(), complex, size_t, Src, T, transformToSymmetric()

### Community 100 - "CartesianSpinors"
Cohesion: 0.10
Nodes (23): CartesianSpinors, large_alpha, large_basis, large_beta, small_alpha, small_basis, small_beta, GaussScalar (+15 more)

### Community 101 - "functional"
Cohesion: 0.29
Nodes (4): functional, READ_RESTART keyword, Restart file (RESTART.NON_REL/X2C_HF/4C), SQP_PNOF_OCC keyword

### Community 102 - "Atom"
Cohesion: 0.16
Nodes (18): nuclearattraction, vector, nuclearAttractionMatrix(), nuclearAttractionPair(), Atom, symbol, x, y (+10 more)

### Community 103 - "ClosedShellSpinOrbitals.cpp"
Cohesion: 0.71
Nodes (6): closedShellSpinOrbitalDensity(), closedShellSpinOrbitalOneElectron(), closedShellSpinOrbitalTwoElectron(), size_t, spatialIndex(), spinOf()

### Community 104 - "test_casfock.cpp"
Cohesion: 0.24
Nodes (25): casfock, buildReference(), check(), checkBackend(), checkGradient(), ciEnergy(), cj(), Eri (+17 more)

### Community 105 - "Casci"
Cohesion: 0.10
Nodes (21): int8_t, Casci, densities, dets_, diagonal, links_, m_, n_ (+13 more)

### Community 106 - "RiDirectEriSource"
Cohesion: 0.10
Nodes (23): JkFunctional, size_t, vector, riJkOnlyFockMatrix(), size_t, vector, riPccdFockMatrix(), PnofFunctional (+15 more)

### Community 107 - "NeoEigenOptions"
Cohesion: 0.29
Nodes (7): NeoEigenOptions, linear_dependence, max_iterations, max_subspace, n_random_guesses, preconditioner_floor, residual_tolerance

### Community 108 - "RestartOrbitals"
Cohesion: 0.22
Nodes (9): main(), complex, RestartOrbitals, c, data, lowdin_applied, min_overlap_eigenvalue, overlap_deviation_final (+1 more)

### Community 109 - "makeAo"
Cohesion: 0.29
Nodes (6): PackedTwoElectronTensor, size_t, uint64_t, makeAo(), Rng, s

### Community 110 - "RdmSource"
Cohesion: 0.09
Nodes (21): FILE, int32_t, ostream, string, complex, function, size_t, RdmFilesResult (+13 more)

### Community 111 - "Casci.cpp"
Cohesion: 0.25
Nodes (21): R, Casci<double>, excitedVectors, Casci<std::complex<double>>, Casci<T>::densities(), Casci<T>::diagonal(), Casci<T>::excitedVectors(), Casci<T>::oneRdm() (+13 more)

### Community 112 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 131 - "UccsdResult"
Cohesion: 0.10
Nodes (21): UccsdResult, converged, e_corr, e_corr_imag, e_mp2, iterations, max_fock_asymmetry, max_fock_ov (+13 more)

### Community 132 - "fock"
Cohesion: 0.31
Nodes (20): T, generalizedFockMatrix(), addPhiHxRi(), addPhiHxRiRef(), addPhiL12Ri(), addPhiL12RiRef(), addPsiHxRi(), addPsiHxRiRef() (+12 more)

### Community 133 - "h"
Cohesion: 0.23
Nodes (18): h, buildPhiRi(), buildPhiRiRef(), buildPsiRi(), buildPsiRiRef(), complex, pair, size_t (+10 more)

### Community 134 - "fock"
Cohesion: 0.10
Nodes (21): fock, pair, size_t, vector, pair, size_t, vector, lowerPairsLocal() (+13 more)

### Community 135 - "solvePccdZAmplitudes"
Cohesion: 0.16
Nodes (20): j, invert(), function, vector, flatten(), string, norm2(), PccdAmplitudeResult (+12 more)

### Community 136 - "Cholesky_Decomposition.cpp"
Cohesion: 0.26
Nodes (17): choleskyDecomposeEri(), choleskyDecomposeEriChecked(), choleskyReconstructEri(), choleskyTransformEri(), choleskyTransformEriMixed(), choleskyTransformVectors(), conjMatrix(), conjugate() (+9 more)

### Community 137 - "cartesianComponents"
Cohesion: 0.15
Nodes (17): vector, vector, MolecularBasis, build, functions_, cartesianComponentIndex(), cartesianComponents(), vector (+9 more)

### Community 138 - "CcWorkspace"
Cohesion: 0.11
Nodes (19): CcWorkspace, act, e_o, e_v, f_oo, f_ov, f_vo, f_vv (+11 more)

### Community 139 - "JkOnlySigmaHessianVector.cpp"
Cohesion: 0.25
Nodes (16): buildPhi(), buildPsi(), checkJkOnlySigmaArgs(), complex, Eri, pair, size_t, T (+8 more)

### Community 140 - "test_casci.cpp"
Cohesion: 0.20
Nodes (17): apply(), bruteForceH(), check(), cj(), size_t, string, T, uint64_t (+9 more)

### Community 141 - "main"
Cohesion: 0.21
Nodes (15): kramersAoOneBodyDeviation(), kramersOneBodyDeviation(), h, check(), column(), size_t, string, uint64_t (+7 more)

### Community 142 - "eri"
Cohesion: 0.43
Nodes (15): addPhiHX(), addPhiL12(), addPsiHX(), addPsiL12(), checkHartreeExchangeSigmaArgs(), conjugateT(), complex, Eri (+7 more)

### Community 143 - "main"
Cohesion: 0.39
Nodes (15): buildPccdPairOf(), complex, Eri, pair, size_t, T, vector, pccdHessianElement() (+7 more)

### Community 144 - "UkbDirectEriSource"
Cohesion: 0.14
Nodes (16): JkFunctional, size_t, vector, ukbJkOnlyFockMatrix(), JkFunctional, size_t, vector, ukbJkOnlyFockMatrixFast() (+8 more)

### Community 145 - "T"
Cohesion: 0.31
Nodes (13): checkedImpl(), copyRow(), PackedTwoElectronTensor, size_t, Src, T, decomposeImpl(), elementAt() (+5 more)

### Community 146 - "X2CDecouplingResult"
Cohesion: 0.13
Nodes (15): c_dhf, complex, rkbCoefficientMatrix(), rkbDensityMatrix(), complex, RkbTwoElectronTensor, dhfOrbitalGradientEfficient(), complex (+7 more)

### Community 147 - "KramersSymmetry.cpp"
Cohesion: 0.43
Nodes (14): s_full, applyOddColumnPhases(), complex, vector, elementwiseConjugate(), fixKramersPairing(), fixKramersPhase(), fixKramersPhaseLarge() (+6 more)

### Community 148 - "ThreeCenterIntegrals.cpp"
Cohesion: 0.25
Nodes (13): size_t, ShellInfo, count, first, auxMetric(), auxMetricOrthogonalization(), size_t, vector (+5 more)

### Community 149 - "DBlockDef"
Cohesion: 0.14
Nodes (14): AsmRow, axes, conj, has_axes, sign, src, array, pair (+6 more)

### Community 150 - "MoIntegralTransform.cpp"
Cohesion: 0.32
Nodes (12): PackedTwoElectronTensor, size_t, densify(), moTwoElectronSymmetric(), moTwoElectronTransformPhysics(), moTwoElectronTransformPhysicsCholesky(), PhysicsFromChemist, transformLeg1() (+4 more)

### Community 151 - "PccdRdm"
Cohesion: 0.23
Nodes (14): buildPccdCoefficients(), buildPccdRdm(), Eri, T, w, pccdPairEnergyFromRdm(), PccdRdm, d (+6 more)

### Community 152 - "test_pccd_amplitudes.cpp"
Cohesion: 0.25
Nodes (12): check(), PccdAmplitudeSolver, size_t, string, uint64_t, main(), randomCoefficients(), Rng (+4 more)

### Community 153 - "PairOp"
Cohesion: 0.17
Nodes (13): FI, FW, size_t, T, uint32_t, vector, makeOp(), PairOp (+5 more)

### Community 154 - "RiMoEri"
Cohesion: 0.31
Nodes (6): denseOf(), complex, size_t, RiMoEri, b_, n_total_

### Community 155 - "writeGaussRdm"
Cohesion: 0.18
Nodes (9): GammaSlabFn, FILE, GaussScalar, int32_t, ostream, string, FortranRecords, f_ (+1 more)

### Community 156 - "GaussRdmResult"
Cohesion: 0.17
Nodes (12): GaussRdmResult, max_imag, n_dm1, n_dm2, n_primitives, n_primitives_large, passes, seconds (+4 more)

### Community 157 - "Term"
Cohesion: 0.20
Nodes (12): cplx, size_t, vector, EqDef, name, terms, fileOrderGlobals(), Term (+4 more)

### Community 158 - "RiNonRelFullOptFock.cpp"
Cohesion: 0.38
Nodes (11): buildSpinOneElectron(), JkFunctional, PnofFunctional, size_t, vector, riJkOnlyFockMatrixNonRel(), riPccdFockMatrixNonRel(), riPnofFockMatrixNonRel() (+3 more)

### Community 159 - "RiSigmaHessianToolkit"
Cohesion: 0.35
Nodes (11): size_t, T, vector, RiSigmaHessianToolkit, bdiag, kb1, kb1diag, kb2 (+3 more)

### Community 160 - "CasciHamiltonian"
Cohesion: 0.18
Nodes (12): CasciHamiltonian, core_energy, n_electrons, n_orbitals, CasciResult, ci, converged, energy (+4 more)

### Community 161 - "PccdSettings"
Cohesion: 0.17
Nodes (12): PccdAmplitudeSolver, vector, PccdSettings, debug, diis_max_iterations, diis_size, fixed_window_occupations, krylov_max_iterations (+4 more)

### Community 162 - "pairContract"
Cohesion: 0.42
Nodes (10): accumU(), size_t, T, vector, diagContract(), fillOp(), gemmNN(), gemmTNacc() (+2 more)

### Community 163 - "OnDemandPairs"
Cohesion: 0.33
Nodes (6): size_t, vector, OnDemandPairs, n_, sqrt_diag_, threshold_

### Community 164 - "Wfn"
Cohesion: 0.20
Nodes (11): string, vector, parseFortranDouble(), readWfn(), Wfn, centre, centre_xyz, exponent (+3 more)

### Community 165 - "DIIS.cpp"
Cohesion: 0.24
Nodes (9): complex, size_t, vector, Diis<double>, Diis<std::complex<double>>, Diis<T>::Diis(), Diis<T>::extrapolate(), realOfProduct() (+1 more)

### Community 166 - "Prim"
Cohesion: 0.20
Nodes (10): Prim, alpha, lx, ly, lz, norm, x, y (+2 more)

### Community 167 - "makeJkOnlyModel"
Cohesion: 0.53
Nodes (9): EriT, JkFunctional, jkOnlyHessianDiagonalImpl(), jkOnlyHessianMatrixImpl(), jkOnlyHessianVectorImpl(), makeJkOnlyModel(), vector, jkExchangeCoupling() (+1 more)

### Community 168 - "Component"
Cohesion: 0.31
Nodes (9): Component, Cc, P, sector, CBLAS_TRANSPOSE, cplx, size_t, gemm() (+1 more)

### Community 169 - "IssueTee"
Cohesion: 0.25
Nodes (6): int_type, streambuf, IssueTee, issues_, line_, target_

### Community 170 - "RkbDerivativeTerms"
Cohesion: 0.22
Nodes (9): size_t, RkbDerivativeTerms, kNone, lower_x, lower_y, lower_z, raise_x, raise_y (+1 more)

### Community 171 - "KramersPairingReport"
Cohesion: 0.22
Nodes (9): size_t, KramersPairingReport, largest_cluster_pairs, max_column_change, max_enforcement_change, max_invariance_residual, n_multi_pair_clusters, partner_error_after (+1 more)

### Community 172 - "SphericalTransformResult"
Cohesion: 0.25
Nodes (9): size_t, vector, SphericalShellLayout, cartesian_offset, l, spherical_offset, SphericalTransformResult, n_spherical (+1 more)

### Community 173 - "X2C_hamiltonian.cpp"
Cohesion: 0.50
Nodes (8): approximateX2COrtho(), buildX2CHamiltonian(), complex, size_t, extractLargeComponentBlock(), hermitianPower(), renormalizedX2CHamiltonian(), subBlock()

### Community 174 - "ShellQuartetBlock"
Cohesion: 0.25
Nodes (7): vector, ShellQuartetBlock, data, ni, nj, nk, nl

### Community 175 - "GaussRdmSettings"
Cohesion: 0.25
Nodes (8): GaussRdmSettings, basis_path, dm1_path, dm2_path, max_memory_gb, threshold, unnormalized_primitives, string

### Community 176 - "RecordWriter"
Cohesion: 0.25
Nodes (5): FILE, int32_t, string, RecordWriter, f_

### Community 177 - "buildRiSigmaHessianToolkit"
Cohesion: 0.38
Nodes (5): buildRiSigmaHessianToolkit(), complex, size_t, T, gemmNN()

### Community 178 - "Reference"
Cohesion: 0.29
Nodes (7): vector, Reference, ci_energy, energy, rdm1, rdm2, state

### Community 179 - "buildSector"
Cohesion: 0.47
Nodes (6): buildSector(), vector, libcintCommonFac(), Sector, prims, weights

### Community 180 - "hartreeExchangeEnergy"
Cohesion: 0.53
Nodes (6): Eri, size_t, T, vector, hartreeExchangeEnergy(), hartreeExchangeFockMatrix()

### Community 181 - "overlap1d"
Cohesion: 0.40
Nodes (5): binom(), doubleFactorial(), overlap1d(), primitiveNorm(), primitiveOverlap()

### Community 182 - "jointOrbitalGradient"
Cohesion: 0.40
Nodes (5): pair, size_t, T, vector, jointOrbitalGradient()

### Community 183 - "RiNonRelDirectSource"
Cohesion: 0.40
Nodes (5): RiNonRelDirectSource, c0, eri3_L, h_core_ao, n_spatial

### Community 184 - "binomial"
Cohesion: 0.60
Nodes (5): binomial(), Casci<T>::Casci(), size_t, uint64_t, popcount()

### Community 185 - "Rng"
Cohesion: 0.40
Nodes (4): uint64_t, rnd(), Rng, s

### Community 186 - "lowerPairsLocal"
Cohesion: 0.40
Nodes (5): pair, size_t, vector, lowerPairsLocal(), main()

### Community 187 - "main"
Cohesion: 0.40
Nodes (5): pair, size_t, vector, lowerPairsLocal(), main()

### Community 188 - "lowerPairsLocal"
Cohesion: 0.40
Nodes (5): pair, size_t, vector, lowerPairsLocal(), main()

### Community 189 - "main"
Cohesion: 0.40
Nodes (5): pair, size_t, vector, lowerPairsLocal(), main()

### Community 190 - "PrimitiveSlot"
Cohesion: 0.50
Nodes (4): PrimitiveSlot, centre, exponent, type_code

### Community 191 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 192 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 193 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 194 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 195 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 196 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 197 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 198 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 199 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 200 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

### Community 201 - "Rng"
Cohesion: 0.50
Nodes (3): uint64_t, Rng, s

## Knowledge Gaps
- **905 isolated node(s):** `exponent`, `am`, `am`, `exponents`, `shells_by_element_` (+900 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1361 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Matrix` connect `Matrix` to `test_neo.cpp`, `test_diis.cpp`, `JK_only.cpp`, `KramersPairing.cpp`, `vector`, `AuxiliaryBasis.cpp`, `PNOFs.cpp`, `Input`, `main.cpp`, `PnofGeminal`, `NEO.cpp`, `Tensor4`, `vector`, `RotationProblem`, `KramersRestriction`, `RiNonRelSpinMoEri`, `RdmftModel`, `runSystem`, `DiracHartreeFockResult`, `ADAM.cpp`, `vector`, `test_cholesky_eri.cpp`, `jkOnlyOrbitalGradient`, `HartreeExchangeHessian.cpp`, `string`, `main`, `OrbitalRotationGradientCheck`, `iomanip`, `X2CHartreeFockResult`, `test_kramers_restriction.cpp`, `PNOFs.h`, `FullOptimization.cpp`, `RkbTwoElectron.cpp`, `ukbHamiltonianMatrix`, `AoCholesky`, `BasisFunction`, `FullOptResult`, `nonRelScfImpl`, `Bowl`, `test_adam.cpp`, `RestartCapture`, `X2C_MoTransform.cpp`, `OrbitalToy`, `vector`, `test_symmetric_transform.cpp`, `Diis`, `solveSqp`, `RkbMoTransform.cpp`, `OrbitalToy`, `stdexcept`, `test_ukb_fock_direct.cpp`, `CholeskyEri.cpp`, `dhfScfImpl`, `makePccdModel`, `Tensor`, `PnofSubspaceOccupationsWithGradient`, `rawHessianTerm`, `C`, `rotateIntegrals`, `rkbCholeskyToMoFused`, `Uccsd.cpp`, `CcRdmSettings`, `t`, `complex`, `diagonalizeHermitian`, `readRestartOrbitals`, `transformToSymmetric`, `CartesianSpinors`, `functional`, `Atom`, `ClosedShellSpinOrbitals.cpp`, `test_casfock.cpp`, `RiDirectEriSource`, `RestartOrbitals`, `RdmSource`, `Casci.cpp`, `UccsdResult`, `fock`, `h`, `solvePccdZAmplitudes`, `Cholesky_Decomposition.cpp`, `cartesianComponents`, `JkOnlySigmaHessianVector.cpp`, `test_casci.cpp`, `main`, `eri`, `main`, `UkbDirectEriSource`, `T`, `X2CDecouplingResult`, `KramersSymmetry.cpp`, `ThreeCenterIntegrals.cpp`, `MoIntegralTransform.cpp`, `PccdRdm`, `test_pccd_amplitudes.cpp`, `PairOp`, `RiMoEri`, `writeGaussRdm`, `RiNonRelFullOptFock.cpp`, `RiSigmaHessianToolkit`, `CasciHamiltonian`, `pairContract`, `DIIS.cpp`, `makeJkOnlyModel`, `Component`, `SphericalTransformResult`, `X2C_hamiltonian.cpp`, `buildRiSigmaHessianToolkit`, `Reference`, `buildSector`, `hartreeExchangeEnergy`, `jointOrbitalGradient`, `RiNonRelDirectSource`?**
  _High betweenness centrality (0.614) - this node is a cross-community bridge._
- **Why does `BasisFunction` connect `BasisFunction` to `AuxiliaryBasis.cpp`, `Input`, `cartesianComponents`, `main.cpp`, `UkbDirectEriSource`, `runSystem`, `ThreeCenterIntegrals.cpp`, `vector`, `string`, `OnDemandPairs`, `RkbTwoElectron.cpp`, `ukbHamiltonianMatrix`, `SpinorBasis`, `AoCholesky`, `BasisSet`, `buildSector`, `RestartCapture`, `test_ukb_fock_direct.cpp`, `CartesianExponents`, `OnDemandUnionCoulombPairs`, `rkbCholeskyToMoFused`, `basisFingerprint`, `CartesianSpinors`, `Atom`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `Casci` connect `Casci` to `CasciHamiltonian`, `test_casfock.cpp`, `test_casci.cpp`, `Casci.cpp`, `complex`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 76 inferred relationships involving `main()` (e.g. with `read` and `BasisFunction`) actually correct?**
  _`main()` has 76 INFERRED edges - model-reasoned connections that need verification._
- **What connects `exponent`, `am`, `am` to the rest of the system?**
  _905 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_neo.cpp` be split into smaller, more focused modules?**
  _Cohesion score 0.05411392405063291 - nodes in this community are weakly interconnected._
- **Should `vector` be split into smaller, more focused modules?**
  _Cohesion score 0.06277665995975855 - nodes in this community are weakly interconnected._