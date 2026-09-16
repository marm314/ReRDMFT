CXX      := g++
# -fopenmp: used by AO_ints/ElectronRepulsion.cpp and C4_DHF/RkbTwoElectron.cpp to
# parallelize the two-electron integral construction (both the raw libcint
# evaluation and the RKB-basis leg transforms are embarrassingly parallel
# over disjoint output blocks). Affects both compilation (pragma
# recognition) and linking (libgomp), since CXXFLAGS is used for both.
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -fopenmp
# -MMD -MP: emit a per-object .d file listing the project headers it
# includes, so `make` rebuilds an object when a header IT USES changes
# -- not just when its own .cpp's mtime changes. Without this, editing
# a widely-#included header (e.g. Input.h) only recompiles the .cpp
# files whose mtime you also touched, leaving every OTHER already-built
# .o compiled against the OLD header silently stale: since these are
# all linked into one binary with no ABI/version check, a stale object
# built against an old class layout (e.g. Input gaining a new member)
# corrupts memory at runtime instead of failing to compile -- hit for
# real once (Input.h gained a `functional_` member; main.o stayed
# stale and segfaulted deep inside Input::read's std::string
# assignment). -MP adds a dummy rule per header so a RENAMED/DELETED
# header doesn't break the build with a "no rule to make target" error.
DEPFLAGS := -MMD -MP
SRC_DIR  := src
# Every file that calls LIBCINT directly to evaluate an AO integral
# (Integrals.h/.cpp's overlap+normalization, NablaIntegrals.h/.cpp,
# NuclearAttraction.h/.cpp, SchrodingerKinetic.h/.cpp,
# ElectronRepulsion.h/.cpp's two-electron tensor): kept in their own
# subdirectory so the LIBCINT-facing layer is easy to find as a whole,
# separate from everything built ON TOP of these AO integrals (RKB/
# spinor transforms, SCF, etc.) elsewhere in the tree.
AO_DIR   := $(SRC_DIR)/AO_ints
# 4-component Dirac-Hartree-Fock two-electron integrals (restricted
# kinetic balance spinor basis): kept in their own subdirectory since they
# are a distinct, self-contained piece of the physics (RkbTwoElectron.h,
# Tensor4.h), built into the same binary. Its own ElectronRepulsion.h/.cpp
# (the actual LIBCINT call) lives in AO_DIR instead, reused via the shared
# include path below.
C4_DIR   := $(SRC_DIR)/C4_DHF
# Nonrelativistic (Large-component-only) Hartree-Fock: its own
# subdirectory for the same reason as C4_DIR, and reuses C4_DIR's
# ElectronRepulsion.h/Tensor4.h (both basis-agnostic) via the shared
# include path below rather than duplicating them.
NON_REL_DIR := $(SRC_DIR)/NON_REL
# Orbital-optimization machinery (generalized Fock matrix, and later the
# orbital Hessian) for RDMFT/CASSCF-style wavefunctions with a general
# (non-idempotent) 1-RDM and an externally-supplied 2-RDM: its own
# subdirectory for the same reason as C4_DIR/NON_REL_DIR, working
# entirely in an orthonormal MO basis (reusing Matrix.h/Tensor4.h via the
# shared include path below, not any AO- or RKB-spinor-specific code).
HESSIAN_DIR := $(SRC_DIR)/Hessian_opt
# Occupation-number optimization machinery (sequential quadratic
# programming, for a fixed orbital basis): its own subdirectory for the
# same reason as HESSIAN_DIR, working with plain Matrix<double>/
# std::vector<double> only (no eri/RKB/basis-specific code).
OCC_DIR := $(SRC_DIR)/Occ_opt
# One-electron X2C ("exact two-component") decoupling of the bare RKB
# Dirac Hamiltonian: its own subdirectory for the same reason as
# C4_DIR, reusing RkbOrthogonalization.h/LinearAlgebra.h via the shared
# include path below.
X2C_DIR := $(SRC_DIR)/X2C_DHF
BUILD_DIR:= build
BIN      := rerdmft

# Optional file for persisting your own settings across builds (git-ignored),
# e.g. a line "LIBCINT := /path/to/libcint.a".
-include Makefile.local

# Path to a compiled LIBCINT static library (libcint.a), see
# https://github.com/sunqm/libcint. Point it at your own installation, e.g.:
#   make LIBCINT=$(HOME)/Installers/libcint/libcint-5.1.6/build/libcint.a
# Its headers (cint.h) are expected in an "include" directory next to it,
# which is where libcint's own CMake build places them; override
# LIBCINT_INC if yours live elsewhere.
LIBCINT     ?=
LIBCINT_INC ?= $(dir $(LIBCINT))include

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(strip $(LIBCINT)),)
$(error LIBCINT is not set. Build with: make LIBCINT=/path/to/libcint.a)
endif
endif

SRCS        := $(wildcard $(SRC_DIR)/*.cpp)
AO_SRCS     := $(wildcard $(AO_DIR)/*.cpp)
C4_SRCS     := $(wildcard $(C4_DIR)/*.cpp)
NON_REL_SRCS:= $(wildcard $(NON_REL_DIR)/*.cpp)
HESSIAN_SRCS:= $(wildcard $(HESSIAN_DIR)/*.cpp)
OCC_SRCS    := $(wildcard $(OCC_DIR)/*.cpp)
X2C_SRCS    := $(wildcard $(X2C_DIR)/*.cpp)
OBJS        := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS)) \
               $(patsubst $(AO_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(AO_SRCS)) \
               $(patsubst $(C4_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(C4_SRCS)) \
               $(patsubst $(NON_REL_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(NON_REL_SRCS)) \
               $(patsubst $(HESSIAN_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(HESSIAN_SRCS)) \
               $(patsubst $(OCC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(OCC_SRCS)) \
               $(patsubst $(X2C_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(X2C_SRCS))

# All seven directories are on the quoted-include search path, so files
# in any one can #include headers from the others without a path prefix.
CPPFLAGS := -I$(LIBCINT_INC) -I$(SRC_DIR) -I$(AO_DIR) -I$(C4_DIR) -I$(NON_REL_DIR) -I$(HESSIAN_DIR) -I$(OCC_DIR) -I$(X2C_DIR)
# LAPACKE (the C interface to LAPACK) is used for the RKB transformation's
# overlap-matrix inverse; installed system-wide via liblapacke-dev.
LDLIBS   := $(LIBCINT) -llapacke -llapack -lblas -lquadmath -lm

GIT_VERSION_HEADER := $(SRC_DIR)/GitVersion.h

.PHONY: all clean force

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(AO_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(C4_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(NON_REL_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(HESSIAN_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(OCC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(X2C_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(CPPFLAGS) -c $< -o $@

# Pull in each object's own header-dependency list generated above (a
# missing .d on a fresh checkout/clean is fine -- `-include` ignores
# that silently instead of erroring).
-include $(OBJS:.o=.d)

# main.cpp prints the current commit SHA at startup, so it needs to be
# rebuilt whenever GitVersion.h's content actually changes (not merely
# regenerated -- see the recipe below, which only rewrites the file, and
# so only updates its mtime, when the SHA differs from what's already
# there).
$(BUILD_DIR)/main.o: $(GIT_VERSION_HEADER)

# Depending on the phony `force` target makes this recipe run on every
# build, but the file itself is only rewritten when its content changes,
# so dependents only see it as "changed" (and get rebuilt) when the
# commit actually differs from the last build.
$(GIT_VERSION_HEADER): force
	@sha=$$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown); \
	content="#define RERDMFT_GIT_SHA \"$$sha\""; \
	if [ ! -f $@ ] || [ "$$(cat $@ 2>/dev/null)" != "$$content" ]; then \
		echo "$$content" > $@; \
	fi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN) $(GIT_VERSION_HEADER)
# ($(BUILD_DIR) already holds the .d files alongside their .o's, so the
# rm -rf above removes them too -- nothing extra needed here.)
