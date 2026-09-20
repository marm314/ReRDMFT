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

# All source files live directly under src/ or exactly one level below
# it (src/<SomeDir>/*.cpp) -- e.g. (not exhaustive, and deliberately not
# kept in sync by hand): AO_basis/ (cartesian AO basis data structures),
# AO_ints/ (every file that calls LIBCINT directly to evaluate an AO
# integral), C4_DHF/ (4-component Dirac-Hartree-Fock, RKB spinor basis),
# NON_REL/ (nonrelativistic HF), Hessian_opt/ (RDMFT/CASSCF orbital
# gradient/Hessian machinery, orthonormal MO basis only), Occ_opt/
# (occupation-number SQP optimization), X2C_DHF/ (X2C decoupling and
# X2C-HF), plus whatever else has been split out since this comment was
# last updated. EVERY such subdirectory is auto-discovered below (both
# for compiling its own .cpp files and for the shared -I search path,
# so files in any one can #include headers from any other with no path
# prefix) -- moving, renaming, or adding a source subdirectory needs NO
# Makefile edit, only that its .cpp basenames stay globally unique
# (object files all land flatly in $(BUILD_DIR), keyed by basename
# alone) and that it stays exactly one level under src/ (a deeper nested
# subdirectory, e.g. src/A/B/*.cpp, is NOT discovered).
SRC_SUBDIRS  := $(patsubst %/,%,$(wildcard $(SRC_DIR)/*/))
ALL_SRC_DIRS := $(SRC_DIR) $(SRC_SUBDIRS)
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

SRCS := $(foreach d,$(ALL_SRC_DIRS),$(wildcard $(d)/*.cpp))
OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(SRCS:.cpp=.o)))

# Every discovered directory is on the quoted-include search path, so
# files in any one can #include headers from any other without a path
# prefix.
CPPFLAGS := -I$(LIBCINT_INC) $(addprefix -I,$(ALL_SRC_DIRS))
# LAPACKE (the C interface to LAPACK) is used for the RKB transformation's
# overlap-matrix inverse; installed system-wide via liblapacke-dev.
LDLIBS   := $(LIBCINT) -llapacke -llapack -lblas -lquadmath -lm

GIT_VERSION_HEADER := $(SRC_DIR)/GitVersion.h

.PHONY: all clean force

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

# One generic pattern rule for every object, regardless of which
# discovered directory its .cpp actually lives in: `vpath` tells make
# where to look for a prerequisite named `%.cpp` that isn't in the
# current directory, trying each of $(ALL_SRC_DIRS) in turn, so `$<`
# below resolves to wherever the file was actually found.
vpath %.cpp $(ALL_SRC_DIRS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
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

# Standalone unit test of Utils/NEO (tests/test_neo.cpp): needs only NEO and
# the LAPACK wrappers, no integrals/basis code. Run: make test_neo LIBCINT=...
.PHONY: test_neo
test_neo: $(BUILD_DIR)/test_neo
	./$(BUILD_DIR)/test_neo

$(BUILD_DIR)/test_neo: tests/test_neo.cpp $(BUILD_DIR)/NEO.o $(BUILD_DIR)/LinearAlgebra.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR) $(BIN) $(GIT_VERSION_HEADER)
# ($(BUILD_DIR) already holds the .d files alongside their .o's, so the
# rm -rf above removes them too -- nothing extra needed here.)
