CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
SRC_DIR  := src
# 4-component Dirac-Hartree-Fock two-electron integrals (restricted
# kinetic balance spinor basis): kept in their own subdirectory since they
# are a distinct, self-contained piece of the physics (RkbTwoElectron.h,
# ElectronRepulsion.h, Tensor4.h), built into the same binary.
C4_DIR   := $(SRC_DIR)/C4_DHF
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

SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
C4_SRCS  := $(wildcard $(C4_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS)) \
            $(patsubst $(C4_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(C4_SRCS))

# Both directories are on the quoted-include search path, so files in
# either one can #include headers from the other without a path prefix.
CPPFLAGS := -I$(LIBCINT_INC) -I$(SRC_DIR) -I$(C4_DIR)
# LAPACKE (the C interface to LAPACK) is used for the RKB transformation's
# overlap-matrix inverse; installed system-wide via liblapacke-dev.
LDLIBS   := $(LIBCINT) -llapacke -llapack -lblas -lquadmath -lm

GIT_VERSION_HEADER := $(SRC_DIR)/GitVersion.h

.PHONY: all clean force

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(C4_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

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
