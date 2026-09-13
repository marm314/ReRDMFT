CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
SRC_DIR  := src
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

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

CPPFLAGS := -I$(LIBCINT_INC)
# LAPACKE (the C interface to LAPACK) is used for the RKB transformation's
# overlap-matrix inverse; installed system-wide via liblapacke-dev.
LDLIBS   := $(LIBCINT) -llapacke -llapack -lblas -lquadmath -lm

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN)
