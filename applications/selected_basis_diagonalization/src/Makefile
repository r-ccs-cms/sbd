# SBD Build System
# Usage: make [GPU=1] [DEBUG=1] [VERBOSE=1]

# User options
GPU ?= 1
TRADMODE ?= 1
DEBUG ?= 0
VERBOSE ?= 0

# Paths
SBD_PATH = ../../..
SBD_INCLUDE_DIR = $(SBD_PATH)/include

# Compiler: Override Make's built-in default (CXX = g++)
# Use := to set default but still allow command-line override: make CXX=other-compiler
ifeq ($(origin CXX), default)
    CXX := CC
endif

# Required flags that must always be present
REQUIRED_CXXFLAGS = -std=c++17 -fopenmp -funroll-loops -I$(SBD_INCLUDE_DIR)
REQUIRED_LDFLAGS = -fopenmp -llapack -lblas

# Prepend optimization flags (these may reset -funroll-loops or -march=native)
ifeq ($(DEBUG),1)
    REQUIRED_CXXFLAGS := -O0 -g -DDEBUG $(REQUIRED_CXXFLAGS)
else
    REQUIRED_CXXFLAGS := -O3 -march=native $(REQUIRED_CXXFLAGS)
endif

# Traditional mode
ifeq ($(TRADMODE),1)
    REQUIRED_CXXFLAGS += -DSBD_TRADMODE
endif

# GPU configuration
#-----------------------------------------------------------------------------
# for AMD GPUs: -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a
# use rocminfo | grep gfx to get the offload architecture
#-----------------------------------------------------------------------------
# for NVIDIA GPUS: -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_100
# NVIDIA A100 sm_80,  H100 sm_90,  GB200 sm_100, etc.
#-----------------------------------------------------------------------------
ifeq ($(GPU),1)
    REQUIRED_CXXFLAGS += -DUSE_GPU -DUSE_DET_CACHE_OMP -DUSE_OMP_OFFLOAD \
		-fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a \
                -fopenmp-offload-mandatory -foffload-lto
    REQUIRED_LDFLAGS += -foffload-lto
else
    REQUIRED_CXXFLAGS += -DUSE_DET_CACHE_OMP
endif

# Allow user override but prepend required flags
override CXXFLAGS := $(REQUIRED_CXXFLAGS) $(CXXFLAGS)
override LDFLAGS := $(REQUIRED_LDFLAGS) $(LDFLAGS)

# Verbosity
ifeq ($(VERBOSE),1)
    $(info === Build Configuration ===)
    $(info Compiler: $(CXX))
    $(info GPU: $(GPU))
    $(info DEBUG: $(DEBUG))
    $(info CXXFLAGS: $(CXXFLAGS))
    $(info LDFLAGS: $(LDFLAGS))
    $(info ==========================)
else
    Q = @
endif

# Targets
TARGET = diag
OBJECTS = main.o

.DEFAULT_GOAL := $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "[LINK] $@"
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	@echo "[CXX]  $<"
	$(Q)$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(Q)rm -f $(TARGET) $(OBJECTS)

.PHONY: clean
