# Fugaku A64FX LLVM/Clang toolchain — mpiclang++ with OpenBLAS
#
# Requires: module load LLVM/llvmorg-22.1.0
# Uses the system Fujitsu MPI wrapper (mpiclang++) which wraps
# aarch64-linux-gnu-clang++ with Fujitsu MPI.
# OpenBLAS is the tested BLAS/LAPACK provider for this toolchain.

set(CMAKE_CXX_COMPILER mpiclang++)
set(CMAKE_C_COMPILER   mpiclang)

# OpenMP via LLVM's libomp
set(OpenMP_CXX_FLAGS      "-fopenmp")
set(OpenMP_CXX_LIB_NAMES  "")
set(OpenMP_CXX_LIBRARIES  "")
set(OpenMP_CXX_FLAGS_INIT "-fopenmp")
