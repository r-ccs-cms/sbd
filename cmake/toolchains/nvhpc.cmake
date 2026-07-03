# NVHPC standalone toolchain — nvc++ with system MPI wrapper
#
# Assumes mpic++ wraps nvc++ and is on PATH.
# BLAS/LAPACK are found via FindBLAS/FindLAPACK (system or module-provided).

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER   mpicc)

# nvc++ uses -mp for OpenMP
set(OpenMP_CXX_FLAGS      "-mp")
set(OpenMP_CXX_LIB_NAMES  "")
set(OpenMP_CXX_LIBRARIES  "")
set(OpenMP_CXX_FLAGS_INIT "-mp")
