# AMD ROCm toolchain — amdclang++ with system MPI wrapper
#
# Assumes the ROCm installation is on PATH (e.g. /opt/rocm/bin) and
# mpicxx wraps amdclang++.
# OpenMP 5 GPU offload uses -fopenmp --offload-arch=<arch>.

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER   mpicc)

set(OpenMP_CXX_FLAGS      "-fopenmp --offload-arch=${SBD_GPU_ARCH}")
set(OpenMP_CXX_LIB_NAMES  "")
set(OpenMP_CXX_LIBRARIES  "")
set(OpenMP_CXX_FLAGS_INIT "-fopenmp")
