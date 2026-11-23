# caop-spectrum, Thrust backend on AMD GPUs, built with hipcc. Use with: make -f Makefile.hip
# Path to the SBD library
SBD_PATH=../..
# MPI C++ wrapper driving hipcc, e.g. for MPICH:  export MPICH_CXX=hipcc   (Open MPI: export OMPI_CXX=hipcc)
# A GPU-aware MPI (built with ROCm/HIP support) is needed for several ranks.
CCCOM=mpicxx
# GPU architecture: gfx90a = MI200 (MI210/MI250/MI250X), gfx942 = MI300
GPU_ARCH=gfx90a
CCFLAGS= --offload-arch=$(GPU_ARCH) -std=c++17 -O3 -fopenmp -DSBD_THRUST -Wno-unused-result -Wno-unused-variable
LDFLAGS= --offload-arch=$(GPU_ARCH) -fopenmp
# BLAS and LAPACK (OpenBLAS here); give the full path if a directory on -L also holds another MPI's libmpi
SYSLIB= -lopenblas
