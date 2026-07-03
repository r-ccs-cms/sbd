cmake_minimum_required(VERSION 3.20)
project(SBD LANGUAGES CXX VERSION 0.1)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── GPU backend ───────────────────────────────────────────────────────────────
set(SBD_GPU_BACKEND "none" CACHE STRING "GPU backend: none | thrust | omp5")
set_property(CACHE SBD_GPU_BACKEND PROPERTY STRINGS none thrust omp5)

# Architecture token: cc90 (GH200/H100), sm_80 (A100), gfx90a (MI250X), …
set(SBD_GPU_ARCH "cc90" CACHE STRING "GPU architecture for the chosen backend")

# ── Feature flags ─────────────────────────────────────────────────────────────
option(SBD_THRUST_SAFE_MPI_ALLREDUCE   "Safe Thrust MPI allreduce"                         OFF)
option(SBD_TRADMODE                    "Traditional (non-Thrust) CPU mode for tpb"         OFF)
option(SBD_USE_NVTX                    "NVTX profiling annotations"                        OFF)
option(SBD_USE_NCCL                    "Link against NCCL"                                 OFF)
option(SBD_USE_CUBLAS                  "Link against cuBLAS"                               OFF)
option(SBD_USE_RANK_DISTRIBUTION       "Enable rank distribution"                          OFF)
option(SBD_USE_BLOCK_RANK_DISTRIBUTION "Enable block rank distribution"                    OFF)
option(SBD_REORDER_INDEX_ARRAY         "Enable index array reordering"                     OFF)
option(SBD_USE_32BIT_PARITY            "Use 32-bit parity"                                 OFF)
option(SBD_USE_THRUST_NOSYNC           "Thrust nosync mode"                                OFF)
option(SBD_SBDIAG_DEBUG                "SBD diagnostic debug output"                       OFF)
option(SBD_COMPLEX                     "Use complex<double> element type (-D_COMPLEX)"     OFF)

# ── Compiler diagnostics ─────────────────────────────────────────────────────
# nvc++ emits noisy warnings for unused variables/functions in template-heavy
# headers. Suppress globally.
# Note: CMAKE_CXX_COMPILER_ID is "NVHPC" (not "NVIDIA") for nvc++ in CMake
# 3.20+, but it is not populated during the main configure pass under Cray PE
# cross-compile mode (compiler check is skipped). PE_ENV=NVIDIA is the
# reliable signal; CMAKE_CXX_COMPILER matches nvc++ for standalone builds.
if(CMAKE_CXX_COMPILER MATCHES "nvc[+][+]" OR "$ENV{PE_ENV}" STREQUAL "NVIDIA")
  set(SBD_USING_NVHPC TRUE)
  add_compile_options(
    "--diag_suppress=declared_but_not_referenced,set_but_not_used,cuda_compile"
  )
endif()

# ── MPI / OpenMP / BLAS / LAPACK ─────────────────────────────────────────────
# With CMAKE_SYSTEM_NAME=CrayLinuxEnvironment, CMake configures search paths
# and compiler wrappers from the active PrgEnv-* module automatically.
find_package(MPI REQUIRED CXX)
find_package(OpenMP REQUIRED CXX)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ── SBD header-only interface target ─────────────────────────────────────────
add_library(sbd_headers INTERFACE)
target_include_directories(sbd_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# ── Diag-app helper ───────────────────────────────────────────────────────────
# Wire up libraries, feature defines, and GPU backend for a diag executable.
function(sbd_configure_diag target)
  target_link_libraries(${target} PRIVATE
    sbd_headers
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    BLAS::BLAS
    LAPACK::LAPACK
  )

  target_compile_definitions(${target} PRIVATE
    $<$<BOOL:${SBD_COMPLEX}>:_COMPLEX>
    $<$<BOOL:${SBD_TRADMODE}>:SBD_TRADMODE>
    $<$<BOOL:${SBD_USE_RANK_DISTRIBUTION}>:SBD_USE_RANK_DISTRIBUTION>
    $<$<BOOL:${SBD_USE_BLOCK_RANK_DISTRIBUTION}>:SBD_USE_BLOCK_RANK_DISTRIBUTION>
    $<$<BOOL:${SBD_REORDER_INDEX_ARRAY}>:SBD_REORDER_INDEX_ARRAY>
    $<$<BOOL:${SBD_USE_32BIT_PARITY}>:SBD_USE_32BIT_PARITY>
    $<$<BOOL:${SBD_SBDIAG_DEBUG}>:SBD_SBDIAG_DEBUG>
  )

  if(SBD_GPU_BACKEND STREQUAL "thrust")
    target_compile_definitions(${target} PRIVATE
      SBD_THRUST
      $<$<BOOL:${SBD_THRUST_SAFE_MPI_ALLREDUCE}>:SBD_THRUST_SAFE_MPI_ALLREDUCE>
      $<$<BOOL:${SBD_USE_NVTX}>:SBD_USE_NVTX>
      $<$<BOOL:${SBD_USE_NCCL}>:SBD_USE_NCCL>
      $<$<BOOL:${SBD_USE_CUBLAS}>:SBD_USE_CUBLAS>
      $<$<BOOL:${SBD_USE_THRUST_NOSYNC}>:SBD_USE_THRUST_NOSYNC>
    )
    target_compile_options(${target} PRIVATE
      -cuda
      "-gpu=${SBD_GPU_ARCH},mem:unified,interceptdeallocations"
    )
    target_link_options(${target} PRIVATE
      -cuda
      "-gpu=${SBD_GPU_ARCH},mem:unified,interceptdeallocations"
    )
  elseif(SBD_GPU_BACKEND STREQUAL "omp5")
    target_compile_definitions(${target} PRIVATE SBD_TRADMODE USE_OMP_OFFLOAD)
    target_compile_options(${target} PRIVATE -mp=gpu "-gpu=${SBD_GPU_ARCH}")
    target_link_options(${target}   PRIVATE -mp=gpu "-gpu=${SBD_GPU_ARCH}")
    # nvc++ requires -cuda to expose __ffsll in host context; __builtin_ffsl is
    # a GCC intrinsic absent from nvc++.  AMD offload uses its own builtins.
    if(SBD_USING_NVHPC)
      target_compile_definitions(${target} PRIVATE "__builtin_ffsl=__ffsll")
      target_compile_options(${target} PRIVATE -cuda)
      target_link_options(${target}   PRIVATE -cuda)
    endif()
  endif()
endfunction()

# ── Apps ──────────────────────────────────────────────────────────────────────
add_subdirectory(apps/caop_selected_basis_diagonalization)
add_subdirectory(apps/chemistry_gdb_selected_basis_diagonalization)
add_subdirectory(apps/chemistry_tpb_selected_basis_diagonalization)
add_subdirectory(apps/gen_dets)
