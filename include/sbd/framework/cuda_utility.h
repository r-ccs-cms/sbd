/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SBD_FRAMEWORK_CUDA_UTILITY_H
#define SBD_FRAMEWORK_CUDA_UTILITY_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

// -----------------------------------------------------------------------------
// CUDA error check
// -----------------------------------------------------------------------------
#ifndef SBD_CHECK_CUDA
#define SBD_CHECK_CUDA(cmd)                                              \
    do {                                                                 \
        cudaError_t e = (cmd);                                           \
        if (e != cudaSuccess) {                                          \
            std::fprintf(stderr,                                         \
                "CUDA error %s at %s:%d\n",                              \
                cudaGetErrorString(e), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)
#endif

// atomic_add: type-safe atomic accumulation for device kernels only.
namespace sbd {

template <typename T>
__device__ inline void atomic_add(T* p, T v)
{
    atomicAdd(p, v);
}

template <>
__device__ inline void atomic_add(std::complex<double>* p, std::complex<double> v)
{
    atomicAdd(reinterpret_cast<double*>(p),     v.real());
    atomicAdd(reinterpret_cast<double*>(p) + 1, v.imag());
}

// Overload: add a real value to the real part of a complex accumulator.
__device__ inline void atomic_add_real(std::complex<double>* p, double v)
{
    atomicAdd(reinterpret_cast<double*>(p), v);
}

// Overload for real types (no-op specialisation needed to avoid ambiguity).
__device__ inline void atomic_add_real(double* p, double v)
{
    atomicAdd(p, v);
}

} // namespace sbd

#endif // SBD_FRAMEWORK_CUDA_UTILITY_H
