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

// atomic_add: type-safe atomic accumulation for device kernels.
// On device: decomposes complex into two double atomics.
// On host:   falls back to non-atomic += (host path is never used
//            for actual computation — only compiled for __host__ __device__
//            correctness).
namespace sbd {

template <typename T>
__host__ __device__ inline void atomic_add(T* p, T v)
{
#ifdef __CUDA_ARCH__
    atomicAdd(p, v);
#else
    *p += v;
#endif
}

template <>
__host__ __device__ inline void atomic_add(std::complex<double>* p, std::complex<double> v)
{
#ifdef __CUDA_ARCH__
    atomicAdd(reinterpret_cast<double*>(p),     v.real());
    atomicAdd(reinterpret_cast<double*>(p) + 1, v.imag());
#else
    *p += v;
#endif
}

// Overload: add a real value to the real part of a complex accumulator.
__host__ __device__ inline void atomic_add_real(std::complex<double>* p, double v)
{
#ifdef __CUDA_ARCH__
    atomicAdd(reinterpret_cast<double*>(p), v);
#else
    *p += std::complex<double>(v, 0.0);
#endif
}

// Overload for real types (no-op specialisation needed to avoid ambiguity).
__host__ __device__ inline void atomic_add_real(double* p, double v)
{
#ifdef __CUDA_ARCH__
    atomicAdd(p, v);
#else
    *p += v;
#endif
}

} // namespace sbd

#endif // SBD_FRAMEWORK_CUDA_UTILITY_H
