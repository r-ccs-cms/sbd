// hipshim, force-included before SBD's sources.
#pragma once
#include <thrust/detail/config.h>
// HIP's device pass doesn't define _OPENMP even with -fopenmp, so rocThrust marks its OpenMP backend as unusable
// there and SBD's host-only thrust::omp calls trip a static assertion. They are host code; say OpenMP is available.
#undef THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
#define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#include <thrust/system/hip/execution_policy.h>
// SBD writes thrust::cuda::par / par_nosync; rocThrust calls that namespace thrust::hip.
namespace thrust { namespace cuda = ::thrust::hip; }
