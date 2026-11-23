// hipshim: compile SBD's CUDA/Thrust code with hipcc on AMD GPUs without editing its sources.
// Maps the CUDA runtime names SBD uses onto HIP. Feasibility work, not upstream code.
#pragma once
#include <hip/hip_runtime.h>
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000   // take the pre-CUDA-13 cudaMemAdvise(ptr, n, advice, int device) branch
#endif
#define cudaError_t                     hipError_t
#define cudaSuccess                     hipSuccess
#define cudaErrorInvalidConfiguration   hipErrorInvalidConfiguration
#define cudaGetErrorString              hipGetErrorString
#define cudaGetLastError                hipGetLastError
#define cudaGetDevice                   hipGetDevice
#define cudaSetDevice                   hipSetDevice
#define cudaGetDeviceCount              hipGetDeviceCount
#define cudaDeviceSynchronize           hipDeviceSynchronize
#define cudaDeviceGetAttribute          hipDeviceGetAttribute
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrPageableMemoryAccess hipDeviceAttributePageableMemoryAccess
#define cudaCpuDeviceId                 hipCpuDeviceId
#define cudaMalloc                      hipMalloc
#define cudaFree                        hipFree
#define cudaMallocHost                  hipHostMalloc
#define cudaFreeHost                    hipHostFree
#define cudaMemGetInfo                  hipMemGetInfo
#define cudaMemcpy                      hipMemcpy
#define cudaMemcpyAsync                 hipMemcpyAsync
#define cudaMemcpy2DAsync               hipMemcpy2DAsync
#define cudaMemsetAsync                 hipMemsetAsync
#define cudaMemcpyDeviceToDevice        hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost          hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice          hipMemcpyHostToDevice
#define cudaMemcpyDefault               hipMemcpyDefault
#define cudaMemAdvise                   hipMemAdvise
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy      hipMemAdviseSetAccessedBy
#define cudaStream_t                    hipStream_t
#define cudaStreamCreateWithFlags       hipStreamCreateWithFlags
#define cudaStreamDestroy               hipStreamDestroy
#define cudaStreamSynchronize           hipStreamSynchronize
#define cudaStreamWaitEvent             hipStreamWaitEvent
#define cudaStreamNonBlocking           hipStreamNonBlocking
#define cudaEvent_t                     hipEvent_t
#define cudaEventCreate                 hipEventCreate
#define cudaEventCreateWithFlags        hipEventCreateWithFlags
#define cudaEventDisableTiming          hipEventDisableTiming
#define cudaEventRecord                 hipEventRecord
#define cudaEventSynchronize            hipEventSynchronize
#define cudaEventElapsedTime            hipEventElapsedTime
#define cudaEventDestroy                hipEventDestroy
#define cudaFuncSetAttribute(f, a, v)   hipFuncSetAttribute(reinterpret_cast<const void*>(f), a, v)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
