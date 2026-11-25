#pragma once
#include "cuda_runtime.h"
#include <hipblas/hipblas.h>

// The cuBLAS names SBD uses (SBD_USE_CUBLAS), mapped onto hipBLAS; argument order is the same.
#define cublasHandle_t                 hipblasHandle_t
#define cublasStatus_t                 hipblasStatus_t
#define cublasCreate                   hipblasCreate
#define cublasDestroy                  hipblasDestroy
#define cublasSetStream                hipblasSetStream
#define cublasSetPointerMode           hipblasSetPointerMode
#define cublasDgemv                    hipblasDgemv
#define cublasSgemv                    hipblasSgemv
#define cublasDscal                    hipblasDscal
#define cublasSscal                    hipblasSscal
#define cublasDdot                     hipblasDdot
#define cublasSdot                     hipblasSdot
#define CUBLAS_OP_N                    HIPBLAS_OP_N
#define CUBLAS_OP_T                    HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_HOST       HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_POINTER_MODE_DEVICE     HIPBLAS_POINTER_MODE_DEVICE
#define CUBLAS_STATUS_SUCCESS          HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED  HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED     HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE    HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH    HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR    HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR   HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED    HIPBLAS_STATUS_NOT_SUPPORTED
// CUBLAS_STATUS_LICENSE_ERROR has no hipBLAS counterpart; SBD only uses it under #ifdef.
