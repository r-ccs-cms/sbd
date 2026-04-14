/**
@file sbd/framework/thrust_kernels.h
@brief kernel classes for Thrust
*/
#ifndef SBD_FRAMEWORK_THRUST_KERNELS_H
#define SBD_FRAMEWORK_THRUST_KERNELS_H

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace sbd
{

// AXPY kernel
template <typename ElemT>
struct AXPY_kernel {
    ElemT a;

    AXPY_kernel(ElemT a_in) : a(a_in) {}

    __host__ __device__ ElemT operator()(const ElemT& x, const ElemT& y) const
    {
        return a * x + y;
    }
};

// AX kernel
template <typename ElemT>
struct AX_kernel {
    ElemT a;

    AX_kernel(ElemT a_in) : a(a_in) {}

    __host__ __device__ ElemT operator()(const ElemT& x) const
    {
        return a * x;
    }
};


// dot product
template <typename ElemT>
struct dot_product_kernel {
    ElemT* A;
    ElemT* B;

    dot_product_kernel(const thrust::device_vector<ElemT>& a, const thrust::device_vector<ElemT>& b)
    {
        A = (ElemT*)thrust::raw_pointer_cast(a.data());
        B = (ElemT*)thrust::raw_pointer_cast(b.data());
    }

    __host__ __device__ ElemT operator()(const size_t i) const
    {
        return A[i] * B[i];
    }
};

template <typename ElemT, typename RealT>
void Normalize(thrust::device_vector<ElemT>& X,
               RealT& res,
               MPI_Comm comm)
{
    res = 0.0;
    RealT sum = 0.0;

#ifdef SBD_NO_CUDA_REDUCE
    // Fallback: copy to host for reduction (avoids raw CUDA kernel issues on nvc++)
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    Normalize(hx, res, comm);
    thrust::copy_n(hx.begin(), hx.size(), X.begin());
    return;
#else
    auto kernel = dot_product_kernel<RealT>(X, X);
    sum = precise_reduce_sum_with_function(kernel, X.size());
#endif

    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    MPI_Allreduce(&sum, &res, 1, DataT, MPI_SUM, comm);
    res = std::sqrt(res);
    ElemT factor = ElemT(1.0 / res);

//    thrust::transform(thrust::device, X.begin(), X.end(), thrust::constant_iterator<ElemT>(factor), X.begin(), thrust::multiplies<ElemT>());
    thrust::transform(thrust::device, X.begin(), X.end(), X.begin(), AX_kernel<ElemT>(factor));
}

template <typename ElemT, typename RealT>
void InnerProduct(const thrust::device_vector<ElemT>& X,
                  const thrust::device_vector<ElemT>& Y,
                  RealT& res,
                  MPI_Comm comm)
{
    res = 0.0;
    RealT sum = 0.0;

#ifdef SBD_NO_CUDA_REDUCE
    // Fallback: copy to host for reduction (avoids raw CUDA kernel issues on nvc++)
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    std::vector<ElemT> hy(Y.size());
    thrust::copy_n(Y.begin(), Y.size(), hy.begin());
    InnerProduct(hx, hy, res, comm);
    return;
#else
    auto kernel = dot_product_kernel<RealT>(X, Y);
    sum = precise_reduce_sum_with_function(kernel, X.size());
#endif

    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    MPI_Allreduce(&sum, &res, 1, DataT, MPI_SUM, comm);
}

}

#endif