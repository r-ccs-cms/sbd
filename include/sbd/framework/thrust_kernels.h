/**
@file sbd/framework/thrust_kernels.h
@brief kernel classes for Thrust
*/
#ifndef SBD_FRAMEWORK_THRUST_KERNELS_H
#define SBD_FRAMEWORK_THRUST_KERNELS_H

#include "sbd/framework/nvtx.h"
#include "sbd/framework/cuda_utility.h"

#ifdef SBD_USE_CUBLAS
#include "sbd/framework/cublas_utility.h"
#endif

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


// Binary element-wise kernel: applies f(X[i], Y[i]) for use with
// precise_reduce_sum_with_function.  The return type is deduced from F.
template <typename ElemT, typename F>
struct binary_kernel {
    const ElemT* X;
    const ElemT* Y;
    F f;

    binary_kernel(const thrust::device_vector<ElemT>& x,
                  const thrust::device_vector<ElemT>& y,
                  F f_in)
        : X(thrust::raw_pointer_cast(x.data()))
        , Y(thrust::raw_pointer_cast(y.data()))
        , f(f_in)
    {}

    __host__ __device__ auto operator()(const size_t i) const
    {
        return f(X[i], Y[i]);
    }
};

template <typename ElemT, typename F>
binary_kernel<ElemT, F> make_binary_kernel(const thrust::device_vector<ElemT>& x,
                                            const thrust::device_vector<ElemT>& y,
                                            F f)
{
    return binary_kernel<ElemT, F>(x, y, f);
}

// squared-norm kernel: returns |A[i]|^2 as double (works for real and complex)
template <typename ElemT>
struct norm_kernel {
    ElemT* A;

    norm_kernel(const thrust::device_vector<ElemT>& a)
    {
        A = (ElemT*)thrust::raw_pointer_cast(a.data());
    }

    __host__ __device__ double operator()(const size_t i) const
    {
        return SquaredNorm(A[i]);
    }
};


template <typename ElemT, typename RealT>
void Normalize(thrust::device_vector<ElemT>& X,
               RealT& res,
               MPI_Comm comm,
               int comm_size = -1,
               cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("Normalize", __LINE__);
    res = 0.0;
    if (X.empty()) return;
    if (comm_size < 0) {
        MPI_Comm_size(comm, &comm_size);
    }

    /*
    // If CUDA native kernel can not be used, use host code
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    Normalize(hx, res, comm);
    thrust::copy_n(hx.begin(), hx.size(), X.begin());
    */

#ifndef SBD_USE_CUBLAS
    auto kernel = norm_kernel<ElemT>(X);
    res = precise_reduce_sum_with_function(kernel, X.size());
    if (comm_size > 1) {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, DataT, MPI_SUM, comm);
    }
    res = std::sqrt(res);
    if (res == RealT(0)) {
        return;
    }
    ElemT factor = ElemT(1.0 / res);
    {
        SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
        // thrust::transform(thrust::device, X.begin(), X.end(), thrust::constant_iterator<ElemT>(factor), X.begin(), thrust::multiplies<ElemT>());
        thrust::transform(thrust::device, X.begin(), X.end(), X.begin(), AX_kernel<ElemT>(factor));
    }
#else
    static_assert(std::is_same<ElemT, RealT>::value,
                  "Normalize with cuBLAS currently requires ElemT == RealT");
    res = sbd::Dot(X, X);
    if (comm_size > 1) {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, DataT, MPI_SUM, comm);
    }
    res = std::sqrt(res);
    if (res == RealT(0)) {
        return;
    }
    ElemT factor = ElemT(1.0 / res);
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    const int n = static_cast<int>(X.size());
    ElemT* x_ptr = thrust::raw_pointer_cast(X.data());
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSscal(ctx.get(), n, &factor, x_ptr, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDscal(ctx.get(), n, &factor, x_ptr, 1));
    }
#endif
}

#ifdef SBD_USE_CUBLAS
template <typename ElemT, typename RealT>
void Normalize2(thrust::device_vector<ElemT>& X0,
                thrust::device_vector<ElemT>& X1,
                RealT& norm0,
                RealT& norm1,
                MPI_Comm comm,
                int comm_size = -1,
                ElemT* ws_ptr = nullptr,
                cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("Normalize2", __LINE__);
    static_assert(std::is_same<ElemT, RealT>::value,
                  "Normalize2 currently requires ElemT == RealT");
    if (comm_size < 0) {
        MPI_Comm_size(comm, &comm_size);
    }

    RealT norms[2]  = {RealT(0), RealT(0)};
    if (ws_ptr == nullptr) {
        if (!X0.empty()) {
            norms[0] = sbd::Dot(X0, X0, stream);
        }
        if (!X1.empty()) {
            norms[1] = sbd::Dot(X1, X1, stream);
        }
    } else {
        if (!X0.empty()) {
            sbd::Dot(X0, X0, ws_ptr, stream);
        }
        if (!X1.empty()) {
            sbd::Dot(X1, X1, ws_ptr+1, stream);
        }
        SBD_CHECK_CUDA(cudaMemcpyAsync(norms, ws_ptr, sizeof(ElemT) * 2,
                                       cudaMemcpyDeviceToHost, stream));
        SBD_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    if (comm_size > 1) {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(MPI_IN_PLACE, norms, 2, DataT, MPI_SUM, comm);
    }
    norm0 = std::sqrt(norms[0]);
    norm1 = std::sqrt(norms[1]);
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    if (norm0 != RealT(0) && !X0.empty()) {
        ElemT factor0 = ElemT(1.0 / norm0);
        const int n0 = static_cast<int>(X0.size());
        ElemT* x0_ptr = thrust::raw_pointer_cast(X0.data());
        if constexpr (std::is_same<ElemT, float>::value) {
            SBD_CHECK_CUBLAS(
                cublasSscal(ctx.get(), n0, &factor0, x0_ptr, 1));
        }
        else if constexpr (std::is_same<ElemT, double>::value) {
            SBD_CHECK_CUBLAS(
                cublasDscal(ctx.get(), n0, &factor0, x0_ptr, 1));
        }
        else {
            static_assert(!sizeof(ElemT), "Unsupported type for Normalize2");
        }
    }
    if (norm1 != RealT(0) && !X1.empty()) {
        ElemT factor1 = ElemT(1.0 / norm1);
        const int n1 = static_cast<int>(X1.size());
        ElemT* x1_ptr = thrust::raw_pointer_cast(X1.data());

        if constexpr (std::is_same<ElemT, float>::value) {
            SBD_CHECK_CUBLAS(
                cublasSscal(ctx.get(), n1, &factor1, x1_ptr, 1));
        }
        else if constexpr (std::is_same<ElemT, double>::value) {
            SBD_CHECK_CUBLAS(
                cublasDscal(ctx.get(), n1, &factor1, x1_ptr, 1));
        }
    }
}
#endif

template <typename ElemT>
void InnerProduct(const thrust::device_vector<ElemT>& X,
                  const thrust::device_vector<ElemT>& Y,
                  ElemT& res,
                  MPI_Comm comm)
{
    using RealT = typename GetRealType<ElemT>::RealT;
    SBD_NVTX_RANGE_COLOR("InnerProduct", __LINE__);

    res = ElemT(0);
    ElemT sum = ElemT(0);

    /*
    // If CUDA native kernel can not be used, use host code
    std::vector<ElemT> hx(X.size());
    thrust::copy_n(X.begin(), X.size(), hx.begin());
    std::vector<ElemT> hy(Y.size());
    thrust::copy_n(Y.begin(), Y.size(), hy.begin());
    InnerProduct(hx, hy, res, comm);
    */

#ifndef SBD_USE_CUBLAS
    if constexpr (std::is_floating_point_v<ElemT>) {
        sum = precise_reduce_sum_with_function(
            make_binary_kernel(X, Y,
                [] __host__ __device__ (const ElemT& x, const ElemT& y) { return x * y; }),
            X.size());
    } else {
        // conj(X)*Y = (xr*yr + xi*yi) + (xr*yi - xi*yr)*i
        // Each component reduced independently (Kahan over one product/element).
        double sum_rr = precise_reduce_sum_with_function(
            make_binary_kernel(X, Y,
                [] __host__ __device__ (const ElemT& x, const ElemT& y) { return x.real() * y.real(); }),
            X.size());
        double sum_ii = precise_reduce_sum_with_function(
            make_binary_kernel(X, Y,
                [] __host__ __device__ (const ElemT& x, const ElemT& y) { return x.imag() * y.imag(); }),
            X.size());
        double sum_ri = precise_reduce_sum_with_function(
            make_binary_kernel(X, Y,
                [] __host__ __device__ (const ElemT& x, const ElemT& y) { return x.real() * y.imag(); }),
            X.size());
        double sum_ir = precise_reduce_sum_with_function(
            make_binary_kernel(X, Y,
                [] __host__ __device__ (const ElemT& x, const ElemT& y) { return x.imag() * y.real(); }),
            X.size());
        sum = ElemT(sum_rr + sum_ii, sum_ri - sum_ir);
    }
#else
    sum = sbd::Dot(X, Y);
#endif
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        MPI_Allreduce(&sum, &res, 1, DataT, MPI_SUM, comm);
    }
}

template <typename ElemT, typename RealT>
void BatchedInnerProduct(std::vector<thrust::device_vector<ElemT>>& X,
                         int num_X,
                         const thrust::device_vector<ElemT>& Y,
                         std::vector<RealT>& Z,
                         MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("BatchedInnerProduct", __LINE__);
    if (num_X == 0) return;
    assert(num_X <= X.size());
    assert(num_X <= Z.size());
#ifndef SBD_USE_CUBLAS
    for (int i = 0; i < num_X; i++) {
        InnerProduct(X[i], Y, Z[i], comm);
    }
#else // #ifndef SBD_USE_CUBLAS
    for (int i = 0; i < num_X; i++) {
        res[i] = sbd::Dot(X[i], Y);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
        MPI_Allreduce(res.data(), Z.data(), num_X, DataT, MPI_SUM, comm);
    }
#endif // #ifndef SBD_USE_CUBLAS
}

#ifdef SBD_USE_CUBLAS

template <typename ElemT, typename RealT>
void BatchedInnerProduct_GEMV(ElemT* X_flat_ptr,
                              int num_X,
                              const thrust::device_vector<ElemT>& Y,
                              std::vector<RealT>& Z,
                              MPI_Comm comm,
                              ElemT* ws_ptr = nullptr,
                              cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedInnerProduct_GEMV", __LINE__);
    if (num_X == 0) return;

    static_assert(std::is_same<ElemT, RealT>::value, "ElemT must match RealT");
    assert(num_X <= Z.size());

    const size_t N = Y.size();
    thrust::device_vector<ElemT> Z_dev;
    ElemT *Z_dev_ptr;
    if (ws_ptr == nullptr) {
        Z_dev.resize(num_X);
        Z_dev_ptr = thrust::raw_pointer_cast(Z_dev.data());
    } else {
        Z_dev_ptr = ws_ptr;
    }
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    const ElemT alpha = static_cast<ElemT>(1.0);
    const ElemT beta  = static_cast<ElemT>(0.0);
    const ElemT* A = X_flat_ptr;
    const ElemT* x = thrust::raw_pointer_cast(Y.data());
    ElemT* y       = Z_dev_ptr;
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(),CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1));
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for GEMV");
    }
    std::vector<ElemT> res(num_X);
    cudaMemcpyAsync(res.data(), Z_dev_ptr, sizeof(ElemT) * num_X,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        MPI_Allreduce(res.data(), Z.data(), num_X, DataT, MPI_SUM, comm);
    }
}

template <typename ElemT, typename RealT>
void BatchedInnerProduct_GEMV(std::vector<thrust::device_vector<ElemT>>& X,
                              int num_X,
                              const thrust::device_vector<ElemT>& Y,
                              std::vector<RealT>& Z,
                              MPI_Comm comm,
                              ElemT* ws_ptr = nullptr,
                              cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedInnerProduct_GEMV", __LINE__);
    if (num_X == 0) return;

    static_assert(std::is_same<ElemT, RealT>::value, "ElemT must match RealT");
    assert(num_X <= X.size());
    assert(num_X <= Z.size());

    const size_t N = Y.size();
    thrust::device_vector<ElemT> X_flat;
    thrust::device_vector<ElemT> Z_dev;
    ElemT *X_flat_ptr;
    ElemT *Z_dev_ptr;
    if (ws_ptr == nullptr) {
        X_flat.resize(num_X * N);
        Z_dev.resize(num_X);
        X_flat_ptr = thrust::raw_pointer_cast(X_flat.data());
        Z_dev_ptr = thrust::raw_pointer_cast(Z_dev.data());
    } else {
        X_flat_ptr = ws_ptr;
        Z_dev_ptr = ws_ptr + (N * num_X);
    }
    for (int i = 0; i < num_X; i++) {
        assert(X[i].size() == N);
        cudaMemcpyAsync(
            X_flat_ptr + (N * i),
            thrust::raw_pointer_cast(X[i].data()),
            sizeof(ElemT) * N,
            cudaMemcpyDeviceToDevice, stream);
    }
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    const ElemT alpha = static_cast<ElemT>(1.0);
    const ElemT beta  = static_cast<ElemT>(0.0);
    const ElemT* A = X_flat_ptr;
    const ElemT* x = thrust::raw_pointer_cast(Y.data());
    ElemT* y       = Z_dev_ptr;
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(),CUBLAS_OP_T, N, num_X,
                        &alpha, A, N, x, 1, &beta, y, 1));
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for GEMV");
    }
    std::vector<ElemT> res(num_X);
    cudaMemcpyAsync(res.data(), Z_dev_ptr, sizeof(ElemT) * num_X,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        MPI_Allreduce(res.data(), Z.data(), num_X, DataT, MPI_SUM, comm);
    }
}

// Internal implementation of batched AXPY via GEMV:
// computes Y = X * alpha + beta * Y after packing X into a column-major matrix.
// This routine only enqueues work onto the given stream; synchronization is
// handled by the caller.
template <typename ElemT>
void BatchedAXPY_GEMV_impl(const ElemT* X_flat_ptr,
                           size_t num_X,
                           const ElemT* alpha_dev_ptr,
                           thrust::device_vector<ElemT>& Y,
                           ElemT beta,
                           cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedAXPY_GEMV_impl", __LINE__);
    if (num_X == 0) return;
    const size_t N = Y.size();
    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);
    const int n = static_cast<int>(N);
    const int m = static_cast<int>(num_X);
    const ElemT one = static_cast<ElemT>(1.0);
    const ElemT* A = X_flat_ptr;
    ElemT* y = thrust::raw_pointer_cast(Y.data());
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &one, A, n, alpha_dev_ptr, 1,
                        &beta, y, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &one, A, n, alpha_dev_ptr, 1,
                        &beta, y, 1));
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for BatchedAXPY_GEMV");
    }
}

template <typename ElemT>
void BatchedAXPY_GEMV_impl(const std::vector<thrust::device_vector<ElemT>>& X,
                           size_t num_X,
                           const ElemT* alpha_dev_ptr,
                           thrust::device_vector<ElemT>& Y,
                           ElemT beta,
                           ElemT* ws_ptr = nullptr,
                           cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedAXPY_GEMV_impl", __LINE__);
    if (num_X == 0) return;
    assert(num_X <= X.size());
    const size_t N = Y.size();
    thrust::device_vector<ElemT> X_flat_local;
    ElemT* X_flat_ptr = nullptr;
    if (ws_ptr == nullptr) {
        X_flat_local.resize(num_X * N);
        X_flat_ptr = thrust::raw_pointer_cast(X_flat_local.data());
    } else {
        X_flat_ptr = ws_ptr;
    }
    for (size_t i = 0; i < num_X; i++) {
        assert(X[i].size() == N);
        SBD_CHECK_CUDA(cudaMemcpyAsync(
                           X_flat_ptr + (N * i), thrust::raw_pointer_cast(X[i].data()),
                           sizeof(ElemT) * N, cudaMemcpyDeviceToDevice, stream));
    }
    BatchedAXPY_GEMV_impl(X_flat_ptr, num_X, alpha_dev_ptr, Y, beta, stream);
}

// When ws_ptr is provided, it must have space for both the packed matrix
// buffer (Y.size() * num_X elements).
template <typename ElemT>
void BatchedAXPY_GEMV(const std::vector<thrust::device_vector<ElemT>>& X,
                      size_t num_X,
                      const thrust::device_vector<ElemT>& alpha_dev,
                      thrust::device_vector<ElemT>& Y,
                      ElemT beta,
                      ElemT* ws_ptr = nullptr,
                      cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedAXPY_GEMV", __LINE__);
    assert(num_X <= alpha_dev.size());
    ElemT *alpha_dev_ptr = thrust::raw_pointer_cast(alpha_dev.data());
    BatchedAXPY_GEMV_impl(X, num_X, alpha_dev_ptr, Y, beta, ws_ptr, stream);
}

// When ws_ptr is provided, it must have space for both the packed matrix
// buffer (Y.size() * num_X elements) and the alpha vector (num_X elements).
template <typename ElemT>
void BatchedAXPY_GEMV(const std::vector<thrust::device_vector<ElemT>>& X,
                      size_t num_X,
                      const std::vector<ElemT>& alpha_host,
                      thrust::device_vector<ElemT>& Y,
                      ElemT beta,
                      ElemT* ws_ptr = nullptr,
                      cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedAXPY_GEMV", __LINE__);
    assert(num_X <= alpha_host.size());
    ElemT *alpha_dev_ptr = nullptr;
    thrust::device_vector<ElemT> alpha_dev;
    if (ws_ptr == nullptr) {
        alpha_dev.resize(num_X);
        alpha_dev_ptr = thrust::raw_pointer_cast(alpha_dev.data());
    } else {
        alpha_dev_ptr = ws_ptr + (Y.size() * num_X);
    }
    SBD_CHECK_CUDA(cudaMemcpyAsync(alpha_dev_ptr, alpha_host.data(),
                                   sizeof(ElemT) * num_X, cudaMemcpyHostToDevice, stream));
    BatchedAXPY_GEMV_impl(X, num_X, alpha_dev_ptr, Y, beta, ws_ptr, stream);
}

template <typename ElemT>
void BatchedAXPY_GEMV(const ElemT* X_flat_ptr,
                      size_t num_X,
                      const std::vector<ElemT>& alpha_host,
                      thrust::device_vector<ElemT>& Y,
                      ElemT beta,
                      ElemT* ws_ptr = nullptr,
                      cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("BatchedAXPY_GEMV", __LINE__);
    assert(num_X <= alpha_host.size());
    ElemT *alpha_dev_ptr = nullptr;
    thrust::device_vector<ElemT> alpha_dev;
    if (ws_ptr == nullptr) {
        alpha_dev.resize(num_X);
        alpha_dev_ptr = thrust::raw_pointer_cast(alpha_dev.data());
    } else {
        alpha_dev_ptr = ws_ptr;
    }
    SBD_CHECK_CUDA(cudaMemcpyAsync(alpha_dev_ptr, alpha_host.data(),
                                   sizeof(ElemT) * num_X, cudaMemcpyHostToDevice, stream));
    BatchedAXPY_GEMV_impl(X_flat_ptr, num_X, alpha_dev_ptr, Y, beta, stream);
}

template <typename ElemT>
void GramSchmidtOrthogonalize_GEMV(const ElemT* X_flat_ptr,
                                   size_t num_X,
                                   thrust::device_vector<ElemT>& Y,
                                   MPI_Comm comm,
                                   int comm_size,
                                   ElemT* ws_ptr = nullptr,
                                   cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("GramSchmidtOrthogonalize_GEMV", __LINE__);
    if (num_X == 0) return;

    const size_t N = Y.size();

    thrust::device_vector<ElemT> res_dev_local;
    ElemT* res_dev_ptr;
    if (ws_ptr == nullptr) {
        res_dev_local.resize(num_X);
        res_dev_ptr = thrust::raw_pointer_cast(res_dev_local.data());
    } else {
        res_dev_ptr = ws_ptr;
    }

    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);

    const ElemT one = static_cast<ElemT>(1.0);
    const ElemT zero = static_cast<ElemT>(0.0);
    const ElemT minus_one = static_cast<ElemT>(-1.0);

    const ElemT* A = X_flat_ptr;
    const ElemT* y_in = thrust::raw_pointer_cast(Y.data());
    ElemT* y_out = thrust::raw_pointer_cast(Y.data());

    const int n = static_cast<int>(N);
    const int m = static_cast<int>(num_X);

    // 1. local overlaps: res_dev = A^T * Y
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_T, n, m,
                        &one, A, n, y_in, 1,
                        &zero, res_dev_ptr, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(), CUBLAS_OP_T, n, m,
                        &one, A, n, y_in, 1,
                        &zero, res_dev_ptr, 1));
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for GramSchmidtOrthogonalize_GEMV");
    }

    // 2. copy overlaps to host and allreduce
    std::vector<ElemT> res_host(num_X);
    SBD_CHECK_CUDA(cudaMemcpyAsync(
                       res_host.data(), res_dev_ptr, sizeof(ElemT) * num_X,
                       cudaMemcpyDeviceToHost, stream));
    SBD_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (comm_size > 1) {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        MPI_Allreduce(MPI_IN_PLACE, res_host.data(), static_cast<int>(num_X), DataT, MPI_SUM, comm);
    }

    // 3. copy back to device
    SBD_CHECK_CUDA(cudaMemcpyAsync(
                       res_dev_ptr, res_host.data(), sizeof(ElemT) * num_X,
                       cudaMemcpyHostToDevice, stream));

    // 4. Y = Y + A * res
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &minus_one, A, n, res_dev_ptr, 1,
                        &one, y_out, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &minus_one, A, n, res_dev_ptr, 1,
                        &one, y_out, 1));
    }
}

#ifdef SBD_USE_NCCL
template <typename ElemT>
void GramSchmidtOrthogonalize_GEMV(const ElemT* X_flat_ptr,
                                   size_t num_X,
                                   thrust::device_vector<ElemT>& Y,
                                   ncclComm_t nccl_comm,
                                   int comm_size,
                                   ElemT* ws_ptr = nullptr,
                                   cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("GramSchmidtOrthogonalize_GEMV", __LINE__);
    if (num_X == 0) return;

    const size_t N = Y.size();

    thrust::device_vector<ElemT> res_dev_local;
    ElemT* res_dev_ptr;
    if (ws_ptr == nullptr) {
        res_dev_local.resize(num_X);
        res_dev_ptr = thrust::raw_pointer_cast(res_dev_local.data());
    } else {
        res_dev_ptr = ws_ptr;
    }

    auto& ctx = sbd::GetCublasContext();
    ctx.set_pointer_mode_host();
    ctx.set_stream(stream);

    const ElemT one = static_cast<ElemT>(1.0);
    const ElemT zero = static_cast<ElemT>(0.0);
    const ElemT minus_one = static_cast<ElemT>(-1.0);

    const ElemT* A = X_flat_ptr;
    const ElemT* y_in = thrust::raw_pointer_cast(Y.data());
    ElemT* y_out = thrust::raw_pointer_cast(Y.data());

    const int n = static_cast<int>(N);
    const int m = static_cast<int>(num_X);

    // 1. local overlaps: res_dev = A^T * Y
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_T, n, m,
                        &one, A, n, y_in, 1,
                        &zero, res_dev_ptr, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(), CUBLAS_OP_T, n, m,
                        &one, A, n, y_in, 1,
                        &zero, res_dev_ptr, 1));
    }
    else {
        static_assert(!sizeof(ElemT), "Unsupported type for GramSchmidtOrthogonalize_GEMV");
    }

    // 2. Reduce overlaps
    if (comm_size > 1) {
        nccl_allreduce(res_dev_ptr, num_X, ncclSum, nccl_comm, stream);
    }

    // 3. Y = Y + A * res
    if constexpr (std::is_same<ElemT, float>::value) {
        SBD_CHECK_CUBLAS(
            cublasSgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &minus_one, A, n, res_dev_ptr, 1,
                        &one, y_out, 1));
    }
    else if constexpr (std::is_same<ElemT, double>::value) {
        SBD_CHECK_CUBLAS(
            cublasDgemv(ctx.get(), CUBLAS_OP_N, n, m,
                        &minus_one, A, n, res_dev_ptr, 1,
                        &one, y_out, 1));
    }
}
#endif  // #ifdef SBD_USE_NCCL

#endif

}

#endif
