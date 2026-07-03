/**
@file sbd/chemistry/basic/davidson.h
@brief davidson for parallel task management for distributed basis
*/
#ifndef SBD_CHEMISTRY_DAVIDSON_THRUST_H
#define SBD_CHEMISTRY_DAVIDSON_THRUST_H

#ifdef SBD_USE_NCCL
#include <nccl.h>
#endif

#include "sbd/framework/nvtx.h"
#include "sbd/framework/cuda_utility.h"

#include "sbd/framework/jacobi.h"
#ifdef __CUDACC__
#include "sbd/framework/cuda_reduce.h"
#else
#include "sbd/framework/hip_reduce.h"
#endif

#include "sbd/framework/thrust_kernels.h"

namespace sbd
{

template <typename ElemT, typename RealT>
struct Determine_kernel {
    RealT eps_reg;
    RealT e0;
    ElemT* C;
    ElemT* R;
    ElemT* dii;
    Determine_kernel(thrust::device_vector<ElemT>& VC, thrust::device_vector<ElemT>& VR, thrust::device_vector<ElemT>& Vd, RealT eps, RealT e) : eps_reg(eps), e0(e)
    {
        C = (ElemT*)thrust::raw_pointer_cast(VC.data());
        R = (ElemT*)thrust::raw_pointer_cast(VR.data());
        dii = (ElemT*)thrust::raw_pointer_cast(Vd.data());
    }
    __host__ __device__ void operator()(int is)
    {
        // Use squared comparison to avoid std::abs(complex) which calls hypot
        // (not translatable to device code by nvc++).
        auto denom = e0 - dii[is];
        if (SquaredNorm(denom) > eps_reg * eps_reg) {
            C[is] = R[is] / denom;
        } else {
            C[is] = R[is] / (denom - eps_reg);
        }
    }
};


template <typename ElemT>
void GetTotalD_Thrust(const thrust::device_vector<ElemT> & hii,
        thrust::device_vector<ElemT>& dii,
        MPI_Comm h_comm) {
    SBD_NVTX_RANGE_COLOR("GetTotalD_Thrust", __LINE__);
    int size_d = hii.size();
    dii.resize(hii.size());
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce((ElemT*)thrust::raw_pointer_cast(hii.data()), (ElemT*)thrust::raw_pointer_cast(dii.data()), size_d, DataT, MPI_SUM, h_comm);
    }
}

/**
     Davidson method for the direct multiplication using TaskHelpers, specialized for the SQD loop calculation.
    @tparam ElemT: Type of the Hamiltonian and wave functions
    @tparam RealT: Real type of ElemT
    @param[in] hii: Diagonal elements for the Hamiltonian matrix
    @param[in/out] W: Initialized wave function in input. Obtained ground state at output.
    @param[in] data: device storage for adets, bdets, helper, I0, I1 and I2
    @param[in] adet_comm_size: number of nodes used to split the alpha-dets
    @param[in] bdet_comm_size: number of nodes used to split the beta-dets
    @param[in] h_comm: Communicator used to cyclicly split the row-index when performing the multiplication of Hamiltonian
    @param[in] b_comm: Communicator to split the wave vector
    @param[in] t_comm: Communicator to split the tasks in column-index when performing the multiplication
    @param[in] max_iteration: Number of maximum interation of the Davidson iteration
    @param[in] num_block: Maximum size of Litz vector space
    @param[in] eps: error torelance (norm of the residual vector)
    @param[in] max_time: Maximum time allowed to perform the calculation
    */

#ifndef SBD_USE_CUBLAS

template <typename ElemT, typename RealT>
void Davidson(const thrust::device_vector<ElemT> &hii,
                std::vector<ElemT> &W,
                MultBase<ElemT>& mult,
                int max_iteration,
                int num_block,
                RealT eps,
                RealT max_time)
{
    SBD_NVTX_RANGE_COLOR("Davidson", __LINE__);
    RealT eps_reg = 1.0e-12;

    std::vector<thrust::device_vector<ElemT>> C(num_block);
    std::vector<thrust::device_vector<ElemT>> HC(num_block);
    for (int i = 0; i < num_block; i++) {
        SBD_NVTX_RANGE_COLOR("for (int i ...", __LINE__ + i);
        C[i].resize(W.size());
        HC[i].resize(W.size());
    }
    thrust::device_vector<ElemT> R(W.size());
    thrust::device_vector<ElemT> dii;
    int mpi_rank_h;
    MPI_Comm_rank(mult.h_comm(), &mpi_rank_h);
    int mpi_size_h;
    MPI_Comm_size(mult.h_comm(), &mpi_size_h);
    int mpi_rank_b;
    MPI_Comm_rank(mult.b_comm(), &mpi_rank_b);
    int mpi_size_b;
    MPI_Comm_size(mult.b_comm(), &mpi_size_b);
    int mpi_rank_t;
    MPI_Comm_rank(mult.t_comm(), &mpi_rank_t);
    int mpi_size_t;
    MPI_Comm_size(mult.t_comm(), &mpi_size_t);
    int mpi_rank_a;
    MPI_Comm_rank(mult.a_comm(), &mpi_rank_a);
    int mpi_size_a;
    MPI_Comm_size(mult.a_comm(), &mpi_size_a);

    ElemT *H = (ElemT *)calloc(num_block * num_block, sizeof(ElemT));
    ElemT *U = (ElemT *)calloc(num_block * num_block, sizeof(ElemT));
    RealT *E = (RealT *)malloc(num_block * sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD_Thrust(hii, dii, mult.h_comm());

#ifdef SBD_DEBUG_DAVIDSON
    std::cout << " diagonal term at mpi process (h,b,t) = ("
                << mpi_rank_h << "," << mpi_rank_b << ","
                << mpi_rank_t << "): ";
    for (size_t id = 0; id < std::min(W.size(), static_cast<size_t>(6)); id++)
    {
        std::cout << " " << dii[id];
    }
    std::cout << std::endl;
#endif

    bool do_continue = true;

    std::vector<double> onestep_times(num_block * max_iteration, 0.0);
    auto start_time = std::chrono::high_resolution_clock::now();

    cudaStream_t stream = 0;
    auto policy_nosync = thrust::cuda::par_nosync.on(stream);
    auto policy_sync = thrust::cuda::par.on(stream);
    // auto policy_sync = thrust::device;
#ifdef SBD_USE_THRUST_NOSYNC
    auto policy = policy_nosync;
#else
    auto policy = policy_sync;
#endif

    // copyin W
    thrust::device_vector<ElemT> W_dev(W.size());
    thrust::copy_n(W.begin(), W.size(), W_dev.begin());

    for (int it = 0; it < max_iteration; it++) {
        SBD_NVTX_RANGE_COLOR("for (int it ...", __LINE__ + it);
        C[0] = W_dev;

        for (int ib = 0; ib < nb; ib++) {
            SBD_NVTX_RANGE_COLOR("for (int ib ...", __LINE__ + ib);
            auto step_start = std::chrono::high_resolution_clock::now();

            //Zero(HC[ib]);
            thrust::fill(HC[ib].begin(), HC[ib].end(), 0);

            {
                SBD_NVTX_RANGE_COLOR("mult.run", __LINE__);
                mult.run(hii, C[ib], HC[ib]);
            }

            for (int jb = 0; jb <= ib; jb++) {
                InnerProduct(C[jb], HC[ib], H[jb + nb * ib], mult.b_comm());
                H[ib + nb * jb] = Conjugate(H[jb + nb * ib]);
            }
            for (int jb = 0; jb <= ib; jb++) {
                for (int kb = 0; kb <= ib; kb++) {
                    U[jb + nb * kb] = H[jb + nb * kb];
                }
            }

#ifdef SBD_NO_LAPACK
            hp_numeric::JacobiHeev(ib + 1, U, nb, E);
#else
            hp_numeric::MatHeev(jobz, uplo, ib + 1, U, nb, E);
#endif

            {
                SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                // ElemT x = U[0];
                // W[is] = C[0][is] * x;
                // thrust::transform(thrust::device, C[0].begin(), C[0].end(), thrust::constant_iterator<ElemT>(U[0]), W_dev.begin(), thrust::multiplies<ElemT>());
                thrust::transform(policy, C[0].begin(), C[0].end(), W_dev.begin(), AX_kernel<ElemT>(U[0]));
            }
            {
                SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                // x = ElemT(-1.0) * U[0];
                // R[is] = HC[0][is] * x;
                // thrust::transform(thrust::device, HC[0].begin(), HC[0].end(), thrust::constant_iterator<ElemT>(-U[0]), R.begin(), thrust::multiplies<ElemT>());
                thrust::transform(policy, HC[0].begin(), HC[0].end(), R.begin(), AX_kernel<ElemT>(-U[0]));
            }
            for (int kb = 1; kb <= ib; kb++) {
                {
                    SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                    // x = U[kb];
                    // W[is] += C[kb][is] * x;
                    thrust::transform(policy, C[kb].begin(), C[kb].end(), W_dev.begin(), W_dev.begin(), AXPY_kernel<ElemT>(U[kb]));
                }
                {
                    SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                    // x = ElemT(-1.0) * U[kb];
                    // R[is] += HC[kb][is] * x;
                    thrust::transform(policy, HC[kb].begin(), HC[kb].end(), R.begin(), R.begin(), AXPY_kernel<ElemT>(-U[kb]));
                }
            }
            {
                SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                // R[is] += E[0] * W[is];
                thrust::transform(policy, W_dev.begin(), W_dev.end(), R.begin(), R.begin(), AXPY_kernel<ElemT>(E[0]));
            }
#ifdef SBD_USE_THRUST_NOSYNC
            cudaStreamSynchronize(stream);
#endif

            /**
                 Patch for stability on Fugaku
                */
            // #ifdef SBD_FUAGKUPATCH
#ifdef SBD_USE_NCCL
            if (mpi_size_a > 1) {
                nccl_allreduce(W_dev, ncclSum, mult.a_nccl_comm());
                nccl_allreduce(R, ncclSum, mult.a_nccl_comm());
            }
#else
            if (mpi_size_t > 1) {
                MpiAllreduce(W_dev, MPI_SUM, mult.t_comm());
                MpiAllreduce(R, MPI_SUM, mult.t_comm());
            }
            if (mpi_size_h > 1) {
                MpiAllreduce(W_dev, MPI_SUM, mult.h_comm());
                MpiAllreduce(R, MPI_SUM, mult.h_comm());
            }
#endif
            if (mpi_size_h * mpi_size_t > 1) {
                ElemT volp(1.0 / (mpi_size_h * mpi_size_t));
                {
                    SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                    // W[is] *= volp;
                    // thrust::transform(thrust::device, W_dev.begin(), W_dev.end(), thrust::constant_iterator<ElemT>(volp), W_dev.begin(), thrust::multiplies<ElemT>());
                    thrust::transform(policy, W_dev.begin(), W_dev.end(), W_dev.begin(), AX_kernel<ElemT>(volp));
                }
                {
                    SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                    // R[is] *= volp;
                    // thrust::transform(thrust::device, R.begin(), R.end(), thrust::constant_iterator<ElemT>(volp), R.begin(), thrust::multiplies<ElemT>());
                    thrust::transform(policy, R.begin(), R.end(), R.begin(), AX_kernel<ElemT>(volp));
                }
#ifdef SBD_USE_THRUST_NOSYNC
                cudaStreamSynchronize(stream);
#endif
            }
            // #endif

            RealT norm_W;
            RealT norm_R;
            Normalize(W_dev, norm_W, mult.b_comm(), mpi_size_b);
            Normalize(R, norm_R, mult.b_comm(), mpi_size_b);
            // std::cout << "  norm_W = " << norm_W << " , norm_R = " << norm_R << std::endl;

#ifdef SBD_DEBUG_DAVIDSON
            std::cout << " Davidson iteration " << it << "." << ib
                        << " at mpi (h,b,t) = ("
                        << mpi_rank_h << "," << mpi_rank_b << ","
                        << mpi_rank_t << "): (tol=" << norm_R << "):";
            for (int p = 0; p < std::min(ib + 1, 4); p++)
            {
                std::cout << " " << E[p];
            }
            std::cout << std::endl;
#else
            if (mpi_rank_h == 0) {
                if (mpi_rank_t == 0) {
                    if (mpi_rank_b == 0) {
                        std::cout << " Davidson iteration " << it << "." << ib
                                    << " (tol=" << norm_R << "):";
                        for (int p = 0; p < std::min(ib + 1, 4); p++) {
                            std::cout << " " << E[p];
                        }
                        std::cout << std::endl;
                    }
                }
            }
#endif
            if (norm_R < eps) {
                do_continue = false;
                break;
            }

            if (ib < nb - 1) {
                // Determine
                auto ci = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(policy, ci, W.size(), Determine_kernel(C[ib + 1], R, dii, eps_reg, E[0]));
                }

                // Gram-Schmidt orthogonalization
                for (int kb = 0; kb < ib + 1; kb++) {
                    ElemT olap;
                    InnerProduct(C[kb], C[ib+1], olap, mult.b_comm());
                    olap *= ElemT(-1.0);
                    {
                        SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                        thrust::transform(policy, C[kb].begin(), C[kb].end(), C[ib + 1].begin(), C[ib + 1].begin(), AXPY_kernel<ElemT>(olap));
                    }
                }
#ifdef SBD_USE_THRUST_NOSYNC
                cudaStreamSynchronize(stream);
#endif

                RealT norm_C;
                Normalize(C[ib + 1], norm_C, mult.b_comm(), mpi_size_b);
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            onestep_times[it * nb + ib] = std::chrono::duration<double>(step_end - step_start).count();
            double ave_time_per_step = 0.0;
            for (int ks = 0; ks <= it * nb + ib; ks++) {
                ave_time_per_step += onestep_times[ks];
            }
            ave_time_per_step /= (it * nb + ib + 1);

            auto current_time = std::chrono::high_resolution_clock::now();
            double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();
            double predicted_next_end = total_elapsed + ave_time_per_step;
            if (mpi_rank_h == 0) {
                if (mpi_rank_t == 0) {
                    MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.b_comm());
                }
                MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.t_comm());
            }
            MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.h_comm());

            if (predicted_next_end > max_time) {
                do_continue = false;
                break;
            }

        } // end for(int ib=0; ib < nb; ib++)

        if (!do_continue) {
            break;
        }

        // Restart with C[0] = W;
        // C[0] = W_dev;
    } // end for(int it=0; it < max_iteration; it++)

    // copyout W
    thrust::copy_n(W_dev.begin(), W.size(), W.begin());

    free(H);
    free(U);
    free(E);
}

#else  // #ifndef SBD_USE_CUBLAS

template <typename ElemT, typename RealT>
void Davidson(const thrust::device_vector<ElemT> &hii,
                std::vector<ElemT> &W,
                MultBase<ElemT>& mult,
                int max_iteration,
                int num_block,
                RealT eps,
                RealT max_time)
{
    SBD_NVTX_RANGE_COLOR("Davidson", __LINE__);
    RealT eps_reg = 1.0e-12;
    ElemT *C, *HC;
    SBD_CHECK_CUDA(cudaMalloc(&C, sizeof(ElemT)*num_block*W.size()));
    SBD_CHECK_CUDA(cudaMalloc(&HC, sizeof(ElemT)*num_block*W.size()));
    thrust::device_vector<ElemT> C_tmp(W.size());
    thrust::device_vector<ElemT> HC_tmp(W.size());
    thrust::device_vector<ElemT> R(W.size());
    thrust::device_vector<ElemT> dii;
    int mpi_rank_h;
    MPI_Comm_rank(mult.h_comm(), &mpi_rank_h);
    int mpi_size_h;
    MPI_Comm_size(mult.h_comm(), &mpi_size_h);
    int mpi_rank_b;
    MPI_Comm_rank(mult.b_comm(), &mpi_rank_b);
    int mpi_size_b;
    MPI_Comm_size(mult.b_comm(), &mpi_size_b);
    int mpi_rank_t;
    MPI_Comm_rank(mult.t_comm(), &mpi_rank_t);
    int mpi_size_t;
    MPI_Comm_size(mult.t_comm(), &mpi_size_t);
    int mpi_rank_a;
    MPI_Comm_rank(mult.a_comm(), &mpi_rank_a);
    int mpi_size_a;
    MPI_Comm_size(mult.a_comm(), &mpi_size_a);

    ElemT *H = (ElemT *)calloc(num_block * num_block, sizeof(ElemT));
    ElemT *U = (ElemT *)calloc(num_block * num_block, sizeof(ElemT));
    RealT *E = (RealT *)malloc(num_block * sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD_Thrust(hii, dii, mult.h_comm());

#ifdef SBD_DEBUG_DAVIDSON
    std::cout << " diagonal term at mpi process (h,b,t) = ("
                << mpi_rank_h << "," << mpi_rank_b << ","
                << mpi_rank_t << "): ";
    for (size_t id = 0; id < std::min(W.size(), static_cast<size_t>(6)); id++)
    {
        std::cout << " " << dii[id];
    }
    std::cout << std::endl;
#endif

    bool do_continue = true;

    std::vector<double> onestep_times(num_block * max_iteration, 0.0);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Workspace for BatchedInnerProduct_GEMV, BatchedAXPY_GEMV, Normalize2,
    // and nccl_allreduce2.
    size_t ws_size = std::max((size_t)nb, 2 * W.size());
    thrust::device_vector<ElemT> workspace(ws_size);

    cudaStream_t stream = 0;
    auto policy_nosync = thrust::cuda::par_nosync.on(stream);
    auto policy_sync = thrust::cuda::par.on(stream);
    // auto policy_sync = thrust::device;
#ifdef SBD_USE_THRUST_NOSYNC
    auto policy = policy_nosync;
#else
    auto policy = policy_sync;
#endif

    // copyin W
    thrust::device_vector<ElemT> W_dev(W.size());
    thrust::copy_n(W.begin(), W.size(), W_dev.begin());

    for (int it = 0; it < max_iteration; it++) {
        SBD_NVTX_RANGE_COLOR("for (int it ...", __LINE__ + it);

        C_tmp = W_dev;

        for (int ib = 0; ib < nb; ib++) {
            SBD_NVTX_RANGE_COLOR("for (int ib ...", __LINE__ + ib);
            auto step_start = std::chrono::high_resolution_clock::now();

            // Copy C_tmp back to C[ib]
            SBD_CHECK_CUDA(cudaMemcpyAsync(C + (W.size() * ib),
                                           thrust::raw_pointer_cast(C_tmp.data()),
                                           sizeof(ElemT) * W.size(),
                                           cudaMemcpyDeviceToDevice, stream));

            // Zero(HC[ib]);
            thrust::fill(policy_nosync, HC_tmp.begin(), HC_tmp.end(), 0);
            SBD_CHECK_CUDA(cudaStreamSynchronize(stream));

            {
                SBD_NVTX_RANGE_COLOR("mult.run", __LINE__);
                mult.run(hii, C_tmp, HC_tmp);
            }

            // Use GEMV to compute multiple inner products in a batched manner,
            // reducing GPU kernel launches and improving memory access
            // efficiency compared to individual dot product evaluations.
            std::vector<ElemT> res(ib+1);
            BatchedInnerProduct_GEMV(C, ib+1, HC_tmp, res, mult.b_comm(),
                                     thrust::raw_pointer_cast(workspace.data()), stream);

            // Copy HC_tmp back to HC[ib]
            SBD_CHECK_CUDA(cudaMemcpyAsync(HC + (W.size() * ib),
                                           thrust::raw_pointer_cast(HC_tmp.data()),
                                           sizeof(ElemT) * W.size(),
                                           cudaMemcpyDeviceToDevice, stream));

            for (int jb = 0; jb <= ib; jb++) {
                H[jb + nb * ib] = res[jb];
                H[ib + nb * jb] = Conjugate(res[jb]);
            }
            for (int jb = 0; jb <= ib; jb++) {
                for (int kb = 0; kb <= ib; kb++) {
                    U[jb + nb * kb] = H[jb + nb * kb];
                }
            }

#ifdef SBD_NO_LAPACK
            hp_numeric::JacobiHeev(ib + 1, U, nb, E);
#else
            hp_numeric::MatHeev(jobz, uplo, ib + 1, U, nb, E);
#endif

            std::vector<ElemT> alpha_W(ib+1);
            std::vector<ElemT> alpha_R(ib+1);
            for (int kb = 0; kb <= ib; kb++) {
                alpha_W[kb] = U[kb];
                alpha_R[kb] = -U[kb];
            }
            // W_dev = C * alpha_W
            BatchedAXPY_GEMV(C, ib + 1, alpha_W, W_dev, ElemT(0),
                             thrust::raw_pointer_cast(workspace.data()), stream);
            // R = HC * alpha_R
            BatchedAXPY_GEMV(HC, ib + 1, alpha_R, R, ElemT(0),
                             thrust::raw_pointer_cast(workspace.data()), stream);
            {
                SBD_NVTX_RANGE_COLOR("thrust::transform", __LINE__);
                // R += E[0] * W
                thrust::transform(policy_nosync,
                                  W_dev.begin(), W_dev.end(), R.begin(), R.begin(),
                                  AXPY_kernel<ElemT>(E[0]));
            }

#ifdef SBD_USE_NCCL
            if (mpi_size_a > 1) {
                // Fuse two AllReduce operations by packing W_dev and R into
                // a single buffer, reducing NCCL collective launch overhead
                // (2 calls -> 1 call).
                nccl_allreduce2(W_dev, R, ncclSum, mult.a_nccl_comm(),
                                thrust::raw_pointer_cast(workspace.data()), stream);
            }
#else // #ifdef SBD_USE_NCCL
            SBD_CHECK_CUDA(cudaStreamSynchronize(stream));
            if (mpi_size_t > 1) {
                MpiAllreduce(W_dev, MPI_SUM, mult.t_comm());
                MpiAllreduce(R, MPI_SUM, mult.t_comm());
            }
            if (mpi_size_h > 1) {
                MpiAllreduce(W_dev, MPI_SUM, mult.h_comm());
                MpiAllreduce(R, MPI_SUM, mult.h_comm());
            }
#endif // #ifdef SBD_USE_NCCL

            RealT norm_W;
            RealT norm_R;
            // The explicit scaling of W_dev and R used in the original code is omitted here.
            // Both vectors are normalized immediately afterwards by Normalize2(), so scaling
            // the vectors themselves is redundant. To preserve norm values consistent with
            // the original behavior, norm_W and norm_R are scaled after Normalize2().
            Normalize2(W_dev, R, norm_W, norm_R, mult.b_comm(), mpi_size_b,
                       thrust::raw_pointer_cast(workspace.data()), stream);
            if (mpi_size_h * mpi_size_t > 1) {
                RealT volp(1.0 / (mpi_size_h * mpi_size_t));
                norm_W *= volp;
                norm_R *= volp;
            }
            // std::cout << "  norm_W = " << norm_W << " , norm_R = " << norm_R << std::endl;

#ifdef SBD_DEBUG_DAVIDSON
            std::cout << " Davidson iteration " << it << "." << ib
                        << " at mpi (h,b,t) = ("
                        << mpi_rank_h << "," << mpi_rank_b << ","
                        << mpi_rank_t << "): (tol=" << norm_R << "):";
            for (int p = 0; p < std::min(ib + 1, 4); p++)
            {
                std::cout << " " << E[p];
            }
            std::cout << std::endl;
#else
            if (mpi_rank_h == 0) {
                if (mpi_rank_t == 0) {
                    if (mpi_rank_b == 0) {
                        std::cout << " Davidson iteration " << it << "." << ib
                                    << " (tol=" << norm_R << "):";
                        for (int p = 0; p < std::min(ib + 1, 4); p++) {
                            std::cout << " " << E[p];
                        }
                        std::cout << std::endl;
                    }
                }
            }
#endif
            if (norm_R < eps) {
                do_continue = false;
                break;
            }

            if (ib < nb - 1) {
                // Determine
                // Initialize C[ib+1]
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(policy_nosync, ci, W.size(),
                                       Determine_kernel(C_tmp, R, dii, eps_reg, E[0]));
                }

                // Gram-Schmidt orthogonalization
#ifdef SBD_USE_NCCL
                GramSchmidtOrthogonalize_GEMV(
                    C, ib + 1, C_tmp, mult.b_nccl_comm(), mpi_size_b,
                    thrust::raw_pointer_cast(workspace.data()), stream);
#else
                GramSchmidtOrthogonalize_GEMV(
                    C, ib + 1, C_tmp, mult.b_comm(), mpi_size_b, 
                    thrust::raw_pointer_cast(workspace.data()), stream);
#endif
                RealT norm_C;
                Normalize(C_tmp, norm_C, mult.b_comm(), mpi_size_b, stream);
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            onestep_times[it * nb + ib] = std::chrono::duration<double>(step_end - step_start).count();
            double ave_time_per_step = 0.0;
            for (int ks = 0; ks <= it * nb + ib; ks++) {
                ave_time_per_step += onestep_times[ks];
            }
            ave_time_per_step /= (it * nb + ib + 1);

            auto current_time = std::chrono::high_resolution_clock::now();
            double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();
            double predicted_next_end = total_elapsed + ave_time_per_step;
            if (mpi_rank_h == 0) {
                if (mpi_rank_t == 0) {
                    MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.b_comm());
                }
                MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.t_comm());
            }
            MPI_Bcast(&predicted_next_end, 1, MPI_DOUBLE, 0, mult.h_comm());

            if (predicted_next_end > max_time) {
                do_continue = false;
                break;
            }

        } // end for(int ib=0; ib < nb; ib++)
        SBD_CHECK_CUDA(cudaStreamSynchronize(stream));

        if (!do_continue) {
            break;
        }

        // Restart with C[0] = W;
        // C[0] = W_dev;
    } // end for(int it=0; it < max_iteration; it++)

    // copyout W
    thrust::copy_n(W_dev.begin(), W.size(), W.begin());

    free(H);
    free(U);
    free(E);

    SBD_CHECK_CUDA(cudaFree(C));
    SBD_CHECK_CUDA(cudaFree(HC));
}

#endif // #ifndef SBD_USE_CUBLAS

}

#endif
