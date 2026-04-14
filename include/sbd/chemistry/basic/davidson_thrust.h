/**
@file sbd/chemistry/basic/davidson.h
@brief davidson for parallel task management for distributed basis
*/
#ifndef SBD_CHEMISTRY_DAVIDSON_THRUST_H
#define SBD_CHEMISTRY_DAVIDSON_THRUST_H

#include "sbd/framework/jacobi.h"
#ifdef __CUDACC__
#include "sbd/framework/cuda_reduce.h"
#else
#include "sbd/framework/hip_reduce.h"
#endif


#include "sbd/framework/thrust_kernels.h"
#include <thrust/execution_policy.h>

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
        if (std::abs(e0 - dii[is]) > eps_reg) {
            C[is] = R[is] / (e0 - dii[is]);
        } else {
            C[is] = R[is] / (e0 - dii[is] - eps_reg);
        }
    }
};


template <typename ElemT>
void GetTotalD_Thrust(const thrust::device_vector<ElemT> & hii,
        thrust::device_vector<ElemT>& dii,
        MPI_Comm h_comm) {
    dii = hii;
    MpiAllreduce(dii, MPI_SUM, h_comm);
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

template <typename ElemT, typename RealT>
void Davidson(const thrust::device_vector<ElemT> &hii,
                std::vector<ElemT> &W,
                MultBase<ElemT>& mult,
                int max_iteration,
                int num_block,
                RealT eps,
                RealT max_time)
{
    RealT eps_reg = 1.0e-12;

    std::vector<thrust::device_vector<ElemT>> C(num_block);
    std::vector<thrust::device_vector<ElemT>> HC(num_block);
    for (int i = 0; i < num_block; i++) {
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

    // copyin W
    thrust::device_vector<ElemT> W_dev(W.size());
    thrust::copy_n(W.begin(), W.size(), W_dev.begin());

    for (int it = 0; it < max_iteration; it++) {
        C[0] = W_dev;

        for (int ib = 0; ib < nb; ib++) {
            auto step_start = std::chrono::high_resolution_clock::now();

            //Zero(HC[ib]);
            thrust::fill(HC[ib].begin(), HC[ib].end(), 0);

            mult.run(hii, C[ib], HC[ib]);

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
            // ElemT x = U[0];
            // W[is] = C[0][is] * x;
            //thrust::transform(thrust::device, C[0].begin(), C[0].end(), thrust::constant_iterator<ElemT>(U[0]), W_dev.begin(), thrust::multiplies<ElemT>());
            thrust::transform(thrust::device, C[0].begin(), C[0].end(), W_dev.begin(), AX_kernel<ElemT>(U[0]));

            // x = ElemT(-1.0) * U[0];
            // R[is] = HC[0][is] * x;
            //thrust::transform(thrust::device, HC[0].begin(), HC[0].end(), thrust::constant_iterator<ElemT>(-U[0]), R.begin(), thrust::multiplies<ElemT>());
            thrust::transform(thrust::device, HC[0].begin(), HC[0].end(), R.begin(), AX_kernel<ElemT>(-U[0]));

            for (int kb = 1; kb <= ib; kb++) {
                // x = U[kb];
                // W[is] += C[kb][is] * x;
                thrust::transform(thrust::device, C[kb].begin(), C[kb].end(), W_dev.begin(), W_dev.begin(), AXPY_kernel<ElemT>(U[kb]));

                // x = ElemT(-1.0) * U[kb];
                // R[is] += HC[kb][is] * x;
                thrust::transform(thrust::device, HC[kb].begin(), HC[kb].end(), R.begin(), R.begin(), AXPY_kernel<ElemT>(-U[kb]));
            }
            // R[is] += E[0] * W[is];
            thrust::transform(thrust::device, W_dev.begin(), W_dev.end(), R.begin(), R.begin(), AXPY_kernel<ElemT>(E[0]));

            /**
                 Patch for stability on Fugaku
                */
            // #ifdef SBD_FUAGKUPATCH
            if (mpi_size_t > 1)
                MpiAllreduce(W_dev, MPI_SUM, mult.t_comm());
            if (mpi_size_h > 1)
                MpiAllreduce(W_dev, MPI_SUM, mult.h_comm());
            if (mpi_size_t > 1)
                MpiAllreduce(R, MPI_SUM, mult.t_comm());
            if (mpi_size_h > 1)
                MpiAllreduce(R, MPI_SUM, mult.h_comm());
            if (mpi_size_h * mpi_size_t > 1) {
                ElemT volp(1.0 / (mpi_size_h * mpi_size_t));
                // W[is] *= volp;
                //thrust::transform(thrust::device, W_dev.begin(), W_dev.end(), thrust::constant_iterator<ElemT>(volp), W_dev.begin(), thrust::multiplies<ElemT>());
                thrust::transform(thrust::device, W_dev.begin(), W_dev.end(), W_dev.begin(), AX_kernel<ElemT>(volp));
                // R[is] *= volp;
                //thrust::transform(thrust::device, R.begin(), R.end(), thrust::constant_iterator<ElemT>(volp), R.begin(), thrust::multiplies<ElemT>());
                thrust::transform(thrust::device, R.begin(), R.end(), R.begin(), AX_kernel<ElemT>(volp));
            }
            // #endif

            RealT norm_W;
            Normalize(W_dev, norm_W, mult.b_comm());

            RealT norm_R;
            Normalize(R, norm_R, mult.b_comm());

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
                thrust::for_each_n(thrust::device, ci, W.size(), Determine_kernel(C[ib + 1], R, dii, eps_reg, E[0]));

                // Gram-Schmidt orthogonalization
                for (int kb = 0; kb < ib + 1; kb++) {
                    ElemT olap;
                    InnerProduct(C[kb], C[ib+1], olap, mult.b_comm());
                    olap *= ElemT(-1.0);
                    thrust::transform(thrust::device, C[kb].begin(), C[kb].end(), C[ib + 1].begin(), C[ib + 1].begin(), AXPY_kernel<ElemT>(olap));
                }

                RealT norm_C;
                Normalize(C[ib + 1], norm_C, mult.b_comm());
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

}

#endif
