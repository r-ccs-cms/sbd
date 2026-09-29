/**
@file sbd/chemistry/basic/lanzcos_thrust.h
@brief lanczos for parallel task management for distributed basis
*/
#ifndef SBD_CHEMISTRY_LANCZOS_THRUST_H
#define SBD_CHEMISTRY_LANCZOS_THRUST_H

#include "sbd/framework/nvtx.h"

#ifdef __CUDACC__
#include "sbd/framework/cuda_reduce.h"
#else
#include "sbd/framework/hip_reduce.h"
#endif

#include "sbd/framework/thrust_kernels.h"
#include "sbd/chemistry/basic/subspace_thrust.h"

namespace sbd
{

// Lanczos body shared by both subspace locations (see DavidsonImpl).
template <typename Space, typename ElemT, typename RealT>
void LanczosImpl(const thrust::device_vector<ElemT> &hii,
				std::vector<ElemT> &W,
				MultBase<ElemT>& mult,
				int max_iteration,
				int num_block,
				RealT eps)
{
    SBD_NVTX_RANGE_COLOR("Lanczos", __LINE__);
    using Vector = typename Space::template vector<ElemT>;
    Space::check(mult.b_comm());
    auto policy = Space::policy();

	char jobz = 'V';
	char uplo = 'U';
	int lda = num_block;
	MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
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

	RealT *A = (RealT *)calloc(num_block * num_block, sizeof(RealT));
	RealT *U = (RealT *)calloc(num_block * num_block, sizeof(RealT));
	RealT *E = (RealT *)malloc(num_block * sizeof(RealT));

//	std::vector<ElemT> HC(W);
//	std::vector<std::vector<ElemT>> C(num_block, W);

    // copyin W
    Vector W_dev(W.size());
    thrust::copy_n(W.begin(), W.size(), W_dev.begin());

    std::vector<Vector> C(num_block);
    Vector HC(W.size());
    SubspaceMult<Space, ElemT> smult(mult, hii, W.size());
    for (int i = 0; i < num_block; i++) {
        C[i].resize(W.size());
		C[i] = W_dev;
    }

	for (int it = 0; it < max_iteration; it++) {
		int n = 0;
		ElemT Aii;
		RealT E_old = 1.0e+8;
		bool stop_it = false;

		//Zero(HC);
		thrust::fill(HC.begin(), HC.end(), 0);

		C[0] = W_dev;

		for (int ib = 0; ib < num_block; ib++) {
			n++;
			int ii = ib + lda * ib;
			int ij = ib + lda * (ib + 1);
			int ji = ib + 1 + lda * ib;
			smult.add(C[ib], HC);

			InnerProduct(C[ib], HC, Aii, mult.b_comm());
			A[ii] = GetReal(Aii);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					U[i + lda * j] = A[i + lda * j];
				}
			}
			hp_numeric::MatHeev(jobz, uplo, n, U, lda, E);

			if (std::abs(E[0] - E_old) < eps) {
				stop_it = true;
				break;
			}
			E_old = E[0];
			if (ib + 1 == num_block) {
				break;
			}

			// HC[is] -= Aii * C[ib][is];
			thrust::transform(policy, C[ib].begin(), C[ib].end(), HC.begin(), HC.begin(), AXPY_kernel<ElemT>(-Aii));

			// C[ib + 1][is] = HC[is];
			C[ib + 1] = HC;

			Normalize(C[ib + 1], A[ij], mult.b_comm());
			A[ji] = A[ij];

			if ((mpi_rank_h == 0) &&
				(mpi_rank_b == 0) &&
				(mpi_rank_t == 0)) {
				std::cout << " Lanczos iteration " << it
							<< ", step " << ib
							<< ": (A,B)=(" << A[ii]
							<< "," << A[ij] << "):";
				for (int p = 0; p < std::min(n, 4); p++) {
					std::cout << " " << E[p];
				}
				std::cout << std::endl;
			}

			if (std::abs(A[ij]) < eps) {
				break;
			}

			ElemT volp(1.0 / (mpi_size_h * mpi_size_t));
			// HC[is] = -A[ij] * volp * C[ib][is];
            thrust::transform(policy, C[ib].begin(), C[ib].end(), HC.begin(), AX_kernel<ElemT>(-A[ij] * volp));

			MpiAllreduce(HC, MPI_SUM, mult.t_comm());
			MpiAllreduce(HC, MPI_SUM, mult.h_comm());

			// HC[is] *= volp;
			thrust::transform(policy, HC.begin(), HC.end(), HC.begin(), AX_kernel<ElemT>(volp));
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				U[i + lda * j] = A[i + lda * j];
			}
		}
		hp_numeric::MatHeev(jobz, uplo, n, U, lda, E);
		for (int ib = 0; ib < n; ib++) {
			if (ib == 0) {
				// W[is] = U[0] * C[ib][is];
				thrust::transform(policy, C[ib].begin(), C[ib].end(), W_dev.begin(), AX_kernel<ElemT>(U[0]));
			} else {
				// W[is] += U[ib] * C[ib][is];
				thrust::transform(policy, C[ib].begin(), C[ib].end(), W_dev.begin(), W_dev.begin(), AXPY_kernel<ElemT>(U[ib]));
			}
		}
		if (stop_it) {
			break;
		}
	}
    // copyout W
    thrust::copy_n(W_dev.begin(), W.size(), W.begin());

	free(A);
	free(E);
	free(U);
}


template <typename ElemT, typename RealT>
void Lanczos(const thrust::device_vector<ElemT> &hii,
				std::vector<ElemT> &W,
				MultBase<ElemT>& mult,
				int max_iteration,
				int num_block,
				RealT eps,
				bool cpu_subspace = false)
{
    if (cpu_subspace)
        LanczosImpl<HostSubspace>(hii, W, mult, max_iteration, num_block, eps);
    else
        LanczosImpl<DeviceSubspace>(hii, W, mult, max_iteration, num_block, eps);
}

}

#endif
