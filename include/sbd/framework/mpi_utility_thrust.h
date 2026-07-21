/// This is a part of qscd
/**
@file mpi_utility_thrust.h
@brief tools for mpi parallelization
 */

#ifndef SBD_FRAMEWORK_MPI_UTILITY_THRUST_H
#define SBD_FRAMEWORK_MPI_UTILITY_THRUST_H

#include "mpi.h"

#include "sbd/framework/nvtx.h"

#ifdef SBD_USE_NCCL
#include "sbd/framework/cuda_utility.h"
#include "sbd/framework/nccl_utility.h"
#endif

namespace sbd
{

#if defined(SBD_NON_CUDA_AWARE_MPI) || defined(SBD_THRUST_SAFE_MPI_ALLREDUCE)

template <typename ElemT>
void MpiAllreduce(thrust::device_vector<ElemT> &A, MPI_Op op, MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("MpiAllreduce", __LINE__);
    // Calling MPI functions directly on the `device_vector` sometimes caused instability,
    // so copy them to the host temporarily.
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    std::vector<ElemT> h_send(A.size()), h_recv(A.size());
    thrust::copy(A.begin(), A.end(), h_send.begin());
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce(h_send.data(), h_recv.data(), static_cast<int>(h_send.size()), DataT, op, comm);
    }
    thrust::copy(h_recv.begin(), h_recv.end(), A.begin());
}

#else

template <typename ElemT>
void MpiAllreduce(thrust::device_vector<ElemT> &A, MPI_Op op, MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("MpiAllreduce", __LINE__);
    std::cout << "   TEST MpiAllreduce" << std::endl;
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#if 1
    thrust::device_vector<ElemT> B(A);
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce", 0);
        MPI_Allreduce((ElemT *)thrust::raw_pointer_cast(B.data()), (ElemT *)thrust::raw_pointer_cast(A.data()), A.size(), DataT, op, comm);
    }
#else
    {
        SBD_NVTX_RANGE_COLOR("MPI_Allreduce (MPI_IN_PLACE)", 0);
        MPI_Allreduce(MPI_IN_PLACE, (ElemT *)thrust::raw_pointer_cast(A.data()), A.size(), DataT, op, comm);
    }
#endif
}

#endif // SBD_NON_CUDA_AWARE_MPI || SBD_THRUST_SAFE_MPI_ALLREDUCE

template <typename ElemT>
void _MpiSlide(const ElemT* A,
    thrust::device_vector<ElemT>& B,
    size_t sizeA,
    int slide,
    MPI_Comm comm)
{
    int mpi_rank;
    MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = sizeA;
    {
        SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
        MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
        MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    }
    {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(2,req_size.data(),sta_size.data());
    }

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#ifdef SBD_NON_CUDA_AWARE_MPI
    std::vector<ElemT> h_send(send_size), h_recv(recv_size);
    if (send_size != 0) {
        cudaMemcpy(h_send.data(), A, send_size * sizeof(ElemT), cudaMemcpyDefault);
    }
    if( send_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
        MPI_Isend(h_send.data(),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
        MPI_Irecv(h_recv.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#else
    if( send_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
        MPI_Isend(A,send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
        MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#endif

    if( send_size != 0 && recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
        SBD_NVTX_RANGE_COLOR("MPI_Waitall", 0);
        MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
#ifdef SBD_NON_CUDA_AWARE_MPI
    if (recv_size != 0) {
        thrust::copy(h_recv.begin(), h_recv.end(), B.begin());
    }
#endif
}

template <typename ElemT>
void MpiSlide(const thrust::device_vector<ElemT>& A,
    thrust::device_vector<ElemT>& B,
    int slide,
    MPI_Comm comm)
{
    _MpiSlide((ElemT*)thrust::raw_pointer_cast(A.data()), B, A.size(), slide, comm);
}

template <typename ElemT>
void MpiSlide(const std::vector<ElemT>& A,
    thrust::device_vector<ElemT>& B,
    int slide,
    MPI_Comm comm)
{
    _MpiSlide(A.data(), B, A.size(), slide, comm);
}



template <>
void MpiSlide(const thrust::device_vector<size_t> & A,
    thrust::device_vector<size_t> & B,
    int slide,
    MPI_Comm comm)
{
    int mpi_rank;
    MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = A.size();
    MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = SBD_MPI_SIZE_T;
#ifdef SBD_NON_CUDA_AWARE_MPI
    std::vector<size_t> h_send(send_size), h_recv(recv_size);
    if (send_size != 0) {
        thrust::copy(A.begin(), A.end(), h_send.begin());
    }
    if( send_size != 0 ) {
        MPI_Isend(h_send.data(),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        MPI_Irecv(h_recv.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#else
    if( send_size != 0 ) {
        MPI_Isend((size_t*)thrust::raw_pointer_cast(A.data()),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
        MPI_Irecv((size_t*)thrust::raw_pointer_cast(B.data()),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#endif

    if( send_size != 0 && recv_size != 0 ) {
        MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
        MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
        MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
#ifdef SBD_NON_CUDA_AWARE_MPI
    if (recv_size != 0) {
        thrust::copy(h_recv.begin(), h_recv.end(), B.begin());
    }
#endif
}



template <typename ElemT>
void _Mpi2dSlide(const ElemT* A,
                thrust::device_vector<ElemT> &B,
                size_t sizeA,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("_Mpi2dSlide", __LINE__);
    // Assuming mpi_rank = x_rank * y_size + y_rank;

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);

    int x_rank = mpi_rank / y_size;
    int y_rank = mpi_rank % y_size;

    int x_dist = (x_rank + x_slide + x_size) % x_size;
    int y_dist = (y_rank + y_slide + y_size) % y_size;
    int mpi_dist = x_dist * y_size + y_dist;

    int x_source = (x_rank - x_slide + x_size) % x_size;
    int y_source = (y_rank - y_slide + y_size) % y_size;
    int mpi_source = x_source * y_size + y_source;

#ifdef SBD_DEBUG_MPI_UTILITY
    std::cout << " Mpi2dSlide at rank " << mpi_rank << " = (" << x_rank << "," << y_rank
                << "): distination rank = " << mpi_dist << " = (" << x_dist << "," << y_dist
                << "), source rank = " << mpi_source << " = (" << x_source << "," << y_source
                << ")" << std::endl;
#endif
    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = sizeA;

    MPI_Isend(size_send.data(), 1, SBD_MPI_SIZE_T,
                mpi_dist, 0, comm, &req_size[0]);
    MPI_Irecv(size_recv.data(), 1, SBD_MPI_SIZE_T,
                mpi_source, 0, comm, &req_size[1]);
    MPI_Waitall(2, req_size.data(), sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#ifdef SBD_NON_CUDA_AWARE_MPI
    std::vector<ElemT> h_send(send_size), h_recv(recv_size);
    if (send_size != 0) {
        cudaMemcpy(h_send.data(), A, send_size * sizeof(ElemT), cudaMemcpyDefault);
    }
    if (send_size != 0) {
        MPI_Isend(h_send.data(), send_size, DataT, mpi_dist, 1, comm, &req_data[0]);
    }
    if (recv_size != 0) {
        MPI_Irecv(h_recv.data(), recv_size, DataT, mpi_source, 1, comm, &req_data[1]);
    }
#else
    if (send_size != 0) {
        MPI_Isend(A, send_size, DataT, mpi_dist, 1, comm, &req_data[0]);
    }
    if (recv_size != 0) {
        MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()), recv_size, DataT, mpi_source, 1, comm, &req_data[1]);
    }
#endif

    if (send_size != 0 && recv_size != 0) {
        MPI_Waitall(2, req_data.data(), sta_data.data());
    }
    else if (send_size != 0 && recv_size == 0) {
        MPI_Waitall(1, &req_data[0], &sta_data[0]);
    }
    else if (send_size == 0 && recv_size != 0) {
        MPI_Waitall(1, &req_data[1], &sta_data[1]);
    }
#ifdef SBD_NON_CUDA_AWARE_MPI
    if (recv_size != 0) {
        thrust::copy(h_recv.begin(), h_recv.end(), B.begin());
    }
#endif
}

template <typename ElemT>
void Mpi2dSlide(const thrust::device_vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("Mpi2dSlide", __LINE__);
    _Mpi2dSlide((ElemT*)thrust::raw_pointer_cast(A.data()), B, A.size(),
                      x_size, y_size, x_slide, y_slide, comm);
}

template <typename ElemT>
void Mpi2dSlide(const std::vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm)
{
    SBD_NVTX_RANGE_COLOR("Mpi2dSlide", __LINE__);
    _Mpi2dSlide(A.data(), B, A.size(),
                      x_size, y_size, x_slide, y_slide, comm);
}

template <typename ElemT>
class Mpi2dSlider {
protected:
    MPI_Request req_send;
    MPI_Request req_recv;
    size_t send_size;
    size_t recv_size;
#ifdef SBD_NON_CUDA_AWARE_MPI
    std::vector<ElemT> h_send_buf;
    std::vector<ElemT> h_recv_buf;
    thrust::device_vector<ElemT>* p_B;
#endif
public:
    Mpi2dSlider()
    {
        send_size = 0;
        recv_size = 0;
#ifdef SBD_NON_CUDA_AWARE_MPI
        p_B = nullptr;
#endif
    }

    void ExchangeAsync(const thrust::device_vector<ElemT> &A,
                thrust::device_vector<ElemT> &B,
                int x_size,
                int y_size,
                int x_slide,
                int y_slide,
                MPI_Comm comm,
                size_t task)
    {
        SBD_NVTX_RANGE_COLOR("ExchangeAsync", __LINE__);
        // Assuming mpi_rank = x_rank * y_size + y_rank;

        int mpi_rank;
        MPI_Comm_rank(comm, &mpi_rank);
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);

        int x_rank = mpi_rank / y_size;
        int y_rank = mpi_rank % y_size;

        int x_dist = (x_rank + x_slide + x_size) % x_size;
        int y_dist = (y_rank + y_slide + y_size) % y_size;
        int mpi_dist = x_dist * y_size + y_dist;

        int x_source = (x_rank - x_slide + x_size) % x_size;
        int y_source = (y_rank - y_slide + y_size) % y_size;
        int mpi_source = x_source * y_size + y_source;

#ifdef SBD_DEBUG_MPI_UTILITY
        std::cout << " Mpi2dSlide at rank " << mpi_rank << " = (" << x_rank << "," << y_rank
                  << "): distination rank = " << mpi_dist << " = (" << x_dist << "," << y_dist
                  << "), source rank = " << mpi_source << " = (" << x_source << "," << y_source
                  << ")" << std::endl;
#endif
        {
            SBD_NVTX_RANGE_COLOR("MPI_Barrier", 0);
            MPI_Barrier(comm);
        }

        std::vector<MPI_Request> req_size(2);
        std::vector<MPI_Status> sta_size(2);
        std::vector<size_t> size_send(1);
        std::vector<size_t> size_recv(1);
        size_send[0] = A.size();

        MPI_Isend(size_send.data(), 1, SBD_MPI_SIZE_T,
                    mpi_dist, task * 2, comm, &req_size[0]);
        MPI_Irecv(size_recv.data(), 1, SBD_MPI_SIZE_T,
                    mpi_source, task * 2, comm, &req_size[1]);
        MPI_Waitall(2, req_size.data(), sta_size.data());

        send_size = size_send[0];
        recv_size = size_recv[0];

        B.resize(recv_size);

        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#ifdef SBD_NON_CUDA_AWARE_MPI
        h_send_buf.resize(send_size);
        h_recv_buf.resize(recv_size);
        p_B = &B;
        if (send_size != 0) {
            cudaMemcpy(h_send_buf.data(), thrust::raw_pointer_cast(A.data()),
                       send_size * sizeof(ElemT), cudaMemcpyDefault);
        }
        if (send_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
            MPI_Isend(h_send_buf.data(), send_size, DataT, mpi_dist, task * 2 + 1, comm, &req_send);
        }
        if (recv_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
            MPI_Irecv(h_recv_buf.data(), recv_size, DataT, mpi_source, task * 2 + 1, comm, &req_recv);
        }
#else
        if (send_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Isend", 0);
            MPI_Isend((ElemT*)thrust::raw_pointer_cast(A.data()), send_size, DataT, mpi_dist, task * 2 + 1, comm, &req_send);
        }
        if (recv_size != 0) {
            SBD_NVTX_RANGE_COLOR("MPI_Irecv", 0);
            MPI_Irecv((ElemT*)thrust::raw_pointer_cast(B.data()), recv_size, DataT, mpi_source, task * 2 + 1, comm, &req_recv);
        }
#endif
    }

    bool Sync(MPI_Comm comm)
    {
        SBD_NVTX_RANGE_COLOR("Sync", __LINE__);
        bool recv = false;
        if (send_size > 0) {
            MPI_Status st;
            SBD_NVTX_RANGE_COLOR("MPI_Wait", 0);
            MPI_Wait(&req_send, &st);
        }
        if (recv_size > 0) {
            MPI_Status st;
            SBD_NVTX_RANGE_COLOR("MPI_Wait", 0);
            MPI_Wait(&req_recv, &st);
            recv = true;
#ifdef SBD_NON_CUDA_AWARE_MPI
            thrust::copy(h_recv_buf.begin(), h_recv_buf.end(), p_B->begin());
#endif
        }
        {
            SBD_NVTX_RANGE_COLOR("MPI_Barrier", 0);
            MPI_Barrier(comm);
        }

        send_size = 0;
        recv_size = 0;
        return recv;
    }
};

}

#endif
