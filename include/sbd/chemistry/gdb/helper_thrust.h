/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis for Thrust
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_THRUST_H
#define SBD_CHEMISTRY_GDB_HELPER_THRUST_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"

namespace sbd
{
namespace gdb
{

/**
       DetBasisMapper is an array to find relation between dets.
*/
class DetIndexMapThrust
{
public:
    size_t* AdetToDetOffset;
    size_t* BdetToDetOffset;
    size_t* AdetIndex;
    size_t* BdetIndex;
    size_t* AdetToBdetSM;
    size_t* AdetToDetSM;
    size_t* BdetToAdetSM;
    size_t* BdetToDetSM;
    size_t size_adet = 0;
    size_t size_bdet = 0;

    __host__ __device__ DetIndexMapThrust() {}

    __host__ __device__ DetIndexMapThrust(const DetIndexMapThrust& other)
    {
        AdetToDetOffset = other.AdetToDetOffset;
        BdetToDetOffset = other.BdetToDetOffset;
        AdetIndex = other.AdetIndex;
        BdetIndex = other.BdetIndex;
        AdetToBdetSM = other.AdetToBdetSM;
        AdetToDetSM = other.AdetToDetSM;
        BdetToAdetSM = other.BdetToAdetSM;
        BdetToDetSM = other.BdetToDetSM;
        size_adet = other.size_adet;
        size_bdet = other.size_bdet;
    }

    DetIndexMapThrust(thrust::device_vector<size_t>& storage, const DetIndexMap& idxmap)
    {
        size_t size = 0;

        std::vector<size_t> offset_adet(idxmap.AdetToDetLen.size() + 1);
        std::vector<size_t> offset_bdet(idxmap.BdetToDetLen.size() + 1);
        for(size_t i=0; i < idxmap.AdetToDetLen.size(); i++) {
            offset_adet[i] = size_adet;
            size_adet += idxmap.AdetToDetLen[i];
        }
        offset_adet[idxmap.AdetToDetLen.size()] = size_adet;

        for(size_t i=0; i < idxmap.BdetToDetLen.size(); i++) {
            offset_bdet[i] = size_bdet;
            size_bdet += idxmap.BdetToDetLen[i];
        }
        offset_bdet[idxmap.BdetToDetLen.size()] = size_bdet;

        size = idxmap.AdetToDetLen.size() + 1 + idxmap.BdetToDetLen.size() + 1 + size_adet * 3 + size_bdet * 3;
        storage.resize(size);
        size_t* base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        // store offset
        AdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_adet.begin(), offset_adet.size(), storage.begin() + count);
        count += offset_adet.size();

        BdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_bdet.begin(), offset_bdet.size(), storage.begin() + count);
        count += offset_bdet.size();

        // set Adet, Bdet indices
        AdetIndex = base_memory + count;
        for(size_t i=0; i < idxmap.AdetToDetLen.size(); i++) {
            thrust::fill_n(storage.begin() + count + offset_adet[i], idxmap.AdetToDetLen[i], i);
        }
        count += size_adet;

        BdetIndex = base_memory + count;
        for(size_t i=0; i < idxmap.BdetToDetLen.size(); i++) {
            thrust::fill_n(storage.begin() + count + offset_bdet[i], idxmap.BdetToDetLen[i], i);
        }
        count += size_bdet;

        // copy SMs
        thrust::copy_n(idxmap.storage.begin(), size_adet * 2 + size_bdet * 2, storage.begin() + count);
        AdetToDetSM = base_memory + count;
        count += size_adet;
        AdetToBdetSM = base_memory + count;
        count += size_adet;
        BdetToDetSM = base_memory + count;
        count += size_bdet;
        BdetToAdetSM = base_memory + count;
    }

    void copy(thrust::device_vector<size_t>& storage, const DetIndexMapThrust& idxmap, const thrust::device_vector<size_t>& src_storage)
    {
        storage = src_storage;

        size_t* base = (size_t*)thrust::raw_pointer_cast(storage.data());
        size_t* src_base = (size_t*)thrust::raw_pointer_cast(src_storage.data());

        AdetToDetOffset = base + (idxmap.AdetToDetOffset - src_base);
        BdetToDetOffset = base + (idxmap.BdetToDetOffset - src_base);
        AdetIndex = base + (idxmap.AdetIndex - src_base);
        BdetIndex = base + (idxmap.BdetIndex - src_base);
        AdetToBdetSM = base + (idxmap.AdetToBdetSM - src_base);
        AdetToDetSM = base + (idxmap.AdetToDetSM - src_base);
        BdetToAdetSM = base + (idxmap.BdetToAdetSM - src_base);
        BdetToDetSM = base + (idxmap.BdetToDetSM - src_base);
        size_adet = idxmap.size_adet;
        size_bdet = idxmap.size_bdet;
    }

    inline __device__ __host__ int64_t adet_lower_bound(size_t jast, size_t jbst, size_t start = 0)
    {
        size_t is = AdetToDetOffset[jast] + start;
        size_t ie = AdetToDetOffset[jast + 1];
        size_t i = (is + ie) / 2;
        while(is != ie) {
            if( AdetToBdetSM[i] < jbst) {
                is = i + 1;
                i = (is + ie) / 2;
            } else {
                ie = i;
                i = (is + ie) / 2;
            }
        }
        if (ie == AdetToDetOffset[jast + 1]) {
            // not found
            return -1;
        }
        return i;
    }

    inline __device__ __host__ int64_t bdet_lower_bound(size_t jbst, size_t jast)
    {
        size_t is = BdetToDetOffset[jbst];
        size_t ie = BdetToDetOffset[jbst + 1];
        size_t i = (is + ie) / 2;
        while(is != ie) {
            if( BdetToAdetSM[i] < jast) {
                is = i + 1;
                i = (is + ie) / 2;
            } else {
                ie = i;
                i = (is + ie) / 2;
            }
        }
        if (ie == BdetToDetOffset[jbst + 1]) {
            // not found
            return -1;
        }
        return i;
    }


};

/**
       Labels connected
*/
class ExcitationLookupThrust
{
public:
    int slide;
    size_t* SelfFromAdetOffset;
    size_t* SelfFromBdetOffset;
    size_t* SinglesFromAdetOffset;
    size_t* SinglesFromBdetOffset;
    size_t* DoublesFromAdetOffset;
    size_t* DoublesFromBdetOffset;
    size_t* SelfFromAdetSM;
    size_t* SelfFromBdetSM;
    size_t* SinglesFromAdetSM;
    int* SinglesAdetCrAnSM;
    size_t* SinglesFromBdetSM;
    int* SinglesBdetCrAnSM;
    size_t* DoublesFromAdetSM;
    int* DoublesAdetCrAnSM;
    size_t* DoublesFromBdetSM;
    int* DoublesBdetCrAnSM;
    size_t size_self_adet = 0;
    size_t size_self_bdet = 0;
    size_t size_single_adet = 0;
    size_t size_single_bdet = 0;
    size_t size_double_adet = 0;
    size_t size_double_bdet = 0;

    __host__ __device__ ExcitationLookupThrust() {}

    __host__ __device__ ExcitationLookupThrust(const ExcitationLookupThrust& other)
    {
        slide = other.slide;
        SelfFromAdetOffset = other.SelfFromAdetOffset;
        SelfFromBdetOffset = other.SelfFromBdetOffset;
        SinglesFromAdetOffset = other.SinglesFromAdetOffset;
        SinglesFromBdetOffset = other.SinglesFromBdetOffset;
        DoublesFromAdetOffset = other.DoublesFromAdetOffset;
        DoublesFromBdetOffset = other.DoublesFromBdetOffset;
        SelfFromAdetSM = other.SelfFromAdetSM;
        SelfFromBdetSM = other.SelfFromBdetSM;
        SinglesFromAdetSM = other.SinglesFromAdetSM;
        SinglesAdetCrAnSM = other.SinglesAdetCrAnSM;
        SinglesFromBdetSM = other.SinglesFromBdetSM;
        SinglesBdetCrAnSM = other.SinglesBdetCrAnSM;
        DoublesFromAdetSM = other.DoublesFromAdetSM;
        DoublesAdetCrAnSM = other.DoublesAdetCrAnSM;
        DoublesFromBdetSM = other.DoublesFromBdetSM;
        DoublesBdetCrAnSM = other.DoublesBdetCrAnSM;

        size_self_adet = other.size_self_adet;
        size_self_bdet = other.size_self_bdet;
        size_single_adet = other.size_single_adet;
        size_single_bdet = other.size_single_bdet;
        size_double_adet = other.size_double_adet;
        size_double_bdet = other.size_double_bdet;
    }

    ExcitationLookupThrust(thrust::device_vector<size_t>& storage, thrust::device_vector<int>& cran_storage, const ExcitationLookup& exidx)
    {
        size_t size = 0;

        slide = exidx.slide;

        std::vector<size_t> offset_self_adet(exidx.SelfFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromAdetLen.size(); i++) {
            offset_self_adet[i] = size_self_adet;
            size_self_adet += exidx.SelfFromAdetLen[i];
            size++;
        }
        offset_self_adet[exidx.SelfFromAdetLen.size()] = size_self_adet;
        size += size_self_adet + 1;

        std::vector<size_t> offset_self_bdet(exidx.SelfFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromBdetLen.size(); i++) {
            offset_self_bdet[i] = size_self_bdet;
            size_self_bdet += exidx.SelfFromBdetLen[i];
            size++;
        }
        offset_self_bdet[exidx.SelfFromBdetLen.size()] = size_self_bdet;
        size += size_self_bdet + 1;

        std::vector<size_t> offset_single_adet(exidx.SinglesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            offset_single_adet[i] = size_single_adet;
            size_single_adet += exidx.SinglesFromAdetLen[i];
            size++;
        }
        offset_single_adet[exidx.SinglesFromAdetLen.size()] = size_single_adet;
        size += size_single_adet + 1;

        std::vector<size_t> offset_double_adet(exidx.DoublesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            offset_double_adet[i] = size_double_adet;
            size_double_adet += exidx.DoublesFromAdetLen[i];
            size++;
        }
        offset_double_adet[exidx.DoublesFromAdetLen.size()] = size_double_adet;
        size += size_double_adet + 1;

        std::vector<size_t> offset_single_bdet(exidx.SinglesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            offset_single_bdet[i] = size_single_bdet;
            size_single_bdet += exidx.SinglesFromBdetLen[i];
            size++;
        }
        offset_single_bdet[exidx.SinglesFromBdetLen.size()] = size_single_bdet;
        size += size_single_bdet + 1;

        std::vector<size_t> offset_double_bdet(exidx.DoublesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            offset_double_bdet[i] = size_double_bdet;
            size_double_bdet += exidx.DoublesFromBdetLen[i];
            size++;
        }
        offset_double_bdet[exidx.DoublesFromBdetLen.size()] = size_double_bdet;
        size += size_double_bdet + 1;

        storage.resize(size);
        size_t* base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        // store offset
        SelfFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_self_adet.begin(), offset_self_adet.size(), storage.begin() + count);
        count += offset_self_adet.size();

        SelfFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_self_bdet.begin(), offset_self_bdet.size(), storage.begin() + count);
        count += offset_self_bdet.size();

        SinglesFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_single_adet.begin(), offset_single_adet.size(), storage.begin() + count);
        count += offset_single_adet.size();

        DoublesFromAdetOffset = base_memory + count;
        thrust::copy_n(offset_double_adet.begin(), offset_double_adet.size(), storage.begin() + count);
        count += offset_double_adet.size();

        SinglesFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_single_bdet.begin(), offset_single_bdet.size(), storage.begin() + count);
        count += offset_single_bdet.size();

        DoublesFromBdetOffset = base_memory + count;
        thrust::copy_n(offset_double_bdet.begin(), offset_double_bdet.size(), storage.begin() + count);
        count += offset_double_bdet.size();

        // copy SMs
        thrust::copy_n(exidx.storage.begin(), size - count, storage.begin() + count);
        SelfFromAdetSM = base_memory + count;
        count += size_self_adet;
        SinglesFromAdetSM = base_memory + count;
        count += size_single_adet;
        DoublesFromAdetSM = base_memory + count;
        count += size_double_adet;
        SelfFromBdetSM = base_memory + count;
        count += size_self_bdet;
        SinglesFromBdetSM = base_memory + count;
        count += size_single_bdet;
        DoublesFromBdetSM = base_memory + count;

        // convert CrAn from AoS to SoA
        size_t size_cran = size_single_adet * 2 + size_double_adet * 4 + size_single_bdet * 2 + size_double_bdet * 4;
        std::vector<int> buf;
        count = 0;

        cran_storage.resize(size_cran);
        int* base_cran = (int*)thrust::raw_pointer_cast(cran_storage.data());

        SinglesAdetCrAnSM = base_cran + count;
        buf.resize(size_single_adet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromAdetLen[i]; j++) {
                size_t offset = offset_single_adet[i] + j;
                buf[offset] = exidx.SinglesAdetCrAnSM[i][j * 2];
                buf[size_single_adet + offset] = exidx.SinglesAdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), size_single_adet * 2, cran_storage.begin() + count);
        count += size_single_adet * 2;


        DoublesAdetCrAnSM = base_cran + count;
        buf.resize(size_double_adet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromAdetLen[i]; j++) {
                size_t offset = offset_double_adet[i] + j;
                buf[offset] = exidx.DoublesAdetCrAnSM[i][j * 4];
                buf[size_double_adet + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 1];
                buf[size_double_adet * 2 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 2];
                buf[size_double_adet * 3 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), size_double_adet * 4, cran_storage.begin() + count);
        count += size_double_adet * 4;

        SinglesBdetCrAnSM = base_cran + count;
        buf.resize(size_single_bdet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromBdetLen[i]; j++) {
                size_t offset = offset_single_bdet[i] + j;
                buf[offset] = exidx.SinglesBdetCrAnSM[i][j * 2];
                buf[size_single_bdet + offset] = exidx.SinglesBdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), size_single_bdet * 2, cran_storage.begin() + count);
        count += size_single_bdet * 2;

        DoublesBdetCrAnSM = base_cran + count;
        buf.resize(size_double_bdet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromBdetLen[i]; j++) {
                size_t offset = offset_double_bdet[i] + j;
                buf[offset] = exidx.DoublesBdetCrAnSM[i][j * 4];
                buf[size_double_bdet + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 1];
                buf[size_double_bdet * 2 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 2];
                buf[size_double_bdet * 3 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), size_double_bdet * 4, cran_storage.begin() + count);
        count += size_double_bdet * 4;
    }


};


void MpiSlide(const DetIndexMapThrust& send_map,
        const thrust::device_vector<size_t>& send_storage,
        DetIndexMapThrust& recv_map,
        thrust::device_vector<size_t>& recv_storage,
        int slide,
        MPI_Comm comm)
{
    std::vector<size_t> send_offset;
    std::vector<size_t> recv_offset;

    size_t* base = (size_t*)thrust::raw_pointer_cast(send_storage.data());
    send_offset.push_back((size_t)(send_map.AdetToDetOffset - base));
    send_offset.push_back((size_t)(send_map.BdetToDetOffset - base));
    send_offset.push_back((size_t)(send_map.AdetIndex - base));
    send_offset.push_back((size_t)(send_map.BdetIndex - base));
    send_offset.push_back((size_t)(send_map.AdetToBdetSM - base));
    send_offset.push_back((size_t)(send_map.AdetToDetSM - base));
    send_offset.push_back((size_t)(send_map.BdetToAdetSM - base));
    send_offset.push_back((size_t)(send_map.BdetToDetSM - base));
    send_offset.push_back(send_map.size_adet);
    send_offset.push_back(send_map.size_bdet);

    sbd::MpiSlide(send_storage, recv_storage, slide, comm);
    sbd::MpiSlide(send_offset, recv_offset, slide, comm);

    base = (size_t*)thrust::raw_pointer_cast(recv_storage.data());
    recv_map.AdetToDetOffset = base + recv_offset[0];
    recv_map.BdetToDetOffset = base + recv_offset[1];
    recv_map.AdetIndex = base + recv_offset[2];
    recv_map.BdetIndex = base + recv_offset[3];
    recv_map.AdetToBdetSM = base + recv_offset[4];
    recv_map.AdetToDetSM = base + recv_offset[5];
    recv_map.BdetToAdetSM = base + recv_offset[6];
    recv_map.BdetToDetSM = base + recv_offset[7];
    recv_map.size_adet = recv_offset[8];
    recv_map.size_bdet = recv_offset[9];
}



template <typename ElemT>
class MpiSlider {
protected:
    MPI_Request req_send;
    MPI_Request req_recv;
    MPI_Request req_send_map;
    MPI_Request req_recv_map;
    size_t send_size;
    size_t recv_size;
    size_t send_size_map;
    size_t recv_size_map;
public:
    MpiSlider()
    {
        send_size = 0;
        recv_size = 0;
        send_size_map = 0;
        recv_size_map = 0;
    }

    void ExchangeAsync(const thrust::device_vector<ElemT> &send,
                thrust::device_vector<ElemT> &recv,
                const DetIndexMapThrust& send_map,
                const thrust::device_vector<size_t> &send_map_storage,
                DetIndexMapThrust& recv_map,
                thrust::device_vector<size_t> &recv_map_storage,
                int slide,
                MPI_Comm comm,
                int id)
    {
        int mpi_rank;
        MPI_Comm_rank(comm,&mpi_rank);
        int mpi_size;
        MPI_Comm_size(comm,&mpi_size);
        int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
        int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

        // exchange data size to be sent
        std::vector<MPI_Request> req_size(2);
        std::vector<MPI_Status> sta_size(2);
        std::vector<size_t> send_offset(12);
        std::vector<size_t> recv_offset(12);
        send_offset[0] = send.size();
        send_offset[1] = send_map_storage.size();

        // exchange idxmap offset
        size_t* base = (size_t*)thrust::raw_pointer_cast(send_map_storage.data());
        send_offset[2] = (size_t)(send_map.AdetToDetOffset - base);
        send_offset[3] = (size_t)(send_map.BdetToDetOffset - base);
        send_offset[4] = (size_t)(send_map.AdetIndex - base);
        send_offset[5] = (size_t)(send_map.BdetIndex - base);
        send_offset[6] = (size_t)(send_map.AdetToBdetSM - base);
        send_offset[7] = (size_t)(send_map.AdetToDetSM - base);
        send_offset[8] = (size_t)(send_map.BdetToAdetSM - base);
        send_offset[9] = (size_t)(send_map.BdetToDetSM - base);
        send_offset[10] = send_map.size_adet;
        send_offset[11] = send_map.size_bdet;

        MPI_Isend(send_offset.data(),12,SBD_MPI_SIZE_T,mpi_dest,id*4,comm,&req_size[0]);
        MPI_Irecv(recv_offset.data(),12,SBD_MPI_SIZE_T,mpi_source,id*4,comm,&req_size[1]);
        MPI_Waitall(2,req_size.data(),sta_size.data());

        send_size = send_offset[0];
        recv_size = recv_offset[0];
        send_size_map = send_offset[1];
        recv_size_map = recv_offset[1];

        // exchange async
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        if( send_size != 0 ) {
            MPI_Isend((ElemT*)thrust::raw_pointer_cast(send.data()),send_size,DataT,mpi_dest,id*4+2,comm,&req_send);
        }
        if( recv_size != 0 ) {
            recv.resize(recv_size);
            MPI_Irecv((ElemT*)thrust::raw_pointer_cast(recv.data()),recv_size,DataT,mpi_source,id*4+2,comm,&req_recv);
        } else {
            recv = send;
        }

        if( send_size_map != 0 ) {
            MPI_Isend((size_t*)thrust::raw_pointer_cast(send_map_storage.data()),send_size_map,SBD_MPI_SIZE_T,mpi_dest,id*4+3,comm,&req_send_map);
        }
        if( recv_size_map != 0 ) {
            recv_map_storage.resize(recv_size_map);
            MPI_Irecv((size_t*)thrust::raw_pointer_cast(recv_map_storage.data()),recv_size_map,SBD_MPI_SIZE_T,mpi_source,id*4+3,comm,&req_recv_map);

        } else {
            recv_map_storage = send_map_storage;
            recv_offset = send_offset;
        }
        base = (size_t*)thrust::raw_pointer_cast(recv_map_storage.data());
        recv_map.AdetToDetOffset = base + recv_offset[2];
        recv_map.BdetToDetOffset = base + recv_offset[3];
        recv_map.AdetIndex = base + recv_offset[4];
        recv_map.BdetIndex = base + recv_offset[5];
        recv_map.AdetToBdetSM = base + recv_offset[6];
        recv_map.AdetToDetSM = base + recv_offset[7];
        recv_map.BdetToAdetSM = base + recv_offset[8];
        recv_map.BdetToDetSM = base + recv_offset[9];
        recv_map.size_adet = recv_offset[10];
        recv_map.size_bdet = recv_offset[11];
    }

    bool Sync(void)
    {
        bool recv = false;
        if (send_size > 0) {
            MPI_Status st;

            MPI_Wait(&req_send, &st);
        }
        if (recv_size > 0) {
            MPI_Status st;

            MPI_Wait(&req_recv, &st);
            recv = true;
        }
        if (send_size_map > 0) {
            MPI_Status st;

            MPI_Wait(&req_send_map, &st);
        }
        if (recv_size_map > 0) {
            MPI_Status st;

            MPI_Wait(&req_recv_map, &st);
            recv = true;
        }

        send_size = 0;
        recv_size = 0;
        send_size_map = 0;
        recv_size_map = 0;
        return recv;
    }
};




} // end namespace gdb

} // end namespace sbd

#endif
