/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis for Thrust
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_THRUST_H
#define SBD_CHEMISTRY_GDB_HELPER_THRUST_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include <stdexcept>
#include <string>

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
    uint32_t* AdetToDetOffset;
    uint32_t* BdetToDetOffset;
    uint32_t* AdetIndex;
    uint32_t* BdetIndex;
    uint32_t* AdetToBdetSM;
    uint32_t* AdetToDetSM;
    uint32_t* BdetToAdetSM;
    uint32_t* BdetToDetSM;
    uint32_t size_adet = 0;
    uint32_t size_bdet = 0;

    DetIndexMapThrust() {}

    DetIndexMapThrust(const DetIndexMapThrust& other)
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

    DetIndexMapThrust(thrust::device_vector<uint32_t>& storage, const DetIndexMap& idxmap)
    {
        const size_t n_adet = idxmap.AdetToDetLen.size();
        const size_t n_bdet = idxmap.BdetToDetLen.size();

        if (n_adet > UINT32_MAX)
            throw std::overflow_error("GDB Thrust: n_adet (" + std::to_string(n_adet) + ") exceeds uint32_t range");
        if (n_bdet > UINT32_MAX)
            throw std::overflow_error("GDB Thrust: n_bdet (" + std::to_string(n_bdet) + ") exceeds uint32_t range");

        std::vector<uint32_t> offset_adet(n_adet + 1);
        std::vector<uint32_t> offset_bdet(n_bdet + 1);
        for (size_t i = 0; i < n_adet; i++) {
            offset_adet[i] = size_adet;
            if (idxmap.AdetToDetLen[i] > UINT32_MAX - (size_t)size_adet)
                throw std::overflow_error("GDB Thrust: AdetToDet offset overflows uint32_t");
            size_adet += static_cast<uint32_t>(idxmap.AdetToDetLen[i]);
        }
        offset_adet[n_adet] = size_adet;

        for (size_t i = 0; i < n_bdet; i++) {
            offset_bdet[i] = size_bdet;
            if (idxmap.BdetToDetLen[i] > UINT32_MAX - (size_t)size_bdet)
                throw std::overflow_error("GDB Thrust: BdetToDet offset overflows uint32_t");
            size_bdet += static_cast<uint32_t>(idxmap.BdetToDetLen[i]);
        }
        offset_bdet[n_bdet] = size_bdet;

        size_t size = (n_adet + 1) + (n_bdet + 1) + (size_t)size_adet * 3 + (size_t)size_bdet * 3;
        storage.resize(size);
        uint32_t* base_memory = thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        // store offsets
        AdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_adet.begin(), offset_adet.size(), storage.begin() + count);
        count += offset_adet.size();

        BdetToDetOffset = base_memory + count;
        thrust::copy_n(offset_bdet.begin(), offset_bdet.size(), storage.begin() + count);
        count += offset_bdet.size();

        // set Adet, Bdet indices — per-element fill_n, one kernel per segment
        AdetIndex = base_memory + count;
        for (size_t i = 0; i < n_adet; i++) {
            thrust::fill_n(storage.begin() + count + offset_adet[i],
                           idxmap.AdetToDetLen[i], static_cast<uint32_t>(i));
        }
        count += size_adet;

        BdetIndex = base_memory + count;
        for (size_t i = 0; i < n_bdet; i++) {
            thrust::fill_n(storage.begin() + count + offset_bdet[i],
                           idxmap.BdetToDetLen[i], static_cast<uint32_t>(i));
        }
        count += size_bdet;

        // copy SMs: convert size_t CPU data to uint32_t on device
        const size_t sm_size = (size_t)size_adet * 2 + (size_t)size_bdet * 2;
        std::vector<uint32_t> sm_buf(sm_size);
        for (size_t i = 0; i < sm_size; i++) {
            if (idxmap.storage[i] > UINT32_MAX)
                throw std::overflow_error("GDB Thrust: idxmap SM value overflows uint32_t");
            sm_buf[i] = static_cast<uint32_t>(idxmap.storage[i]);
        }
        thrust::copy_n(sm_buf.begin(), sm_buf.size(), storage.begin() + count);
        AdetToDetSM = base_memory + count;
        count += size_adet;
        AdetToBdetSM = base_memory + count;
        count += size_adet;
        BdetToDetSM = base_memory + count;
        count += size_bdet;
        BdetToAdetSM = base_memory + count;
    }

    void copy(thrust::device_vector<uint32_t>& storage, const DetIndexMapThrust& idxmap, const thrust::device_vector<uint32_t>& src_storage)
    {
        storage = src_storage;

        uint32_t* base = thrust::raw_pointer_cast(storage.data());
        const uint32_t* src_base = thrust::raw_pointer_cast(src_storage.data());

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

    inline __device__ __host__ std::pair<bool, uint32_t> adet_lower_bound(uint32_t i_lower, uint32_t i_upper, uint32_t jbst)
    {
        uint32_t is = i_lower;
        uint32_t ie = i_upper;
        while (is != ie) {
            uint32_t i = (is + ie) / 2;
            if (AdetToBdetSM[i] < jbst)
                is = i + 1;
            else
                ie = i;
        }
        return {ie != i_upper && AdetToBdetSM[ie] == jbst, ie};
    }

    inline __device__ __host__ std::pair<bool, uint32_t> bdet_lower_bound(uint32_t i_lower, uint32_t i_upper, uint32_t jast)
    {
        uint32_t is = i_lower;
        uint32_t ie = i_upper;
        while (is != ie) {
            uint32_t i = (is + ie) / 2;
            if (BdetToAdetSM[i] < jast)
                is = i + 1;
            else
                ie = i;
        }
        return {ie != i_upper && BdetToAdetSM[ie] == jast, ie};
    }


};

/**
       Labels connected
*/
class ExcitationLookupThrust
{
public:
    int slide;
    uint32_t* SelfFromAdetOffset;
    uint32_t* SelfFromBdetOffset;
    uint32_t* SinglesFromAdetOffset;
    uint32_t* SinglesFromBdetOffset;
    uint32_t* DoublesFromAdetOffset;
    uint32_t* DoublesFromBdetOffset;
    uint32_t* SelfFromAdetSM;
    uint32_t* SelfFromBdetSM;
    uint32_t* SinglesFromAdetSM;
    int* SinglesAdetCrAnSM;
    uint32_t* SinglesFromBdetSM;
    int* SinglesBdetCrAnSM;
    uint32_t* DoublesFromAdetSM;
    int* DoublesAdetCrAnSM;
    uint32_t* DoublesFromBdetSM;
    int* DoublesBdetCrAnSM;
    uint32_t size_self_adet = 0;
    uint32_t size_self_bdet = 0;
    uint32_t size_single_adet = 0;
    uint32_t size_single_bdet = 0;
    uint32_t size_double_adet = 0;
    uint32_t size_double_bdet = 0;

    ExcitationLookupThrust() {}

    ExcitationLookupThrust(const ExcitationLookupThrust& other)
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

    ExcitationLookupThrust(thrust::device_vector<uint32_t>& storage, thrust::device_vector<int>& cran_storage, const ExcitationLookup& exidx)
    {
        size_t size = 0;

        slide = exidx.slide;

        std::vector<uint32_t> offset_self_adet(exidx.SelfFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromAdetLen.size(); i++) {
            offset_self_adet[i] = size_self_adet;
            if (exidx.SelfFromAdetLen[i] > UINT32_MAX - (size_t)size_self_adet)
                throw std::overflow_error("GDB Thrust: SelfFromAdet offset overflows uint32_t");
            size_self_adet += static_cast<uint32_t>(exidx.SelfFromAdetLen[i]);
            size++;
        }
        offset_self_adet[exidx.SelfFromAdetLen.size()] = size_self_adet;
        size += (size_t)size_self_adet + 1;

        std::vector<uint32_t> offset_self_bdet(exidx.SelfFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SelfFromBdetLen.size(); i++) {
            offset_self_bdet[i] = size_self_bdet;
            if (exidx.SelfFromBdetLen[i] > UINT32_MAX - (size_t)size_self_bdet)
                throw std::overflow_error("GDB Thrust: SelfFromBdet offset overflows uint32_t");
            size_self_bdet += static_cast<uint32_t>(exidx.SelfFromBdetLen[i]);
            size++;
        }
        offset_self_bdet[exidx.SelfFromBdetLen.size()] = size_self_bdet;
        size += (size_t)size_self_bdet + 1;

        std::vector<uint32_t> offset_single_adet(exidx.SinglesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            offset_single_adet[i] = size_single_adet;
            if (exidx.SinglesFromAdetLen[i] > UINT32_MAX - (size_t)size_single_adet)
                throw std::overflow_error("GDB Thrust: SinglesFromAdet offset overflows uint32_t");
            size_single_adet += static_cast<uint32_t>(exidx.SinglesFromAdetLen[i]);
            size++;
        }
        offset_single_adet[exidx.SinglesFromAdetLen.size()] = size_single_adet;
        size += (size_t)size_single_adet + 1;

        std::vector<uint32_t> offset_double_adet(exidx.DoublesFromAdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            offset_double_adet[i] = size_double_adet;
            if (exidx.DoublesFromAdetLen[i] > UINT32_MAX - (size_t)size_double_adet)
                throw std::overflow_error("GDB Thrust: DoublesFromAdet offset overflows uint32_t");
            size_double_adet += static_cast<uint32_t>(exidx.DoublesFromAdetLen[i]);
            size++;
        }
        offset_double_adet[exidx.DoublesFromAdetLen.size()] = size_double_adet;
        size += (size_t)size_double_adet + 1;

        std::vector<uint32_t> offset_single_bdet(exidx.SinglesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            offset_single_bdet[i] = size_single_bdet;
            if (exidx.SinglesFromBdetLen[i] > UINT32_MAX - (size_t)size_single_bdet)
                throw std::overflow_error("GDB Thrust: SinglesFromBdet offset overflows uint32_t");
            size_single_bdet += static_cast<uint32_t>(exidx.SinglesFromBdetLen[i]);
            size++;
        }
        offset_single_bdet[exidx.SinglesFromBdetLen.size()] = size_single_bdet;
        size += (size_t)size_single_bdet + 1;

        std::vector<uint32_t> offset_double_bdet(exidx.DoublesFromBdetLen.size() + 1);
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            offset_double_bdet[i] = size_double_bdet;
            if (exidx.DoublesFromBdetLen[i] > UINT32_MAX - (size_t)size_double_bdet)
                throw std::overflow_error("GDB Thrust: DoublesFromBdet offset overflows uint32_t");
            size_double_bdet += static_cast<uint32_t>(exidx.DoublesFromBdetLen[i]);
            size++;
        }
        offset_double_bdet[exidx.DoublesFromBdetLen.size()] = size_double_bdet;
        size += (size_t)size_double_bdet + 1;

        storage.resize(size);
        uint32_t* base_memory = thrust::raw_pointer_cast(storage.data());

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

        // copy SMs (exidx.storage is size_t; convert to uint32_t with overflow assertions)
        size_t sm_count = size - count;
        std::vector<uint32_t> sm_buf(sm_count);
        for (size_t i = 0; i < sm_count; i++) {
            if (exidx.storage[i] > UINT32_MAX)
                throw std::overflow_error("GDB Thrust: exidx SM value overflows uint32_t");
            sm_buf[i] = static_cast<uint32_t>(exidx.storage[i]);
        }
        thrust::copy_n(sm_buf.begin(), sm_count, storage.begin() + count);
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
        size_t size_cran = (size_t)size_single_adet * 2 + (size_t)size_double_adet * 4 + (size_t)size_single_bdet * 2 + (size_t)size_double_bdet * 4;
        std::vector<int> buf;
        count = 0;

        cran_storage.resize(size_cran);
        int* base_cran = (int*)thrust::raw_pointer_cast(cran_storage.data());

        SinglesAdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_single_adet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromAdetLen[i]; j++) {
                size_t offset = offset_single_adet[i] + j;
                buf[offset] = exidx.SinglesAdetCrAnSM[i][j * 2];
                buf[(size_t)size_single_adet + offset] = exidx.SinglesAdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_single_adet * 2, cran_storage.begin() + count);
        count += (size_t)size_single_adet * 2;


        DoublesAdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_double_adet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromAdetLen[i]; j++) {
                size_t offset = offset_double_adet[i] + j;
                buf[offset] = exidx.DoublesAdetCrAnSM[i][j * 4];
                buf[(size_t)size_double_adet + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 1];
                buf[(size_t)size_double_adet * 2 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 2];
                buf[(size_t)size_double_adet * 3 + offset] = exidx.DoublesAdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_double_adet * 4, cran_storage.begin() + count);
        count += (size_t)size_double_adet * 4;

        SinglesBdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_single_bdet * 2);
#pragma omp parallel for
        for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.SinglesFromBdetLen[i]; j++) {
                size_t offset = offset_single_bdet[i] + j;
                buf[offset] = exidx.SinglesBdetCrAnSM[i][j * 2];
                buf[(size_t)size_single_bdet + offset] = exidx.SinglesBdetCrAnSM[i][j * 2 + 1];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_single_bdet * 2, cran_storage.begin() + count);
        count += (size_t)size_single_bdet * 2;

        DoublesBdetCrAnSM = base_cran + count;
        buf.resize((size_t)size_double_bdet * 4);
#pragma omp parallel for
        for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
            for (int j=0; j <  exidx.DoublesFromBdetLen[i]; j++) {
                size_t offset = offset_double_bdet[i] + j;
                buf[offset] = exidx.DoublesBdetCrAnSM[i][j * 4];
                buf[(size_t)size_double_bdet + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 1];
                buf[(size_t)size_double_bdet * 2 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 2];
                buf[(size_t)size_double_bdet * 3 + offset] = exidx.DoublesBdetCrAnSM[i][j * 4 + 3];
            }
        }
        thrust::copy_n(buf.begin(), (size_t)size_double_bdet * 4, cran_storage.begin() + count);
        count += (size_t)size_double_bdet * 4;
    }


};


void MpiSlide(const DetIndexMapThrust& send_map,
        const thrust::device_vector<uint32_t>& send_storage,
        DetIndexMapThrust& recv_map,
        thrust::device_vector<uint32_t>& recv_storage,
        int slide,
        MPI_Comm comm)
{
    std::vector<size_t> send_offset;
    std::vector<size_t> recv_offset;

    {
        const uint32_t* send_base = thrust::raw_pointer_cast(send_storage.data());
        send_offset.push_back((size_t)(send_map.AdetToDetOffset - send_base));
        send_offset.push_back((size_t)(send_map.BdetToDetOffset - send_base));
        send_offset.push_back((size_t)(send_map.AdetIndex - send_base));
        send_offset.push_back((size_t)(send_map.BdetIndex - send_base));
        send_offset.push_back((size_t)(send_map.AdetToBdetSM - send_base));
        send_offset.push_back((size_t)(send_map.AdetToDetSM - send_base));
        send_offset.push_back((size_t)(send_map.BdetToAdetSM - send_base));
        send_offset.push_back((size_t)(send_map.BdetToDetSM - send_base));
    }
    send_offset.push_back((size_t)send_map.size_adet);
    send_offset.push_back((size_t)send_map.size_bdet);

    sbd::MpiSlide(send_storage, recv_storage, slide, comm);
    sbd::MpiSlide(send_offset, recv_offset, slide, comm);

    uint32_t* base = thrust::raw_pointer_cast(recv_storage.data());
    recv_map.AdetToDetOffset = base + recv_offset[0];
    recv_map.BdetToDetOffset = base + recv_offset[1];
    recv_map.AdetIndex = base + recv_offset[2];
    recv_map.BdetIndex = base + recv_offset[3];
    recv_map.AdetToBdetSM = base + recv_offset[4];
    recv_map.AdetToDetSM = base + recv_offset[5];
    recv_map.BdetToAdetSM = base + recv_offset[6];
    recv_map.BdetToDetSM = base + recv_offset[7];
    recv_map.size_adet = static_cast<uint32_t>(recv_offset[8]);
    recv_map.size_bdet = static_cast<uint32_t>(recv_offset[9]);
}



template <typename ElemT>
class MpiSlider {
protected:
    MPI_Request req_send;
    MPI_Request req_recv;
    MPI_Request req_send_map;
    MPI_Request req_recv_map;
    MPI_Request req_size_send;
    MPI_Request req_size_recv;
    std::vector<size_t> send_offset_buf;
    std::vector<size_t> recv_offset_buf;
    DetIndexMapThrust*  recv_map_ptr;
    uint32_t*           recv_map_base_saved;
    bool have_req_send;
    bool have_req_recv;
    bool have_req_send_map;
    bool have_req_recv_map;
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
        send_offset_buf.resize(12, 0);
        recv_offset_buf.resize(12, 0);
        recv_map_ptr = nullptr;
        recv_map_base_saved = nullptr;
        have_req_send = false;
        have_req_recv = false;
        have_req_send_map = false;
        have_req_recv_map = false;
    }

    // Return the ket/map recv sizes recorded by the most recent ExchangeAsync call.
    // Valid after ExchangeAsync returns and before Sync() clears them.
    size_t get_recv_size()     const { return recv_size; }
    size_t get_recv_size_map() const { return recv_size_map; }

    // ExchangeAsync with caller-managed recv buffer capacity.
    //
    // send_ket_size    — actual number of ket elements in send (the caller
    //                    tracks this separately because send.size() may equal
    //                    global_max_ket_size after pre-allocation).
    // send_map_size    — actual number of map elements in send_map_storage.
    //
    // recv / recv_map_storage — both device_vectors, pre-allocated by the
    //                    caller to global_max_ket_size / global_max_map_size
    //                    respectively and kept at that size throughout the
    //                    task loop.  No resize is performed here; MPI_Irecv
    //                    writes directly into the pre-allocated device memory.
    //
    // The caller is responsible for:
    //   1. Before the first call: resize both recv buffers to the global max
    //      (MPI_Allreduce MAX of local sizes), then cudaDeviceSynchronize()
    //      so the fill kernels complete before MPI_Irecv fires.
    //   2. After Sync() returns true: update the tracked ket/map sizes via
    //      get_recv_size() / get_recv_size_map(), then swap active/recv buffers.
    void ExchangeAsync(const thrust::device_vector<ElemT> &send,
                size_t send_ket_size,
                thrust::device_vector<ElemT> &recv,
                const DetIndexMapThrust& send_map,
                const thrust::device_vector<uint32_t> &send_map_storage,
                size_t send_map_size,
                DetIndexMapThrust& recv_map,
                thrust::device_vector<uint32_t> &recv_map_storage,
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

        // Build the 12-element offset packet to send:
        //   [0]  = send_ket_size  (actual ket elements, not global_max)
        //   [1]  = send_map_size  (actual map elements, not global_max)
        //   [2..9] = recv_map field byte-offsets relative to send_map_storage base
        //   [10] = size_adet, [11] = size_bdet
        // send_offset_buf is a member so its lifetime extends through Sync().
        send_offset_buf[0] = send_ket_size;
        send_offset_buf[1] = send_map_size;
        {
            const uint32_t* send_base = thrust::raw_pointer_cast(send_map_storage.data());
            send_offset_buf[2] = (size_t)(send_map.AdetToDetOffset - send_base);
            send_offset_buf[3] = (size_t)(send_map.BdetToDetOffset - send_base);
            send_offset_buf[4] = (size_t)(send_map.AdetIndex - send_base);
            send_offset_buf[5] = (size_t)(send_map.BdetIndex - send_base);
            send_offset_buf[6] = (size_t)(send_map.AdetToBdetSM - send_base);
            send_offset_buf[7] = (size_t)(send_map.AdetToDetSM - send_base);
            send_offset_buf[8] = (size_t)(send_map.BdetToAdetSM - send_base);
            send_offset_buf[9] = (size_t)(send_map.BdetToDetSM - send_base);
        }
        send_offset_buf[10] = (size_t)send_map.size_adet;
        send_offset_buf[11] = (size_t)send_map.size_bdet;

        // Post offset send and recv non-blocking.  The Waitall is deferred to
        // Sync() so ExchangeAsync returns immediately without stalling on the
        // ring-sender having called ExchangeAsync itself.
        MPI_Isend(send_offset_buf.data(),12,SBD_MPI_SIZE_T,mpi_dest,id*4,comm,&req_size_send);
        MPI_Irecv(recv_offset_buf.data(),12,SBD_MPI_SIZE_T,mpi_source,id*4,comm,&req_size_recv);
        // No MPI_Waitall here — Sync() will wait for the offset and then set
        // recv_size, recv_size_map, and recv_map field pointers.

        send_size     = send_ket_size;
        send_size_map = send_map_size;
        // recv_size and recv_size_map are unknown until Sync() reads recv_offset_buf.

        // Post large ket and map sends/recvs.
        // MPI_Irecv count uses recv.size() / recv_map_storage.size() (= global_max),
        // which is always >= the actual recv_size the sender will send.  MPI allows
        // receive count >= send count; only the sent elements are written.
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        have_req_send = (send_size != 0);
        if (have_req_send) {
            MPI_Isend((ElemT*)thrust::raw_pointer_cast(send.data()),send_size,DataT,mpi_dest,id*4+2,comm,&req_send);
        }
        size_t max_recv_ket = recv.size();
        have_req_recv = (max_recv_ket != 0);
        if (have_req_recv) {
            MPI_Irecv((ElemT*)thrust::raw_pointer_cast(recv.data()),max_recv_ket,DataT,mpi_source,id*4+2,comm,&req_recv);
        }

        have_req_send_map = (send_size_map != 0);
        if (have_req_send_map) {
            MPI_Isend((uint32_t*)thrust::raw_pointer_cast(send_map_storage.data()),send_size_map,MPI_UINT32_T,mpi_dest,id*4+3,comm,&req_send_map);
        }
        size_t max_recv_map = recv_map_storage.size();
        have_req_recv_map = (max_recv_map != 0);
        if (have_req_recv_map) {
            MPI_Irecv((uint32_t*)thrust::raw_pointer_cast(recv_map_storage.data()),max_recv_map,MPI_UINT32_T,mpi_source,id*4+3,comm,&req_recv_map);
        }

        // Save recv_map pointer and base address so Sync() can set up the field
        // pointers after the offset message arrives with the actual offsets.
        recv_map_ptr        = &recv_map;
        recv_map_base_saved = thrust::raw_pointer_cast(recv_map_storage.data());
    }

    // CPU-staging variant of ExchangeAsync.
    //
    // MPI send/recv use CPU-pinned host pointers so the NIC reads/writes CPU
    // DRAM (LPDDR5X) rather than GPU HBM, eliminating bandwidth contention with
    // the compute kernels.  After Sync() the caller copies h_recv/h_recv_map to
    // GPU via cudaMemcpyAsync on a copy_stream, then records a cudaEvent that
    // the compute_stream waits on before launching the next task's kernels.
    //
    // send_gpu_base: thrust::raw_pointer_cast(tidxmap_storage[active_buf].data())
    //   — needed to encode field offsets into the offset packet (same as
    //   ExchangeAsync uses send_map_storage.data() for this).
    // recv_map_gpu_base: thrust::raw_pointer_cast(tidxmap_storage[recv_buf].data())
    //   — Sync() applies received offsets to this GPU base to fix up recv_map
    //   field pointers.  The caller's subsequent cudaMemcpyAsync fills that GPU
    //   storage, making the pointers valid before compute_stream uses them.
    void ExchangeAsyncHost(
                const ElemT*             h_send,
                size_t                   send_ket_size,
                ElemT*                   h_recv,
                size_t                   max_recv_ket,
                const DetIndexMapThrust& send_map,
                const uint32_t*          send_gpu_base,
                const uint32_t*          h_send_map,
                size_t                   send_map_size,
                DetIndexMapThrust&       recv_map,
                uint32_t*                h_recv_map,
                size_t                   max_recv_map,
                uint32_t*                recv_map_gpu_base,
                int slide,
                MPI_Comm comm,
                int id)
    {
        int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
        int mpi_size; MPI_Comm_size(comm, &mpi_size);
        int mpi_dest   = (mpi_size + mpi_rank + slide) % mpi_size;
        int mpi_source = (mpi_size + mpi_rank - slide) % mpi_size;

        // Offset packet encoding — identical to ExchangeAsync, using the GPU
        // base pointer for field-offset arithmetic (layout is the same).
        send_offset_buf[0]  = send_ket_size;
        send_offset_buf[1]  = send_map_size;
        send_offset_buf[2]  = (size_t)(send_map.AdetToDetOffset - send_gpu_base);
        send_offset_buf[3]  = (size_t)(send_map.BdetToDetOffset - send_gpu_base);
        send_offset_buf[4]  = (size_t)(send_map.AdetIndex       - send_gpu_base);
        send_offset_buf[5]  = (size_t)(send_map.BdetIndex       - send_gpu_base);
        send_offset_buf[6]  = (size_t)(send_map.AdetToBdetSM    - send_gpu_base);
        send_offset_buf[7]  = (size_t)(send_map.AdetToDetSM     - send_gpu_base);
        send_offset_buf[8]  = (size_t)(send_map.BdetToAdetSM    - send_gpu_base);
        send_offset_buf[9]  = (size_t)(send_map.BdetToDetSM     - send_gpu_base);
        send_offset_buf[10] = (size_t)send_map.size_adet;
        send_offset_buf[11] = (size_t)send_map.size_bdet;

        MPI_Isend(send_offset_buf.data(), 12, SBD_MPI_SIZE_T, mpi_dest,   id*4, comm, &req_size_send);
        MPI_Irecv(recv_offset_buf.data(), 12, SBD_MPI_SIZE_T, mpi_source, id*4, comm, &req_size_recv);

        send_size     = send_ket_size;
        send_size_map = send_map_size;

        // Data send/recv via host pointers — NIC reads/writes CPU DRAM only.
        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        have_req_send = (send_size != 0);
        if (have_req_send)
            MPI_Isend(const_cast<ElemT*>(h_send), static_cast<int>(send_size),
                      DataT, mpi_dest, id*4+2, comm, &req_send);
        have_req_recv = (max_recv_ket != 0);
        if (have_req_recv)
            MPI_Irecv(h_recv, static_cast<int>(max_recv_ket),
                      DataT, mpi_source, id*4+2, comm, &req_recv);

        have_req_send_map = (send_size_map != 0);
        if (have_req_send_map)
            MPI_Isend(const_cast<uint32_t*>(h_send_map), static_cast<int>(send_size_map),
                      MPI_UINT32_T, mpi_dest, id*4+3, comm, &req_send_map);
        have_req_recv_map = (max_recv_map != 0);
        if (have_req_recv_map)
            MPI_Irecv(h_recv_map, static_cast<int>(max_recv_map),
                      MPI_UINT32_T, mpi_source, id*4+3, comm, &req_recv_map);

        recv_map_ptr        = &recv_map;
        recv_map_base_saved = recv_map_gpu_base;
    }

    bool Sync(void)
    {
        // Wait for the offset message from the ring-sender.  Previously this
        // Waitall was inside ExchangeAsync, blocking before GPU kernels started.
        // Moving it here overlaps the offset handshake with GPU kernel execution.
        {
            MPI_Status st;
            MPI_Wait(&req_size_recv, &st);
        }
        recv_size     = recv_offset_buf[0];
        recv_size_map = recv_offset_buf[1];

        // Set up recv_map field pointers from the received offsets.
        if (recv_map_ptr) {
            uint32_t* base = recv_map_base_saved;
            recv_map_ptr->AdetToDetOffset = base + recv_offset_buf[2];
            recv_map_ptr->BdetToDetOffset = base + recv_offset_buf[3];
            recv_map_ptr->AdetIndex       = base + recv_offset_buf[4];
            recv_map_ptr->BdetIndex       = base + recv_offset_buf[5];
            recv_map_ptr->AdetToBdetSM    = base + recv_offset_buf[6];
            recv_map_ptr->AdetToDetSM     = base + recv_offset_buf[7];
            recv_map_ptr->BdetToAdetSM    = base + recv_offset_buf[8];
            recv_map_ptr->BdetToDetSM     = base + recv_offset_buf[9];
            recv_map_ptr->size_adet       = static_cast<uint32_t>(recv_offset_buf[10]);
            recv_map_ptr->size_bdet       = static_cast<uint32_t>(recv_offset_buf[11]);
            recv_map_ptr        = nullptr;
            recv_map_base_saved = nullptr;
        }

        if (have_req_send) {
            MPI_Status st;
            MPI_Wait(&req_send, &st);
            have_req_send = false;
        }
        if (have_req_recv) {
            MPI_Status st;
            MPI_Wait(&req_recv, &st);
            have_req_recv = false;
        }
        if (have_req_send_map) {
            MPI_Status st;
            MPI_Wait(&req_send_map, &st);
            have_req_send_map = false;
        }
        if (have_req_recv_map) {
            MPI_Status st;
            MPI_Wait(&req_recv_map, &st);
            have_req_recv_map = false;
        }
        // Wait for offset send last — send_offset_buf must stay alive until here.
        {
            MPI_Status st;
            MPI_Wait(&req_size_send, &st);
        }

        send_size = 0;
        send_size_map = 0;
        // recv_size and recv_size_map remain valid for get_recv_size() /
        // get_recv_size_map() calls after Sync() returns.
        return (recv_size > 0);
    }
};




} // end namespace gdb

} // end namespace sbd

#endif
