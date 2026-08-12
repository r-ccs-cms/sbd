/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_THRUST_H
#define SBD_CHEMISTRY_TPB_MULT_THRUST_H

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#ifdef SBD_USE_NCCL
#include <nccl.h>
#endif

#include "sbd/framework/nvtx.h"

// per thread DetI, DetJ storage size (1GB max)
#define MAX_DET_SIZE 134217728

namespace sbd
{

template <typename ElemT>
class MultTPBThrust : public MultBase<ElemT> {
public:
    thrust::device_vector<size_t> adets;
    thrust::device_vector<size_t> bdets;
    thrust::device_vector<size_t> dets;
    size_t adets_size;
    size_t bdets_size;
    size_t dets_size;
    size_t bra_adets_begin;
    size_t bra_adets_end;
    size_t bra_bdets_begin;
    size_t bra_bdets_end;
    size_t ket_adets_begin;
    size_t ket_adets_end;
    size_t ket_bdets_begin;
    size_t ket_bdets_end;
    ElemT I0;
    oneInt_Thrust<ElemT> I1;
    thrust::device_vector<ElemT> I1_store;
    twoInt_Thrust<ElemT> I2;
    thrust::device_vector<ElemT> I2_store;
    thrust::device_vector<ElemT> I2_dm;
    thrust::device_vector<ElemT> I2_em;
    std::vector<thrust::device_vector<size_t>> helper_storage;
    std::vector<thrust::device_vector<int>> CrAn_storage;
    size_t num_max_threads;
    std::vector<TaskHelpersThrust<ElemT>> helper;
    bool use_precalculated_dets = true;
    bool collapse_loop = true;
    int max_memory_gb_for_determinants;
    size_t adet_comm_size;
    size_t bdet_comm_size;

    MultTPBThrust() {}

    void Init(
        const det_vector<size_t, det_kind::half> &adets_in,
        const det_vector<size_t, det_kind::half> &bdets_in,
        const uint32_t bit_length_in,
        const size_t norbs_in,
        const size_t adet_comm_size_in,
        const size_t bdet_comm_size_in,
        const std::vector<TaskHelpers> &helper_in,
        const ElemT &I0_in,
        const oneInt<ElemT> &I1_in,
        const twoInt<ElemT> &I2_in,
        MPI_Comm h_comm_in,
        MPI_Comm b_comm_in,
        MPI_Comm t_comm_in,
        MPI_Comm a_comm_in,
#ifdef SBD_USE_NCCL
        ncclComm_t h_nccl_comm,
        ncclComm_t b_nccl_comm,
        ncclComm_t t_nccl_comm,
        ncclComm_t a_nccl_comm,
#endif
        bool use_pre_dets,
        int max_gb_dets,
        bool collapse);

    void UpdateDet(size_t task);

    void run(const thrust::device_vector<ElemT> &hii,
                    const thrust::device_vector<ElemT> &Wk,
                    thrust::device_vector<ElemT> &Wb) override;

    void makeQChamDiagTerms(thrust::device_vector<ElemT> &hii);

    void correlation(const std::vector<ElemT> & w,
                     std::vector<std::vector<ElemT>> & onebody_out,
                     std::vector<std::vector<ElemT>> & twobody_out);
};

// contructor for Mult data
template <typename ElemT>
void MultTPBThrust<ElemT>::Init(
    const det_vector<size_t, det_kind::half> &adets_in,
    const det_vector<size_t, det_kind::half> &bdets_in,
    const uint32_t bit_length_in,
    const size_t norbs_in,
    const size_t adet_comm_size_in,
    const size_t bdet_comm_size_in,
    const std::vector<TaskHelpers> &helper_in,
    const ElemT &I0_in,
    const oneInt<ElemT> &I1_in,
    const twoInt<ElemT> &I2_in,
    MPI_Comm h_comm_in,
    MPI_Comm b_comm_in,
    MPI_Comm t_comm_in,
    MPI_Comm a_comm_in,
#ifdef SBD_USE_NCCL
    ncclComm_t h_nccl_comm,
    ncclComm_t b_nccl_comm,
    ncclComm_t t_nccl_comm,
    ncclComm_t a_nccl_comm,
#endif
    bool use_pre_dets,
    int max_gb_dets,
    bool collapse)
{
    SBD_NVTX_RANGE_COLOR("Init", __LINE__);

#ifdef SBD_USE_RANK_DISTRIBUTION
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
    printf("[%s,%d] Block rank distribution enabled\n", __FILE__, __LINE__);
#else
    printf("[%s,%d] Cyclic (strided) rank distribution enabled\n", __FILE__, __LINE__);
#endif
#ifdef SBD_USE_VECTORIZATION
    printf("[%s,%d] Vectorization enabled\n", __FILE__, __LINE__);
#endif
#endif

#ifdef SBD_USE_32BIT_PARITY
    printf("[%s,%d] 32-bit version of parity used (bit_length = %u)\n",
           __FILE__, __LINE__, bit_length_in);
    if (bit_length_in > 32) {
        fprintf(stderr, "[ERROR] bit_length is too large for 32-bit version\n");
        exit(-1);
    }
#endif

    this->bit_length_ = bit_length_in;
    this->norbs_ = norbs_in;
    this->D_size_ = (2 * norbs_in + bit_length_in - 1) / bit_length_in;
    this->D_half_size_ = (norbs_in + bit_length_in - 1) / bit_length_in;

    this->h_comm_ = h_comm_in;
    this->b_comm_ = b_comm_in;
    this->t_comm_ = t_comm_in;
    this->a_comm_ = a_comm_in;
#ifdef SBD_USE_NCCL
    this->h_nccl_comm_ = h_nccl_comm;
    this->b_nccl_comm_ = b_nccl_comm;
    this->t_nccl_comm_ = t_nccl_comm;
    this->a_nccl_comm_ = a_nccl_comm;
#endif

    adet_comm_size = adet_comm_size_in;
    bdet_comm_size = bdet_comm_size_in;

    I0 = I0_in;
    // copyin I1
    I1_store.resize(I1_in.store.size());
    thrust::copy_n(I1_in.store.begin(), I1_in.store.size(), I1_store.begin());
    I1 = oneInt_Thrust<ElemT>(I1_store, I1_in.norbs);

    // copyin I2
    I2_store.resize(I2_in.store.size());
    thrust::copy_n(I2_in.store.begin(), I2_in.store.size(), I2_store.begin());
    I2_dm.resize(I2_in.DirectMat.size());
    thrust::copy_n(I2_in.DirectMat.begin(), I2_in.DirectMat.size(), I2_dm.begin());
    I2_em.resize(I2_in.ExchangeMat.size());
    thrust::copy_n(I2_in.ExchangeMat.begin(), I2_in.ExchangeMat.size(), I2_em.begin());
    I2 = twoInt_Thrust<ElemT>(I2_store, I2_in.norbs, I2_dm, I2_em, I2_in.zero, I2_in.maxEntry);

    adets_size = 0;
    bdets_size = 0;
    bra_adets_begin = 0;
    bra_adets_end = 0;
    bra_bdets_begin = 0;
    bra_bdets_end = 0;
    ket_adets_begin = 0;
    ket_adets_end = 0;
    ket_bdets_begin = 0;
    ket_bdets_end = 0;

    use_precalculated_dets = use_pre_dets;
    max_memory_gb_for_determinants = max_gb_dets;
    collapse_loop = collapse;

    // copyin helpers
    helper.clear();
    helper_storage.resize(helper_in.size());
    CrAn_storage.resize(helper_in.size());
    for (size_t task = 0; task < helper_in.size(); task++) {
        helper.push_back(TaskHelpersThrust<ElemT>(helper_storage[task], CrAn_storage[task], helper_in[task], (!use_precalculated_dets) || (!collapse_loop)));

        adets_size = std::max(adets_size, helper[task].braAlphaEnd - helper[task].braAlphaStart);
        bdets_size = std::max(bdets_size, helper[task].braBetaEnd - helper[task].braBetaStart);
    }

    // copyin adets, bdets: staged through a host buffer for one bulk H2D transfer.
    adets.resize(this->D_half_size_ * adets_in.size());
    bdets.resize(this->D_half_size_ * bdets_in.size());
    {
        std::vector<size_t> buf(this->D_half_size_ * adets_in.size());
#pragma omp parallel for
        for (size_t i = 0; i < adets_in.size(); i++) {
            std::copy_n(adets_in[i].begin(), this->D_half_size_, buf.begin() + i * this->D_half_size_);
        }
        thrust::copy_n(buf.begin(), buf.size(), adets.begin());

        buf.resize(this->D_half_size_ * bdets_in.size());
#pragma omp parallel for
        for (size_t i = 0; i < bdets_in.size(); i++) {
            std::copy_n(bdets_in[i].begin(), this->D_half_size_, buf.begin() + i * this->D_half_size_);
        }
        thrust::copy_n(buf.begin(), buf.size(), bdets.begin());
    }

    dets_size = 0;
    if (use_precalculated_dets) {
        // allocate pre-calculated DetI
        dets.resize(this->D_size_ * adets_size * bdets_size);

        num_max_threads = 0;

        // check overflow
        if (collapse_loop) {
            for (size_t task = 0; task < helper.size(); task++) {
                size_t braAlphaSize = helper[task].braAlphaEnd - helper[task].braAlphaStart;
                size_t braBetaSize = helper[task].braBetaEnd - helper[task].braBetaStart;
                if (helper[task].taskType == 2) {
                    size_t s = helper[task].size_single_alpha * braBetaSize;
                    if (s / helper[task].size_single_alpha != braBetaSize) {
                        collapse_loop = false;
                        break;
                    }
                    s = helper[task].size_double_alpha * braBetaSize;
                    if (s / helper[task].size_double_alpha != braBetaSize) {
                        collapse_loop = false;
                        break;
                    }
                } else if (helper[task].taskType == 1) {
                    size_t s = helper[task].size_single_beta * braAlphaSize;
                    if (s / helper[task].size_single_beta != braAlphaSize) {
                        collapse_loop = false;
                        break;
                    }
                    s = helper[task].size_double_beta * braAlphaSize;
                    if (s / helper[task].size_double_beta != braAlphaSize) {
                        collapse_loop = false;
                        break;
                    }
                } else {
                    size_t s = helper[task].size_single_alpha * helper[task].size_single_beta;
                    if (s / helper[task].size_single_alpha != helper[task].size_single_beta) {
                        collapse_loop = false;
                        break;
                    }
                }
            }
        }
    } else {
        for (size_t task = 0; task < helper.size(); task++) {
            size_t braAlphaSize = helper[task].braAlphaEnd - helper[task].braAlphaStart;
            size_t braBetaSize = helper[task].braBetaEnd - helper[task].braBetaStart;
            dets_size = std::max(dets_size, braAlphaSize * braBetaSize);
        }
        num_max_threads = dets_size;

        // number of max threads, this is enabled when per thread DetI and DetJ storage is used (non-pre calculate)
        if (max_memory_gb_for_determinants > 0) {
            if (dets_size * this->D_size_ * sizeof(size_t) > (size_t)max_memory_gb_for_determinants * 1024 * 1024 * 1024) {
                dets_size = ((size_t)max_memory_gb_for_determinants * 1024 * 1024 * 1024 / (this->D_size_ * sizeof(size_t))) & (~1023ULL);
                num_max_threads = dets_size;
            }
        }
        // allocate per thread storage for DetI
        dets.resize(this->D_size_ * dets_size);
    }
}


template <typename ElemT>
class MultKernelBase : public DeterminantKernels<ElemT> {
protected:
    ElemT *Wb;
    ElemT* T;
    size_t adets_size;
    size_t bdets_size;
    size_t mpi_rank_h;
    size_t mpi_size_h;
    size_t* adets;
    size_t* bdets;
    size_t* det_I;
public:
    MultKernelBase() {}

    MultKernelBase( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0, data.I1, data.I2)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        T = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
        det_I = (size_t*)thrust::raw_pointer_cast(data.dets.data());

        adets_size = data.adets_size;
        bdets_size = data.bdets_size;
    }

    MultKernelBase(const MultTPBThrust<ElemT>& data)
                 : DeterminantKernels<ElemT>(data.bit_length(), data.norbs(), data.I0, data.I1, data.I2)
    {
        adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
        det_I = (size_t*)thrust::raw_pointer_cast(data.dets.data());

        adets_size = data.adets_size;
        bdets_size = data.bdets_size;
    }

    void set_mpi_size(size_t h_rank, size_t h_size)
    {
        mpi_rank_h = h_rank;
        mpi_size_h = h_size;
    }
};

template <typename ElemT>
class DetFromAlphaBetaKernel : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    bool update_I;
public:
    DetFromAlphaBetaKernel(const TaskHelpersThrust<ElemT>& h, const MultTPBThrust<ElemT>& data, bool i)
                        : MultKernelBase<ElemT>(data)
    {
        helper = h;
        update_I = i;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t a = i / this->bdets_size;
        size_t b = i - a * this->bdets_size;
        size_t* Det;
        if (update_I) {
            Det = this->det_I + i * this->D_size;
            size_t ia = a + helper.braAlphaStart;
            size_t ib = b + helper.braBetaStart;

            if (ia < helper.braAlphaEnd && ib < helper.braBetaEnd)
                this->DetFromAlphaBeta(Det, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
        }
    }
};


template <typename ElemT>
void MultTPBThrust<ElemT>::UpdateDet(size_t task)
{
    // precalculate DetI (if update needed)
    bool update_I =  bra_adets_begin != helper[task].braAlphaStart || bra_bdets_begin != helper[task].braBetaStart ||
                    bra_adets_end != helper[task].braAlphaEnd || bra_bdets_end != helper[task].braBetaEnd;
    if (update_I) {
        bra_adets_begin = helper[task].braAlphaStart;
        bra_bdets_begin = helper[task].braBetaStart;
        bra_adets_end = helper[task].braAlphaEnd;
        bra_bdets_end = helper[task].braBetaEnd;
        ket_adets_begin = helper[task].ketAlphaStart;
        ket_bdets_begin = helper[task].ketBetaStart;
        ket_adets_end = helper[task].ketAlphaEnd;
        ket_bdets_end = helper[task].ketBetaEnd;

        DetFromAlphaBetaKernel det_kernel(helper[task], *this, update_I);
        auto det_ci = thrust::counting_iterator<size_t>(0);
        thrust::for_each_n(thrust::device, det_ci, adets_size * bdets_size, det_kernel);
    }
}

template <typename ElemT>
class MultAlphaBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultAlphaBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    __device__ inline void loop_body(size_t i, int64_t& braIdx, ElemT& eij) {
        braIdx = -1;
        eij = 0;
        size_t j = i / helper.size_single_beta;  // 0 .. size_single_alpha-1
        size_t k = i % helper.size_single_beta;  // 0 .. size_single_beta-1
        if (j >= helper.size_single_alpha) return;
        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];
        braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) +
                  ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        eij = this->TwoExcite(
            DetI,
            helper.SinglesAlphaCrAnSM[j],
            helper.SinglesBetaCrAnSM[k],
            helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha],
            helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
        if (eij != 0) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                           + jb - helper.ketBetaStart;
            eij *= this->T[ketIdx];
        }
    }        

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        int64_t braIdx;
        ElemT eij;
        loop_body(i, braIdx, eij);
        if (braIdx < 0 || eij == 0) return;
        atomicAdd(this->Wb + braIdx, eij);
    }
};


// NOTE: "Vec" here refers to processing multiple elements per thread (thread
// coarsening), not SIMD vectorization. Memory accesses are not contiguous and
// no vector instructions are used. The main purpose is to increase work per
// thread and reduce the number of global atomic operations, as adjacent
// elements often update the same memory locations and can be accumulated within
// a thread before issuing a single atomic update.
template <typename ElemT, int VecLen = 2>
class MultAlphaBeta_Vec : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultAlphaBeta_Vec(const TaskHelpersThrust<ElemT>& h,
                      const thrust::device_vector<ElemT>& v_wb,
                      const thrust::device_vector<ElemT>& v_t,
                      const MultTPBThrust<ElemT>& data
        ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    // NOTE: loop_body is intentionally duplicated in both MultAlphaBeta and
    // MultAlphaBeta_Vec. While it would be cleaner to share the implementation
    // via inheritance, doing so resulted in noticeable performance degradation,
    // so the duplication is kept to preserve performance.
    __device__ inline void loop_body(size_t i, int64_t& braIdx, ElemT& eij) {
        braIdx = -1;
        eij = 0;
        size_t j = i / helper.size_single_beta;  // 0 .. size_single_alpha-1
        size_t k = i % helper.size_single_beta;  // 0 .. size_single_beta-1
        if (j >= helper.size_single_alpha) return;
        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];
        braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) +
                  ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        eij = this->TwoExcite(
            DetI,
            helper.SinglesAlphaCrAnSM[j],
            helper.SinglesBetaCrAnSM[k],
            helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha],
            helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
        if (eij != 0) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                           + jb - helper.ketBetaStart;
            eij *= this->T[ketIdx];
        }
    }        

    // kernel entry point
    __device__ __host__ void operator()(size_t i0)
    {
        size_t i = (VecLen * i0);
        int64_t braIdx;
        ElemT eij;
        loop_body(i, braIdx, eij);
        for (size_t i1 = 1; i1 < VecLen; i1++) {
            i = (VecLen * i0) + i1;
            int64_t braIdx_new;
            ElemT eij_new;
            loop_body(i, braIdx_new, eij_new);
            if (braIdx == braIdx_new) {
                if (braIdx >= 0) {
                    eij += eij_new;
                }
                continue;
            }
            if ((braIdx >= 0) && (eij != 0)) {
                atomicAdd(this->Wb + braIdx, eij);
            }
            braIdx = braIdx_new;
            eij = eij_new;
        }
        if ((braIdx >= 0) && (eij != 0)) {
            atomicAdd(this->Wb + braIdx, eij);
        }
    }
};


template <typename ElemT>
class MultSingleAlpha : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultSingleAlpha(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t k = i / helper.size_single_alpha;
        size_t j = i - k * helper.size_single_alpha;

        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = k + helper.braBetaStart;
        size_t jb = ib;

        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                       + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;

        ElemT eij = this->OneExcite(DetI, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
        if (eij == 0) return;

        atomicAdd(this->Wb + braIdx, eij * this->T[ketIdx]);
    }
};

template <typename ElemT>
class MultDoubleAlpha : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultDoubleAlpha(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t k = i / helper.size_double_alpha;
        size_t j = i - k * helper.size_double_alpha;

        size_t ia = helper.DoublesFromAlphaBraIndex[j];
        size_t ja = helper.DoublesFromAlphaKetIndex[j];
        size_t ib = k + helper.braBetaStart;
        size_t jb = ib;

        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                       + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;

        ElemT eij = this->TwoExcite(DetI,
                                    helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
        if (eij == 0) return;

        atomicAdd(this->Wb + braIdx, eij * this->T[ketIdx]);
    }
};

template <typename ElemT>
class MultSingleBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultSingleBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = i / helper.size_single_beta;
        size_t k = i - j * helper.size_single_beta;

        size_t ia = j + helper.braAlphaStart;
        size_t ja = ia;
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];

        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                       + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;

        ElemT eij = this->OneExcite(DetI, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
        if (eij == 0) return;

        atomicAdd(this->Wb + braIdx, eij * this->T[ketIdx]);
    }
};

template <typename ElemT>
class MultDoubleBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultDoubleBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = i / helper.size_double_beta;
        size_t k = i - j * helper.size_double_beta;

        size_t ia = j + helper.braAlphaStart;
        size_t ja = ia;
        size_t ib = helper.DoublesFromBetaBraIndex[k];
        size_t jb = helper.DoublesFromBetaKetIndex[k];
        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                       + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;

        ElemT eij = this->TwoExcite(DetI,
                                    helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                    helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
        if (eij == 0) return;

        atomicAdd(this->Wb + braIdx, eij * this->T[ketIdx]);
    }
};

#define SB_MULT_ALPHA 0
#define SB_MULT_BETA  1
#define SB_MULT_SINGLE 0
#define SB_MULT_DOUBLE 1

// MultUnified unifies multiple kernel variants into a single template-based
// implementation, allowing shared application of optimizations (e.g., thread
// coarsening) and avoiding code duplication across individual kernels.
template <typename ElemT, int AlphaOrBeta, int SingleOrDouble>
class MultUnified : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultUnified(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data
        ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }

    __device__ inline void loop_body(size_t i, int64_t& braIdx, ElemT& eij) {
        braIdx = -1;
        eij = 0;
        size_t k;
        size_t j;
        size_t ia;
        size_t ja;
        size_t ib;
        size_t jb;
        if (AlphaOrBeta == SB_MULT_ALPHA) {
            size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleAlpha
                k = i / helper.size_single_alpha;
                j = i % helper.size_single_alpha;
                ia = helper.SinglesFromAlphaBraIndex[j];
                ja = helper.SinglesFromAlphaKetIndex[j];
            } else {
                // DoubleAlpha
                k = i / helper.size_double_alpha;
                j = i % helper.size_double_alpha;
                ia = helper.DoublesFromAlphaBraIndex[j];
                ja = helper.DoublesFromAlphaKetIndex[j];
            }
            ib = k + helper.braBetaStart;
            jb = ib;
            if (k >= braBetaSize) return;
        } else {
            size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleBeta
                j = i / helper.size_single_beta;
                k = i % helper.size_single_beta;
                ib = helper.SinglesFromBetaBraIndex[k];
                jb = helper.SinglesFromBetaKetIndex[k];
            } else {
                // DoubleBeta
                j = i / helper.size_double_beta;
                k = i % helper.size_double_beta;
                ib = helper.DoublesFromBetaBraIndex[k];
                jb = helper.DoublesFromBetaKetIndex[k];
            }
            ia = j + helper.braAlphaStart;
            ja = ia;
            if (j >= braAlphaSize) return;
        }

        braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart)
                + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        if (AlphaOrBeta == SB_MULT_ALPHA) {
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleAlpha
                eij = this->OneExcite(
                    DetI,
                    helper.SinglesAlphaCrAnSM[j],
                    helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            } else {
                // DoubleAlpha
                eij = this->TwoExcite(
                    DetI,
                    helper.DoublesAlphaCrAnSM[j],
                    helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha],
                    helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
            }
        } else {
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleBeta
                eij = this->OneExcite(
                    DetI,
                    helper.SinglesBetaCrAnSM[k],
                    helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            } else {
                // DoubleBeta
                eij = this->TwoExcite(
                    DetI,
                    helper.DoublesBetaCrAnSM[k],
                    helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                    helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta],
                    helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
            }
        }
        if (eij != 0) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                           + jb - helper.ketBetaStart;
            eij *= this->T[ketIdx];
        }
    }
    
    // kernel entry point
    __device__ void operator()(size_t i)
    {
        int64_t braIdx;
        ElemT eij;
        loop_body(i, braIdx, eij);
        if ((braIdx < 0) || (eij == 0)) return;
        atomicAdd(this->Wb + braIdx, eij);
    }
};

// MultUnified_Vec extends MultUnified by applying thread coarsening (multiple
// elements per thread). This is not SIMD vectorization; memory accesses are not
// contiguous. The goal is to reduce global atomic operations by accumulating
// contributions within a thread before issuing atomic updates.
template <typename ElemT, int AlphaOrBeta, int SingleOrDouble, int VecLen = 2>
class MultUnified_Vec : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MultUnified_Vec(const TaskHelpersThrust<ElemT>& h,
                    const thrust::device_vector<ElemT>& v_wb,
                    const thrust::device_vector<ElemT>& v_t,
                    const MultTPBThrust<ElemT>& data
        ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
    }
    
    // NOTE: loop_body is intentionally duplicated in both MultUnified and
    // MultUnified_Vec. While it would be cleaner to share the implementation
    // via inheritance, doing so resulted in noticeable performance degradation,
    // so the duplication is kept to preserve performance.
    __device__ inline void loop_body(size_t i, int64_t& braIdx, ElemT& eij) {
        braIdx = -1;
        eij = 0;
        size_t k;
        size_t j;
        size_t ia;
        size_t ja;
        size_t ib;
        size_t jb;
        if (AlphaOrBeta == SB_MULT_ALPHA) {
            size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleAlpha
                k = i / helper.size_single_alpha;
                j = i % helper.size_single_alpha;
                ia = helper.SinglesFromAlphaBraIndex[j];
                ja = helper.SinglesFromAlphaKetIndex[j];
            } else {
                // DoubleAlpha
                k = i / helper.size_double_alpha;
                j = i % helper.size_double_alpha;
                ia = helper.DoublesFromAlphaBraIndex[j];
                ja = helper.DoublesFromAlphaKetIndex[j];
            }
            ib = k + helper.braBetaStart;
            jb = ib;
            if (k >= braBetaSize) return;
        } else {
            size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleBeta
                j = i / helper.size_single_beta;
                k = i % helper.size_single_beta;
                ib = helper.SinglesFromBetaBraIndex[k];
                jb = helper.SinglesFromBetaKetIndex[k];
            } else {
                // DoubleBeta
                j = i / helper.size_double_beta;
                k = i % helper.size_double_beta;
                ib = helper.DoublesFromBetaBraIndex[k];
                jb = helper.DoublesFromBetaKetIndex[k];
            }
            ia = j + helper.braAlphaStart;
            ja = ia;
            if (j >= braAlphaSize) return;
        }

        braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart)
                + ib - helper.braBetaStart;
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        if (AlphaOrBeta == SB_MULT_ALPHA) {
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleAlpha
                eij = this->OneExcite(
                    DetI,
                    helper.SinglesAlphaCrAnSM[j],
                    helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            } else {
                // DoubleAlpha
                eij = this->TwoExcite(
                    DetI,
                    helper.DoublesAlphaCrAnSM[j],
                    helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha],
                    helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
            }
        } else {
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleBeta
                eij = this->OneExcite(
                    DetI,
                    helper.SinglesBetaCrAnSM[k],
                    helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            } else {
                // DoubleBeta
                eij = this->TwoExcite(
                    DetI,
                    helper.DoublesBetaCrAnSM[k],
                    helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                    helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta],
                    helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
            }
        }
        if (eij != 0) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                           + jb - helper.ketBetaStart;
            eij *= this->T[ketIdx];
        }
    }
    
    // kernel entry point
    __device__ void operator()(size_t i0)
    {
        size_t i = (VecLen * i0);
        int64_t braIdx;
        ElemT eij;
        loop_body(i, braIdx, eij);
        for (size_t i1 = 1; i1 < VecLen; i1++) {
            i = (VecLen * i0) + i1;
            int64_t braIdx_new;
            ElemT eij_new;
            loop_body(i, braIdx_new, eij_new);
            if (braIdx == braIdx_new) {
                if (braIdx >= 0) {
                    eij += eij_new;
                }
                continue;
            }
            if ((braIdx >= 0) && (eij != 0)) {
                atomicAdd(this->Wb + braIdx, eij);
            }
            braIdx = braIdx_new;
            eij = eij_new;
        }
        if ((braIdx >= 0) && (eij != 0)) {
            atomicAdd(this->Wb + braIdx, eij);
        }
    }
};


template <typename ElemT>
class MultTask0 : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    MultTask0(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data, size_t o,
                bool pre) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
        use_pre_dets = pre;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        ElemT e = 0.0;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                    size_t jb = helper.SinglesFromBetaKetIndex[k];
                    size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                    + jb - helper.ketBetaStart;
                    ElemT eij = this->TwoExcite(DetI,
                                        helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                        helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
                    e += eij * this->T[ketIdx];
                }
            }
            this->Wb[braIdx] += e;
        }
    }
};

template <typename ElemT>
class MultTask1 : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    MultTask1(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data, size_t o,
                bool pre) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
        use_pre_dets = pre;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        ElemT e = 0.0;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }

            for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                size_t ja = ia;
                size_t jb = helper.SinglesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT eij = this->OneExcite(DetI, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
                e += eij * this->T[ketIdx];
            }
            for (size_t k = helper.DoublesFromBetaOffset[b]; k < helper.DoublesFromBetaOffset[b + 1]; k++) {
                size_t ja = ia;
                size_t jb = helper.DoublesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT eij = this->TwoExcite(DetI,
                                    helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                    helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
                e += eij * this->T[ketIdx];
            }
            this->Wb[braIdx] += e;
        }
    }
};

template <typename ElemT>
class MultTask2 : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    MultTask2(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data, size_t o,
                bool pre) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
        use_pre_dets = pre;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        ElemT e = 0.0;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT eij = this->OneExcite(DetI, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
                e += eij * this->T[ketIdx];
            }
            for (size_t j = helper.DoublesFromAlphaOffset[a]; j < helper.DoublesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.DoublesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT eij = this->TwoExcite(DetI,
                                    helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
                e += eij * this->T[ketIdx];
            }
            this->Wb[braIdx] += e;
        }
    }
};


// kernel for Wb initialization
template <typename ElemT>
class Wb_init_kernel {
protected:
    ElemT* Wb;
    ElemT* hii;
    ElemT* T;
public:
    Wb_init_kernel(thrust::device_vector<ElemT>& Wb_in, const thrust::device_vector<ElemT>& hii_in, const thrust::device_vector<ElemT>& T_in)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(Wb_in.data());
        hii = (ElemT*)thrust::raw_pointer_cast(hii_in.data());
        T = (ElemT*)thrust::raw_pointer_cast(T_in.data());
    }
    __host__ __device__ void operator()(size_t i)
    {
        Wb[i] += hii[i] * T[i];
    }
};


template <typename ElemT>
void MultTPBThrust<ElemT>::run(
            const thrust::device_vector<ElemT> &hii,
            const thrust::device_vector<ElemT> &Wk,
            thrust::device_vector<ElemT> &Wb)
{

#ifdef SBD_DEBUG_TUNING
    std::cout << " multiplication by Robert is called " << std::endl;
#endif

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(this->h_comm_, &mpi_rank_h);
    MPI_Comm_size(this->h_comm_, &mpi_size_h);

    int mpi_size_b;
    MPI_Comm_size(this->b_comm_, &mpi_size_b);
    int mpi_rank_b;
    MPI_Comm_rank(this->b_comm_, &mpi_rank_b);
    int mpi_size_t;
    MPI_Comm_size(this->t_comm_, &mpi_size_t);
    int mpi_rank_t;
    MPI_Comm_rank(this->t_comm_, &mpi_rank_t);
    int mpi_size_a;
    MPI_Comm_size(this->a_comm_, &mpi_size_a);
    int mpi_rank_a;
    MPI_Comm_rank(this->a_comm_, &mpi_rank_a);

    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;

    size_t adet_min = 0;
    size_t adet_max = adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    thrust::device_vector<ElemT> T[2];
    int active_T = 0;
    int recv_T = 1;
    size_t task_sent = 0;
    Mpi2dSlider<ElemT> mpi2dslider;

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    if (helper.size() != 0) {
        if (mpi_size_b > 1) {
            Mpi2dSlide(Wk, T[active_T], adet_comm_size, bdet_comm_size,
                        -helper[0].adetShift, -helper[0].bdetShift, this->b_comm_);
        } else {
            T[active_T] = Wk;
        }
    }
    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    if (mpi_rank_t == 0) {
        auto ci = thrust::counting_iterator<size_t>(0);
        {
            SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
            thrust::for_each_n(thrust::device, ci, T[active_T].size(), Wb_init_kernel(Wb, hii, T[active_T]));
        }
    }

    double time_slid = 0.0;
    for (size_t task = 0; task < helper.size(); task++) {
        SBD_NVTX_RANGE_COLOR("for (size_t task ...", __LINE__ + task);
#ifdef SBD_DEBUG_MULT
        auto time_task_start = std::chrono::high_resolution_clock::now();
        std::cout << " Start multiplication for task " << task << " at (h,b,t) = ("
                    << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "): task type = "
                    << helper[task].taskType << ", bra-adet range = ["
                    << helper[task].braAlphaStart << "," << helper[task].braAlphaEnd << "), bra-bdet range = ["
                    << helper[task].braBetaStart << "," << helper[task].braBetaEnd << "), ket-adet range = ["
                    << helper[task].ketAlphaStart << "," << helper[task].ketAlphaEnd << "), ket-bdet range = ["
                    << helper[task].ketBetaStart << "," << helper[task].ketBetaEnd << "), ket wf =";
        for (size_t i = 0; i < std::min(static_cast<size_t>(4), T[active_T].size()); i++) {
            std::cout << " " << T[active_T][i];
        }
        std::cout << std::endl;
#endif

        // synchronize 2d slide
        if (task_sent == task) {
            if (task_sent != 0) {
                auto time_slid_start = std::chrono::high_resolution_clock::now();
                if (mpi2dslider.Sync(this->b_comm_)) {
                    int t = active_T;
                    active_T = recv_T;
                    recv_T = t;
                }
                auto time_slid_end = std::chrono::high_resolution_clock::now();
                auto time_slid_count = std::chrono::duration_cast<std::chrono::microseconds>(time_slid_end - time_slid_start).count();
                time_slid += 1.0e-6 * time_slid_count;
            }

            // exchange T asynchronously for later tasks
            for (size_t extask = task_sent; extask < helper.size() - 1; extask++) {
                if (helper[extask].taskType == 0) {
#ifdef SBD_DEBUG_MULT
                    size_t adet_rank = mpi_rank_b / bdet_comm_size;
                    size_t bdet_rank = mpi_rank_b % bdet_comm_size;
                    size_t adet_rank_task = (adet_rank + helper[extask].adetShift) % adet_comm_size;
                    size_t bdet_rank_task = (bdet_rank + helper[extask].bdetShift) % bdet_comm_size;
                    size_t adet_rank_next = (adet_rank + helper[extask + 1].adetShift) % adet_comm_size;
                    size_t bdet_rank_next = (bdet_rank + helper[extask + 1].bdetShift) % bdet_comm_size;
                    std::cout << " mult: task " << task << " at mpi process (h,b,t) = ("
                                << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t
                                << "): two-dimensional slide communication from ("
                                << adet_rank_task << "," << bdet_rank_task << ") to ("
                                << adet_rank_next << "," << bdet_rank_next << ")"
                                << std::endl;
#endif
                    int adetslide = helper[extask].adetShift - helper[extask + 1].adetShift;
                    int bdetslide = helper[extask].bdetShift - helper[extask + 1].bdetShift;
                    mpi2dslider.ExchangeAsync(T[active_T], T[recv_T], adet_comm_size, bdet_comm_size, adetslide, bdetslide, this->b_comm_, extask);
                    task_sent = extask + 1;
                    break;
                }
            }
        }

        braAlphaSize = helper[task].braAlphaEnd - helper[task].braAlphaStart;
        braBetaSize = helper[task].braBetaEnd - helper[task].braBetaStart;

        size_t offset;
        size_t size;
        if (use_precalculated_dets && collapse_loop) {
            // precalculate DetI (if update needed)
            UpdateDet(task);

            if (helper[task].taskType == 2) {
                // SingleAlpha
                size = helper[task].size_single_alpha * braBetaSize;
#ifndef SBD_USE_RANK_DISTRIBUTION
                MultSingleAlpha single_kernel(helper[task], Wb, T[active_T], *this);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::counting_iterator<size_t>(0);
#else // #ifndef SBD_USE_RANK_DISTRIBUTION
                size = (size + mpi_size_h - 1) / mpi_size_h;
#ifndef SBD_USE_VECTORIZATION
                MultUnified<ElemT, SB_MULT_ALPHA, SB_MULT_SINGLE>
                    single_kernel(helper[task], Wb, T[active_T], *this);
#else
                constexpr int VecLenS = 2;
                size = (size + VecLenS - 1) / VecLenS;
                MultUnified_Vec<ElemT, SB_MULT_ALPHA, SB_MULT_SINGLE, VecLenS>
                    single_kernel(helper[task], Wb, T[active_T], *this);
#endif
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::make_transform_iterator(
                    thrust::counting_iterator<size_t>(0),
                    [=] __host__ __device__ (size_t t) {
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
                        // Contiguous block distribution
                        return t + (size * mpi_rank_h);
#else
                        // Cyclic (strided) distribution
                        return mpi_rank_h + (mpi_size_h * t);
#endif
                    });
#endif // #ifndef SBD_USE_RANK_DISTRIBUTION
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cis, size, single_kernel);
                }

                // DoubleAlpha
                size = helper[task].size_double_alpha * braBetaSize;
#ifndef SBD_USE_RANK_DISTRIBUTION
                MultDoubleAlpha double_kernel(helper[task], Wb, T[active_T], *this);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::counting_iterator<size_t>(0);
#else // #ifndef SBD_USE_RANK_DISTRIBUTION
                size = (size + mpi_size_h - 1) / mpi_size_h;
#ifndef SBD_USE_VECTORIZATION
                MultUnified<ElemT, SB_MULT_ALPHA, SB_MULT_DOUBLE>
                    double_kernel(helper[task], Wb, T[active_T], *this);
#else
                constexpr int VecLenD = 2;
                size = (size + VecLenD - 1) / VecLenD;
                MultUnified_Vec<ElemT, SB_MULT_ALPHA, SB_MULT_DOUBLE, VecLenD>
                    double_kernel(helper[task], Wb, T[active_T], *this);
#endif
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::make_transform_iterator(
                    thrust::counting_iterator<size_t>(0),
                    [=] __host__ __device__ (size_t t) {
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
                        // Contiguous block distribution
                        return t + (size * mpi_rank_h);
#else
                        // Cyclic (strided) distribution
                        return mpi_rank_h + (mpi_size_h * t);
#endif
                    });
#endif // #ifndef SBD_USE_RANK_DISTRIBUTION
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cid, size, double_kernel);
                }
            } else if(helper[task].taskType == 1) {
                // SingleBeta
                size = helper[task].size_single_beta * braAlphaSize;
#ifndef SBD_USE_RANK_DISTRIBUTION
                MultSingleBeta single_kernel(helper[task], Wb, T[active_T], *this);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::counting_iterator<size_t>(0);
#else // #ifndef SBD_USE_RANK_DISTRIBUTION
                size = (size + mpi_size_h - 1) / mpi_size_h;
#ifndef SBD_USE_VECTORIZATION
                MultUnified<ElemT, SB_MULT_BETA, SB_MULT_SINGLE>
                    single_kernel(helper[task], Wb, T[active_T], *this);
#else
                constexpr int VecLenS = 2;
                size = (size + VecLenS - 1) / VecLenS;
                MultUnified_Vec<ElemT, SB_MULT_BETA, SB_MULT_SINGLE, VecLenS>
                    single_kernel(helper[task], Wb, T[active_T], *this);
#endif
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::make_transform_iterator(
                    thrust::counting_iterator<size_t>(0),
                    [=] __host__ __device__ (size_t t) {
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
                        // Contiguous block distribution
                        return t + (size * mpi_rank_h);
#else
                        // Cyclic (strided) distribution
                        return mpi_rank_h + (mpi_size_h * t);
#endif
                    });
#endif // #ifndef SBD_USE_RANK_DISTRIBUTION
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cis, size, single_kernel);
                }

                // DoubleBeta
                size = helper[task].size_double_beta * braAlphaSize;
#ifndef SBD_USE_RANK_DISTRIBUTION
                MultDoubleBeta double_kernel(helper[task], Wb, T[active_T], *this);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::counting_iterator<size_t>(0);
#else // #ifndef SBD_USE_RANK_DISTRIBUTION
                size = (size + mpi_size_h - 1) / mpi_size_h;
#ifndef SBD_USE_VECTORIZATION
                MultUnified<ElemT, SB_MULT_BETA, SB_MULT_DOUBLE>
                    double_kernel(helper[task], Wb, T[active_T], *this);
#else
                constexpr int VecLenD = 2;
                size = (size + VecLenD - 1) / VecLenD;
                MultUnified_Vec<ElemT, SB_MULT_BETA, SB_MULT_DOUBLE, VecLenD>
                    double_kernel(helper[task], Wb, T[active_T], *this);
#endif
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::make_transform_iterator(
                    thrust::counting_iterator<size_t>(0),
                    [=] __host__ __device__ (size_t t) {
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
                        // Contiguous block distribution
                        return t + (size * mpi_rank_h);
#else
                        // Cyclic (strided) distribution
                        return mpi_rank_h + (mpi_size_h * t);
#endif
                    });
#endif // #ifndef SBD_USE_RANK_DISTRIBUTION
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cid, size, double_kernel);
                }
            } else {
                //
                size = helper[task].size_single_alpha * helper[task].size_single_beta;
#ifndef SBD_USE_RANK_DISTRIBUTION
                MultAlphaBeta kernel(helper[task], Wb, T[active_T], *this);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto ci = thrust::counting_iterator<size_t>(0);
#else // #ifndef SBD_USE_RANK_DISTRIBUTION
                size = (size + mpi_size_h - 1) / mpi_size_h;
#ifndef SBD_USE_VECTORIZATION
                MultAlphaBeta kernel(helper[task], Wb, T[active_T], *this);
#else
                constexpr int VecLen = 2;
                size = (size + VecLen - 1) / VecLen;
                MultAlphaBeta_Vec<ElemT, VecLen> kernel(helper[task], Wb, T[active_T], *this);
#endif
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto ci = thrust::make_transform_iterator(
                    thrust::counting_iterator<size_t>(0),
                    [=] __host__ __device__ (size_t t) {
#ifdef SBD_USE_BLOCK_RANK_DISTRIBUTION
                        // Contiguous block distribution
                        return t + (size * mpi_rank_h);
#else
                        // Cyclic (strided) distribution
                        return mpi_rank_h + (mpi_size_h * t);
#endif
                    });
#endif // #ifndef SBD_USE_RANK_DISTRIBUTION
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, ci, size, kernel);
                }
            }
        } else {
            size = braAlphaSize * braBetaSize;
            if (use_precalculated_dets) {
                num_max_threads = size;
                // precalculate DetI (if update needed)
                UpdateDet(task);
            }

            if (helper[task].taskType == 2) {
                offset = 0;
                while (offset < size) {
                    size_t num_threads = num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }
                    MultTask2 kernel(helper[task], Wb, T[active_T], *this, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    {
                        SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                        thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    }
                    offset += num_threads;
                }
            } else if(helper[task].taskType == 1) {
                offset = 0;
                while (offset < size) {
                    size_t num_threads = num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    MultTask1 kernel(helper[task], Wb, T[active_T], *this, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    {
                        SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                        thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    }
                    offset += num_threads;
                }
            } else {
                offset = 0;
                while (offset < size) {
                    size_t num_threads = num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    MultTask0 kernel(helper[task], Wb, T[active_T], *this, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    {
                        SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                        thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    }
                    offset += num_threads;
                }
            }
        }

#ifdef SBD_DEBUG_MULT
        auto time_task_end = std::chrono::high_resolution_clock::now();
        auto time_task_count = std::chrono::duration_cast<std::chrono::microseconds>(time_task_end - time_task_start).count();
        std::cout << "     time for task " << task << " [" << helper[task].taskType << "] : " << 1.0e-6 * time_task_count << std::endl;
#endif
    } // end for(size_t task=0; task < helper.size(); task++)

    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
#ifdef SBD_USE_NCCL
    if (mpi_size_a > 1) {
        nccl_allreduce(Wb, ncclSum, this->a_nccl_comm_);
    }
#else
    if (mpi_size_t > 1) {
        MpiAllreduce(Wb, MPI_SUM, this->t_comm_);
    }
    if (mpi_size_h > 1) {
        MpiAllreduce(Wb, MPI_SUM, this->h_comm_);
    }
#endif
    auto time_comm_end = std::chrono::high_resolution_clock::now();

#ifdef SBD_DEBUG_MULT
    auto time_copy_count = std::chrono::duration_cast<std::chrono::microseconds>(time_copy_end - time_copy_start).count();
    auto time_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_mult_end - time_mult_start).count();
    auto time_comm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_comm_end - time_comm_start).count();

    double time_copy = 1.0e-6 * time_copy_count;
    double time_mult = 1.0e-6 * time_mult_count;
    double time_comm = 1.0e-6 * time_comm_count;
    std::cout << " mult: time for first copy     = " << time_copy << std::endl;
    std::cout << " mult: time for multiplication = " << time_mult << std::endl;
    std::cout << " mult: time for 2d slide comm  = " << time_slid << std::endl;
    std::cout << " mult: time for allreduce comm = " << time_comm << std::endl;
#endif

} // end function


template <typename ElemT>
class MakeQChamDiagTermKernel : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    ElemT* hii;
    size_t offset;
    bool use_pre_dets;
public:
    MakeQChamDiagTermKernel(thrust::device_vector<ElemT>& hii_in, const TaskHelpersThrust<ElemT>& h, const MultTPBThrust<ElemT>& data, size_t o, bool pre)
                        : MultKernelBase<ElemT>(data)
    {
        helper = h;
        hii = (ElemT*)thrust::raw_pointer_cast(hii_in.data());
        offset = o;
        use_pre_dets = pre;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        if (braIdx % this->mpi_size_h == this->mpi_rank_h) {
            size_t* Det = this->det_I + i * this->D_size;
            if (!use_pre_dets) {
                size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
                size_t a = braIdx / braBetaSize;
                size_t b = braIdx - a * braBetaSize;
                size_t ia = a + helper.braAlphaStart;
                size_t ib = b + helper.braBetaStart;
                this->DetFromAlphaBeta(Det, this->adets + ia * this->D_size, this->bdets + ib * this->D_size);
            }
            hii[braIdx] = this->ZeroExcite(Det);
        }
    }
};


template <typename ElemT>
void MultTPBThrust<ElemT>::makeQChamDiagTerms(thrust::device_vector<ElemT> &hii)
{
    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(this->h_comm_, &mpi_rank_h);
    MPI_Comm_size(this->h_comm_, &mpi_size_h);

    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;
    if( helper.size() != 0 ) {
        braAlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
        braBetaSize = helper[0].braBetaEnd - helper[0].braBetaStart;
    }
    size_t braSize = braAlphaSize * braBetaSize;

    if (use_precalculated_dets)
        UpdateDet(0);

    hii.resize(braSize, ElemT(0.0));

    if (use_precalculated_dets) {
        MakeQChamDiagTermKernel kernel(hii, helper[0], *this, 0, true);
        kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
        auto ci = thrust::counting_iterator<size_t>(0);
        thrust::for_each_n(thrust::device, ci, braSize, kernel);
    } else {
        size_t offset = 0;
        while (offset < braSize) {
            size_t num_threads = num_max_threads;
            if (offset + num_threads > braSize) {
                num_threads = braSize - offset;
            }

            MakeQChamDiagTermKernel kernel(hii, helper[0], *this, offset, use_precalculated_dets);
            kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
            auto ci = thrust::counting_iterator<size_t>(0);
            thrust::for_each_n(thrust::device, ci, num_threads, kernel);
            offset += num_threads;
        }
    }
}


}

#endif
