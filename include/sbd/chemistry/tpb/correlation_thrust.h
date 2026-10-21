/**
@file sbd/chemistry/tpb/correlation_thrust.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_TPB_CORRELATION_THRUST_H
#define SBD_CHEMISTRY_TPB_CORRELATION_THRUST_H

namespace sbd
{

template <typename ElemT>
class CorrelationKernelBase : public MultKernelBase<ElemT> {
protected:
    CorrelationKernels<ElemT> correlation;
public:
    CorrelationKernelBase() {}

    CorrelationKernelBase(const MultTPBThrust<ElemT>& data,
                        const thrust::device_vector<ElemT>& v_wb,
                        const thrust::device_vector<ElemT>& v_t,
                        thrust::device_vector<ElemT>& b1,
                        thrust::device_vector<ElemT>& b2)
                         : MultKernelBase<ElemT>(v_wb, v_t, data),
                           correlation(data.bit_length(), data.norbs(),  data.I0, data.I1, data.I2, b1, b2)
    {
    }

};

template <typename ElemT>
class CorrelationInit : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationInit(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;

        if( (i % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t* DetI = this->det_I + i * this->D_size;
            this->correlation.ZeroDiffCorrelation(DetI, this->Wb[i]);
        }
    }
};

template <typename ElemT>
class CorrelationInitNoCache : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    CorrelationInitNoCache(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o, bool pre ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
        use_pre_dets = pre;
    }

    __device__ __host__ size_t size(void)
    {
        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t size = braAlphaSize * braBetaSize;
        return size;
    }

    // kernel entry point
    __device__ void run(size_t* Det, size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t* DetI;
            if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = Det + threadIdx.x * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            this->correlation.ZeroDiffCorrelation(DetI, this->Wb[braIdx]);
        }
    }
};

template <typename ElemT>
__global__ void correlation_init_kernel(CorrelationInitNoCache<ElemT> corr)
{
    extern __shared__ size_t det[];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nid = blockDim.x * gridDim.x;
    size_t size = corr.size();

    for (size_t i = tid; i < size; i += nid) {
        corr.run(det, i);
    }
}

template <typename ElemT>
class CorrelationAlphaBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationAlphaBeta(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = i / helper.size_single_beta;
        size_t k = i - j * helper.size_single_beta;

        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];

        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
            ElemT WeightI = this->Wb[braIdx];
            ElemT WeightJ = this->T[ketIdx];

            this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
        }
    }
};


template <typename ElemT>
class CorrelationSingleAlpha : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationSingleAlpha(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
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
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
            ElemT WeightI = this->Wb[braIdx];
            ElemT WeightJ = this->T[ketIdx];
            this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
        }
    }
};

template <typename ElemT>
class CorrelationDoubleAlpha : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationDoubleAlpha(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
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
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
            ElemT WeightI = this->Wb[braIdx];
            ElemT WeightJ = this->T[ketIdx];

            this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
        }
    }
};

template <typename ElemT>
class CorrelationSingleBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationSingleBeta(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
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
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
            ElemT WeightI = this->Wb[braIdx];
            ElemT WeightJ = this->T[ketIdx];
            this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
        }
    }
};

template <typename ElemT>
class CorrelationDoubleBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationDoubleBeta(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
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
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
            ElemT WeightI = this->Wb[braIdx];
            ElemT WeightJ = this->T[ketIdx];

            this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                        helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
        }
    }
};


template <typename ElemT>
class CorrelationKernel : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    CorrelationKernel(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o, bool pre ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
        use_pre_dets = pre;
    }
    __device__ __host__ size_t size(void)
    {
        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t size = braAlphaSize * braBetaSize;
        return size;
    }

    __device__ void task_0(size_t* shared_det, size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        int tid = threadIdx.x;
        int nid = blockDim.x;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = shared_det;
                __syncthreads();
                if (tid == 0)
                    this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
                __syncthreads();
            }
            ElemT WeightI = this->Wb[braIdx];

            size_t idx_offset = tid;

            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                size_t k;
                for (k = helper.SinglesFromBetaOffset[b] + idx_offset; k < helper.SinglesFromBetaOffset[b + 1]; k += nid) {
                    size_t jb = helper.SinglesFromBetaKetIndex[k];
                    size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                    + jb - helper.ketBetaStart;
                    ElemT WeightJ = this->T[ketIdx];
                    this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                        helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
                }
                idx_offset = k - helper.SinglesFromBetaOffset[b + 1];
            }
        }
    }

    __device__ void task_1(size_t* shared_det, size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        int tid = threadIdx.x;
        int nid = blockDim.x;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = shared_det;
                __syncthreads();
                if (tid == 0)
                    this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
                __syncthreads();
            }
            ElemT WeightI = this->Wb[braIdx];

            size_t idx_offset = tid;
            size_t k;
            for (k = helper.SinglesFromBetaOffset[b] + idx_offset; k < helper.SinglesFromBetaOffset[b + 1]; k += nid) {
                size_t ja = ia;
                size_t jb = helper.SinglesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            }
            idx_offset = k - helper.SinglesFromBetaOffset[b + 1];
            for (k = helper.DoublesFromBetaOffset[b] + idx_offset; k < helper.DoublesFromBetaOffset[b + 1]; k += nid) {
                size_t ja = ia;
                size_t jb = helper.DoublesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                        helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
           }
        }
    }

    __device__ void task_2(size_t* shared_det, size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;
        int tid = threadIdx.x;
        int nid = blockDim.x;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = shared_det;
                __syncthreads();
                if (tid == 0)
                    this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
                __syncthreads();
            }
            ElemT WeightI = this->Wb[braIdx];

            size_t idx_offset = tid;
            size_t j;
            for (j = helper.SinglesFromAlphaOffset[a] + idx_offset; j < helper.SinglesFromAlphaOffset[a + 1]; j += nid) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            }
            idx_offset = j - helper.SinglesFromAlphaOffset[a + 1];
            for (j = helper.DoublesFromAlphaOffset[a] + idx_offset; j < helper.DoublesFromAlphaOffset[a + 1]; j += nid) {
                size_t ja = helper.DoublesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                    helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
            }
        }
    }
};


template <typename ElemT>
__global__ void correlation_kernel(CorrelationKernel<ElemT> corr, int type)
{
    extern __shared__ size_t det[];

    size_t tid = blockIdx.x;
    size_t nid = gridDim.x;
    size_t size = corr.size();

    if (type == 2) {
        for (size_t i = tid; i < size; i += nid) {
            corr.task_2(det, i);
        }
    } else if (type == 1) {
        for (size_t i = tid; i < size; i += nid) {
            corr.task_1(det, i);
        }
    } else {
        for (size_t i = tid; i < size; i += nid) {
            corr.task_0(det, i);
        }
    }
}


/**
    Function to evaluate the two-particle correlation functions
*/
template <typename ElemT>
void MultTPBThrust<ElemT>::correlation(const std::vector<ElemT> &W_in,
                    std::vector<std::vector<ElemT>> &onebody_out,
                    std::vector<std::vector<ElemT>> &twobody_out)
{
    thrust::device_vector<ElemT> onebody(this->norbs() * this->norbs() * 2, ElemT(0.0));
    thrust::device_vector<ElemT> twobody(this->norbs() * this->norbs() * this->norbs() * this->norbs() * 4, ElemT(0.0));

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(this->h_comm(), &mpi_rank_h);
    MPI_Comm_size(this->h_comm(), &mpi_size_h);

    int mpi_size_b;
    MPI_Comm_size(this->b_comm(), &mpi_size_b);
    int mpi_rank_b;
    MPI_Comm_rank(this->b_comm(), &mpi_rank_b);
    int mpi_size_t;
    MPI_Comm_size(this->t_comm(), &mpi_size_t);
    int mpi_rank_t;
    MPI_Comm_rank(this->t_comm(), &mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;
    if (helper.size() != 0) {
        braAlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
        braBetaSize = helper[0].braBetaEnd - helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    thrust::device_vector<ElemT> T(max_det_size);
    thrust::device_vector<ElemT> R;
    thrust::device_vector<ElemT> W(W_in.size());
    thrust::copy_n(W_in.begin(), W_in.size(), W.begin());

    if (helper.size() != 0) {
        Mpi2dSlide(W, T, adet_comm_size, bdet_comm_size,
                    -helper[0].adetShift, -helper[0].bdetShift, this->b_comm());
    }

    size_t offset = 0;
    size_t size = 0;
    if (mpi_rank_t == 0) {
        if (use_precalculated_dets && collapse_loop) {
            // precalculate DetI (if update needed)
            UpdateDet(0);

            size = braAlphaSize * braBetaSize;
            CorrelationInit kernel(helper[0], *this, W, T, onebody, twobody);
            kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

            auto ci = thrust::counting_iterator<size_t>(0);
            thrust::for_each_n(thrust::device, ci, size, kernel);
        } else {
            size = braAlphaSize * braBetaSize;
            CorrelationInitNoCache kernel(helper[0], *this, W, T, onebody, twobody, 0, false);
            kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

            size_t max_threads = SBD_MAX_DYNAMIC_SHARED_BYTES / (this->D_size_ * sizeof(size_t));
            if (max_threads > 1024)
                max_threads = 1024;
            size_t nthreads = 1;
            while(nthreads * 2 <= max_threads) {
                nthreads *= 2;
            }
            size_t nblocks = std::max(((size + nthreads - 1) / nthreads + 3) / 4, (size_t)1);

            correlation_init_kernel<<<nblocks, nthreads, nthreads * this->D_size_ * sizeof(size_t)>>>(kernel);
            SBD_CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    for (size_t task = 0; task < helper.size(); task++) {
        size_t ketAlphaSize = helper[task].ketAlphaEnd - helper[task].ketAlphaStart;
        size_t ketBetaSize = helper[task].ketBetaEnd - helper[task].ketBetaStart;

        if (use_precalculated_dets && collapse_loop) {
            // precalculate DetI (if update needed)
            UpdateDet(task);

            if (helper[task].taskType == 2) { // beta range are same
                size = helper[task].size_single_alpha * braBetaSize;
                CorrelationSingleAlpha single_kernel(helper[task], *this, W, T, onebody, twobody);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cis = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cis, size, single_kernel);

                size = helper[task].size_double_alpha * braBetaSize;
                CorrelationDoubleAlpha double_kernel(helper[task], *this, W, T, onebody, twobody);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cid = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cid, size, double_kernel);
            }
            else if (helper[task].taskType == 1) {
                size = helper[task].size_single_beta * braAlphaSize;
                CorrelationSingleBeta single_kernel(helper[task], *this, W, T, onebody, twobody);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cis = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cis, size, single_kernel);

                size = helper[task].size_double_beta * braAlphaSize;
                CorrelationDoubleBeta double_kernel(helper[task], *this, W, T, onebody, twobody);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cid = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cid, size, double_kernel);
            } else {
                size = helper[task].size_single_alpha * helper[task].size_single_beta;
                CorrelationAlphaBeta kernel(helper[task], *this, W, T, onebody, twobody);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto ci = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, ci, size, kernel);
            }
        } else {    // no collapse
            size = braAlphaSize * braBetaSize;
            if (size != 0) {
                CorrelationKernel kernel(helper[task], *this, W, T, onebody, twobody, 0, false);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                size_t num_threads = 32;
                size_t max_threads = helper_storage[task].size() / size;
                if (max_threads > 1024)
                    max_threads = 1024;
                while(num_threads * 2 <= max_threads) {
                    num_threads *= 2;
                }
                size_t num_blocks = std::max((size + 3) / 4, (size_t)1);

                correlation_kernel<<<num_blocks, num_threads, this->D_size_ * sizeof(size_t)>>>(kernel, helper[task].taskType);
                SBD_CHECK_CUDA(cudaDeviceSynchronize());
            }
        }

        if (helper[task].taskType == 0 && task != helper.size() - 1) {
            int adetslide = helper[task].adetShift - helper[task + 1].adetShift;
            int bdetslide = helper[task].bdetShift - helper[task + 1].bdetShift;
            R.resize(T.size());
            R = T;
            Mpi2dSlide(R, T, adet_comm_size, bdet_comm_size, adetslide, bdetslide, this->b_comm());
        }
    } // end for(size_t task=0; task < helper.size(); task++)

    if (mpi_size_b > 1)
        MpiAllreduce(onebody, MPI_SUM, this->b_comm());
    if (mpi_size_t > 1)
        MpiAllreduce(onebody, MPI_SUM, this->t_comm());
    if (mpi_size_h > 1)
        MpiAllreduce(onebody, MPI_SUM, this->h_comm());

    if (mpi_size_b > 1)
        MpiAllreduce(twobody, MPI_SUM, this->b_comm());
    if (mpi_size_t > 1)
        MpiAllreduce(twobody, MPI_SUM, this->t_comm());
    if (mpi_size_h > 1)
        MpiAllreduce(twobody, MPI_SUM, this->h_comm());


    // copy out onebody, twobody
    onebody_out.resize(2);
    size = this->norbs() * this->norbs();
    offset = 0;
    for(int s=0; s < 2; s++) {
        onebody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(onebody.begin() + offset, size, onebody_out[s].begin());
        offset += size;
    }

    twobody_out.resize(4);
    size = this->norbs() * this->norbs() * this->norbs() * this->norbs();
    offset = 0;
    for(int s=0; s < 4; s++) {
        twobody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(twobody.begin() + offset, size, twobody_out[s].begin());
        offset += size;
    }
}

} // end namespace sbd

#endif // end if for #ifndef SBD_CHEMISTRY_PTMB_CORRELATION_H
