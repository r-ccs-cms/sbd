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

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            this->correlation.ZeroDiffCorrelation(DetI, this->Wb[braIdx]);
        }
    }
};


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

    __device__ inline void loop_body(size_t i, int64_t& braIdx) {
        braIdx = -1;
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

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];

        this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                            helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                            helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        int64_t braIdx;
        loop_body(i, braIdx);
    }

    /*__device__ __host__ void operator()(size_t i)
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
    }*/
};

template <typename ElemT>
class CorrelationAlphaBeta_Vec : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationAlphaBeta_Vec(const TaskHelpersThrust<ElemT>& h,
                const MultTPBThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
    }

    __device__ inline void loop_body(size_t i, int64_t& braIdx) {
        braIdx = -1;
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

        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];

        this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                            helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                            helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i0)
    {
        size_t i = (VecLen * i0);
        int64_t braIdx;
        loop_body(i, braIdx);
        for (size_t i1 = 1; i1 < VecLen; i1++) {
            i = (VecLen * i0) + i1;
            int64_t braIdx_new;
            loop_body(i, braIdx_new);
            braIdx = braIdx_new;
        }
    }
};


#define SB_CORR_ALPHA 0
#define SB_CORR_BETA  1
#define SB_CORR_SINGLE 0
#define SB_CORR_DOUBLE 1

// MultUnified unifies multiple kernel variants into a single template-based
// implementation, allowing shared application of optimizations (e.g., thread
// coarsening) and avoiding code duplication across individual kernels.
template <typename ElemT, int AlphaOrBeta, int SingleOrDouble>
class CorrelationUnified : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    CorrelationUnified(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultTPBThrust<ElemT>& data,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2
                ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
    }

    __device__ inline void loop_body(size_t i, int64_t& braIdx) {
        braIdx = -1;
        size_t k;
        size_t j;
        size_t ia;
        size_t ja;
        size_t ib;
        size_t jb;
        if (AlphaOrBeta == SB_CORR_ALPHA) {
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
        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];
        if (AlphaOrBeta == SB_CORR_ALPHA) {
            if (SingleOrDouble == SB_CORR_SINGLE) {
                // SingleAlpha
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            } else {
                // DoubleAlpha
                this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
            }
        } else {
            if (SingleOrDouble == SB_MULT_SINGLE) {
                // SingleBeta
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            } else {
                // DoubleBeta
                this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                        helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
            }
        }
    }

    // kernel entry point
    __device__ void operator()(size_t i)
    {
        int64_t braIdx;
        loop_body(i, braIdx);
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
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];
        this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
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
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];

        this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                            helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                            helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
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
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];
        this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
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
#ifndef SBD_USE_RANK_DISTRIBUTION
        if( (braIdx % this->mpi_size_h) != this->mpi_rank_h ) return;
#endif
        size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                        + jb - helper.ketBetaStart;

        size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->D_size;
        ElemT WeightI = this->Wb[braIdx];
        ElemT WeightJ = this->T[ketIdx];

        this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                    helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                    helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
    }
};


template <typename ElemT>
class CorrelationTask0 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    CorrelationTask0(const TaskHelpersThrust<ElemT>& h,
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

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            ElemT WeightI = this->Wb[braIdx];

            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                    size_t jb = helper.SinglesFromBetaKetIndex[k];
                    size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                    + jb - helper.ketBetaStart;
                    ElemT WeightJ = this->T[ketIdx];
                    this->correlation.TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                        helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
                }
            }
        }
    }
};

template <typename ElemT>
class CorrelationTask1 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    CorrelationTask1(const TaskHelpersThrust<ElemT>& h,
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

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            ElemT WeightI = this->Wb[braIdx];

            for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                size_t ja = ia;
                size_t jb = helper.SinglesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            }
            for (size_t k = helper.DoublesFromBetaOffset[b]; k < helper.DoublesFromBetaOffset[b + 1]; k++) {
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
};

template <typename ElemT>
class CorrelationTask2 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
    bool use_pre_dets;
public:
    CorrelationTask2(const TaskHelpersThrust<ElemT>& h,
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

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
             if (use_pre_dets)
                DetI = this->det_I + braIdx * this->D_size;
            else {
                DetI = this->det_I + i * this->D_size;
                this->DetFromAlphaBeta(DetI, this->adets + ia * this->D_half_size, this->bdets + ib * this->D_half_size);
            }
            ElemT WeightI = this->Wb[braIdx];

            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->correlation.OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            }
            for (size_t j = helper.DoublesFromAlphaOffset[a]; j < helper.DoublesFromAlphaOffset[a + 1]; j++) {
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
            {
                SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                thrust::for_each_n(thrust::device, ci, size, kernel);
            }
        } else {
            size = braAlphaSize * braBetaSize;
            if (use_precalculated_dets) {
                num_max_threads = size;
                // precalculate DetI (if update needed)
                UpdateDet(0);
            }

            offset = 0;
            while (offset < size) {
                size_t num_threads = num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                CorrelationInitNoCache kernel(helper[0], *this, W, T, onebody, twobody, offset, use_precalculated_dets);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto ci = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                offset += num_threads;
            }
        }
    }

    for (size_t task = 0; task < helper.size(); task++) {
        size_t ketAlphaSize = helper[task].ketAlphaEnd - helper[task].ketAlphaStart;
        size_t ketBetaSize = helper[task].ketBetaEnd - helper[task].ketBetaStart;

        if (use_precalculated_dets && collapse_loop) {
            // precalculate DetI (if update needed)
            UpdateDet(task);

            if (helper[task].taskType == 2) {
                // SingleAlpha
                size = helper[task].size_single_alpha * braBetaSize;
                CorrelationSingleAlpha single_kernel(helper[task], *this, W, T, onebody, twobody);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cis, size, single_kernel);
                }

                // DoubleAlpha
                size = helper[task].size_double_alpha * braBetaSize;
                CorrelationDoubleAlpha double_kernel(helper[task], *this, W, T, onebody, twobody);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cid, size, double_kernel);
                }
            } else if(helper[task].taskType == 1) {
                // SingleBeta
                size = helper[task].size_single_beta * braAlphaSize;
                CorrelationSingleBeta single_kernel(helper[task], *this, W, T, onebody, twobody);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cis = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cis, size, single_kernel);
                }

                // DoubleBeta
                size = helper[task].size_double_beta * braAlphaSize;
                CorrelationDoubleBeta double_kernel(helper[task], *this, W, T, onebody, twobody);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto cid = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, cid, size, double_kernel);
                }
            } else {
                //
                size = helper[task].size_single_alpha * helper[task].size_single_beta;
                CorrelationAlphaBeta kernel(helper[task], *this, W, T, onebody, twobody);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
                auto ci = thrust::counting_iterator<size_t>(0);
                {
                    SBD_NVTX_RANGE_COLOR("thrust::for_each_n", __LINE__);
                    thrust::for_each_n(thrust::device, ci, size, kernel);
                }
            }
        } else {    // no collapse
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
                    CorrelationTask2 kernel(helper[task], *this, W, T, onebody, twobody, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } else if(helper[task].taskType == 1) {
                offset = 0;
                while (offset < size) {
                    size_t num_threads = num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationTask1 kernel(helper[task], *this, W, T, onebody, twobody, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } else {
                offset = 0;
                while (offset < size) {
                    size_t num_threads = num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationTask0 kernel(helper[task], *this, W, T, onebody, twobody, offset, use_precalculated_dets);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
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
