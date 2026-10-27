/**
@file sbd/chemistry/tpb/helpers.h
@brief Helper array to construct Hamiltonian for parallel taskers for distributed basis
 */
#ifndef SBD_CHEMISTRY_TPB_HELPER_THRUST_H
#define SBD_CHEMISTRY_TPB_HELPER_THRUST_H

#include <cassert>
#include <vector>

#include "sbd/chemistry/tpb/helper.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "sbd/chemistry/tpb/excitation_thrust.h"


namespace sbd {

template <typename ElemT>
class TaskHelpersThrust {
public:
    size_t braAlphaStart;
    size_t braAlphaEnd;
    size_t ketAlphaStart;
    size_t ketAlphaEnd;
    size_t braBetaStart;
    size_t braBetaEnd;
    size_t ketBetaStart;
    size_t ketBetaEnd;
    size_t taskType;
    size_t adetShift;
    size_t bdetShift;

//    size_t* base_memory;
    size_t* SinglesFromAlphaBraIndex;
    size_t* SinglesFromAlphaKetIndex;
    size_t* DoublesFromAlphaBraIndex;
    size_t* DoublesFromAlphaKetIndex;
    size_t* SinglesFromBetaBraIndex;
    size_t* SinglesFromBetaKetIndex;
    size_t* DoublesFromBetaBraIndex;
    size_t* DoublesFromBetaKetIndex;

    // pointer to cached c and d for energy calculation
//    int* base_cran;
    int* SinglesAlphaCrAnSM;
    int* SinglesBetaCrAnSM;
    int* DoublesAlphaCrAnSM;
    int* DoublesBetaCrAnSM;

    // offset for non-load balanced mode
    size_t* SinglesFromAlphaOffset;
    size_t* SinglesFromBetaOffset;
    size_t* DoublesFromAlphaOffset;
    size_t* DoublesFromBetaOffset;

    size_t size_single_alpha = 0;
    size_t size_double_alpha = 0;
    size_t size_single_beta = 0;
    size_t size_double_beta = 0;

    size_t num_mult_threads = 1;

    TaskHelpersThrust() {}

    TaskHelpersThrust(const TaskHelpersThrust<ElemT>& other)
    {
        braAlphaStart = other.braAlphaStart;
        braAlphaEnd = other.braAlphaEnd;
        ketAlphaStart = other.ketAlphaStart;
        ketAlphaEnd = other.ketAlphaEnd;
        braBetaStart = other.braBetaStart;
        braBetaEnd = other.braBetaEnd;
        ketBetaStart = other.ketBetaStart;
        ketBetaEnd = other.ketBetaEnd;
        taskType = other.taskType;
        adetShift = other.adetShift;
        bdetShift = other.bdetShift;

//        base_memory = other.base_memory;
        SinglesFromAlphaBraIndex = other.SinglesFromAlphaBraIndex;
        SinglesFromAlphaKetIndex = other.SinglesFromAlphaKetIndex;
        DoublesFromAlphaBraIndex = other.DoublesFromAlphaBraIndex;
        DoublesFromAlphaKetIndex = other.DoublesFromAlphaKetIndex;
        SinglesFromBetaBraIndex = other.SinglesFromBetaBraIndex;
        SinglesFromBetaKetIndex = other.SinglesFromBetaKetIndex;
        DoublesFromBetaBraIndex = other.DoublesFromBetaBraIndex;
        DoublesFromBetaKetIndex = other.DoublesFromBetaKetIndex;

//        base_cran = other.base_cran;
        SinglesAlphaCrAnSM = other.SinglesAlphaCrAnSM;
        SinglesBetaCrAnSM = other.SinglesBetaCrAnSM;
        DoublesAlphaCrAnSM = other.DoublesAlphaCrAnSM;
        DoublesBetaCrAnSM = other.DoublesBetaCrAnSM;

        SinglesFromAlphaOffset = other.SinglesFromAlphaOffset;
        SinglesFromBetaOffset = other.SinglesFromBetaOffset;
        DoublesFromAlphaOffset = other.DoublesFromAlphaOffset;
        DoublesFromBetaOffset = other.DoublesFromBetaOffset;

        size_single_alpha = other.size_single_alpha;
        size_double_alpha = other.size_double_alpha;
        size_single_beta = other.size_single_beta;
        size_double_beta = other.size_double_beta;

        num_mult_threads = other.num_mult_threads;
    }

    // store_offset:    emit the per-bra CSR offset arrays (SinglesFromAlphaOffset,
    //                  ...), required by the offset-based kernels
    //                  (MultSharedDet, MultTask0/1/2, CorrelationTask0/1/2).
    // store_bra_index: emit the flat BraIndex arrays (SinglesFromAlphaBraIndex,
    //                  ...), required by the collapsed-loop kernels
    //                  (MultAlphaBeta, MultSingleAlpha, CorrelationAlphaBeta, ...).
    // The two are independent: a build that uses MultSharedDet for H*v but the
    // collapsed kernels for the RDMs needs both, so neither may be inferred from
    // the other.
    //
    // This only takes pointers into data owned by ExcitationThrust; several task
    // helpers alias the same block.  SBD_REORDER_INDEX_ARRAY therefore lives in
    // ExcitationThrust::Init (reorder_by_ket_block), where it is applied once per
    // block on host data before the H2D upload -- doing it here would permute
    // shared arrays once per referring task.
    TaskHelpersThrust(const ExcitationThrust& ext, const int iblock, const int task, bool store_offset = false, bool store_bra_index = true)
    {
        size_t alpha_block = ext.alphaBlockSchedule[iblock];
        size_t beta_block = ext.betaBlockSchedule[iblock];

        braAlphaStart = ext.braAlphaStart;
        braAlphaEnd = ext.braAlphaEnd;
        ketAlphaStart = ext.ketAlphaStart[alpha_block];
        ketAlphaEnd = ext.ketAlphaEnd[alpha_block];
        braBetaStart = ext.braBetaStart;
        braBetaEnd = ext.braBetaEnd;
        ketBetaStart = ext.ketBetaStart[beta_block];
        ketBetaEnd = ext.ketBetaEnd[beta_block];
        taskType = task;

        adetShift = ext.adetShift[iblock];
        bdetShift = ext.bdetShift[iblock];

        size_t braAlphaSize = braAlphaEnd - braAlphaStart;
        size_t braBetaSize = braBetaEnd - braBetaStart;

        size_single_alpha = ext.size_single_alpha[alpha_block];
        size_double_alpha = ext.size_double_alpha[alpha_block];
        size_single_beta = ext.size_single_beta[beta_block];
        size_double_beta = ext.size_double_beta[beta_block];

        SinglesFromAlphaKetIndex = ext.get_single_alpha_ptr(alpha_block);
        DoublesFromAlphaKetIndex = ext.get_double_alpha_ptr(alpha_block);
        SinglesFromBetaKetIndex = ext.get_single_beta_ptr(beta_block);
        DoublesFromBetaKetIndex = ext.get_double_beta_ptr(beta_block);

        SinglesFromAlphaBraIndex = nullptr;
        DoublesFromAlphaBraIndex = nullptr;
        SinglesFromBetaBraIndex = nullptr;
        DoublesFromBetaBraIndex = nullptr;
        if (store_bra_index) {
            SinglesFromAlphaBraIndex = ext.get_single_alpha_bra_ptr(alpha_block);
            DoublesFromAlphaBraIndex = ext.get_double_alpha_bra_ptr(alpha_block);
            SinglesFromBetaBraIndex = ext.get_single_beta_bra_ptr(beta_block);
            DoublesFromBetaBraIndex = ext.get_double_beta_bra_ptr(beta_block);
        }

        SinglesAlphaCrAnSM = ext.get_single_alpha_cran_ptr(alpha_block);
        DoublesAlphaCrAnSM = ext.get_double_alpha_cran_ptr(alpha_block);
        SinglesBetaCrAnSM = ext.get_single_beta_cran_ptr(beta_block);
        DoublesBetaCrAnSM = ext.get_double_beta_cran_ptr(beta_block);

        SinglesFromAlphaOffset = nullptr;
        DoublesFromAlphaOffset = nullptr;
        SinglesFromBetaOffset = nullptr;
        DoublesFromBetaOffset = nullptr;
        if (store_offset) {
            SinglesFromAlphaOffset = ext.get_single_alpha_offset_ptr(alpha_block);
            DoublesFromAlphaOffset = ext.get_double_alpha_offset_ptr(alpha_block);
            SinglesFromBetaOffset = ext.get_single_beta_offset_ptr(beta_block);
            DoublesFromBetaOffset = ext.get_double_beta_offset_ptr(beta_block);
        }

        size_t max_threads = 1;
        if (taskType == 3) {
            // taskType == 3 calculates all tasks in the block at once
            max_threads = (size_single_alpha * size_single_beta
                            + size_single_alpha + size_double_alpha
                            + size_single_beta + size_double_beta)
                        / (braAlphaSize * braBetaSize);
        } else if (taskType == 2) {
            if (braAlphaSize != 0)
                max_threads = (size_single_alpha + size_double_alpha) / braAlphaSize;
        } else if (taskType == 1) {
            if (braBetaSize != 0)
                max_threads = (size_single_beta + size_double_beta) / braBetaSize;
        } else {
            if (braAlphaSize != 0 && braBetaSize != 0)
                max_threads = (size_single_alpha / braAlphaSize) * (size_single_beta / braBetaSize);
        }
        if (max_threads > 1024)
            max_threads = 1024;
        num_mult_threads = 32;
        while (num_mult_threads * 4 <= max_threads)
            num_mult_threads *= 2;
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
