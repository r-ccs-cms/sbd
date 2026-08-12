/**
@file sbd/chemistry/tpb/helpers.h
@brief Helper array to construct Hamiltonian for parallel taskers for distributed basis
 */
#ifndef SBD_CHEMISTRY_TPB_HELPER_THRUST_H
#define SBD_CHEMISTRY_TPB_HELPER_THRUST_H

#include <cassert>

#include "sbd/chemistry/tpb/helper.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

    size_t* base_memory;
    size_t* SinglesFromAlphaBraIndex;
    size_t* SinglesFromAlphaKetIndex;
    size_t* DoublesFromAlphaBraIndex;
    size_t* DoublesFromAlphaKetIndex;
    size_t* SinglesFromBetaBraIndex;
    size_t* SinglesFromBetaKetIndex;
    size_t* DoublesFromBetaBraIndex;
    size_t* DoublesFromBetaKetIndex;

    // pointer to cached c and d for energy calculation
    int* base_cran;
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

        base_memory = other.base_memory;
        SinglesFromAlphaBraIndex = other.SinglesFromAlphaBraIndex;
        SinglesFromAlphaKetIndex = other.SinglesFromAlphaKetIndex;
        DoublesFromAlphaBraIndex = other.DoublesFromAlphaBraIndex;
        DoublesFromAlphaKetIndex = other.DoublesFromAlphaKetIndex;
        SinglesFromBetaBraIndex = other.SinglesFromBetaBraIndex;
        SinglesFromBetaKetIndex = other.SinglesFromBetaKetIndex;
        DoublesFromBetaBraIndex = other.DoublesFromBetaBraIndex;
        DoublesFromBetaKetIndex = other.DoublesFromBetaKetIndex;

        base_cran = other.base_cran;
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
    }

    TaskHelpersThrust(thrust::device_vector<size_t>& storage, thrust::device_vector<int>& cran_storage, const TaskHelpers& helper, bool store_offset = false)
    {
        braAlphaStart = helper.braAlphaStart;
        braAlphaEnd = helper.braAlphaEnd;
        ketAlphaStart = helper.ketAlphaStart;
        ketAlphaEnd = helper.ketAlphaEnd;
        braBetaStart = helper.braBetaStart;
        braBetaEnd = helper.braBetaEnd;
        ketBetaStart = helper.ketBetaStart;
        ketBetaEnd = helper.ketBetaEnd;
        taskType = helper.taskType;
        adetShift = helper.adetShift;
        bdetShift = helper.bdetShift;

        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        thrust::host_vector<size_t> offset_ij;

        size_t size = 0;
        size_t size_cran = 0;

        size_single_alpha = 0;
        size_double_alpha = 0;
        size_single_beta = 0;
        size_double_beta = 0;

        std::vector<size_t> offset_single_alpha(braAlphaSize + 1);
        std::vector<size_t> offset_double_alpha(braAlphaSize + 1);
        if (taskType != 1) {
            for(size_t i=0; i < braAlphaSize; i++) {
                offset_single_alpha[i] = size_single_alpha;
                size_single_alpha += helper.SinglesFromAlphaLen[i];
                if (taskType == 2) {
                    offset_double_alpha[i] = size_double_alpha;
                    size_double_alpha += helper.DoublesFromAlphaLen[i];
                }
            }
            offset_single_alpha[braAlphaSize] = size_single_alpha;
            offset_double_alpha[braAlphaSize] = size_double_alpha;
            size += size_single_alpha + size_double_alpha;
            size_cran += size_single_alpha * 2 + size_double_alpha * 4;

            if (store_offset) {
                size += (braAlphaSize + 1);
                if (taskType == 2)
                    size += (braAlphaSize + 1);
            }
        }

        std::vector<size_t> offset_single_beta(braBetaSize + 1);
        std::vector<size_t> offset_double_beta(braBetaSize + 1);
        if (taskType != 2) {
            for(size_t i=0; i < braBetaSize; i++) {
                offset_single_beta[i] = size_single_beta;
                size_single_beta += helper.SinglesFromBetaLen[i];
                if (taskType == 1) {
                    offset_double_beta[i] = size_double_beta;
                    size_double_beta += helper.DoublesFromBetaLen[i];
                }
            }
            offset_single_beta[braBetaSize] = size_single_beta;
            offset_double_beta[braBetaSize] = size_double_beta;
            size += size_single_beta + size_double_beta;
            size_cran += size_single_beta * 2 + size_double_beta * 4;

            if (store_offset) {
                size += (braBetaSize + 1);
                if (taskType == 1)
                    size += (braBetaSize + 1);
            }
        }

        if (!store_offset)
            size *= 2;
        storage.resize(size);
        base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        if (store_offset) {
            // store offsets for non-collapsed loop calculation
            if (taskType != 1) {
                SinglesFromAlphaOffset = base_memory + count;
                thrust::copy_n(offset_single_alpha.begin(), braAlphaSize + 1, storage.begin() + count);
                count += braAlphaSize + 1;

                if (taskType == 2) {
                    DoublesFromAlphaOffset = base_memory + count;
                    thrust::copy_n(offset_double_alpha.begin(), braAlphaSize + 1, storage.begin() + count);
                    count += braAlphaSize + 1;
                }
            }

            if (taskType != 2) {
                SinglesFromBetaOffset = base_memory + count;
                thrust::copy_n(offset_single_beta.begin(), braBetaSize + 1, storage.begin() + count);
                count += braBetaSize + 1;

                if (taskType == 1) {
                    DoublesFromBetaOffset = base_memory + count;
                    thrust::copy_n(offset_double_beta.begin(), braBetaSize + 1, storage.begin() + count);
                    count += braBetaSize + 1;
                }
            }
        }

        SinglesFromAlphaKetIndex = nullptr;
        SinglesFromAlphaBraIndex = nullptr;
        DoublesFromAlphaKetIndex = nullptr;
        DoublesFromAlphaBraIndex = nullptr;
        if (taskType != 1) {
            if (store_offset) {
                SinglesFromAlphaKetIndex = base_memory + count;
            } else {
                SinglesFromAlphaKetIndex = base_memory + count;
                SinglesFromAlphaBraIndex = base_memory + count + size_single_alpha;
            }
            for(size_t i=0; i < braAlphaSize; i++) {
                thrust::copy_n(helper.SinglesFromAlphaSM[i], helper.SinglesFromAlphaLen[i], storage.begin() + count + offset_single_alpha[i]);
#ifdef SBD_DEBUG
                // **** DEBUG ****
                printf("[%s,%d] SinglesFromAlphaKetIndex: i=%llu, n=%llu, ", __FILE__, __LINE__,
                       i, helper.SinglesFromAlphaLen[i]);
                for(size_t j=0; j <helper.SinglesFromAlphaLen[i];j++) {
                    printf(" %llu:%llu,", j, helper.SinglesFromAlphaSM[i][j]);
                }
                printf("\n");
#endif
            }
            if (!store_offset) {
                for(size_t i=0; i < braAlphaSize; i++) {
                    thrust::fill_n(storage.begin() + size_single_alpha + count + offset_single_alpha[i], helper.SinglesFromAlphaLen[i], i + helper.braAlphaStart);
#ifdef SBD_DEBUG
                    // **** DEBUG ****
                    printf("[%s,%d] SinglesFromAlphaBraIndex: i=%llu, n=%llu, offset=%llu\n", __FILE__, __LINE__,
                           i, helper.SinglesFromAlphaLen[i], offset_single_alpha[i] );
#endif
                }
            }
            count += size_single_alpha;
            if (!store_offset) {
                count += size_single_alpha;
            }

            if (taskType == 2) {
                if (store_offset) {
                    DoublesFromAlphaKetIndex = base_memory + count;
                } else {
                    DoublesFromAlphaKetIndex = base_memory + count;
                    DoublesFromAlphaBraIndex = base_memory + count + size_double_alpha;
                }
                for(size_t i=0; i < braAlphaSize; i++) {
                    thrust::copy_n(helper.DoublesFromAlphaSM[i], helper.DoublesFromAlphaLen[i], storage.begin() + count + offset_double_alpha[i]);
                    if (!store_offset) {
                        thrust::fill_n(storage.begin() + count + size_double_alpha + offset_double_alpha[i], helper.DoublesFromAlphaLen[i], i + helper.braAlphaStart);
                    }
                }
                count += size_double_alpha;
                if (!store_offset) {
                    count += size_double_alpha;
                }
            }
        }

        SinglesFromBetaKetIndex = nullptr;
        SinglesFromBetaBraIndex = nullptr;
        DoublesFromBetaKetIndex = nullptr;
        DoublesFromBetaBraIndex = nullptr;
        if (taskType != 2) {
            if (store_offset) {
                SinglesFromBetaKetIndex = base_memory + count;
            } else {
                SinglesFromBetaKetIndex = base_memory + count;
                SinglesFromBetaBraIndex = base_memory + count + size_single_beta;
            }
            for(size_t i=0; i < braBetaSize; i++) {
                thrust::copy_n(helper.SinglesFromBetaSM[i], helper.SinglesFromBetaLen[i], storage.begin() + count + offset_single_beta[i]);
#ifdef SBD_DEBUG
                // **** DEBUG ****
                printf("[%s,%d] SinglesFromBetaKetIndex: i=%llu, n=%llu, ", __FILE__, __LINE__,
                       i, helper.SinglesFromBetaLen[i]);
                for(size_t j=0; j <helper.SinglesFromBetaLen[i];j++) {
                    printf(" %llu:%llu,", j, helper.SinglesFromBetaSM[i][j]);
                }
                printf("\n");
#endif
            }
            if (!store_offset) {
                for(size_t i=0; i < braBetaSize; i++) {
                    thrust::fill_n(storage.begin() + count + size_single_beta + offset_single_beta[i], helper.SinglesFromBetaLen[i], i + helper.braBetaStart);
#ifdef SBD_DEBUG
                    // **** DEBUG ****
                    printf("[%s,%d] SinglesFromBetaBraIndex: i=%llu, n=%llu, offset=%llu\n", __FILE__, __LINE__,
                           i, helper.SinglesFromBetaLen[i], offset_single_beta[i] );
#endif
                }
            }
            count += size_single_beta;
            if (!store_offset) {
                count += size_single_beta;
            }

            if (taskType == 1) {
                if (store_offset) {
                    DoublesFromBetaKetIndex = base_memory + count;
                } else {
                    DoublesFromBetaKetIndex = base_memory + count;
                    DoublesFromBetaBraIndex = base_memory + count + size_double_beta;
                }
                for(size_t i=0; i < braBetaSize; i++) {
                    thrust::copy_n(helper.DoublesFromBetaSM[i], helper.DoublesFromBetaLen[i], storage.begin() + count + offset_double_beta[i]);
                    if (!store_offset) {
                        thrust::fill_n(storage.begin() + count + size_double_beta + offset_double_beta[i], helper.DoublesFromBetaLen[i], i + helper.braBetaStart);
                    }
                }
                count += size_double_beta;
                if (!store_offset) {
                    count += size_double_beta;
                }
            }
        }

        // convert CrAn from AoS to SoA
        size_t count_cran = 0;
        std::vector<int> buf;

        cran_storage.resize(size_cran);
        base_cran = (int*)thrust::raw_pointer_cast(cran_storage.data());

        SinglesAlphaCrAnSM = nullptr;
        DoublesAlphaCrAnSM = nullptr;
        if (taskType != 1) {
            SinglesAlphaCrAnSM = base_cran + count_cran;
            buf.resize(size_single_alpha * 2);
#pragma omp parallel for
            for(size_t i=0; i < braAlphaSize; i++) {
                for (int j=0; j < helper.SinglesFromAlphaLen[i]; j++) {
                    buf[offset_single_alpha[i] + j] = helper.SinglesAlphaCrAnSM[i][j * 2];
                    buf[size_single_alpha + offset_single_alpha[i] + j] = helper.SinglesAlphaCrAnSM[i][j * 2 + 1];
                }
            }
#ifdef SBD_DEBUG
            // **** DEBUG ****
            for(size_t i=0; i < braAlphaSize; i++) {
                printf("[%s,%d] SinglesAlphaCrAnSM: i=%llu, n=%llu, ", __FILE__, __LINE__,
                       i, helper.SinglesFromAlphaLen[i]);
                for(size_t j=0; j <helper.SinglesFromAlphaLen[i];j++) {
                    printf(" %llu:%llu-%llu,", j,
                           helper.SinglesAlphaCrAnSM[i][j * 2],
                           helper.SinglesAlphaCrAnSM[i][j * 2 + 1]);
                }
                printf("\n");
            }
#endif
            thrust::copy_n(buf.begin(), size_single_alpha * 2, cran_storage.begin() + count_cran);
            count_cran += size_single_alpha * 2;

            if (taskType == 2) {
                DoublesAlphaCrAnSM = base_cran + count_cran;
                buf.resize(size_double_alpha * 4);
#pragma omp parallel for
                for(size_t i=0; i < braAlphaSize; i++) {
                    for (int j=0; j < helper.DoublesFromAlphaLen[i]; j++) {
                        buf[offset_double_alpha[i] + j] = helper.DoublesAlphaCrAnSM[i][j * 4];
                        buf[offset_double_alpha[i] + j + size_double_alpha] = helper.DoublesAlphaCrAnSM[i][j * 4 + 1];
                        buf[offset_double_alpha[i] + j + size_double_alpha * 2] = helper.DoublesAlphaCrAnSM[i][j * 4 + 2];
                        buf[offset_double_alpha[i] + j + size_double_alpha * 3] = helper.DoublesAlphaCrAnSM[i][j * 4 + 3];
                    }
                }
                thrust::copy_n(buf.begin(), size_double_alpha * 4, cran_storage.begin() + count_cran);
                count_cran += size_double_alpha * 4;
            }
        }

        SinglesBetaCrAnSM = nullptr;
        DoublesBetaCrAnSM = nullptr;
        if (taskType != 2) {
            SinglesBetaCrAnSM = base_cran + count_cran;
            buf.resize(size_single_beta * 2);
#pragma omp parallel for
            for(size_t i=0; i < braBetaSize; i++) {
                for (int j=0; j < helper.SinglesFromBetaLen[i]; j++) {
                    buf[offset_single_beta[i] + j] = helper.SinglesBetaCrAnSM[i][j * 2];
                    buf[size_single_beta + offset_single_beta[i] + j] = helper.SinglesBetaCrAnSM[i][j * 2 + 1];
                }
            }
#ifdef SBD_DEBUG
            // **** DEBUG ****
            for(size_t i=0; i < braBetaSize; i++) {
                printf("[%s,%d] SinglesBetaCrAnSM: i=%llu, n=%llu, ", __FILE__, __LINE__,
                       i, helper.SinglesFromBetaLen[i]);
                for(size_t j=0; j <helper.SinglesFromBetaLen[i];j++) {
                    printf(" %llu:%llu-%llu,", j,
                           helper.SinglesBetaCrAnSM[i][j * 2],
                           helper.SinglesBetaCrAnSM[i][j * 2 + 1]);
                }
                printf("\n");
            }
#endif
            thrust::copy_n(buf.begin(), size_single_beta * 2, cran_storage.begin() + count_cran);
            count_cran += size_single_beta * 2;

            if (taskType == 1) {
                DoublesBetaCrAnSM = base_cran + count_cran;
                buf.resize(size_double_beta * 4);
#pragma omp parallel for
                for(size_t i=0; i < braBetaSize; i++) {
                    for (int j=0; j < helper.DoublesFromBetaLen[i]; j++) {
                        buf[offset_double_beta[i] + j] = helper.DoublesBetaCrAnSM[i][j * 4];
                        buf[offset_double_beta[i] + j + size_double_beta] = helper.DoublesBetaCrAnSM[i][j * 4 + 1];
                        buf[offset_double_beta[i] + j + size_double_beta * 2] = helper.DoublesBetaCrAnSM[i][j * 4 + 2];
                        buf[offset_double_beta[i] + j + size_double_beta * 3] = helper.DoublesBetaCrAnSM[i][j * 4 + 3];
                    }
                }
                thrust::copy_n(buf.begin(), size_double_beta * 4, cran_storage.begin() + count_cran);
                count_cran += size_double_beta * 4;
            }
        }

#ifdef SBD_REORDER_INDEX_ARRAY
        //
        // Reorder excitation entries using a block-based permutation
        // derived from KetIndex.
        //
        // Motivation:
        //   - Accesses based on BraIndex are already relatively local,
        //     since SinglesFrom{Alpha,Beta}BraIndex are sorted.
        //   - In contrast, accesses derived from KetIndex (e.g., T[ketIdx])
        //     tend to be random and can hurt memory performance.
        //
        // A full sort by KetIndex is not used, because it would destroy
        // the locality of BraIndex and lead to more scattered accesses
        // for DetI / Wb updates.
        //
        // Instead, we apply a block-based grouping:
        //
        //     block_id = KetIndex / block_size
        //
        // Entries in the same block are placed contiguously, improving
        // locality of KetIndex-derived accesses while preserving some
        // of the original ordering (and thus BraIndex locality).
        //
        std::vector<size_t> permutation;
        // constexpr size_t block_size = 16;
        constexpr size_t block_size = 32;
#ifdef SBD_DEBUG_HELPER
        printf("[%s,%d] Reordering index arrays (block_size=%zu)\n",
               __FILE__, __LINE__, block_size);
#endif
        // NOTE:
        // block_size controls the trade-off between improving locality
        // of KetIndex-based accesses and preserving BraIndex locality.
        //
        // - Smaller block_size:
        //     preserves more of the original ordering (better for BraIndex),
        //     but gives limited improvement for KetIndex locality.
        //
        // - Larger block_size:
        //     improves grouping of nearby ket indices (better for T[ketIdx]),
        //     but increases the risk of disrupting BraIndex locality.
        //
        // Cr/An arrays are accessed in a streaming manner and are not
        // significantly affected by this permutation.
        //
        // The optimal value is workload- and architecture-dependent,
        // and should be determined empirically.

        //
        // Build a stable permutation that groups entries by KetIndex block.
        //
        // Steps:
        //   1. Copy KetIndex from device to host.
        //   2. Build a histogram of the number of entries in each block.
        //   3. Compute prefix sums to obtain per-block write offsets.
        //   4. Scatter original indices into 'permutation' using the prefix
        //      sum array as running write pointers.
        //
        // After this function, permutation[k] gives the original index of the
        // k-th entry in block-grouped order.
        //
        auto setup_permutation = [&](size_t* ket_index_ptr, size_t ket_index_size) {
            assert(ket_index_ptr);
            permutation.assign(ket_index_size, ket_index_size);
            std::vector<size_t> ket_index_vector(permutation.size());
            thrust::copy_n(thrust::device_pointer_cast(ket_index_ptr),
                           permutation.size(), ket_index_vector.begin());
            assert(!ket_index_vector.empty());
            size_t ket_index_limit = *std::max_element(ket_index_vector.begin(), ket_index_vector.end()) + 1;
            size_t n_blocks = (ket_index_limit + block_size - 1) / block_size;
            std::vector<size_t> histo(n_blocks, 0);
            std::vector<size_t> csum(n_blocks + 1, 0);
            std::vector<size_t> csum_org(n_blocks + 1);
            for (size_t i = 0; i < permutation.size(); i++) {
                size_t ket_index = ket_index_vector[i];
                assert(ket_index < ket_index_limit);
                size_t block_id = ket_index / block_size;
                assert(block_id < n_blocks);
                histo[block_id] += 1;
            }
            for (size_t i = 0; i < n_blocks; i++) {
                csum[i+1] = csum[i] + histo[i];
            }
            assert(csum[n_blocks] == permutation.size());
            std::copy(csum.begin(), csum.end(), csum_org.begin());
            for (size_t i = 0; i < permutation.size(); i++) {
                size_t ket_index = ket_index_vector[i];
                assert(ket_index < ket_index_limit);
                size_t block_id = ket_index / block_size;
                assert(block_id < n_blocks);
                size_t pos = csum[block_id]++;
                assert(csum_org[block_id] <= pos && pos < csum_org[block_id+1]);
                permutation[pos] = i;
            }
            for (size_t i = 0; i < n_blocks; i++) {
                assert(csum[i] == csum_org[i] + histo[i]);
            }
            for (size_t i = 0; i < permutation.size(); i++) {
                assert(permutation[i] < permutation.size());
            }
        };

        // Reorder a device array using the permutation built above.
        //
        // The array is copied once to host memory, permuted on host, and then
        // copied back to device memory.
        //
        auto reorder_device_array = [&](auto* ptr) {
            if (!ptr) return;
            using T = std::remove_pointer_t<decltype(ptr)>;
            auto dev_ptr = thrust::device_pointer_cast(ptr);
            std::vector<T> original_vector(permutation.size());
            std::vector<T> permuted_vector(permutation.size());
            thrust::copy_n(dev_ptr, permutation.size(), original_vector.begin());
            for (size_t i = 0; i < permutation.size(); i++) {
                permuted_vector[i] = original_vector[permutation[i]];
            }
            thrust::copy_n(permuted_vector.begin(), permutation.size(), dev_ptr);
        };

        // Reorder alpha single-excitation arrays
        if (size_single_alpha > 0 && SinglesFromAlphaKetIndex) {
#ifdef SBD_DEBUG_HELPER
            printf("[%s,%d] Reordering index arrays (size_single_alpha=%zu)\n",
                   __FILE__, __LINE__, size_single_alpha);
#endif
            setup_permutation(SinglesFromAlphaKetIndex, size_single_alpha);
            reorder_device_array(SinglesFromAlphaKetIndex);
            reorder_device_array(SinglesFromAlphaBraIndex);
            if (SinglesAlphaCrAnSM) {
                reorder_device_array(SinglesAlphaCrAnSM);
                reorder_device_array(SinglesAlphaCrAnSM + size_single_alpha);
            }
        }

        // Reorder beta single-excitation arrays
        if (size_single_beta > 0 && SinglesFromBetaKetIndex) {
#ifdef SBD_DEBUG_HELPER
            printf("[%s,%d] Reordering index arrays (size_single_beta=%zu)\n",
                   __FILE__, __LINE__, size_single_beta);
#endif
            setup_permutation(SinglesFromBetaKetIndex, size_single_beta);
            reorder_device_array(SinglesFromBetaKetIndex);
            reorder_device_array(SinglesFromBetaBraIndex);
            if (SinglesBetaCrAnSM) {
                reorder_device_array(SinglesBetaCrAnSM);
                reorder_device_array(SinglesBetaCrAnSM + size_single_beta);
            }
        }

        // Reorder alpha double-excitation arrays
        if (size_double_alpha > 0 && DoublesFromAlphaKetIndex) {
#ifdef SBD_DEBUG_HELPER
            printf("[%s,%d] Reordering index arrays (size_double_alpha=%zu)\n",
                   __FILE__, __LINE__, size_double_alpha);
#endif
            setup_permutation(DoublesFromAlphaKetIndex, size_double_alpha);
            reorder_device_array(DoublesFromAlphaKetIndex);
            reorder_device_array(DoublesFromAlphaBraIndex);
            if (DoublesAlphaCrAnSM) {
                reorder_device_array(DoublesAlphaCrAnSM);
                reorder_device_array(DoublesAlphaCrAnSM + size_double_alpha);
                reorder_device_array(DoublesAlphaCrAnSM + size_double_alpha * 2);
                reorder_device_array(DoublesAlphaCrAnSM + size_double_alpha * 3);
            }
        }

        // Reorder beta double-excitation arrays
        if (size_double_beta > 0 && DoublesFromBetaKetIndex) {
#ifdef SBD_DEBUG_HELPER
            printf("[%s,%d] Reordering index arrays (size_double_beta=%zu)\n",
                   __FILE__, __LINE__, size_double_beta);
#endif
            setup_permutation(DoublesFromBetaKetIndex, size_double_beta);
            reorder_device_array(DoublesFromBetaKetIndex);
            reorder_device_array(DoublesFromBetaBraIndex);
            if (DoublesBetaCrAnSM) {
                reorder_device_array(DoublesBetaCrAnSM);
                reorder_device_array(DoublesBetaCrAnSM + size_double_beta);
                reorder_device_array(DoublesBetaCrAnSM + size_double_beta * 2);
                reorder_device_array(DoublesBetaCrAnSM + size_double_beta * 3);
            }
        }
#endif  // #ifdef SBD_REORDER_INDEX_ARRAY
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
