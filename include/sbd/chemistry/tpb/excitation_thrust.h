/**
@file sbd/chemistry/tpb/excitation_thrust.h
@brief class for excitation storage for GPU
*/
#ifndef SBD_CHEMISTRY_TPB_EXCITATION_THRUST_H
#define SBD_CHEMISTRY_TPB_EXCITATION_THRUST_H

#include <algorithm>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "sbd/framework/det_vector.h"
#include "sbd/framework/mpi_utility.h"      // get_mpi_range
#include "sbd/chemistry/basic/determinants.h" // difference, OrbitalDifference

namespace sbd
{

class ExcitationThrust {
public:
    size_t braAlphaStart;
    size_t braAlphaEnd;
    size_t braBetaStart;
    size_t braBetaEnd;
    size_t local_adets_size;
    size_t local_bdets_size;

    thrust::device_vector<size_t> SinglesFromAlpha;
    thrust::device_vector<size_t> DoublesFromAlpha;
    thrust::device_vector<size_t> SinglesFromBeta;
    thrust::device_vector<size_t> DoublesFromBeta;
    thrust::device_vector<size_t> SinglesFromAlphaOffset;
    thrust::device_vector<size_t> DoublesFromAlphaOffset;
    thrust::device_vector<size_t> SinglesFromBetaOffset;
    thrust::device_vector<size_t> DoublesFromBetaOffset;
    thrust::device_vector<size_t> SinglesFromAlphaBraIndex;
    thrust::device_vector<size_t> DoublesFromAlphaBraIndex;
    thrust::device_vector<size_t> SinglesFromBetaBraIndex;
    thrust::device_vector<size_t> DoublesFromBetaBraIndex;
    thrust::device_vector<int> SinglesFromAlphaCrAn;
    thrust::device_vector<int> DoublesFromAlphaCrAn;
    thrust::device_vector<int> SinglesFromBetaCrAn;
    thrust::device_vector<int> DoublesFromBetaCrAn;

    // block ranges for this process (used to be stored in helper)
    size_t num_alpha_blocks = 1;
    size_t num_beta_blocks = 1;
    std::vector<size_t> ketAlphaStart;
    std::vector<size_t> ketAlphaEnd;
    std::vector<size_t> ketBetaStart;
    std::vector<size_t> ketBetaEnd;
    std::vector<size_t> alphaBlockSchedule;
    std::vector<size_t> betaBlockSchedule;
    std::vector<size_t> adetShift;
    std::vector<size_t> bdetShift;
    size_t num_total_blocks = 1;

    std::vector<size_t> single_alpha_offset;
    std::vector<size_t> double_alpha_offset;
    std::vector<size_t> single_beta_offset;
    std::vector<size_t> double_beta_offset;

    // size for each block
    std::vector<size_t> size_single_alpha;
    std::vector<size_t> size_double_alpha;
    std::vector<size_t> size_single_beta;
    std::vector<size_t> size_double_beta;

    ExcitationThrust()
    {

    }

private:
    // The host-side *_offset arrays are one global prefix sum over all blocks, so
    // entry [b*n + i] counts every excitation in blocks 0..b-1 as well.  The
    // offset-based kernels index them per block as [a] and [a+1] with a pointer
    // already advanced to the block, so each block needs its own prefix sum
    // starting at 0 and needs n+1 entries (the kernel reads [a+1] with a == n-1).
    // Lay the device copy out as num_blocks contiguous (n+1)-entry runs.
    static void upload_block_relative_offsets(const std::vector<size_t>& host_offset,
                                              size_t block_len,
                                              size_t num_blocks,
                                              thrust::device_vector<size_t>& out)
    {
        const size_t stride = block_len + 1;
        std::vector<size_t> rel(num_blocks * stride);
        for (size_t b = 0; b < num_blocks; b++) {
            const size_t base = host_offset[b * block_len];
#pragma omp parallel for
            for (size_t i = 0; i <= block_len; i++) {
                rel[b * stride + i] = host_offset[b * block_len + i] - base;
            }
        }
        out.resize(rel.size());
        thrust::copy_n(rel.begin(), rel.size(), out.begin());
    }

#ifdef SBD_REORDER_INDEX_ARRAY
    //
    // Group excitation entries by ket-index block to improve locality of the
    // T[ketIdx] loads in the mult kernels.
    //
    // Motivation (unchanged from the original formulation in helper_thrust.h):
    //   - Accesses based on BraIndex are already relatively local, since the
    //     BraIndex arrays are sorted.
    //   - Accesses derived from KetIndex (e.g. T[ketIdx]) tend to be random and
    //     hurt memory performance.
    // A full sort by KetIndex is not used, because it would destroy BraIndex
    // locality and scatter the DetI / Wb accesses.  Instead entries are grouped
    // by block_id = KetIndex / block_size, which clusters nearby ket indices
    // while preserving much of the original ordering.
    //
    // Two differences from the original, both deliberate:
    //
    //  1. The permutation is applied independently *within each bra
    //     determinant's* sub-range [offset[a], offset[a+1]) instead of across the
    //     whole array.  This is what makes it compatible with the offset-based
    //     kernels (MultSharedDet, MultTask0/1/2, CorrelationTask0/1/2): per-bra
    //     counts are untouched, so the offset arrays remain valid without being
    //     rebuilt, and BraIndex is constant inside a sub-range.  A permutation
    //     that crossed bra boundaries -- what the original did -- interleaves
    //     entries of different bra determinants and simply cannot be described by
    //     the offset arrays, which is why it silently produced a wrong H*v.
    //
    //  2. It runs here, once per block, on host data before the single H2D
    //     upload.  ExcitationThrust owns one copy of each block's arrays and many
    //     task helpers alias it, so permuting in the helper constructor reordered
    //     shared data once per referring task -- repeatedly, and underneath
    //     helpers already built.  Doing it here also drops the original's
    //     per-array D2H/H2D round trip.
    //
    // ket_index / bra_index / cran are the full host arrays for one spin channel;
    // cran is SoA with ncomp components, based at (block element offset * ncomp)
    // and strided by the block's own element count.  bra_index may be empty when
    // the BraIndex arrays were not requested.
    //
    // NOTE: block_size trades ket locality against preserving the original order
    // (and thus BraIndex locality).  Larger groups nearby ket indices more
    // aggressively; smaller preserves more of the original ordering.  Cr/An are
    // streamed and largely insensitive.  The optimum is workload- and
    // architecture-dependent and should be determined empirically.
    static void reorder_by_ket_block(std::vector<size_t>& ket_index,
                                     std::vector<size_t>& bra_index,
                                     std::vector<int>& cran,
                                     size_t ncomp,
                                     const std::vector<size_t>& offset,
                                     size_t block_len,
                                     size_t num_blocks)
    {
        // constexpr size_t block_size = 16;
        constexpr size_t block_size = 32;
        const bool has_bra = !bra_index.empty();

        for (size_t b = 0; b < num_blocks; b++) {
            const size_t base_elem = offset[b * block_len];
            const size_t n_elem = offset[(b + 1) * block_len] - base_elem;
            if (n_elem == 0) continue;
            const size_t cran_base = base_elem * ncomp;

#pragma omp parallel
            {
                // permutation[k] = index (within the sub-range) of the entry that
                // belongs at position k after grouping.
                std::vector<size_t> permutation;
                std::vector<size_t> histo;
                std::vector<size_t> csum;
                std::vector<size_t> scratch_s;
                std::vector<int> scratch_i;

#pragma omp for
                for (size_t a = 0; a < block_len; a++) {
                    const size_t lo = offset[b * block_len + a];
                    const size_t hi = offset[b * block_len + a + 1];
                    const size_t n = hi - lo;
                    if (n < 2) continue;  // nothing to reorder

                    // Build a stable permutation grouping [lo, hi) by ket block.
                    // Group ids are taken relative to the smallest ket index in
                    // the sub-range: these are global det indices, so anchoring at
                    // 0 would size the histogram by the largest index rather than
                    // by the span actually present.
                    size_t ket_min = ket_index[lo];
                    size_t ket_max = ket_index[lo];
                    for (size_t i = lo + 1; i < hi; i++) {
                        ket_min = std::min(ket_min, ket_index[i]);
                        ket_max = std::max(ket_max, ket_index[i]);
                    }
                    const size_t group_base = ket_min / block_size;
                    const size_t n_groups = ket_max / block_size - group_base + 1;

                    histo.assign(n_groups, 0);
                    for (size_t i = lo; i < hi; i++)
                        histo[ket_index[i] / block_size - group_base]++;

                    csum.assign(n_groups + 1, 0);
                    for (size_t g = 0; g < n_groups; g++)
                        csum[g + 1] = csum[g] + histo[g];

                    permutation.assign(n, n);
                    for (size_t i = lo; i < hi; i++)
                        permutation[csum[ket_index[i] / block_size - group_base]++] = i - lo;

                    // Apply it to KetIndex, BraIndex and every CrAn component.
                    scratch_s.resize(n);
                    for (size_t k = 0; k < n; k++)
                        scratch_s[k] = ket_index[lo + permutation[k]];
                    std::copy_n(scratch_s.begin(), n, ket_index.begin() + lo);

                    if (has_bra) {
                        // Constant across the sub-range today, permuted anyway so
                        // the three arrays stay consistent by construction.
                        for (size_t k = 0; k < n; k++)
                            scratch_s[k] = bra_index[lo + permutation[k]];
                        std::copy_n(scratch_s.begin(), n, bra_index.begin() + lo);
                    }

                    scratch_i.resize(n);
                    const size_t sub = lo - base_elem;  // within-block element index
                    for (size_t c = 0; c < ncomp; c++) {
                        const size_t comp_base = cran_base + c * n_elem + sub;
                        for (size_t k = 0; k < n; k++)
                            scratch_i[k] = cran[comp_base + permutation[k]];
                        std::copy_n(scratch_i.begin(), n, cran.begin() + comp_base);
                    }
                }
            }
        }
    }
#endif  // SBD_REORDER_INDEX_ARRAY

public:

    void Init(  const det_vector<size_t, det_kind::half> &adets,
                const det_vector<size_t, det_kind::half> &bdets,
                const uint32_t bit_length,
                const size_t norbs,
                const size_t adet_comm_size,
                const size_t bdet_comm_size,
                MPI_Comm b_comm,
                MPI_Comm a_comm,
                bool with_bra, bool store_offset
    )
    {
        int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
        int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
        int bra_rank = mpi_rank_b;
        int bra_adet_rank = bra_rank / bdet_comm_size;
        int bra_bdet_rank = bra_rank % bdet_comm_size;

        num_total_blocks = adet_comm_size * bdet_comm_size;

        // block scheduling
        alphaBlockSchedule.resize(num_total_blocks);
        betaBlockSchedule.resize(num_total_blocks);
        adetShift.resize(num_total_blocks);
        bdetShift.resize(num_total_blocks);
        size_t block_count = 0;
        for (int na = 0; na < adet_comm_size; na++) {
            for (int nb = 0; nb < bdet_comm_size; nb++) {
                alphaBlockSchedule[block_count] = (bra_adet_rank + na) % adet_comm_size;
                betaBlockSchedule[block_count] = (bra_bdet_rank + nb) % bdet_comm_size;
                adetShift[block_count] = na;
                bdetShift[block_count] = nb;
                block_count++;
            }
        }

        // range for each block
        braAlphaStart = 0;
        braAlphaEnd = adets.size();
        braBetaStart = 0;
        braBetaEnd = bdets.size();

        get_mpi_range(adet_comm_size, bra_adet_rank, braAlphaStart, braAlphaEnd);
        get_mpi_range(bdet_comm_size, bra_bdet_rank, braBetaStart, braBetaEnd);
        local_adets_size = braAlphaEnd - braAlphaStart;
        local_bdets_size = braBetaEnd - braBetaStart;

        num_alpha_blocks = adet_comm_size;
        ketAlphaStart.assign(num_alpha_blocks, 0);
        ketAlphaEnd.assign(num_alpha_blocks, adets.size());
        for (size_t ia = 0; ia < num_alpha_blocks; ia++) {
            get_mpi_range(adet_comm_size, ia, ketAlphaStart[ia], ketAlphaEnd[ia]);
        }

        num_beta_blocks = bdet_comm_size;
        ketBetaStart.assign(num_beta_blocks, 0);
        ketBetaEnd.assign(num_beta_blocks, bdets.size());
        for (size_t ib = 0; ib < num_beta_blocks; ib++) {
            get_mpi_range(bdet_comm_size, ib, ketBetaStart[ib], ketBetaEnd[ib]);
        }

        // make alpha excitation on host
        single_alpha_offset.resize(num_alpha_blocks * local_adets_size + 1);
        double_alpha_offset.resize(num_alpha_blocks * local_adets_size + 1);
        single_alpha_offset[0] = 0;
        double_alpha_offset[0] = 0;

        for (size_t ba = 0; ba < num_alpha_blocks; ba++) {
#pragma omp parallel for
            for (size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
                size_t scount = 0;
                size_t dcount = 0;
                for (size_t ja = ketAlphaStart[ba]; ja < ketAlphaEnd[ba]; ja++) {
                    int d = difference(adets[ia], adets[ja], bit_length, norbs);
                    if ( d == 2 ) scount++;
                    else if ( d == 4 ) dcount++;
                }
                single_alpha_offset[ba * local_adets_size + (ia - braAlphaStart) + 1] = scount;
                double_alpha_offset[ba * local_adets_size + (ia - braAlphaStart) + 1] = dcount;
            }
        }
        for (size_t i = 0; i < num_alpha_blocks * local_adets_size; i++) {
            single_alpha_offset[i + 1] += single_alpha_offset[i];
            double_alpha_offset[i + 1] += double_alpha_offset[i];
        }
        size_t num_single_alpha = single_alpha_offset[num_alpha_blocks * local_adets_size];
        size_t num_double_alpha = double_alpha_offset[num_alpha_blocks * local_adets_size];

        std::vector<size_t> single_alpha_index(num_single_alpha);
        std::vector<size_t> double_alpha_index(num_double_alpha);
        std::vector<size_t> single_alpha_bra_index;
        std::vector<size_t> double_alpha_bra_index;
        if (with_bra) {
            single_alpha_bra_index.resize(num_single_alpha);
            double_alpha_bra_index.resize(num_double_alpha);
        }

        std::vector<int> single_alpha_cran(num_single_alpha * 2);
        std::vector<int> double_alpha_cran(num_double_alpha * 4);
        size_single_alpha.resize(num_alpha_blocks);
        size_double_alpha.resize(num_alpha_blocks);
        for (size_t ba = 0; ba < num_alpha_blocks; ba++) {
            const size_t soffset = single_alpha_offset[ba * local_adets_size];
            const size_t doffset = double_alpha_offset[ba * local_adets_size];
            const size_t nsingle = single_alpha_offset[(ba + 1) * local_adets_size] - soffset;
            const size_t ndouble = double_alpha_offset[(ba + 1) * local_adets_size] - doffset;
            size_single_alpha[ba] = nsingle;
            size_double_alpha[ba] = ndouble;
            const size_t scran_base = soffset * 2;
            const size_t dcran_base = doffset * 4;
#pragma omp parallel
            {
                std::vector<int> cr; cr.reserve(2);
                std::vector<int> an; an.reserve(2);
#pragma omp for
                for (size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
                    size_t scount = single_alpha_offset[ba * local_adets_size + ia - braAlphaStart] - soffset;
                    size_t dcount = double_alpha_offset[ba * local_adets_size + ia - braAlphaStart] - doffset;
                    for (size_t ja = ketAlphaStart[ba]; ja < ketAlphaEnd[ba]; ja++) {
                        int d = difference(adets[ia], adets[ja], bit_length, norbs);
                        if ( d == 2 ) {
                    	    OrbitalDifference(adets[ia], adets[ja], bit_length, norbs, cr, an);
                            single_alpha_index[soffset + scount] = ja;
                            single_alpha_cran[scran_base + scount] = 2 * cr[0];
                            single_alpha_cran[scran_base + nsingle + scount] = 2 * an[0];
                            if (with_bra)
                                single_alpha_bra_index[soffset + scount] = ia;
                            scount++;
                        } else if ( d == 4 ) {
                    	    OrbitalDifference(adets[ia], adets[ja], bit_length, norbs, cr, an);
                            double_alpha_index[doffset + dcount] = ja;
                            double_alpha_cran[dcran_base + dcount] = 2 * cr[0];
                            double_alpha_cran[dcran_base + ndouble + dcount] = 2 * cr[1];
                            double_alpha_cran[dcran_base + ndouble * 2 + dcount] = 2 * an[0];
                            double_alpha_cran[dcran_base + ndouble * 3 + dcount] = 2 * an[1];
                            if (with_bra)
                                double_alpha_bra_index[doffset + dcount] = ia;
                            dcount++;
                        }
                    }
                }
            }
        }

#ifdef SBD_REORDER_INDEX_ARRAY
        // Reorder once here, for every block, before the H2D upload.
        reorder_by_ket_block(single_alpha_index, single_alpha_bra_index,
                            single_alpha_cran, 2,
                            single_alpha_offset, local_adets_size, num_alpha_blocks);
        reorder_by_ket_block(double_alpha_index, double_alpha_bra_index,
                            double_alpha_cran, 4,
                            double_alpha_offset, local_adets_size, num_alpha_blocks);
#endif

        SinglesFromAlpha.resize(num_single_alpha);
        DoublesFromAlpha.resize(num_double_alpha);
        SinglesFromAlphaCrAn.resize(num_single_alpha * 2);
        DoublesFromAlphaCrAn.resize(num_double_alpha * 4);
        thrust::copy_n(single_alpha_index.begin(), num_single_alpha, SinglesFromAlpha.begin());
        thrust::copy_n(double_alpha_index.begin(), num_double_alpha, DoublesFromAlpha.begin());
        thrust::copy_n(single_alpha_cran.begin(), num_single_alpha * 2 , SinglesFromAlphaCrAn.begin());
        thrust::copy_n(double_alpha_cran.begin(), num_double_alpha * 4 , DoublesFromAlphaCrAn.begin());
        if (store_offset) {
            upload_block_relative_offsets(single_alpha_offset, local_adets_size,
                                        num_alpha_blocks, SinglesFromAlphaOffset);
            upload_block_relative_offsets(double_alpha_offset, local_adets_size,
                                        num_alpha_blocks, DoublesFromAlphaOffset);
        }
        if (with_bra) {
            SinglesFromAlphaBraIndex.resize(num_single_alpha);
            DoublesFromAlphaBraIndex.resize(num_double_alpha);
            thrust::copy_n(single_alpha_bra_index.begin(), num_single_alpha, SinglesFromAlphaBraIndex.begin());
            thrust::copy_n(double_alpha_bra_index.begin(), num_double_alpha, DoublesFromAlphaBraIndex.begin());
        }

        // make beta excitation on host
        single_beta_offset.resize(num_beta_blocks * local_bdets_size + 1);
        double_beta_offset.resize(num_beta_blocks * local_bdets_size + 1);
        single_beta_offset[0] = 0;
        double_beta_offset[0] = 0;

        for (size_t bb = 0; bb < num_beta_blocks; bb++) {
#pragma omp parallel for
            for (size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
                size_t scount = 0;
                size_t dcount = 0;
                for (size_t jb = ketBetaStart[bb]; jb < ketBetaEnd[bb]; jb++) {
                    int d = difference(bdets[ib], bdets[jb], bit_length, norbs);
                    if ( d == 2 ) scount++;
                    else if ( d == 4 ) dcount++;
                }
                single_beta_offset[bb * local_bdets_size + (ib - braBetaStart) + 1] = scount;
                double_beta_offset[bb * local_bdets_size + (ib - braBetaStart) + 1] = dcount;
            }
        }
        for (size_t i = 0; i < num_beta_blocks * local_bdets_size; i++) {
            single_beta_offset[i + 1] += single_beta_offset[i];
            double_beta_offset[i + 1] += double_beta_offset[i];
        }
        size_t num_single_beta = single_beta_offset[num_beta_blocks * local_bdets_size];
        size_t num_double_beta = double_beta_offset[num_beta_blocks * local_bdets_size];

        std::vector<size_t> single_beta_index(num_single_beta);
        std::vector<size_t> double_beta_index(num_double_beta);
        std::vector<int> single_beta_cran(num_single_beta * 2);
        std::vector<int> double_beta_cran(num_double_beta * 4);
        std::vector<size_t> single_beta_bra_index;
        std::vector<size_t> double_beta_bra_index;
        if (with_bra) {
            single_beta_bra_index.resize(num_single_beta);
            double_beta_bra_index.resize(num_double_beta);
        }

        size_single_beta.resize(num_beta_blocks);
        size_double_beta.resize(num_beta_blocks);
        for (size_t bb = 0; bb < num_beta_blocks; bb++) {
            const size_t soffset = single_beta_offset[bb * local_bdets_size];
            const size_t doffset = double_beta_offset[bb * local_bdets_size];
            const size_t nsingle = single_beta_offset[(bb + 1) * local_bdets_size] - soffset;
            const size_t ndouble = double_beta_offset[(bb + 1) * local_bdets_size] - doffset;
            size_single_beta[bb] = nsingle;
            size_double_beta[bb] = ndouble;
            const size_t scran_base = soffset * 2;
            const size_t dcran_base = doffset * 4;
#pragma omp parallel
            {
                std::vector<int> cr; cr.reserve(2);
                std::vector<int> an; an.reserve(2);
#pragma omp for
                for (size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
                    size_t scount = single_beta_offset[bb * local_bdets_size + ib - braBetaStart] - soffset;
                    size_t dcount = double_beta_offset[bb * local_bdets_size + ib - braBetaStart] - doffset;
                    for (size_t jb = ketBetaStart[bb]; jb < ketBetaEnd[bb]; jb++) {
                        int d = difference(bdets[ib], bdets[jb], bit_length, norbs);
                        if ( d == 2 ) {
                    	    OrbitalDifference(bdets[ib], bdets[jb], bit_length, norbs, cr, an);
                            single_beta_index[soffset + scount] = jb;
                            single_beta_cran[scran_base + scount] = 2 * cr[0] + 1;
                            single_beta_cran[scran_base + nsingle + scount] = 2 * an[0] + 1;
                            if (with_bra)
                                single_beta_bra_index[soffset + scount] = ib;
                            scount++;
                        } else if ( d == 4 ) {
                    	    OrbitalDifference(bdets[ib], bdets[jb], bit_length, norbs, cr, an);
                            double_beta_index[doffset + dcount] = jb;
                            double_beta_cran[dcran_base + dcount] = 2 * cr[0] + 1;
                            double_beta_cran[dcran_base + ndouble + dcount] = 2 * cr[1] + 1;
                            double_beta_cran[dcran_base + ndouble * 2 + dcount] = 2 * an[0] + 1;
                            double_beta_cran[dcran_base + ndouble * 3 + dcount] = 2 * an[1] + 1;
                            if (with_bra)
                                double_beta_bra_index[doffset + dcount] = ib;
                            dcount++;
                        }
                    }
                }
            }
        }
#ifdef SBD_REORDER_INDEX_ARRAY
        reorder_by_ket_block(single_beta_index, single_beta_bra_index,
                            single_beta_cran, 2,
                            single_beta_offset, local_bdets_size, num_beta_blocks);
        reorder_by_ket_block(double_beta_index, double_beta_bra_index,
                            double_beta_cran, 4,
                            double_beta_offset, local_bdets_size, num_beta_blocks);
#endif

        SinglesFromBeta.resize(num_single_beta);
        DoublesFromBeta.resize(num_double_beta);
        SinglesFromBetaCrAn.resize(num_single_beta * 2);
        DoublesFromBetaCrAn.resize(num_double_beta * 4);
        thrust::copy_n(single_beta_index.begin(), num_single_beta, SinglesFromBeta.begin());
        thrust::copy_n(double_beta_index.begin(), num_double_beta, DoublesFromBeta.begin());
        thrust::copy_n(single_beta_cran.begin(), num_single_beta * 2 , SinglesFromBetaCrAn.begin());
        thrust::copy_n(double_beta_cran.begin(), num_double_beta * 4 , DoublesFromBetaCrAn.begin());
        if (store_offset) {
            upload_block_relative_offsets(single_beta_offset, local_bdets_size,
                                        num_beta_blocks, SinglesFromBetaOffset);
            upload_block_relative_offsets(double_beta_offset, local_bdets_size,
                                        num_beta_blocks, DoublesFromBetaOffset);
        }
        if (with_bra) {
            SinglesFromBetaBraIndex.resize(num_single_beta);
            DoublesFromBetaBraIndex.resize(num_double_beta);
            thrust::copy_n(single_beta_bra_index.begin(), num_single_beta, SinglesFromBetaBraIndex.begin());
            thrust::copy_n(double_beta_bra_index.begin(), num_double_beta, DoublesFromBetaBraIndex.begin());
        }
    }


    size_t* get_single_alpha_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromAlpha.data()) + single_alpha_offset[ba * local_adets_size];
    }
    size_t* get_double_alpha_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromAlpha.data()) + double_alpha_offset[ba * local_adets_size];
    }

    size_t* get_single_beta_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromBeta.data()) + single_beta_offset[bb * local_bdets_size];
    }
    size_t* get_double_beta_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromBeta.data()) + double_beta_offset[bb * local_bdets_size];
    }

    size_t* get_single_alpha_bra_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromAlphaBraIndex.data()) + single_alpha_offset[ba * local_adets_size];
    }
    size_t* get_double_alpha_bra_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromAlphaBraIndex.data()) + double_alpha_offset[ba * local_adets_size];
    }

    size_t* get_single_beta_bra_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromBetaBraIndex.data()) + single_beta_offset[bb * local_bdets_size];
    }
    size_t* get_double_beta_bra_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromBetaBraIndex.data()) + double_beta_offset[bb * local_bdets_size];
    }

    int* get_single_alpha_cran_ptr(size_t ba) const
    {
        return (int*)thrust::raw_pointer_cast(SinglesFromAlphaCrAn.data()) + single_alpha_offset[ba * local_adets_size] * 2;
    }
    int* get_double_alpha_cran_ptr(size_t ba) const
    {
        return (int*)thrust::raw_pointer_cast(DoublesFromAlphaCrAn.data()) + double_alpha_offset[ba * local_adets_size] * 4;
    }

    int* get_single_beta_cran_ptr(size_t bb) const
    {
        return (int*)thrust::raw_pointer_cast(SinglesFromBetaCrAn.data()) + single_beta_offset[bb * local_bdets_size] * 2;
    }
    int* get_double_beta_cran_ptr(size_t bb) const
    {
        return (int*)thrust::raw_pointer_cast(DoublesFromBetaCrAn.data()) + double_beta_offset[bb * local_bdets_size] * 4;
    }

    size_t* get_single_alpha_offset_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromAlphaOffset.data()) + ba * (local_adets_size + 1);
    }
    size_t* get_double_alpha_offset_ptr(size_t ba) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromAlphaOffset.data()) + ba * (local_adets_size + 1);
    }

    size_t* get_single_beta_offset_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(SinglesFromBetaOffset.data()) + bb * (local_bdets_size + 1);
    }
    size_t* get_double_beta_offset_ptr(size_t bb) const
    {
        return (size_t*)thrust::raw_pointer_cast(DoublesFromBetaOffset.data()) + bb * (local_bdets_size + 1);
    }

};




}

#endif//SBD_CHEMISTRY_TPB_EXCITATION_THRUST_H