/**
@file sbd/caop/basic/mult_thrust.h
@brief Thrust/CUDA GPU implementation of CAOP mult() — subwarp ballot-dispatch kernel.
Included by mult.h when SBD_THRUST is defined. Same call signature as the CPU version.
*/
#ifndef SBD_CAOP_BASIC_MULT_THRUST_H
#define SBD_CAOP_BASIC_MULT_THRUST_H

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <stdexcept>

#include "sbd/caop/basic/generalop.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/mpi_utility_thrust.h"

namespace sbd {

// ============================================================
// Tunable kernel parameters (override at compile time)
// ============================================================
#ifndef SBD_CAOP_MULT_BLOCK_SIZE
  #define SBD_CAOP_MULT_BLOCK_SIZE 64
#endif
#ifndef SBD_CAOP_MULT_SUBWARP_SIZE
  #define SBD_CAOP_MULT_SUBWARP_SIZE 32
#endif
#ifndef SBD_CAOP_MULT_MIN_BLOCKS_PER_SM
  #define SBD_CAOP_MULT_MIN_BLOCKS_PER_SM 32
#endif

static_assert(SBD_CAOP_MULT_BLOCK_SIZE % 32 == 0,
              "SBD_CAOP_MULT_BLOCK_SIZE must be a multiple of 32 (full warp)");
static_assert(32 % SBD_CAOP_MULT_SUBWARP_SIZE == 0,
              "SBD_CAOP_MULT_SUBWARP_SIZE must divide 32 (must be 1, 2, 4, 8, 16, or 32)");

// ============================================================
// CaopBufferSlider<ElemT>: CPU-staging MPI slider for twk.
// ============================================================
template <typename ElemT>
class CaopBufferSlider {
    MPI_Request req_sz_send_, req_sz_recv_, req_data_send_, req_data_recv_;
    size_t sz_send_buf_, sz_recv_buf_;
    bool have_send_, have_recv_;
    size_t recv_size_;
public:
    CaopBufferSlider() : sz_send_buf_(0), sz_recv_buf_(0),
                         have_send_(false), have_recv_(false), recv_size_(0) {}

    size_t get_recv_size() const { return recv_size_; }

    void ExchangeAsyncHost(const ElemT* h_send, size_t send_size,
                           ElemT* h_recv,  size_t max_recv,
                           int slide, MPI_Comm comm, int tag_base) {
        int mpi_rank, mpi_size;
        MPI_Comm_rank(comm, &mpi_rank);
        MPI_Comm_size(comm, &mpi_size);
        int dest   = (mpi_size + mpi_rank + slide) % mpi_size;
        int source = (mpi_size + mpi_rank - slide) % mpi_size;

        sz_send_buf_ = send_size;
        MPI_Isend(&sz_send_buf_, 1, SBD_MPI_SIZE_T, dest,   tag_base,   comm, &req_sz_send_);
        MPI_Irecv(&sz_recv_buf_, 1, SBD_MPI_SIZE_T, source, tag_base,   comm, &req_sz_recv_);

        MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
        have_send_ = (send_size > 0);
        if (have_send_)
            MPI_Isend(const_cast<ElemT*>(h_send), (int)send_size, DataT,
                      dest, tag_base + 1, comm, &req_data_send_);
        have_recv_ = (max_recv > 0);
        if (have_recv_)
            MPI_Irecv(h_recv, (int)max_recv, DataT,
                      source, tag_base + 1, comm, &req_data_recv_);
    }

    bool Sync() {
        MPI_Status st;
        MPI_Wait(&req_sz_send_, &st);
        MPI_Wait(&req_sz_recv_, &st);
        recv_size_ = sz_recv_buf_;
        if (have_send_) { MPI_Wait(&req_data_send_, &st); have_send_ = false; }
        if (have_recv_) { MPI_Wait(&req_data_recv_, &st); have_recv_ = false; }
        return true;
    }
};

// ============================================================
// Wb_init_kernel: scale wb and add diagonal term (for t_rank==0)
// ============================================================
template <typename ElemT>
struct CaopWbInitKernel {
    ElemT* d_wb;
    const ElemT* d_hd;
    const ElemT* d_twk;
    ElemT volp;
    bool add_diag;

    __host__ __device__ void operator()(size_t i) const {
        d_wb[i] *= volp;
        if (add_diag) d_wb[i] += d_hd[i] * d_twk[i];
    }
};

// ============================================================
// CaopMultKernel<ElemT>: GPU functor, one subwarp per bra.
// ============================================================
template <typename ElemT>
struct CaopMultKernel {
    static constexpr int SUBWARP       = SBD_CAOP_MULT_SUBWARP_SIZE;
    static constexpr int BlockSize     = SBD_CAOP_MULT_BLOCK_SIZE;
    static constexpr int GROUPS        = BlockSize / SUBWARP;
    static constexpr int BUF_PG        = 2 * SUBWARP;
    static constexpr int MinBlocksPerSM = SBD_CAOP_MULT_MIN_BLOCKS_PER_SM;

    // bra / output
    const size_t* d_bs;     // flat bra dets [n_bras * elem_size]
    ElemT*        d_wb;     // output [n_bras]
    int           n_bras;
    int           elem_size;

    // ket (current task)
    const size_t* d_tbs;    // flat ket dets [n_kets * elem_size]
    const ElemT*  d_twk;    // ket coefficients [n_kets]
    int           n_kets;

    // operator data (init-time)
    const size_t* d_m1;             // must-be-occupied  [n_terms * elem_size]
    const size_t* d_m2;             // must-be-unoccupied [n_terms * elem_size]
    const ElemT*  d_coeff;          // coefficients+reordering sign [n_terms]
    const int*    d_fops_bpos;      // bit positions (word-desc sorted) [total_fops]
    const int*    d_word_start;     // [n_terms * (elem_size+1)]
    const int*    d_ndag_per_word;  // creation ops per word [n_terms * elem_size]
    int           n_terms;
    bool          sign_flag;

    __device__ void operator()(size_t arg_i) {
        constexpr int SW   = SUBWARP;
        constexpr int G    = GROUPS;
        constexpr int BPGP = BUF_PG;

        const int ib   = (int)(arg_i / SW);
        const int lane = (int)(arg_i % SW);
        const int grp  = (int)(threadIdx.x / SW);

        if (ib >= n_bras) return;

        // Dynamic shared memory layout:
        //   s_bra  [G * (ES+1)] size_t   — bra cache, padded stride
        //   s_n    [G * BPGP]   int      — ballot candidate buffer
        //   wr_tmp [G]  WarpReduce::TempStorage
        // vk_scr removed: compute_contribution tracks cur_w in a register per lane.
        const int ps = elem_size + 1;  // padded stride
        extern __shared__ char raw[];
        size_t* s_bra = (size_t*)raw;
        int*    s_n   = (int*)(s_bra + G * ps);

        using WR = cub::WarpReduce<ElemT, SW>;
        typename WR::TempStorage* wr_tmp =
            (typename WR::TempStorage*)(s_n + G * BPGP);

        size_t* my_bra = s_bra + grp * ps;
        int*    my_n   = s_n   + grp * BPGP;

        // Per-subwarp ballot mask (handles SW < 32 with multiple groups per warp)
        const unsigned lane_in_warp = (unsigned)threadIdx.x % 32u;
        const unsigned grp_in_warp  = lane_in_warp / (unsigned)SW;
        const unsigned sw_lanes     = (SW == 32) ? 0xFFFFFFFFu : ((1u << SW) - 1u);
        const unsigned sw_mask      = sw_lanes << (grp_in_warp * SW);

        // Cooperative bra load (handles elem_size > SW)
        for (int w = lane; w < elem_size; w += SW)
            my_bra[w] = d_bs[(size_t)ib * elem_size + w];
        __syncwarp(sw_mask);

        ElemT thread_sum(0);
        int s_count = 0;

        for (int n_base = 0; n_base < n_terms; n_base += SW) {
            int n = n_base + lane;
            bool survives = false;
            if (n < n_terms) {
                survives = true;
                const size_t* m1 = d_m1 + (size_t)n * elem_size;
                const size_t* m2 = d_m2 + (size_t)n * elem_size;
                for (int w = 0; w < elem_size && survives; w++) {
                    if ((~my_bra[w] & m1[w]) || (my_bra[w] & m2[w]))
                        survives = false;
                }
            }
            unsigned ballot = __ballot_sync(sw_mask, survives);
            if (survives) {
                int rank = __popc(ballot & ((1u << lane_in_warp) - 1u));
                my_n[s_count + rank] = n;
            }
            s_count += __popc(ballot);

            if (s_count >= SW) {
                __syncwarp(sw_mask);
                if (lane < s_count)
                    thread_sum += compute_contribution(my_bra, my_n[lane]);
                // compact: shift remainder left by SW
                int next_idx = lane + SW;
                int next_val = (next_idx < s_count) ? my_n[next_idx] : 0;
                // fence before write
                __syncwarp(sw_mask);
                my_n[lane] = next_val;
                s_count -= SW;
            }
        }

        // drain remainder
        if (s_count > 0) {
            __syncwarp(sw_mask);
            if (lane < s_count)
                thread_sum += compute_contribution(my_bra, my_n[lane]);
        }

        ElemT total = WR(wr_tmp[grp]).Sum(thread_sum);
        if (lane == 0) d_wb[ib] += total;
    }

    __device__ ElemT compute_contribution(const size_t* my_bra, int m) {
        // n_occ_base starts as the full-bra popcount across all words.  At the
        // top of each w iteration we subtract __popcll(cur_w) — which equals
        // __popcll(my_bra[w]) before any ops touch cur_w — leaving the sum for
        // words w2 < w, which is what the sign logic needs.
        int sign = 1;
        int lo = 0, hi = n_kets;

        int n_occ_base = 0;
        if (sign_flag) {
            for (int w2 = 0; w2 < elem_size; w2++)
                n_occ_base += __popcll((unsigned long long)my_bra[w2]);
        }

        for (int w = elem_size - 1; w >= 0; w--) {
            int wi       = elem_size - 1 - w;
            int op_start = d_word_start[m * (elem_size + 1) + wi];
            int op_end   = d_word_start[m * (elem_size + 1) + wi + 1];
            int n_dag_w  = (op_start < op_end) ? d_ndag_per_word[m * elem_size + wi] : 0;

            size_t cur_w = my_bra[w];  // register: evolves as ops are applied

            if (sign_flag)
                n_occ_base -= __popcll((unsigned long long)cur_w);

            // creation ops
            for (int k = op_start, ke = op_start + n_dag_w; k < ke; k++) {
                size_t bit = (size_t)1 << d_fops_bpos[k];
                cur_w ^= bit;
                if (sign_flag) {
                    int n_occ = __popcll((unsigned long long)(cur_w & (bit - 1))) + n_occ_base;
                    if (n_occ & 1) sign = -sign;
                }
            }
            // annihilation ops
            for (int k = op_start + n_dag_w; k < op_end; k++) {
                size_t bit = (size_t)1 << d_fops_bpos[k];
                cur_w |= bit;
                if (sign_flag) {
                    int n_occ = __popcll((unsigned long long)(cur_w & (bit - 1))) + n_occ_base;
                    if (n_occ & 1) sign = -sign;
                }
            }

            int lo0 = lo;  // save before lower_bound narrows it

            // lower_bound for cur_w
            {
                int L = lo, R = hi;
                while (L < R) {
                    int mid = L + (R - L) / 2;
                    if (d_tbs[mid * elem_size + w] < cur_w) L = mid + 1;
                    else R = mid;
                }
                lo = L;
            }
            if (lo >= hi || d_tbs[lo * elem_size + w] != cur_w) return ElemT(0);

            if (w > 0) {
                // upper_bound — start from lo0 (same as lower_bound) for uniform access
                int L = lo0, R = hi;
                while (L < R) {
                    int mid = L + (R - L) / 2;
                    if (d_tbs[mid * elem_size + w] <= cur_w) L = mid + 1;
                    else R = mid;
                }
                hi = L;
            }
        }

        return d_coeff[m] * ElemT(sign) * d_twk[lo];
    }
};

// ============================================================
// Kernel launcher with dynamic shared memory
// ============================================================
template <typename F>
__global__ __launch_bounds__(F::BlockSize, F::MinBlocksPerSM)
void caop_mult_kernel_fn(size_t n_threads, F f) {
    size_t i = (size_t)blockIdx.x * F::BlockSize + threadIdx.x;
    if (i < n_threads) f(i);
}

template <typename F>
inline void launch_caop_mult(size_t n_bras, F f, size_t smem_bytes, cudaStream_t stream) {
    if (n_bras == 0) return;
    constexpr int BS = F::BlockSize;
    constexpr int SW = F::SUBWARP;
    size_t n_threads = n_bras * SW;
    size_t grid = (n_threads + BS - 1) / BS;
    caop_mult_kernel_fn<F><<<grid, BS, smem_bytes, stream>>>(n_threads, f);
}

// ============================================================
// CaopMultThrust<ElemT>: host-side driver
// ============================================================
template <typename ElemT>
class CaopMultThrust {
    // init-time GPU data
    thrust::device_vector<size_t> d_bs_;
    thrust::device_vector<ElemT>  d_hd_;
    thrust::device_vector<size_t> d_m1_;
    thrust::device_vector<size_t> d_m2_;
    thrust::device_vector<ElemT>  d_coeff_;
    thrust::device_vector<int>    d_fops_bpos_;
    thrust::device_vector<int>    d_word_start_;
    thrust::device_vector<int>    d_ndag_per_word_;

    // per-task ket det sequences — populated lazily on the first run() call
    // via live MPI slides, then cached on GPU for all subsequent iterations.
    std::vector<thrust::device_vector<size_t>> d_tbs_seq_;
    std::vector<size_t> n_kets_per_task_;
    bool tbs_initialized_;
    bool init_done_;

    // local bra det_vector (CPU) — stored at Init for seeding the first-run tbs slide
    det_vector<size_t> bs_;

    // global max n_kets across b_comm ranks (Allreduce at Init; fixed for all Davidson iters)
    size_t global_max_n_;

    int n_bras_;
    int n_terms_;
    int elem_size_;
    bool sign_;
    size_t smem_bytes_;

    std::vector<int> slide_;
    MPI_Comm h_comm_, b_comm_, t_comm_;

    // Persistent run() resources — allocated once on first run(), reused every Davidson iter.
    thrust::device_vector<ElemT> d_twk_[2];
    ElemT*          h_twk_[2];
    cudaStream_t    compute_stream_, copy_stream_;
    cudaEvent_t     copy_done_[2], compute_done_[2];

    // Compute tbs_seq via live MPI slides, upload to GPU.
    // Called once on the first run(); results cached in d_tbs_seq_ / n_kets_per_task_.
    void precompute_tbs_seq() {
        size_t n_tasks = slide_.size();
        d_tbs_seq_.resize(n_tasks);
        n_kets_per_task_.resize(n_tasks, 0);

        if (n_tasks == 0) return;

        det_vector<size_t> cur_tbs;
        if (slide_[0] != 0) {
            MpiSlide(bs_, cur_tbs, slide_[0], b_comm_);
        } else {
            cur_tbs = bs_;
        }

        for (size_t task = 0; task < n_tasks; task++) {
            n_kets_per_task_[task] = cur_tbs.size();
            const auto& flat = cur_tbs.cflat();
            d_tbs_seq_[task].resize(flat.size());
            if (!flat.empty())
                thrust::copy_n(flat.begin(), flat.size(), d_tbs_seq_[task].begin());

            if (task + 1 < n_tasks) {
                int bslide = slide_[task] - slide_[task + 1];
                det_vector<size_t> next_tbs;
                MpiSlide(cur_tbs, next_tbs, bslide, b_comm_);
                cur_tbs = std::move(next_tbs);
            }
        }
    }

    void free_run_resources() {
        if (h_twk_[0] != nullptr) {
            cudaFreeHost(h_twk_[0]); h_twk_[0] = nullptr;
            cudaFreeHost(h_twk_[1]); h_twk_[1] = nullptr;
            cudaEventDestroy(copy_done_[0]);
            cudaEventDestroy(copy_done_[1]);
            cudaEventDestroy(compute_done_[0]);
            cudaEventDestroy(compute_done_[1]);
            cudaStreamDestroy(copy_stream_);  copy_stream_  = nullptr;
            cudaStreamDestroy(compute_stream_); compute_stream_ = nullptr;
        }
    }

public:
    CaopMultThrust() : tbs_initialized_(false), init_done_(false), global_max_n_(0),
                       n_bras_(0), n_terms_(0), elem_size_(0),
                       sign_(false), smem_bytes_(0),
                       h_twk_{ nullptr, nullptr },
                       compute_stream_(nullptr), copy_stream_(nullptr),
                       copy_done_{ nullptr, nullptr },
                       compute_done_{ nullptr, nullptr } {}

    ~CaopMultThrust() { free_run_resources(); }

    bool is_initialized() const { return init_done_; }

    // Reset to uninitialized state. Must be called if H or the basis size changes
    // between sbd::diag() calls, to discard stale cached data.
    void reset() {
        tbs_initialized_ = false;
        init_done_ = false;
        d_tbs_seq_.clear();
        n_kets_per_task_.clear();
        free_run_resources();
    }

    // Init: uploads Hamiltonian and basis data to the GPU.
    // Idempotent: returns immediately if already initialized for the same problem.
    // Throws std::logic_error if dimensions changed without reset() being called.
    // Does NOT precompute tbs_seq — that happens lazily on the first run() call.
    void Init(const std::vector<ElemT>& hd,
              const det_vector<size_t>& bs,
              int bit_length,
              const GeneralOp<ElemT>& H,
              bool sign,
              const std::vector<int>& slide,
              MPI_Comm h_comm, MPI_Comm b_comm, MPI_Comm t_comm) {
        // Idempotent: return if already initialized for the same problem.
        // Throw if dimensions changed without reset() being called first.
        if (init_done_) {
            if ((int)H.o_.size() != n_terms_ ||
                (int)bs.size()   != n_bras_  ||
                slide.size()     != slide_.size())
                throw std::logic_error(
                    "CaopMultThrust::Init: problem dimensions changed; call reset() before reinitializing");
            return;
        }
        // Clear lazy state and free persistent run() resources for the new problem.
        tbs_initialized_ = false;
        d_tbs_seq_.clear();
        n_kets_per_task_.clear();
        free_run_resources();
        h_comm_ = h_comm; b_comm_ = b_comm; t_comm_ = t_comm;
        sign_   = sign;
        slide_  = slide;
        n_bras_ = (int)bs.size();
        elem_size_ = (n_bras_ > 0) ? (int)bs.elem_size() : 0;

        // Store local bra det_vector for seeding the first-run tbs slide
        bs_ = bs;

        // Upload bras
        {
            const auto& flat = bs.cflat();
            d_bs_.resize(flat.size());
            thrust::copy_n(flat.begin(), flat.size(), d_bs_.begin());
        }
        // Upload diagonal
        d_hd_.resize(hd.size());
        thrust::copy_n(hd.begin(), hd.size(), d_hd_.begin());

        n_terms_ = (int)H.o_.size();

        if (n_terms_ > 0 && elem_size_ > 0) {
            // Compute and upload masks
            det_vector<size_t> m1, m2;
            H.PrecomputeMasks(bit_length, m1, m2);
            {
                const auto& fm1 = m1.cflat();
                d_m1_.resize(fm1.size());
                thrust::copy_n(fm1.begin(), fm1.size(), d_m1_.begin());
            }
            {
                const auto& fm2 = m2.cflat();
                d_m2_.resize(fm2.size());
                thrust::copy_n(fm2.begin(), fm2.size(), d_m2_.begin());
            }

            // Build sorted operator tables
            const int ES = elem_size_;
            std::vector<ElemT> coeff_host(n_terms_);
            std::vector<int>   word_start_host(n_terms_ * (ES + 1), 0);
            std::vector<int>   ndag_per_word_host(n_terms_ * ES, 0);
            std::vector<int>   fops_bpos_host;
            fops_bpos_host.reserve(n_terms_ * 4);

            int fpos = 0;
            for (int m = 0; m < n_terms_; m++) {
                const auto& op = H.o_[m];
                int n_fops = (int)op.fops_.size();
                int n_dag  = op.n_dag_;

                struct OpInfo { int orig_idx, word, bpos, type; };
                std::vector<OpInfo> ops(n_fops);
                for (int k = 0; k < n_fops; k++) {
                    int q = op.fops_[k].q_;
                    ops[k] = { k, q / bit_length, q % bit_length, (k < n_dag) ? 0 : 1 };
                }

                // Sort: word DESC, creation (0) before annihilation (1), original order within
                std::stable_sort(ops.begin(), ops.end(), [](const OpInfo& a, const OpInfo& b) {
                    if (a.word != b.word) return a.word > b.word;
                    if (a.type != b.type) return a.type < b.type;
                    return a.orig_idx < b.orig_idx;
                });

                // Count inversions (number of anticommuting swaps from original to sorted order)
                std::vector<int> perm(n_fops);
                for (int i = 0; i < n_fops; i++) perm[i] = ops[i].orig_idx;
                int inv = 0;
                for (int i = 0; i < n_fops; i++)
                    for (int j = i + 1; j < n_fops; j++)
                        if (perm[i] > perm[j]) inv++;
                coeff_host[m] = H.c_[m] * ((sign && (inv & 1)) ? ElemT(-1) : ElemT(1));

                // Fill word_start and ndag_per_word
                int cur_wi   = 0;
                int cur_word = ES - 1;
                word_start_host[m * (ES + 1) + 0] = fpos;

                for (int i = 0; i < n_fops; i++) {
                    while (ops[i].word < cur_word) {
                        cur_wi++;
                        cur_word--;
                        word_start_host[m * (ES + 1) + cur_wi] = fpos;
                    }
                    fops_bpos_host.push_back(ops[i].bpos);
                    if (ops[i].type == 0) ndag_per_word_host[m * ES + cur_wi]++;
                    fpos++;
                }
                for (int wi = cur_wi + 1; wi <= ES; wi++)
                    word_start_host[m * (ES + 1) + wi] = fpos;
            }

            d_coeff_.resize(n_terms_);
            thrust::copy_n(coeff_host.begin(), n_terms_, d_coeff_.begin());
            d_fops_bpos_.resize(fops_bpos_host.size());
            if (!fops_bpos_host.empty())
                thrust::copy_n(fops_bpos_host.begin(), fops_bpos_host.size(), d_fops_bpos_.begin());
            d_word_start_.resize(word_start_host.size());
            thrust::copy_n(word_start_host.begin(), word_start_host.size(), d_word_start_.begin());
            d_ndag_per_word_.resize(ndag_per_word_host.size());
            thrust::copy_n(ndag_per_word_host.begin(), ndag_per_word_host.size(), d_ndag_per_word_.begin());
        }

        // Compute global max n_kets for buffer sizing (fixed across all Davidson iterations)
        {
            size_t local_n = (size_t)n_bras_;
            MPI_Allreduce(&local_n, &global_max_n_, 1, SBD_MPI_SIZE_T, MPI_MAX, b_comm_);
        }

        // Compute smem per block
        constexpr int SW   = CaopMultKernel<ElemT>::SUBWARP;
        constexpr int G    = CaopMultKernel<ElemT>::GROUPS;
        constexpr int BPGP = CaopMultKernel<ElemT>::BUF_PG;
        using WR = cub::WarpReduce<ElemT, SW>;
        int ps = elem_size_ + 1;
        smem_bytes_ = (size_t)G * ps * sizeof(size_t)          // s_bra
                    + (size_t)G * BPGP * sizeof(int)            // s_n
                    + (size_t)G * sizeof(typename WR::TempStorage); // wr_tmp
        init_done_ = true;
    }

    void run(const std::vector<ElemT>& wk, std::vector<ElemT>& wb) {
        if (slide_.empty()) return;

        // First call: allocate persistent buffers and compute tbs_seq.
        // All subsequent calls reuse d_twk_/h_twk_/streams/events and d_tbs_seq_.
        if (!tbs_initialized_) {
            d_twk_[0].resize(global_max_n_);
            d_twk_[1].resize(global_max_n_);
            cudaMallocHost(&h_twk_[0], global_max_n_ * sizeof(ElemT));
            cudaMallocHost(&h_twk_[1], global_max_n_ * sizeof(ElemT));
            cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&copy_stream_,    cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&copy_done_[0],    cudaEventDisableTiming);
            cudaEventCreateWithFlags(&copy_done_[1],    cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done_[0], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done_[1], cudaEventDisableTiming);
            // Pre-signal both compute_done slots so the first copy_stream wait doesn't hang.
            cudaEventRecord(compute_done_[0], 0);
            cudaEventRecord(compute_done_[1], 0);
            cudaDeviceSynchronize();
            precompute_tbs_seq();
            tbs_initialized_ = true;
        }

        // Verify mask sizes are consistent with elem_size_ set at Init().
        if (n_terms_ != 0 && elem_size_ != 0 &&
            ((int)d_m1_.size() != n_terms_ * elem_size_ ||
             (int)d_m2_.size() != n_terms_ * elem_size_))
            throw std::logic_error(
                "CaopMultThrust::run: mask size mismatch (n_terms_ or elem_size_ changed after Init())");

        int mpi_size_h; MPI_Comm_size(h_comm_, &mpi_size_h);
        int mpi_rank_h; MPI_Comm_rank(h_comm_, &mpi_rank_h);
        int mpi_size_t; MPI_Comm_size(t_comm_, &mpi_size_t);
        int mpi_rank_t; MPI_Comm_rank(t_comm_, &mpi_rank_t);

        // Upload wb to GPU
        thrust::device_vector<ElemT> d_wb(wb.size());
        thrust::copy_n(wb.begin(), wb.size(), d_wb.begin());

        // Initial slide of wk — when slide_[0]==0 (common case) use wk directly;
        // no intermediate copy.
        std::vector<ElemT> twk_init_buf;
        const ElemT* twk0_data;
        size_t twk0_size;
        if (slide_[0] != 0) {
            MpiSlide(wk, twk_init_buf, slide_[0], b_comm_);
            twk0_data = twk_init_buf.data();
            twk0_size = twk_init_buf.size();
        } else {
            twk0_data = wk.data();
            twk0_size = wk.size();
        }

        int active_buf = 0, recv_buf = 1;
        size_t twk_size[2] = { twk0_size, global_max_n_ };

        // Seed active buffer: CPU → pinned → GPU
        if (twk0_size > 0)
            std::memcpy(h_twk_[active_buf], twk0_data, twk0_size * sizeof(ElemT));
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_twk_[active_buf].data()),
                        h_twk_[active_buf],
                        twk0_size * sizeof(ElemT),
                        cudaMemcpyHostToDevice, copy_stream_);
        cudaEventRecord(copy_done_[active_buf], copy_stream_);
        cudaEventRecord(copy_done_[recv_buf],   copy_stream_);

        // Wb_init: scale + add diagonal (on compute_stream, after d_twk[active_buf] is ready)
        {
            ElemT volp = ElemT(1.0 / ((double)mpi_size_h * (double)mpi_size_t));
            CaopWbInitKernel<ElemT> wbinit;
            wbinit.d_wb  = thrust::raw_pointer_cast(d_wb.data());
            wbinit.d_hd  = thrust::raw_pointer_cast(d_hd_.data());
            wbinit.d_twk = thrust::raw_pointer_cast(d_twk_[active_buf].data());
            wbinit.volp  = volp;
            wbinit.add_diag = (mpi_rank_t == 0);
            auto ci = thrust::counting_iterator<size_t>(0);
            cudaStreamWaitEvent(compute_stream_, copy_done_[active_buf]);
            thrust::for_each_n(thrust::cuda::par.on(compute_stream_), ci, (size_t)n_bras_, wbinit);
        }

        CaopBufferSlider<ElemT> slider;

        for (size_t task = 0; task < slide_.size(); task++) {
            cudaStreamWaitEvent(compute_stream_, copy_done_[active_buf]);

            if (n_terms_ > 0) {
                CaopMultKernel<ElemT> kernel;
                kernel.d_bs          = thrust::raw_pointer_cast(d_bs_.data());
                kernel.d_wb          = thrust::raw_pointer_cast(d_wb.data());
                kernel.n_bras        = n_bras_;
                kernel.elem_size     = elem_size_;
                kernel.d_tbs         = thrust::raw_pointer_cast(d_tbs_seq_[task].data());
                kernel.d_twk         = thrust::raw_pointer_cast(d_twk_[active_buf].data());
                kernel.n_kets        = (int)n_kets_per_task_[task];
                kernel.d_m1          = thrust::raw_pointer_cast(d_m1_.data());
                kernel.d_m2          = thrust::raw_pointer_cast(d_m2_.data());
                kernel.d_coeff       = thrust::raw_pointer_cast(d_coeff_.data());
                kernel.d_fops_bpos   = thrust::raw_pointer_cast(d_fops_bpos_.data());
                kernel.d_word_start  = thrust::raw_pointer_cast(d_word_start_.data());
                kernel.d_ndag_per_word = thrust::raw_pointer_cast(d_ndag_per_word_.data());
                kernel.n_terms       = n_terms_;
                kernel.sign_flag     = sign_;
                launch_caop_mult((size_t)n_bras_, kernel, smem_bytes_, compute_stream_);
            }

            cudaEventRecord(compute_done_[active_buf], compute_stream_);

            if (task + 1 < slide_.size()) {
                int bslide = slide_[task] - slide_[task + 1];
                cudaEventSynchronize(copy_done_[recv_buf]);
                slider.ExchangeAsyncHost(
                    h_twk_[active_buf], twk_size[active_buf],
                    h_twk_[recv_buf],   global_max_n_,
                    bslide, b_comm_, (int)task * 4);
                if (slider.Sync()) {
                    twk_size[recv_buf] = slider.get_recv_size();
                    cudaStreamWaitEvent(copy_stream_, compute_done_[recv_buf]);
                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_twk_[recv_buf].data()),
                                    h_twk_[recv_buf],
                                    twk_size[recv_buf] * sizeof(ElemT),
                                    cudaMemcpyHostToDevice, copy_stream_);
                    cudaEventRecord(copy_done_[recv_buf], copy_stream_);
                    std::swap(active_buf, recv_buf);
                }
            } else {
                cudaEventSynchronize(compute_done_[active_buf]);
            }
        }

        // Allreduce
        MpiAllreduce(d_wb, MPI_SUM, t_comm_);
        MpiAllreduce(d_wb, MPI_SUM, h_comm_);

        // Download wb
        thrust::copy_n(d_wb.begin(), wb.size(), wb.begin());
    }
};

// ============================================================
// Free mult() — caller supplies a CaopMultThrust<ElemT>& to own GPU resources.
// driver.Init() is idempotent: returns immediately if already initialized for
// the same problem; throws if dimensions changed without reset().
// Call reset() on the driver before reuse if H or the basis changes.
// ============================================================
template <typename ElemT>
void mult(const std::vector<ElemT>& hd,
          const std::vector<ElemT>& wk,
          std::vector<ElemT>& wb,
          const det_vector<size_t>& bs,
          const int bit_length,
          const std::vector<int>& slide,
          const GeneralOp<ElemT>& H,
          bool sign,
          MPI_Comm h_comm, MPI_Comm b_comm, MPI_Comm t_comm,
          CaopMultThrust<ElemT>& driver) {
    driver.Init(hd, bs, bit_length, H, sign, slide, h_comm, b_comm, t_comm);
    driver.run(wk, wb);
}

} // namespace sbd

#endif // SBD_CAOP_BASIC_MULT_THRUST_H
