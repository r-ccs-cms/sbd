/**
@file sbd/chemistry/tpb/mult_factored_thrust.h
@brief Factored Hamiltonian multiplication for the tensor-product basis (--factored_mult 1)

In a tensor-product basis the determinants are all pairs (a, b) of an alpha string list and a beta
string list, and the Hamiltonian factors into string operators and integrals:

  H = H_a (x) I + I (x) H_b + sum_{pq,rs} (pq|rs) E^a_pq (x) E^b_rs

H_a and H_b are the same-spin string Hamiltonians (one-body part, same-spin two-body part and, for
H_a, the core energy), with 1 + S + D entries per string for S singles and D doubles within the
list. E_pq moves one electron from q to p, with p == q giving the occupation. H·C is formed from
these factors; no element of H is stored:

  - same-spin terms, H_a C and C H_b^T: H_a and H_b are kept as their nonzero 128 x 64 tiles, and
    each thread accumulates one output row over 32 columns from C staged in shared memory
    (spmm_same_spin);
  - cross term: for each bra string a, X_a[rs, b'] = sum_k sgn_k (pq_k|rs) C[a'_k, b'] over the
    excitations a = E_pq a' of a, computed by FP64 tensor-core MMA on 64 x 64 tiles. Only the rs
    rows that some b' in a 64-wide tile can use are computed, and only the entries with E_rs b' in
    the beta list are added into H·C[a, b] (fused_cross); X never reaches GPU memory.

Within a string the orbitals are ordered ascending, and the factored form takes determinants as the
alpha string followed by the beta string. sbd interleaves the spins (spin orbital 2i is alpha i,
2i+1 is beta i), so run() multiplies by the phase P(a, b) = (-1)^(sum over i in a of the beta
electrons below i) on the way in and on the way out; the vectors outside stay in sbd's convention.
Inside, the strings of each spin are ordered by their bit-reversed value, which keeps the beta
tiles of the cross term compact; the same passes apply that permutation.

Limits of this version: the alpha strings split over the ranks (bdet_comm_size, task_comm_size
and h_comm of size 1), real coefficients, at most 64 orbitals.
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_FACTORED_THRUST_H
#define SBD_CHEMISTRY_TPB_MULT_FACTORED_THRUST_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "sbd/framework/nvtx.h"

#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

namespace sbd {
namespace factored {

// ---------------------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------------------

#define SBD_FM_LD 72    // row stride (doubles) of the V, C and X tiles: 64 + 8 avoids bank conflicts
#define SBD_FM_LDC 65   // row stride of the same-spin C tiles; odd so transposed fills spread over banks
#define SBD_FM_TR 128   // rows per same-spin tile (columns per tile: 64)
#ifndef SBD_FM_SMEM_LIMIT
#define SBD_FM_SMEM_LIMIT 0  // bytes of shared memory a cross-term block may use; 0: the device's limit
#endif

__device__ __forceinline__ void fm_cp_async8(double* dst, const double* src, bool valid)
{
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(d), "l"(src), "r"(valid ? 8 : 0));  // zero-fills when not valid
}
__device__ __forceinline__ void fm_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void fm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ void fm_dmma(double& d0, double& d1, double a, double b)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
                 : "+d"(d0), "+d"(d1) : "d"(a), "d"(b));
}

// V tile t of a chunk (rows k of the tile, 64 rs columns) into a ring stage, by cp.async
__device__ __forceinline__ void fm_load_v_tile(double* stage, const double* Vs, const int* rl,
                                               const int* vr_s, int K, int t, int m, int kh)
{
    const int k0 = t * 32, kc = min(32, K - k0);
    double* dst = stage + m;
    const double* src = Vs + rl[m];
    for (int k = kh; k < kc; k += 2) fm_cp_async8(dst + k * SBD_FM_LD, src + vr_s[k0 + k], true);
    fm_cp_commit();
}

/**
   Cross term. One block per (segment of the excitation list of bra a, 64-column tile of b'); a list
   too long for shared memory is split into segments whose sums add into the same sigma row. The C
   tile C[a'_k, b'-tile] (K x 64) stays in shared memory; the sign of each excitation is folded into the V row it reads (Vs holds
   V, -V and a zero row for padding), so the C tile and the 32 x 64 V tiles (two-stage ring, which
   also holds the X tile) arrive by cp.async. For each 64-row chunk of the rows the tile can use
   (rlist), 4 warps of 32 x 32 form X with mma.sync m8n8k4 f64, and the entries with E_rs b' in the
   list are added into sigma (gent: rs_local 6 | b'_local 6 | sign 1 | b 19, grouped by chunk).
   C holds the beta columns from ccol0 of every alpha row, with row stride ldc; the launch covers the
   beta tiles from bt0.
*/
static __global__ void __launch_bounds__(128)
fused_cross(const double* __restrict__ C, const double* __restrict__ Vs, int ldv,
            const long long* __restrict__ kptr, const int* __restrict__ kket,
            const int* __restrict__ krow, const int* __restrict__ kbra, const int* __restrict__ sidx,
            const long long* __restrict__ goff, const unsigned* __restrict__ gent,
            const int* __restrict__ cbase, const int* __restrict__ rlist, long long nb,
            long long a0, double* sigma,  // sigma holds this rank's bras a0.. (rows a - a0)
            long long ldc, long long ccol0, int bt0)
{
    extern __shared__ double sm[];
    const int bt = blockIdx.x + bt0;
    const int sg = sidx[blockIdx.y];
    const long long a = kbra[sg];
    const long long kp = kptr[sg];
    const int K = (int)(kptr[sg + 1] - kp);  // multiple of 4, padded with the zero row
    const int c0 = cbase[bt], nch = cbase[bt + 1] - c0;
    const int T = (K + 31) / 32;            // 32-deep V tiles per chunk
    double* Bs = sm;                        // K x LD
    double* ring = Bs + K * SBD_FM_LD;      // 2 x 32 x LD V tiles, or the 64 x LD X tile
    long long* cr_s = (long long*)(ring + 64 * SBD_FM_LD);  // K offsets of the C rows (a'_k * nb)
    int* vr_s = (int*)(cr_s + K);           // K offsets of the signed V rows (row * ldv)
    int* rl_s = vr_s + K;                   // nch x 64 rs columns
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;  // 2 x 2 warps, 32 x 32 outputs each
    const int r4 = lane & 3, q8 = lane >> 2;  // fragment row/column within an 8 x 4 / 4 x 8 piece
    for (int i = tid; i < K; i += 128) {
        cr_s[i] = (long long)kket[kp + i] * ldc;
        vr_s[i] = krow[kp + i] * ldv;
    }
    for (int i = tid; i < nch * 64; i += 128) rl_s[i] = rlist[(long long)c0 * 64 + i];
    __syncthreads();

    const int m = tid & 63, kh = tid >> 6;  // each thread copies one column, every other row
    const long long b0 = (long long)bt * 64;
    {   // C tile: one cp.async group
        const bool ok = b0 + m < nb;
        for (int k = kh; k < K; k += 2)
            fm_cp_async8(Bs + k * SBD_FM_LD + m, C + cr_s[k] + (ok ? b0 + m - ccol0 : 0), ok);
        fm_cp_commit();
    }
    if (nch > 0) fm_load_v_tile(ring, Vs, rl_s, vr_s, K, 0, m, kh);

    double* srow = sigma + (a - a0) * nb;
    for (int c = 0; c < nch; c++) {
        double acc[4][4][2];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j][0] = acc[i][j][1] = 0.0;
        for (int t = 0; t < T; t++) {
            if (t + 1 < T) {
                fm_load_v_tile(ring + ((t + 1) & 1) * 32 * SBD_FM_LD, Vs, rl_s + c * 64, vr_s, K, t + 1, m, kh);
                fm_cp_wait1();
            }
            else fm_cp_wait0();
            __syncthreads();  // tile t (and, first time, the C tile) visible to all warps
            const double* As = ring + (t & 1) * 32 * SBD_FM_LD + wm * 32 + q8;
            const double* Bt = Bs + (t * 32) * SBD_FM_LD + wn * 32 + q8;
            const int kc = min(32, K - t * 32);
            for (int kk = 0; kk < kc; kk += 4) {
                double fa[4], fb[4];
#pragma unroll
                for (int i = 0; i < 4; i++) fa[i] = As[(kk + r4) * SBD_FM_LD + i * 8];  // A[m][k], .row
#pragma unroll
                for (int j = 0; j < 4; j++) fb[j] = Bt[(kk + r4) * SBD_FM_LD + j * 8];  // B[k][n], .col
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) fm_dmma(acc[i][j][0], acc[i][j][1], fa[i], fb[j]);
            }
            __syncthreads();  // stage (t & 1) free for the load two tiles ahead / for X
        }
        double* Xs = ring;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                double* x = Xs + (wm * 32 + i * 8 + q8) * SBD_FM_LD + wn * 32 + j * 8 + 2 * r4;
                x[0] = acc[i][j][0];
                x[1] = acc[i][j][1];
            }
        __syncthreads();
        const long long e1 = goff[c0 + c + 1];
        for (long long e = goff[c0 + c] + tid; e < e1; e += 128) {
            unsigned u = gent[e];
            double v = Xs[(u & 63) * SBD_FM_LD + ((u >> 6) & 63)];
            atomicAdd(srow + (u >> 13), (u >> 12) & 1 ? -v : v);
        }
        __syncthreads();  // X read before the ring is refilled
        if (c + 1 < nch) fm_load_v_tile(ring, Vs, rl_s + (c + 1) * 64, vr_s, K, 0, m, kh);
    }
}

// 64 x 64 slice of C for same-spin tile t into a ring stage (mode 0: rows J0.. of C; mode 1:
// columns J0.., transposed), by cp.async. Element (alpha row i, beta column j) of C is at
// C[(i - crow0) * ldc + j - ccol0]; the tile's H column index J0 + c outside [cw0, cw1) reads zero.
__device__ __forceinline__ void fm_load_c_tile(double* dst, const double* C, long long J0, long long y0,
                                               long long nrow, long long nchk, int mode, int tid,
                                               long long cw0, long long cw1, long long crow0,
                                               long long ccol0, long long ldc)
{
    const int u = tid & 63;  // consecutive threads read consecutive global addresses
    for (int v = tid >> 6; v < 64; v += 2 * SBD_FM_TR / 64) {
        if (mode == 0) {      // Cs[c][y] = C[J0 + c, y0 + y], c = v, y = u
            bool ok = J0 + v < nrow && J0 + v >= cw0 && J0 + v < cw1 && y0 + u < nchk;
            fm_cp_async8(dst + v * SBD_FM_LDC + u, C + (ok ? (J0 + v - crow0) * ldc + y0 + u - ccol0 : 0), ok);
        } else {              // Cs[c][y] = C[y0 + y, J0 + c], y = v, c = u
            bool ok = y0 + v < nchk && J0 + u < nrow && J0 + u >= cw0 && J0 + u < cw1;
            fm_cp_async8(dst + u * SBD_FM_LDC + v, C + (ok ? (y0 + v - crow0) * ldc + J0 + u - ccol0 : 0), ok);
        }
    }
    fm_cp_commit();
}

/**
   Same-spin products with a sparse, symmetric string Hamiltonian stored as nonzero TR x 64 tiles of
   (column, value) entries sorted by row (rp: TR + 1 row pointers per tile). C is na x nb; sigma holds
   the alpha rows this rank computes, row a at a - sbase.
     mode 0: sigma[I rows, chunk] += H_a[I rows, :] C[:, chunk]      (I over alpha tiles built from
             row rbase, rows outside [rlo, rhi) dropped)
     mode 1: sigma[chunk, I cols] += C[chunk, :] H_b[:, I cols]       (I over beta, chunks of [rlo, rhi))
   The chunks cover [ylo, yhi) (beta strings in mode 0, alpha rows in mode 1); only the tiles whose
   columns of H overlap [cw0, cw1) are read; C is addressed as in fm_load_c_tile.
   One block per (tile row I, 64-wide chunk); blockIdx.x = I runs fastest, so the blocks in flight
   share one slice of C, which stays in L2. Each nonzero tile's 64 x 64 slice of C arrives in a
   two-stage cp.async ring; thread (row, half) owns one output row and 32 output columns in
   registers, so each entry of its row drives 32 multiply-adds.
*/
static __global__ void __launch_bounds__(2 * SBD_FM_TR, 2)  // 2 blocks per SM: at most 128 registers
spmm_same_spin(const double* __restrict__ C, double* sigma, const int* __restrict__ trp,
               const int* __restrict__ tj, const int* __restrict__ rp,
               const unsigned char* __restrict__ ec, const double* __restrict__ ev,
               long long na, long long nb, long long rlo, long long rhi, long long sbase, long long rbase,
               int mode, long long ylo, long long yhi, long long cw0, long long cw1, long long crow0,
               long long ccol0, long long ldc)
{
    extern __shared__ double smem[];  // two stages of 64 x LDC doubles
    const int I = blockIdx.x, ch = blockIdx.y;
    const int tid = threadIdx.x;
    const int row = tid % SBD_FM_TR, half = tid / SBD_FM_TR;
    const long long nrow = mode == 0 ? na : nb;   // rows and columns of this H
    const long long nchk = yhi;                   // end of the chunked index
    const long long y0 = ylo + (long long)ch * 64;
    int t0 = trp[I], t1 = trp[I + 1];
    {   // tiles ascend by column: keep those overlapping [cw0, cw1)
        int lo = t0, hi = t1;
        while (lo < hi) { const int m = (lo + hi) / 2; if ((long long)(tj[m] + 1) * 64 <= cw0) lo = m + 1; else hi = m; }
        t0 = lo;
        hi = t1;
        while (lo < hi) { const int m = (lo + hi) / 2; if ((long long)tj[m] * 64 < cw1) lo = m + 1; else hi = m; }
        t1 = lo;
    }
    double acc[32];
#pragma unroll
    for (int k = 0; k < 32; k++) acc[k] = 0.0;
    if (t0 < t1) fm_load_c_tile(smem, C, (long long)tj[t0] * 64, y0, nrow, nchk, mode, tid, cw0, cw1, crow0, ccol0, ldc);
    for (int t = t0; t < t1; t++) {
        const int st = (t - t0) & 1;
        if (t + 1 < t1) {
            fm_load_c_tile(smem + (st ^ 1) * 64 * SBD_FM_LDC, C, (long long)tj[t + 1] * 64, y0, nrow, nchk,
                           mode, tid, cw0, cw1, crow0, ccol0, ldc);
            fm_cp_wait1();
        }
        else fm_cp_wait0();
        __syncthreads();
        const double* cs = smem + st * 64 * SBD_FM_LDC + half * 32;
        int e = rp[(long long)t * (SBD_FM_TR + 1) + row];
        const int e1 = rp[(long long)t * (SBD_FM_TR + 1) + row + 1];
        if (e < e1) {
            int c = ec[e];
            double v = ev[e];
            for (; e < e1; e++) {
                const double* crow = cs + c * SBD_FM_LDC;
                const double vv = v;
                if (e + 1 < e1) { c = ec[e + 1]; v = ev[e + 1]; }  // next entry's loads overlap
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += vv * crow[k];
            }
        }
        __syncthreads();  // stage st free for the load two tiles ahead
    }
    const long long R = rbase + (long long)I * SBD_FM_TR + row;
    if (R >= nrow || (mode == 0 && (R < rlo || R >= rhi))) return;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const long long y = y0 + half * 32 + k;
        if (y < nchk) {
            if (mode == 0) sigma[(R - sbase) * nb + y] += acc[k];
            else sigma[(y - sbase) * nb + R] += acc[k];
        }
    }
}

// Between sbd's layout and the engine's: the engine orders the strings of each spin by their
// bit-reversed value (alpha within each rank's block, see Init) and its determinants alpha string
// then beta string, so element (a', b') of an engine vector is P(a, b) times element (pa[a'], pb[b'])
// of an sbd vector, with P(a, b) = (-1)^(sum over i in a of the beta electrons below i) from sbd's
// interleaved spins.
__device__ __forceinline__ int fm_phase(unsigned long long A, unsigned long long B)
{
    int s = 0;
    while (A) {
        const int o = __ffsll((long long)A) - 1;
        A &= A - 1;
        s += __popcll(B & ((1ull << o) - 1));
    }
    return s & 1;
}

// T[a' - tbase, b'] = P Cg[gofs[a'] - shift, pb[b']] for the engine rows r0 <= a' < r1; Cg is the gathered
// sbd vector (shift 0), its row for engine row a' at gofs[a'] (rank blocks padded to the same
// length), or this rank's block of it (shift = the block's first row)
static __global__ void gather_phase(const unsigned long long* __restrict__ xa,
                                    const unsigned long long* __restrict__ xb,
                                    const long long* __restrict__ gofs, const int* __restrict__ pb,
                                    long long r0, long long r1, long long nb, long long shift,
                                    long long tbase, const double* __restrict__ Cg,
                                    double* __restrict__ T)
{
    const long long n = (r1 - r0) * nb;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += (long long)gridDim.x * blockDim.x) {
        const long long i = r0 * nb + j;
        const long long a = i / nb, b = i - a * nb;
        const double v = Cg[(gofs[a] - shift) * nb + pb[b]];
        T[i - tbase * nb] = fm_phase(xa[a], xb[b]) ? -v : v;
    }
}

// W[pa[a'] - obase, pb[b']] += P S[a' - sbase, b'] for engine rows r0 <= a' < r1 that this rank owns
// (its W starts at sbd row obase; run() adds H C into its output, as Lanczos expects)
static __global__ void scatter_phase(const unsigned long long* __restrict__ xa,
                                     const unsigned long long* __restrict__ xb,
                                     const int* __restrict__ pa, const int* __restrict__ pb,
                                     long long r0, long long r1, long long nb, long long sbase,
                                     long long obase, const double* __restrict__ S,
                                     double* __restrict__ W)
{
    const long long n = (r1 - r0) * nb;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / nb, b = i - r * nb, a = r0 + r;
        const double v = S[(a - sbase) * nb + b];
        W[(pa[a] - obase) * nb + pb[b]] += fm_phase(xa[a], xb[b]) ? -v : v;
    }
}

// hii[a, b] = I0 + d_a[a] + d_b[b] + sum over p in a, r in b of (pp|rr)  (Jb[b * L + p])
static __global__ void diagonal_terms(const unsigned long long* __restrict__ xa,
                                      const double* __restrict__ da, const double* __restrict__ db,
                                      const double* __restrict__ Jb, int L, long long na, long long nb,
                                      double I0, double* hii)
{
    const long long n = na * nb;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        const long long a = i / nb, b = i - a * nb;
        unsigned long long A = xa[a];
        double v = I0 + da[a] + db[b];
        const double* J = Jb + b * L;
        while (A) {
            const int o = __ffsll((long long)A) - 1;
            A &= A - 1;
            v += J[o];
        }
        hii[i] = v;
    }
}

// a reproducible value in [-1, 1) from an index (splitmix64), for the reference check
struct hash_value {
    __host__ __device__ double operator()(size_t i) const
    {
        unsigned long long z = i + 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
    }
};

// ---------------------------------------------------------------------------------------------
// Host-side tables
// ---------------------------------------------------------------------------------------------

inline int fm_popcount(uint64_t x) { return __builtin_popcountll(x); }
// (-1)^(occupied orbitals of x below i)
inline int fm_sign_below(uint64_t x, int i) { return (fm_popcount(x & ((1ull << i) - 1)) & 1) ? -1 : 1; }

/**
   Strings of one spin, their same-spin Hamiltonian (CSR, diagonal first in each row) and their
   excitation lists E (for each bra string: the occupied orbitals as p == q entries, then the single
   excitations a = E_pq a' within the list, with sign).
*/
struct SpinStrings {
    size_t n = 0;
    std::vector<uint64_t> x;
    std::vector<int64_t> hptr;
    std::vector<int32_t> hcol;
    std::vector<double> hval;
    std::vector<double> hdiag;
    std::vector<int64_t> eptr;
    std::vector<int32_t> eket;
    std::vector<int32_t> epq;
    std::vector<double> esg;
};

inline void BuildSpinStrings(SpinStrings &s, int L, const std::vector<double> &h1,
                             const std::vector<double> &eri)
{
    const size_t n = s.n;
    auto V = [&](int p, int q, int r, int t) {
        return eri[((static_cast<size_t>(p) * L + q) * L + r) * L + t];
    };
    std::vector<int64_t> ns(n, 0), nd(n, 0);
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n; i++) {
        int64_t cs = 0, cd = 0;
        for (size_t j = 0; j < n; j++) {
            int c = fm_popcount(s.x[i] ^ s.x[j]);
            cs += (c == 2);
            cd += (c == 4);
        }
        ns[i] = cs;
        nd[i] = cd;
    }
    const int nel = fm_popcount(s.x[0]);
    s.hptr.assign(n + 1, 0);
    s.eptr.assign(n + 1, 0);
    for (size_t i = 0; i < n; i++) {
        s.hptr[i + 1] = s.hptr[i] + 1 + ns[i] + nd[i];
        s.eptr[i + 1] = s.eptr[i] + ns[i] + nel;
    }
    s.hcol.resize(s.hptr[n]);
    s.hval.resize(s.hptr[n]);
    s.hdiag.resize(n);
    s.eket.resize(s.eptr[n]);
    s.epq.resize(s.eptr[n]);
    s.esg.resize(s.eptr[n]);
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n; i++) {
        const uint64_t a = s.x[i];
        int occ[64];
        int k = 0;
        for (int o = 0; o < L; o++)
            if ((a >> o) & 1) occ[k++] = o;
        int64_t hp = s.hptr[i], ep = s.eptr[i];
        double d = 0.0;
        for (int u = 0; u < k; u++) {
            const int p = occ[u];
            d += h1[p * L + p];
            for (int w = 0; w < k; w++) {
                const int r = occ[w];
                d += 0.5 * (V(p, p, r, r) - V(p, r, r, p));
            }
            s.eket[ep] = static_cast<int32_t>(i);
            s.epq[ep] = p * L + p;
            s.esg[ep] = 1.0;
            ep++;
        }
        s.hdiag[i] = d;
        s.hcol[hp] = static_cast<int32_t>(i);
        s.hval[hp] = d;
        hp++;
        for (size_t j = 0; j < n; j++) {
            const uint64_t ap = s.x[j];
            const uint64_t diff = a ^ ap;
            const int c = fm_popcount(diff);
            if (c == 2) {
                const int p = __builtin_ctzll(a & diff);   // in the bra, not the ket
                const int q = __builtin_ctzll(ap & diff);  // in the ket, not the bra
                // a = E_pq a' = a+_p a_q a'
                const uint64_t y = ap & ~(1ull << q);
                const int sg = fm_sign_below(ap, q) * fm_sign_below(y, p);
                double v = h1[p * L + q];
                for (int u = 0; u < k; u++) {
                    const int kk = occ[u];
                    if (kk != p) v += V(p, q, kk, kk) - V(p, kk, kk, q);
                }
                s.hcol[hp] = static_cast<int32_t>(j);
                s.hval[hp] = sg * v;
                hp++;
                s.eket[ep] = static_cast<int32_t>(j);
                s.epq[ep] = p * L + q;
                s.esg[ep] = sg;
                ep++;
            } else if (c == 4) {
                uint64_t cr = a & diff, an = ap & diff;
                const int p = __builtin_ctzll(cr); cr &= cr - 1;
                const int r = __builtin_ctzll(cr);
                const int q = __builtin_ctzll(an); an &= an - 1;
                const int t = __builtin_ctzll(an);
                // a = a+_p a+_r a_t a_q a'
                uint64_t y = ap;
                int sg = fm_sign_below(y, q);
                y &= ~(1ull << q);
                sg *= fm_sign_below(y, t);
                y &= ~(1ull << t);
                sg *= fm_sign_below(y, r);
                y |= 1ull << r;
                sg *= fm_sign_below(y, p);
                s.hcol[hp] = static_cast<int32_t>(j);
                s.hval[hp] = sg * (V(p, q, r, t) - V(p, t, r, q));
                hp++;
            }
        }
    }
}

/**
   Nonzero TR x 64 tiles of a same-spin Hamiltonian, entries (column within the tile, value)
   ordered by row; diag_shift is added to the diagonal (the core energy, once, on the alpha side).
*/
struct SpinTiles {
    size_t nT = 0;
    size_t r0 = 0;       // first row of tile 0
    size_t ntiles = 0;   // nonzero tiles
    thrust::device_vector<int> trp, tj, rp;
    thrust::device_vector<unsigned char> ec;
    thrust::device_vector<double> ev;
};

// rows [ra, rb) of the string Hamiltonian (all rows by default), in tiles of TR rows from ra
inline void BuildSpinTiles(SpinTiles &t, const SpinStrings &s, double diag_shift, size_t ra = 0,
                           size_t rb = static_cast<size_t>(-1))
{
    const size_t n = s.n;
    rb = std::min(rb, n);
    const size_t TR = SBD_FM_TR;
    const size_t nT = (rb - ra + TR - 1) / TR, nC = (n + 63) / 64;
    std::vector<int> trp(nT + 1, 0), tj, rp;
    std::vector<unsigned char> ec(s.hptr[rb] - s.hptr[ra]);
    std::vector<double> ev(s.hptr[rb] - s.hptr[ra]);
    std::vector<int64_t> cnt(nC), tile_of(nC), pos(nC);
    int64_t base = 0;
    for (size_t I = 0; I < nT; I++) {
        const size_t r0 = ra + I * TR, r1 = std::min(rb, r0 + TR);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t r = r0; r < r1; r++)
            for (int64_t e = s.hptr[r]; e < s.hptr[r + 1]; e++) cnt[s.hcol[e] / 64]++;
        const size_t first = tj.size();
        for (size_t ct = 0; ct < nC; ct++) {
            if (cnt[ct] == 0) continue;
            tile_of[ct] = static_cast<int64_t>(tj.size());
            pos[ct] = base;
            base += cnt[ct];
            tj.push_back(static_cast<int>(ct));
        }
        const size_t ntile = tj.size() - first;
        trp[I + 1] = static_cast<int>(tj.size());
        std::vector<int> rowcnt(ntile * TR, 0);
        std::vector<int64_t> start(ntile);
        for (size_t u = 0; u < ntile; u++) start[u] = pos[tj[first + u]];
        for (size_t r = r0; r < r1; r++)
            for (int64_t e = s.hptr[r]; e < s.hptr[r + 1]; e++) {
                const int ct = s.hcol[e] / 64;
                const int64_t at = pos[ct]++;
                ec[at] = static_cast<unsigned char>(s.hcol[e] % 64);
                ev[at] = s.hval[e] + (static_cast<size_t>(s.hcol[e]) == r ? diag_shift : 0.0);
                rowcnt[(tile_of[ct] - first) * TR + (r - r0)]++;
            }
        for (size_t u = 0; u < ntile; u++) {
            int64_t p = start[u];
            rp.push_back(static_cast<int>(p));
            for (size_t r = 0; r < TR; r++) {
                p += rowcnt[u * TR + r];
                rp.push_back(static_cast<int>(p));
            }
        }
    }
    if (base > INT32_MAX) throw std::runtime_error("factored_mult: same-spin Hamiltonian too large for 32-bit offsets");
    t.nT = nT;
    t.r0 = ra;
    t.ntiles = tj.size();
    t.trp = trp;
    t.tj = tj;
    t.rp = rp;
    t.ec = ec;
    t.ev = ev;
}

// nonzero 64-column tiles of each TR-row tile row (tiles from row 0): the alpha same-spin kernel's
// work goes by these (a 64 x 64 slice of C per tile), not by the entries
inline std::vector<int> ColumnTilesPerTileRow(const SpinStrings &s)
{
    const size_t TR = SBD_FM_TR, n = s.n, nT = (n + TR - 1) / TR, nC = (n + 63) / 64;
    std::vector<int> out(nT, 0);
    std::vector<char> seen(nC);
    for (size_t I = 0; I < nT; I++) {
        std::fill(seen.begin(), seen.end(), 0);
        for (size_t r = I * TR; r < std::min(n, (I + 1) * TR); r++)
            for (int64_t e = s.hptr[r]; e < s.hptr[r + 1]; e++) seen[s.hcol[e] / 64] = 1;
        for (char c : seen) out[I] += c;
    }
    return out;
}

/**
   Tables of the cross term: per bra alpha string its excitation list, in segments (sptr) of at most
   kseg entries, each padded to a multiple of 4 (C rows kket, signed V rows krow, bra kbra), the
   signed integral rows Vs, and per 64-wide tile of beta kets
   the rs rows it can use (rlist, in 64-row chunks from cbase) with its scatter entries (gent from
   goff, one range per chunk).
*/
struct CrossTables {
    int ldv = 0;
    size_t nbt = 0;
    int max_chunks = 0;
    thrust::device_vector<double> Vs;
    thrust::device_vector<long long> kptr, goff;
    thrust::device_vector<int> kket, krow, kbra, cbase, rlist;
    thrust::device_vector<unsigned> gent;
    int kseg = 0;           // longest segment that fits the shared memory of a block
    std::vector<int> Kp;    // padded excitation count of each segment (host)
    std::vector<size_t> sptr;  // segments of bra a: sptr[a] .. sptr[a + 1] (host)
    std::vector<thrust::device_vector<int>> bucket_ids;
    std::vector<size_t> bucket_smem;
    std::vector<int> bucket_group;   // index into fm_kp_bounds
    std::vector<double> bucket_kp;   // sum of the padded excitation counts of its bras
    double flops = 0.0;
    size_t total_chunks = 0;
};

// bras of the cross term are launched in groups of padded excitation count up to these bounds
static const int fm_kp_bounds[] = {64, 96, 128, 160, 192, 256, 1 << 30};

// shared memory of a fused_cross block for bras of at most km excitations
inline size_t fm_cross_smem(int km, int max_chunks)
{
    return static_cast<size_t>(km + 64) * SBD_FM_LD * sizeof(double)
           + static_cast<size_t>(km) * (sizeof(long long) + sizeof(int))
           + static_cast<size_t>(max_chunks) * 64 * sizeof(int);
}

inline void BuildCrossTables(CrossTables &x, const SpinStrings &sa, const SpinStrings &sb, int L,
                             const std::vector<double> &eri, size_t smem_limit)
{
    const int nrs = L * L;
    const int nrc = nrs / 64 + 1;  // > nrs / 64, so column nrs (padding) exists and is zero
    x.ldv = nrc * 64;
    std::vector<double> Vs(static_cast<size_t>(2 * nrs + 1) * x.ldv, 0.0);
    for (int pq = 0; pq < nrs; pq++)
        for (int rs = 0; rs < nrs; rs++) {
            const double v = eri[static_cast<size_t>(pq) * nrs + rs];
            Vs[static_cast<size_t>(pq) * x.ldv + rs] = v;
            Vs[static_cast<size_t>(nrs + pq) * x.ldv + rs] = -v;
        }
    x.Vs = Vs;

    // beta side: rows used per 64-wide tile of kets, scatter entries grouped by chunk
    const size_t nb = sb.n;
    if (nb >= (1u << 19)) throw std::runtime_error("factored_mult: more than 2^19 beta strings");
    x.nbt = (nb + 63) / 64;
    std::vector<int> rank(x.nbt * nrs, -1), nused(x.nbt, 0);
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++) rank[(sb.eket[e] / 64) * nrs + sb.epq[e]] = 0;
    std::vector<int> cbase(x.nbt + 1, 0);
    for (size_t t = 0; t < x.nbt; t++) {
        int u = 0;
        for (int rs = 0; rs < nrs; rs++)
            if (rank[t * nrs + rs] == 0) rank[t * nrs + rs] = u++;
        nused[t] = u;
        cbase[t + 1] = cbase[t] + (u + 63) / 64;
        x.max_chunks = std::max(x.max_chunks, (u + 63) / 64);
    }
    std::vector<int> rlist(static_cast<size_t>(cbase[x.nbt]) * 64, nrs);
    for (size_t t = 0; t < x.nbt; t++)
        for (int rs = 0; rs < nrs; rs++) {
            const int u = rank[t * nrs + rs];
            if (u >= 0) rlist[static_cast<size_t>(cbase[t]) * 64 + u] = rs;
        }
    std::vector<long long> goff(cbase[x.nbt] + 1, 0);
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++) {
            const int t = sb.eket[e] / 64;
            goff[cbase[t] + rank[t * nrs + sb.epq[e]] / 64 + 1]++;
        }
    for (size_t c = 0; c < static_cast<size_t>(cbase[x.nbt]); c++) goff[c + 1] += goff[c];
    std::vector<long long> fill(goff.begin(), goff.end() - 1);
    std::vector<unsigned> gent(goff.back());
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++) {
            const int t = sb.eket[e] / 64;
            const int u = rank[t * nrs + sb.epq[e]];
            gent[fill[cbase[t] + u / 64]++] = static_cast<unsigned>(u % 64)
                                              | (static_cast<unsigned>(sb.eket[e] % 64) << 6)
                                              | (static_cast<unsigned>(sb.esg[e] < 0) << 12)
                                              | (static_cast<unsigned>(b) << 13);
        }
    x.cbase = cbase;
    x.rlist = rlist;
    x.goff = goff;
    x.gent = gent;
    x.total_chunks = cbase[x.nbt];

    // alpha side: lists longer than the shared memory allows are cut into near-equal segments
    while (fm_cross_smem(x.kseg + 4, x.max_chunks) <= smem_limit) x.kseg += 4;
    if (x.kseg == 0) throw std::runtime_error("factored_mult: shared memory too small for the cross term");
    const size_t na = sa.n;
    x.sptr.assign(na + 1, 0);
    std::vector<int> nseg(na);
    for (size_t a = 0; a < na; a++) {
        nseg[a] = static_cast<int>(std::max<int64_t>(1, (sa.eptr[a + 1] - sa.eptr[a] + x.kseg - 1) / x.kseg));
        x.sptr[a + 1] = x.sptr[a] + nseg[a];
    }
    const size_t ns = x.sptr[na];
    std::vector<long long> kptr(ns + 1, 0);
    std::vector<int> Kp(ns), kbra(ns);
    std::vector<int64_t> e0(ns + 1);  // first excitation of each segment
    for (size_t a = 0; a < na; a++) {
        const int64_t n = sa.eptr[a + 1] - sa.eptr[a];
        for (int j = 0; j <= nseg[a]; j++) e0[x.sptr[a] + j] = sa.eptr[a] + n * j / nseg[a];
        for (size_t g = x.sptr[a]; g < x.sptr[a + 1]; g++) {
            Kp[g] = static_cast<int>((e0[g + 1] - e0[g] + 3) / 4 * 4);
            kptr[g + 1] = kptr[g] + Kp[g];
            kbra[g] = static_cast<int>(a);
        }
    }
    std::vector<int> kket(kptr[ns], 0), krow(kptr[ns], 2 * nrs);
    for (size_t g = 0; g < ns; g++)
        for (int64_t e = e0[g]; e < e0[g + 1]; e++) {
            const long long at = kptr[g] + (e - e0[g]);
            kket[at] = sa.eket[e];
            krow[at] = sa.esg[e] > 0 ? sa.epq[e] : nrs + sa.epq[e];
        }
    x.kptr = kptr;
    x.kket = kket;
    x.krow = krow;
    x.kbra = kbra;
    x.Kp = Kp;
}

// launches for the segments of the bras a0 <= a < a1 that this rank computes, grouped by padded
// excitation count so that the shared memory of a launch fits its segments
inline void BuildCrossBuckets(CrossTables &x, size_t a0, size_t a1)
{
    x.bucket_ids.clear();
    x.bucket_smem.clear();
    x.bucket_group.clear();
    x.bucket_kp.clear();
    int lo = 0;
    double ksum = 0.0;
    for (int g = 0; g < static_cast<int>(sizeof(fm_kp_bounds) / sizeof(int)); g++) {
        const int hi = fm_kp_bounds[g];
        std::vector<int> ids;
        int km = 0;
        double kp = 0.0;
        for (size_t sg = x.sptr[a0]; sg < x.sptr[a1]; sg++)
            if (x.Kp[sg] > lo && x.Kp[sg] <= hi) {
                ids.push_back(static_cast<int>(sg));
                km = std::max(km, x.Kp[sg]);
                kp += x.Kp[sg];
            }
        lo = hi;
        if (ids.empty()) continue;
        x.bucket_ids.emplace_back(ids);
        x.bucket_smem.push_back(fm_cross_smem(km, x.max_chunks));
        x.bucket_group.push_back(g);
        x.bucket_kp.push_back(kp);
    }
    for (size_t sg = x.sptr[a0]; sg < x.sptr[a1]; sg++) ksum += x.Kp[sg];
    x.flops = 2.0 * 64 * 64 * static_cast<double>(x.total_chunks) * ksum;
}

} // namespace factored

/**
   H·C from the factored Hamiltonian; see the file comment.

   Several ranks split the alpha strings (adet_comm_size = number of ranks, bdet_comm_size,
   task_comm_size and h_comm of size 1): each rank owns the alpha rows of its helper's bra range and
   all beta strings. Every multiplication needs nearly all of C on every rank (the doubles of a block
   of bras reach almost every row), so run() gathers the whole input vector first, with
   ncclAllGather when built with SBD_USE_NCCL and GPU-aware MPI_Allgather otherwise. The rows of H·C
   are then computed in ranges split by estimated work (BalanceRows), which differ from the owned
   ranges by a few rows at each boundary; those rows go to their owners. No reduction follows.
*/
template <typename ElemT>
class MultTPBFactored : public MultBase<ElemT> {
public:
    MultTPBFactored() {}
    ~MultTPBFactored()
    {
#ifdef SBD_USE_NCCL
        if (nccl_ready_) {
            ncclCommDestroy(nccl_);
            cudaEventDestroy(gathered_);
            cudaEventDestroy(local_ready_);
            for (int b = 0; b < 2; b++) {
                cudaEventDestroy(pass_packed_[b]);
                cudaEventDestroy(pass_in_[b]);
            }
            cudaStreamDestroy(comm_);
        }
#endif
    }

    void Init(const det_vector<size_t, det_kind::half> &adets,
              const det_vector<size_t, det_kind::half> &bdets,
              const uint32_t bit_length,
              const size_t norbs,
              const std::vector<TaskHelpers> &helper,
              const ElemT &I0,
              const oneInt<ElemT> &I1,
              const twoInt<ElemT> &I2,
              MPI_Comm h_comm, MPI_Comm b_comm, MPI_Comm t_comm, MPI_Comm a_comm, int passes = 1,
              size_t reserve_bytes = 0)
    {
        SBD_NVTX_RANGE_COLOR("MultTPBFactored::Init", __LINE__);
        if (!std::is_same<ElemT, double>::value)
            throw std::runtime_error("factored_mult: real coefficients only");
        int sh, st;
        MPI_Comm_size(h_comm, &sh);
        MPI_Comm_size(t_comm, &st);
        MPI_Comm_size(b_comm, &P_);
        MPI_Comm_rank(b_comm, &rank_);
        if (sh != 1 || st != 1)
            throw std::runtime_error("factored_mult: task_comm_size and h_comm must be 1");
        if (helper.empty()) throw std::runtime_error("factored_mult: no task helper on this rank");
        if (helper[0].braBetaStart != 0 || helper[0].braBetaEnd != bdets.size())
            throw std::runtime_error("factored_mult: split the alpha strings only (bdet_comm_size 1)");
        if (norbs > 64) throw std::runtime_error("factored_mult: at most 64 orbitals");
        this->bit_length_ = bit_length;
        this->norbs_ = norbs;
        this->h_comm_ = h_comm;
        this->b_comm_ = b_comm;
        this->t_comm_ = t_comm;
        this->a_comm_ = a_comm;
        L_ = static_cast<int>(norbs);
        I0_ = I0;
        a0_ = helper[0].braAlphaStart;
        a1_ = helper[0].braAlphaEnd;

        // every rank's alpha range, in rank order
        std::vector<long long> mine = {static_cast<long long>(a0_), static_cast<long long>(a1_)};
        std::vector<long long> ranges(2 * P_);
        MPI_Allgather(mine.data(), 2, MPI_LONG_LONG, ranges.data(), 2, MPI_LONG_LONG, b_comm);
        maxrows_ = 0;
        for (int r = 0; r < P_; r++) {
            maxrows_ = std::max<size_t>(maxrows_, ranges[2 * r + 1] - ranges[2 * r]);
            if (ranges[2 * r] != (r == 0 ? 0 : ranges[2 * r - 1]))
                throw std::runtime_error("factored_mult: alpha ranges are not contiguous in rank order");
        }

        auto strings = [&](const det_vector<size_t, det_kind::half> &d, factored::SpinStrings &s) {
            s.n = d.size();
            s.x.resize(s.n);
            for (size_t i = 0; i < s.n; i++) {
                uint64_t v = 0;
                for (int o = 0; o < L_; o++)
                    if ((d[i][o / bit_length] >> (o % bit_length)) & 1) v |= 1ull << o;
                s.x[i] = v;
            }
        };
        strings(adets, sa_);
        strings(bdets, sb_);
        na_ = sa_.n;
        nb_ = sb_.n;
        if (static_cast<long long>(na_) != ranges[2 * P_ - 1])
            throw std::runtime_error("factored_mult: alpha ranges do not cover the alpha strings");
        // Engine order: strings sorted by their bit-reversed value (orbital 0 most significant).
        // sbd sorts with the highest orbital most significant; in that order a 64-wide tile of beta
        // kets mixes strings that share fewer orbitals, and the cross term computes 1.8x the rows
        // (507 against 285 padded rows per tile at 10,000 strings). Alpha strings are sorted within
        // each rank's range, so a rank's bras stay contiguous.
        const std::vector<uint64_t> xa_ext = sa_.x, xb_ext = sb_.x;
        auto reversed = [&](uint64_t v) {
            uint64_t r = 0;
            for (int o = 0; o < L_; o++)
                if ((v >> o) & 1) r |= 1ull << (L_ - 1 - o);
            return r;
        };
        auto engine_order = [&](factored::SpinStrings &s, std::vector<int> &perm,
                                const std::vector<long long> &blocks) {
            perm.resize(s.n);
            for (size_t i = 0; i < s.n; i++) perm[i] = static_cast<int>(i);
            std::vector<uint64_t> key(s.n);
            for (size_t i = 0; i < s.n; i++) key[i] = reversed(s.x[i]);
            for (size_t k = 0; k + 1 < blocks.size(); k++)
                std::stable_sort(perm.begin() + blocks[k], perm.begin() + blocks[k + 1],
                                 [&](int i, int j) { return key[i] < key[j]; });
            std::vector<uint64_t> x(s.n);
            for (size_t i = 0; i < s.n; i++) x[i] = s.x[perm[i]];
            s.x = x;
        };
        std::vector<long long> ablocks(P_ + 1);
        for (int r = 0; r < P_; r++) ablocks[r] = ranges[2 * r];
        ablocks[P_] = na_;
        std::vector<int> pa, pb;
        engine_order(sa_, pa, ablocks);
        engine_order(sb_, pb, {0, static_cast<long long>(nb_)});
        pa_ = pa;
        pb_ = pb;
        // row of the gathered vector (rank blocks padded to maxrows) for each engine alpha row
        std::vector<long long> gofs(na_);
        for (int r = 0; r < P_; r++)
            for (long long i = ranges[2 * r]; i < ranges[2 * r + 1]; i++)
                gofs[i] = static_cast<long long>(r) * maxrows_ + (pa[i] - ranges[2 * r]);
        gofs_ = gofs;

        std::vector<double> h1(static_cast<size_t>(L_) * L_);
        std::vector<double> eri(static_cast<size_t>(L_) * L_ * L_ * L_);
        for (int p = 0; p < L_; p++)
            for (int q = 0; q < L_; q++) h1[p * L_ + q] = I1.Value(2 * p, 2 * q);
        for (int p = 0; p < L_; p++)
            for (int q = 0; q < L_; q++)
                for (int r = 0; r < L_; r++)
                    for (int s = 0; s < L_; s++)
                        eri[((static_cast<size_t>(p) * L_ + q) * L_ + r) * L_ + s] =
                            I2.Value(2 * p, 2 * q, 2 * r, 2 * s);

        factored::BuildSpinStrings(sa_, L_, h1, eri);
        factored::BuildSpinStrings(sb_, L_, h1, eri);
        factored::BuildSpinTiles(tb_, sb_, 0.0);  // the alpha tiles cover the computed rows (SplitRows)
        tcnt_ = factored::ColumnTilesPerTileRow(sa_);
        int dev = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        size_t smem_limit = static_cast<size_t>(optin);
#if SBD_FM_SMEM_LIMIT > 0
        smem_limit = std::min<size_t>(smem_limit, SBD_FM_SMEM_LIMIT);  // to test the segments on a larger GPU
#endif
        factored::BuildCrossTables(cross_, sa_, sb_, L_, eri, smem_limit);

        // Shared memory of the largest cross-term launch on any rank, so that every rank sets the
        // same limit and the occupancy of each group of segments can be queried for the work estimate.
        std::vector<int> km_all;  // largest padded excitation count in each group, over all segments
        for (size_t g = 0, lo = 0; g < sizeof(factored::fm_kp_bounds) / sizeof(int); g++) {
            const int hi = factored::fm_kp_bounds[g];
            int km = 0;
            for (int kp : cross_.Kp)
                if (kp > static_cast<int>(lo) && kp <= hi) km = std::max(km, kp);
            km_all.push_back(km);
            lo = hi;
        }
        size_t smem = 0;
        for (int km : km_all)
            if (km > 0) smem = std::max(smem, factored::fm_cross_smem(km, cross_.max_chunks));
        if (smem > smem_limit)
            throw std::runtime_error("factored_mult: excitation lists too long for shared memory");
        cudaFuncSetAttribute(factored::fused_cross, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
        cudaFuncSetAttribute(factored::spmm_same_spin, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * 64 * SBD_FM_LDC * static_cast<int>(sizeof(double)));
        ranges_ = ranges;
        BalanceRows(km_all);
        if (rank_ == 0)
            std::printf(" factored_mult: %zu alpha x %zu beta strings on %d rank(s); same-spin entries "
                        "%lld / %lld; cross term %.1f rs rows per 64-wide beta tile, %.3e flop per "
                        "multiplication on rank 0\n",
                        na_, nb_, P_, static_cast<long long>(sa_.hptr[na_]),
                        static_cast<long long>(sb_.hptr[nb_]), 64.0 * cross_.total_chunks / cross_.nbt,
                        cross_.flops);
        if (rank_ == 0 && cross_.sptr[na_] > na_) {
            size_t nsplit = 0;
            for (size_t a = 0; a < na_; a++) nsplit += cross_.sptr[a + 1] - cross_.sptr[a] > 1;
            std::printf(" factored_mult: %zu excitation lists longer than %d entries (the shared memory of a "
                        "block) run as %zu segments in all\n", nsplit, cross_.kseg, cross_.sptr[na_]);
        }

        std::vector<unsigned long long> xa(sa_.x.begin(), sa_.x.end()), xb(sb_.x.begin(), sb_.x.end());
        xa_ = xa;
        xb_ = xb;
        // the diagonal is made in sbd's order, for this rank's rows
        std::vector<unsigned long long> xae(xa_ext.begin() + a0_, xa_ext.begin() + a1_);
        xa_ext_ = xae;
        std::vector<double> da(na_), db(nb_);
        for (size_t i = 0; i < na_; i++) da[pa[i]] = sa_.hdiag[i];
        for (size_t i = 0; i < nb_; i++) db[pb[i]] = sb_.hdiag[i];
        std::vector<double> da_local(da.begin() + a0_, da.begin() + a1_);
        da_ = da_local;
        db_ = db;
        std::vector<double> Jb(nb_ * L_, 0.0);  // Jb[b, p] = sum over r in b of (pp|rr)
        for (size_t b = 0; b < nb_; b++)
            for (int p = 0; p < L_; p++)
                for (int r = 0; r < L_; r++)
                    if ((xb_ext[b] >> r) & 1)
                        Jb[b * L_ + p] += eri[((static_cast<size_t>(p) * L_ + p) * L_ + r) * L_ + r];
        Jb_ = Jb;

        if (P_ > 1) {
            // Across nodes the gather needs NCCL: its allgather runs as 8 rings that cross between
            // nodes at different GPUs, so it uses all four NICs of a node, 84 GB/s per GPU on two GH200
            // nodes against 24 GB/s for MPI_Allgather; Davidson at 1e9 determinants on 8 GPUs took
            // 51.5 s against 120.9 s.
            MPI_Comm node;
            MPI_Comm_split_type(b_comm, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &node);
            int on_node = 0;
            MPI_Comm_size(node, &on_node);
            MPI_Comm_free(&node);
            int one_node = on_node == P_, all_one_node = 0;
            MPI_Allreduce(&one_node, &all_one_node, 1, MPI_INT, MPI_LAND, b_comm);
#ifndef SBD_USE_NCCL
            if (!all_one_node)
                throw std::runtime_error("factored_mult: ranks on more than one node need a build with "
                                         "SBD_USE_NCCL=ON (the MPI_Allgather gather is too slow between nodes)");
#ifdef SBD_NON_CUDA_AWARE_MPI
            // Without NCCL the gather and row exchange pass device memory to MPI; copying them through
            // host memory instead left 2 GPUs no faster than 1 (job 3479048).
            throw std::runtime_error("factored_mult: several ranks need a build with SBD_USE_NCCL=ON or "
                                     "with GPU-aware MPI (SBD_NON_CUDA_AWARE_MPI=OFF)");
#else
            // A GPU-aware build still fails if MPI's GPU support is off at run time: Cray MPICH without
            // MPICH_GPU_SUPPORT_ENABLED=1 hangs on an A100 and aborts on GH200. Cray's MPIX_Query_cuda_support
            // returns 0 even when it is on, so read the variable there; ask Open MPI.
#if defined(CRAY_MPICH_VERSION)
            const char* gpu = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
            if (!gpu || std::strcmp(gpu, "1") != 0)
                throw std::runtime_error("factored_mult: several ranks need MPICH_GPU_SUPPORT_ENABLED=1 "
                                         "or a build with SBD_USE_NCCL=ON");
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
            if (!MPIX_Query_cuda_support())
                throw std::runtime_error("factored_mult: several ranks need MPI with CUDA support enabled "
                                         "or a build with SBD_USE_NCCL=ON");
#endif
#endif
#endif
            if (maxrows_ * nb_ > static_cast<size_t>(INT32_MAX))
                throw std::runtime_error("factored_mult: a rank's part of the vector exceeds 2^31 elements");
#ifdef SBD_USE_NCCL
            ncclUniqueId id;
            if (rank_ == 0) ncclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, b_comm);
            if (ncclCommInitRank(&nccl_, P_, id, rank_) != ncclSuccess)
                throw std::runtime_error("factored_mult: ncclCommInitRank failed");
            nccl_ready_ = true;
            // the gather's stream must not wait for the legacy default stream, which runs the kernels
            cudaStreamCreateWithFlags(&comm_, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&local_ready_, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&gathered_, cudaEventDisableTiming);
            for (int b = 0; b < 2; b++) {
                cudaEventCreateWithFlags(&pass_packed_[b], cudaEventDisableTiming);
                cudaEventCreateWithFlags(&pass_in_[b], cudaEventDisableTiming);
            }
#endif
        }
        ChoosePasses(passes, reserve_bytes);
    }

    /**
       Passes: a multiplication brings in the beta columns [bg_[q], bg_[q + 1]) of the input at a time
       (whole 64-wide tiles), into two buffers of 1/Q of it each, the next pass's broadcast running
       behind the current pass's kernels. With passes <= 0 Q is the smallest from 2 whose buffers fit
       in the free GPU memory, after reserve_bytes for what the caller allocates later (the Davidson
       or Lanczos vectors, the diagonal) and this engine's own rows (phased input, computed output,
       received rows), with 5% to spare. Two passes were faster than one gather at 1e9 determinants
       on 8 GPUs (43.9 against 45.2 s) and four nearly as fast (44.4 s). One rank needs no buffers.
    */
    void ChoosePasses(int passes, size_t reserve_bytes)
    {
        const int nbt = static_cast<int>(cross_.nbt);
        auto buffers = [&](int q) {  // bytes of the pass buffers for q passes
            size_t maxw = 0;
            for (int k = 0; k < q; k++) {
                const long long g0 = std::min<long long>(nb_, static_cast<long long>(cross_.nbt * k / q) * 64);
                const long long g1 = std::min<long long>(nb_, static_cast<long long>(cross_.nbt * (k + 1) / q) * 64);
                maxw = std::max<size_t>(maxw, g1 - g0);
            }
            return static_cast<size_t>(std::min(q, 2)) * na_ * maxw * sizeof(double);
        };
        const char* how = "requested";
        if (P_ == 1) {
            Q_ = 1;
            how = "one rank";
        } else if (passes > 0) {
            Q_ = std::min(passes, nbt);
        } else {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const double own = 8.0 * nb_ * ((a1_ - a0_) + 1.2 * (c1_ - c0_));  // phased rows, output, received
            const double avail = static_cast<double>(free_b) - 0.05 * total_b - own - static_cast<double>(reserve_bytes);
            int q = 2;
            while (q < nbt && static_cast<double>(buffers(q)) > avail) q++;
            int qmax = q;  // the same Q on every rank: the largest any rank needs
            MPI_Allreduce(&q, &qmax, 1, MPI_INT, MPI_MAX, this->b_comm_);
            if (static_cast<double>(buffers(qmax)) > avail && qmax >= nbt)
                throw std::runtime_error("factored_mult: the input vector does not fit in GPU memory in any number of passes");
            Q_ = qmax;
            how = "chosen for the free GPU memory";
        }
        bg_.assign(Q_ + 1, 0);
        for (int q = 0; q <= Q_; q++)
            bg_[q] = std::min<long long>(nb_, static_cast<long long>(cross_.nbt * q / Q_) * 64);
        if (rank_ == 0)
            std::printf(" factored_mult: %d pass(es) per multiplication (%s); input vector buffers %.2f GB per "
                        "GPU, own rows %.2f GB\n",
                        Q_, how, P_ == 1 ? 0.0 : 1e-9 * buffers(Q_), 8e-9 * (a1_ - a0_) * nb_);
    }

    /// Diagonal of H for this rank's rows, in sbd's layout (the phase does not touch it).
    void makeQChamDiagTerms(thrust::device_vector<ElemT> &hii)
    {
        hii.resize((a1_ - a0_) * nb_);
        factored::diagonal_terms<<<1024, 256>>>(
            ptr(xa_ext_), ptr(da_), ptr(db_), ptr(Jb_), L_, static_cast<long long>(a1_ - a0_),
            static_cast<long long>(nb_), I0_, ptr(hii));
        check("diagonal_terms");
    }

    /// Wb += H Wk for this rank's rows; hii is not needed (the diagonal is part of the factors).
    void run(const thrust::device_vector<ElemT> &hii,
             const thrust::device_vector<ElemT> &Wk,
             thrust::device_vector<ElemT> &Wb) override
    {
        (void)hii;
        SBD_NVTX_RANGE_COLOR("MultTPBFactored::run", __LINE__);
        const size_t nloc = (a1_ - a0_) * nb_;
        const size_t ncomp = (c1_ - c0_) * nb_;
        if (Wb.size() != nloc) Wb.resize(nloc);
        if (TL_.size() != nloc) TL_.resize(nloc);
        size_t maxw = 0;
        for (int q = 0; q < Q_; q++) maxw = std::max<size_t>(maxw, bg_[q + 1] - bg_[q]);
        const int nbuf = P_ == 1 ? 0 : std::min(Q_, 2);  // one rank reads its own rows directly
        for (int b = 0; b < nbuf; b++)
            if (Tb_[b].size() != na_ * maxw) Tb_[b].resize(na_ * maxw);
        if (S_.size() != ncomp) S_.resize(ncomp);
        if (R_.size() != nrecv_rows_ * nb_) R_.resize(nrecv_rows_ * nb_);
        const long long na = static_cast<long long>(na_), nb = static_cast<long long>(nb_);
        const long long a0 = static_cast<long long>(a0_), a1 = static_cast<long long>(a1_);
        const long long c0 = static_cast<long long>(c0_), c1 = static_cast<long long>(c1_);
        const long long m0 = std::max(a0, c0), m1 = std::min(a1, c1);  // rows owned and computed here
        double* TL = ptr(TL_);
        double* S = ptr(S_);
        const int ss_smem = 2 * 64 * SBD_FM_LDC * sizeof(double);
        // the second multiplication times the kernels the work split models (Calibrate)
        const size_t NG = cg_.size();
        const bool timing = P_ > 1 && nrun_ == 1;
        std::vector<std::pair<size_t, std::pair<cudaEvent_t, cudaEvent_t>>> marks;
        auto timed = [&](size_t what, auto &&launch) {
            if (!timing) { launch(); return; }
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, 0);
            launch();
            cudaEventRecord(e1, 0);
            marks.push_back({what, {e0, e1}});
        };
        // beta same-spin terms for the computed rows [r0, r1) from the columns [cw0, cw1) of C, whose
        // element (alpha row i, beta column j) is at C[(i - crow0) * ldc + j - ccol0]
        auto beta_terms = [&](long long r0, long long r1, const double* C, long long crow0, long long ccol0,
                              long long ldc, long long cw0, long long cw1) {
            if (r1 > r0)
                timed(NG + 1, [&] {
                    factored::spmm_same_spin<<<dim3(tb_.nT, (r1 - r0 + 63) / 64), 2 * SBD_FM_TR, ss_smem>>>(
                        C, S, ptr(tb_.trp), ptr(tb_.tj), ptr(tb_.rp), ptr(tb_.ec), ptr(tb_.ev), na, nb, r0, r1,
                        c0, 0, 1, r0, r1, cw0, cw1, crow0, ccol0, ldc);
                });
        };

        // This rank's rows in engine order and phase (TL). Pass q brings in the beta columns
        // [bg_[q], bg_[q + 1]) of every alpha row: each rank copies its rows' slab into Tb_[q % 2],
        // and the owners broadcast them on comm_ while the previous pass computes. Each beta tile of
        // the cross term and each beta chunk of the alpha terms is computed in the one pass that holds
        // its columns; the beta terms of the rows a rank owns and computes need only TL and run during
        // the first broadcast.
        factored::gather_phase<<<1024, 256>>>(ptr(xa_), ptr(xb_), ptr(gofs_), ptr(pb_), a0, a1, nb,
                                               static_cast<long long>(rank_ * maxrows_), a0, ptr(Wk), TL);
        cudaMemsetAsync(S, 0, ncomp * sizeof(double), 0);
        auto pack = [&](int q) {  // this rank's rows of pass q into its buffer, on the compute stream
            const long long w = bg_[q + 1] - bg_[q];
            cudaMemcpy2DAsync(ptr(Tb_[q & 1]) + a0 * w, w * sizeof(double), TL + bg_[q], nb * sizeof(double),
                              w * sizeof(double), a1 - a0, cudaMemcpyDeviceToDevice, 0);
        };
#ifdef SBD_USE_NCCL
        auto broadcast = [&](int q) {  // after pack(q), on comm_
            const int b = q & 1;
            const long long w = bg_[q + 1] - bg_[q];
            cudaEventRecord(pass_packed_[b], 0);
            cudaStreamWaitEvent(comm_, pass_packed_[b], 0);
            double* buf = ptr(Tb_[b]);
            ncclGroupStart();
            for (int r = 0; r < P_; r++) {
                const long long lo = ranges_[2 * r], hi = ranges_[2 * r + 1];
                if (lo < hi)
                    ncclBroadcast(buf + lo * w, buf + lo * w, (hi - lo) * w, ncclDouble, r, nccl_, comm_);
            }
            if (ncclGroupEnd() != ncclSuccess) throw std::runtime_error("factored_mult: pass broadcast failed");
            cudaEventRecord(pass_in_[b], comm_);
        };
        if (P_ > 1) {
            SBD_NVTX_RANGE_COLOR("MultTPBFactored::gather", __LINE__);
            for (int q = 0; q < std::min(Q_, 2); q++) {
                pack(q);
                broadcast(q);
            }
        }
#endif
        beta_terms(m0, m1, TL, a0, 0, nb, 0, nb);
        for (int q = 0; q < Q_; q++) {
            const long long g0 = bg_[q], g1 = bg_[q + 1], w = g1 - g0;
            const double* C = TL;
            long long crow0 = a0, ccol0 = 0, ldc = nb;
            if (P_ > 1) {
                C = ptr(Tb_[q & 1]);
                crow0 = 0;
                ccol0 = g0;
                ldc = w;
#ifdef SBD_USE_NCCL
                cudaStreamWaitEvent(0, pass_in_[q & 1], 0);
#else
                // MPI: the broadcasts of a pass, after the kernels queued so far
                pack(q);
                cudaStreamSynchronize(0);
                std::vector<MPI_Request> req;
                for (int r = 0; r < P_; r++) {
                    const long long lo = ranges_[2 * r], hi = ranges_[2 * r + 1];
                    if (lo >= hi) continue;
                    req.emplace_back();
                    MPI_Ibcast(ptr(Tb_[q & 1]) + lo * w, static_cast<int>((hi - lo) * w), MPI_DOUBLE, r,
                               this->b_comm_, &req.back());
                }
                MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
#endif
                // beta terms of the computed rows owned by other ranks, from this pass's columns
                beta_terms(c0, std::min(c1, a0), C, crow0, ccol0, ldc, g0, g1);
                beta_terms(std::max(c0, a1), c1, C, crow0, ccol0, ldc, g0, g1);
            }
            if (ta_.nT > 0 && w > 0)
                timed(NG, [&] {
                    factored::spmm_same_spin<<<dim3(ta_.nT, (w + 63) / 64), 2 * SBD_FM_TR, ss_smem>>>(
                        C, S, ptr(ta_.trp), ptr(ta_.tj), ptr(ta_.rp), ptr(ta_.ec), ptr(ta_.ev), na, nb, c0, c1, c0,
                        static_cast<long long>(ta_.r0), 0, g0, g1, 0, na, crow0, ccol0, ldc);
                });
            const int bt0 = static_cast<int>(g0 / 64), nbt = static_cast<int>((w + 63) / 64);
            for (size_t k = 0; k < cross_.bucket_ids.size() && nbt > 0; k++) {
                const int* ids = ptr(cross_.bucket_ids[k]);
                const size_t count = cross_.bucket_ids[k].size();
                timed(cross_.bucket_group[k], [&] {
                    for (size_t y0 = 0; y0 < count; y0 += 65535) {
                        const unsigned ny = static_cast<unsigned>(std::min<size_t>(65535, count - y0));
                        factored::fused_cross<<<dim3(nbt, ny), 128, cross_.bucket_smem[k]>>>(
                            C - crow0 * ldc, ptr(cross_.Vs), cross_.ldv, ptr(cross_.kptr), ptr(cross_.kket),
                            ptr(cross_.krow), ptr(cross_.kbra), ids + y0, ptr(cross_.goff), ptr(cross_.gent), ptr(cross_.cbase),
                            ptr(cross_.rlist), nb, c0, S, ldc, ccol0, bt0);
                    }
                });
            }
#ifdef SBD_USE_NCCL
            if (P_ > 1 && q + 2 < Q_) {
                pack(q + 2);  // into this pass's buffer, after its kernels
                broadcast(q + 2);
            }
#endif
        }
        if (m1 > m0)
            factored::scatter_phase<<<1024, 256>>>(ptr(xa_), ptr(xb_), ptr(pa_), ptr(pb_), m0, m1, nb, c0, a0,
                                                   S, ptr(Wb));
        if (!sends_.empty() || !recvs_.empty()) {
            // computed rows to their owners
            SBD_NVTX_RANGE_COLOR("MultTPBFactored::exchange", __LINE__);
            double* R = ptr(R_);
#ifdef SBD_USE_NCCL
            cudaEventRecord(local_ready_, 0);
            cudaStreamWaitEvent(comm_, local_ready_, 0);
            ncclGroupStart();
            for (const Piece &p : sends_)
                ncclSend(S + p.off * nb, (p.r1 - p.r0) * nb_, ncclDouble, p.peer, nccl_, comm_);
            for (const Piece &p : recvs_)
                ncclRecv(R + p.off * nb, (p.r1 - p.r0) * nb_, ncclDouble, p.peer, nccl_, comm_);
            if (ncclGroupEnd() != ncclSuccess) throw std::runtime_error("factored_mult: row exchange failed");
            cudaEventRecord(gathered_, comm_);
            cudaStreamWaitEvent(0, gathered_, 0);
#else
            cudaStreamSynchronize(0);
            std::vector<MPI_Request> req;
            for (const Piece &p : sends_) {
                req.emplace_back();
                MPI_Isend(S + p.off * nb, static_cast<int>((p.r1 - p.r0) * nb), MPI_DOUBLE, p.peer, 0,
                          this->b_comm_, &req.back());
            }
            for (const Piece &p : recvs_) {
                req.emplace_back();
                MPI_Irecv(R + p.off * nb, static_cast<int>((p.r1 - p.r0) * nb), MPI_DOUBLE, p.peer, 0,
                          this->b_comm_, &req.back());
            }
            MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
#endif
            for (const Piece &p : recvs_)
                factored::scatter_phase<<<1024, 256>>>(ptr(xa_), ptr(xb_), ptr(pa_), ptr(pb_), p.r0, p.r1, nb,
                                                       p.r0 - p.off, a0, R, ptr(Wb));
        }
        check("run");
        if (timing) {
            std::vector<double> t(NG + 2, 0.0);
            for (auto &m : marks) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, m.second.first, m.second.second);
                t[m.first] += 1e-3 * ms;
                cudaEventDestroy(m.second.first);
                cudaEventDestroy(m.second.second);
            }
            Calibrate(t);
        }
        nrun_++;
    }

    /// Relative differences from a reference multiplier (method 0) in the diagonal and in H x for a
    /// pseudo-random x, over all ranks (--factored_mult_check 1).
    void CompareWithReference(MultBase<ElemT> &ref, const thrust::device_vector<ElemT> &hii,
                              const thrust::device_vector<ElemT> &hii_ref, double &diag_diff,
                              double &mult_diff)
    {
        const size_t nloc = (a1_ - a0_) * nb_;
        diag_diff = rel_diff(hii, hii_ref);
        thrust::device_vector<ElemT> x(nloc), y(nloc), y_ref(nloc);
        thrust::transform(thrust::counting_iterator<size_t>(a0_ * nb_),
                          thrust::counting_iterator<size_t>(a0_ * nb_ + nloc), x.begin(),
                          factored::hash_value());
        run(hii, x, y);
        ref.run(hii_ref, x, y_ref);
        mult_diff = rel_diff(y, y_ref);
    }

    /// Flops of this rank's cross-term tensor-core work per multiplication.
    double cross_flops() const { return cross_.flops; }

private:
    /**
       The rows each rank computes, [c0_, c1_), split by work rather than by count; the vector
       layout (each rank owns [a0_, a1_)) stays sbd's even split, and run() sends the rows computed by
       another rank to their owner. The work of an engine row a is
         cg_[group of a] x (padded excitations of a)   cross term, by launch group
         + ca_ x (nonzero column tiles of its tile row) / 128    alpha same-spin terms
         + cr_                                        beta same-spin terms, the same for every row
       in seconds per multiplication. The alpha kernel loads a 64 x 64 slice of C per nonzero tile,
       so its time goes by tiles, not entries (5.2-6.3 ms per 128-row tile row on every rank of job
       3444566, against 169-207 ms per rank). Init starts from GH200 kernel times (jobs 3444161 and
       3444566, 1e9 determinants on 8 GPUs): the cross term 0.55 ns per excitation and 64-row chunk
       of the beta tiles, x sqrt(2 / blocks per SM of its launch) (one block per SM costs ~1.4x);
       ~24.5 ns per column tile and 64 beta strings (6 ms per tile row if all 495 column tiles are
       nonzero); 0.775 ps per nonzero of H_b. The second
       multiplication times these kernels on every rank, and Calibrate() refits the coefficients and
       splits the rows again.
       With even counts the work of 8 ranks differed by 13% on Fe4S4, the ranks with the lowest
       strings (most excitations within the list) doing the most; every rank then waits for those
       at the next gather.
    */
    void BalanceRows(const std::vector<int> &km_all)
    {
        const size_t G = km_all.size();
        cg_.assign(G, 0.0);
        for (size_t g = 0; g < G; g++) {
            if (km_all[g] == 0) continue;
            int nblk = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, factored::fused_cross, 128,
                                                          factored::fm_cross_smem(km_all[g], cross_.max_chunks));
            cg_[g] = 0.55e-9 * cross_.total_chunks * std::sqrt(2.0 / std::max(nblk, 1));
        }
        ca_ = 24.5e-9 * static_cast<double>((nb_ + 63) / 64);
        cr_ = 0.775e-12 * static_cast<double>(sb_.hptr[nb_]);
        cb_.assign(P_ + 1, 0);
        for (int r = 0; r < P_; r++) cb_[r + 1] = ranges_[2 * r + 1];
        SplitRows("estimated work");
    }

    /// Computed ranges from the coefficients, the exchange plan and this rank's cross-term launches.
    void SplitRows(const char* how)
    {
        std::vector<double> pre(na_ + 1, 0.0);
        for (size_t a = 0; a < na_; a++) {
            pre[a + 1] = pre[a] + ca_ * tcnt_[a / SBD_FM_TR] / SBD_FM_TR + cr_;
            for (size_t sg = cross_.sptr[a]; sg < cross_.sptr[a + 1]; sg++) {
                const int kp = cross_.Kp[sg];
                size_t g = 0;
                while (kp > factored::fm_kp_bounds[g]) g++;
                pre[a + 1] += cg_[g] * kp;
            }
        }
        double even = 0.0, before = 0.0, after = 0.0;
        for (int q = 0; q < P_; q++) {
            even = std::max(even, pre[ranges_[2 * q + 1]] - pre[ranges_[2 * q]]);
            before = std::max(before, pre[cb_[q + 1]] - pre[cb_[q]]);
        }
        if (P_ > 1) {
            for (int r = 1; r < P_; r++) {
                cb_[r] = std::lower_bound(pre.begin(), pre.end(), pre[na_] * r / P_) - pre.begin();
                cb_[r] = std::max(cb_[r], cb_[r - 1]);
            }
            cb_[P_] = na_;
        }
        c0_ = cb_[rank_];
        c1_ = cb_[rank_ + 1];
        sends_.clear();
        recvs_.clear();
        nrecv_rows_ = 0;
        long long moved = 0;
        for (int q = 0; q < P_; q++) {
            const long long o0 = ranges_[2 * q], o1 = ranges_[2 * q + 1];
            after = std::max(after, pre[cb_[q + 1]] - pre[cb_[q]]);
            moved += std::abs(static_cast<long long>(cb_[q]) - o0);
            if (q == rank_) continue;
            const long long s0 = std::max<long long>(c0_, o0), s1 = std::min<long long>(c1_, o1);
            if (s0 < s1) sends_.push_back({q, s0, s1, s0 - static_cast<long long>(c0_)});
            const long long r0 = std::max<long long>(cb_[q], a0_), r1 = std::min<long long>(cb_[q + 1], a1_);
            if (r0 < r1) {
                recvs_.push_back({q, r0, r1, static_cast<long long>(nrecv_rows_)});
                nrecv_rows_ += r1 - r0;
            }
        }
        factored::BuildCrossBuckets(cross_, c0_, c1_);
        factored::BuildSpinTiles(ta_, sa_, I0_, c0_, c1_);
        if (rank_ == 0 && P_ > 1)
            std::printf(" factored_mult: rows split by %s: slowest rank %.1f ms per multiplication with "
                        "even counts, %.1f ms with the previous split, %.1f ms now (mean %.1f ms); %lld "
                        "boundary rows moved\n",
                        how, 1e3 * even, 1e3 * before, 1e3 * after, 1e3 * pre[na_] / P_, moved);
    }

    /// Refit cg_, ca_ and cr_ from the kernel times of one multiplication on all ranks
    /// (t: seconds per launch group, then alpha and beta same-spin terms) and split again.
    void Calibrate(const std::vector<double> &t)
    {
        const size_t G = cg_.size();
        std::vector<double> v(2 * G + 4, 0.0);  // times and work: groups, alpha, beta
        for (size_t k = 0; k < cross_.bucket_group.size(); k++) v[G + cross_.bucket_group[k]] += cross_.bucket_kp[k];
        for (size_t g = 0; g < G; g++) v[g] = t[g];
        v[2 * G] = t[G];
        v[2 * G + 1] = static_cast<double>(ta_.ntiles);
        v[2 * G + 2] = t[G + 1];
        v[2 * G + 3] = static_cast<double>(c1_ - c0_);
        std::vector<double> sum(v.size());
        MPI_Allreduce(v.data(), sum.data(), static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM, this->b_comm_);
        double mine = 0.0, slowest = 0.0;
        for (double x : t) mine += x;
        MPI_Allreduce(&mine, &slowest, 1, MPI_DOUBLE, MPI_MAX, this->b_comm_);
        double total = 0.0;
        for (size_t g = 0; g < G; g++) total += sum[g];
        total += sum[2 * G] + sum[2 * G + 2];
        if (rank_ == 0)
            std::printf(" factored_mult: measured kernel time per multiplication: slowest rank %.1f ms, mean "
                        "%.1f ms\n", 1e3 * slowest, 1e3 * total / P_);
        for (size_t g = 0; g < G; g++)
            if (sum[G + g] > 0.0) cg_[g] = sum[g] / sum[G + g];
        if (sum[2 * G + 1] > 0.0) ca_ = sum[2 * G] / sum[2 * G + 1];
        if (sum[2 * G + 3] > 0.0) cr_ = sum[2 * G + 2] / sum[2 * G + 3];
        if (rank_ == 0) {
            std::printf(" factored_mult: fitted costs: cross term");
            for (size_t g = 0; g < G; g++)
                if (sum[G + g] > 0.0)
                    std::printf(" %.3f", 1e9 * cg_[g] / cross_.total_chunks);
            std::printf(" ns per excitation and chunk by launch group; alpha %.2f ns per column tile and "
                        "64 beta strings; beta %.2f us per row\n",
                        1e9 * ca_ / ((nb_ + 63) / 64), 1e6 * cr_);
        }
        SplitRows("measured kernel times");
    }

    template <typename T>
    static T* ptr(thrust::device_vector<T> &v) { return thrust::raw_pointer_cast(v.data()); }
    template <typename T>
    static const T* ptr(const thrust::device_vector<T> &v) { return thrust::raw_pointer_cast(v.data()); }
    double rel_diff(const thrust::device_vector<ElemT> &a, const thrust::device_vector<ElemT> &b) const
    {
        thrust::device_vector<ElemT> d(a.size());
        thrust::transform(a.begin(), a.end(), b.begin(), d.begin(), thrust::minus<ElemT>());
        double local[2] = {thrust::inner_product(d.begin(), d.end(), d.begin(), 0.0),
                           thrust::inner_product(b.begin(), b.end(), b.begin(), 0.0)};
        double sum[2];
        MPI_Allreduce(local, sum, 2, MPI_DOUBLE, MPI_SUM, this->b_comm_);
        return sum[1] > 0.0 ? std::sqrt(sum[0] / sum[1]) : std::sqrt(sum[0]);
    }
    static void check(const char* where)
    {
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("factored_mult: ") + where + ": " + cudaGetErrorString(err));
    }

    int L_ = 0;
    double I0_ = 0.0;
    size_t na_ = 0, nb_ = 0;
    int P_ = 1, rank_ = 0;
    size_t a0_ = 0, a1_ = 0;    // this rank's alpha rows (same positions in sbd and engine order)
    size_t c0_ = 0, c1_ = 0;    // the engine rows this rank computes (BalanceRows)
    std::vector<size_t> cb_;    // every rank's first computed row, and na
    std::vector<long long> ranges_;  // every rank's owned rows, [ranges_[2r], ranges_[2r + 1])
    std::vector<double> cg_;    // work model (BalanceRows)
    std::vector<int> tcnt_;     // nonzero column tiles of each 128-row tile row of H_a, tiles from row 0
    double ca_ = 0.0, cr_ = 0.0;
    int nrun_ = 0;              // multiplications so far; the second one calibrates the split
    struct Piece {
        int peer;
        long long r0, r1, off;  // engine rows [r0, r1), at row off of S (sends) or R_ (receives)
    };
    std::vector<Piece> sends_, recvs_;  // computed rows owned by another rank, and the reverse
    size_t nrecv_rows_ = 0;
    size_t maxrows_ = 0;        // largest alpha range over the ranks
    factored::SpinStrings sa_, sb_;
    factored::SpinTiles ta_, tb_;
    factored::CrossTables cross_;
    thrust::device_vector<unsigned long long> xa_, xb_;  // engine order
    thrust::device_vector<unsigned long long> xa_ext_;   // sbd order, this rank's rows (diagonal)
    thrust::device_vector<int> pa_, pb_;                 // engine index -> sbd index
    thrust::device_vector<long long> gofs_;              // engine alpha row -> row of the gathered vector
    thrust::device_vector<double> da_, db_, Jb_;         // sbd order (diagonal)
    thrust::device_vector<double> TL_, S_;               // own rows in engine order and phase, computed rows
    thrust::device_vector<double> Tb_[2];                // a pass's beta columns of every alpha row
    int Q_ = 1;                                          // passes per multiplication
    std::vector<long long> bg_;                          // first beta column of each pass, and nb
    thrust::device_vector<double> R_;                    // received rows of this rank's output
#ifdef SBD_USE_NCCL
    ncclComm_t nccl_;
    bool nccl_ready_ = false;
    cudaStream_t comm_;                 // the gather
    cudaEvent_t local_ready_, gathered_;
    cudaEvent_t pass_packed_[2], pass_in_[2];  // a pass's own rows copied; its input arrived
#endif
};

} // namespace sbd

#endif
