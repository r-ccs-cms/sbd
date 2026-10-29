/**
@file sbd/chemistry/gdb/mult_staircase_thrust.h
@brief GDB H·C on a staircase of string products (--factored_mult 1)

A GDB basis S is a list of (alpha string, beta string) pairs. Ranked by how many determinants each
string appears in, heat-bath lists fill dyadic buckets of (alpha rank, beta rank) almost completely
or almost not at all, along a hyperbola: a staircase. With the strings in rank order, the staircase
gives each alpha row a a width w(a), non-increasing in a; its cells (a, b < w(a)) hold 97-98% of the
list at about 1.4x its size. The rest of the list is the remainder.

H·C on S, with C zero outside S, is

  sigma = P_S (H_a C + C H_b^T + sum (pq|rs) E^a_pq C E^b_rs^T) + I0 C

(H_a, H_b the same-spin string Hamiltonians, E_pq the string excitations), computed in two parts:

  - staircase kets to staircase bras, from the TPB factors on the staircase layout (row a of C at
    roff[a], w(a) entries):
      same-spin terms by sparse tiles of H_a and H_b (spmm_stair, masked to the widths);
      the cross term pair by pair of alpha strings (bra a = E_pq ket a'), in the cheaper direction:
      bra-driven (cross_bra, TPB's fused_cross with the bra's kets sorted by width, so a 64-wide tile
      of kets' beta columns b' uses only the kets wider than the tile start, and outputs b >= w(a)
      dropped) when w(a') <= w(a), and ket-driven (cross_ket: for a ket row a' and a 64-wide tile of
      output columns b, gather T[rs, b] = sgn C[a', E_rs^-1 b], form U = V T for the ket's bras, and
      add the rows of the bras wider than the tile start) when w(a) < w(a'). Either way a pair costs
      about min(w(a), w(a')) times the rs rows of its tiles;
  - every pair with a remainder determinant, per element (rem_elem): for each remainder bra, all of
    its connected kets (same-spin rows of H_a and H_b, and the products of its alpha and beta
    excitation lists), adding H_bk C_k into sigma_b and, for staircase kets k, H_kb C_b into sigma_k.

The engine order of a determinant is alpha string then beta string; sbd's interleaved determinants
differ by the sign factored::fm_phase.

Several ranks: sbd's determinants are split over b_comm (and repeated over h_comm and t_comm). Every rank
builds the staircase from the whole list, gathers the whole of C before each multiplication (GPU-aware
MPI_Allgatherv over b_comm), and does a contiguous share of each part's work list (the same-spin tile rows,
the cross-term blocks, the remainder determinants), cut at equal estimated cost over all the ranks; the
partial sigmas are summed (MPI_Allreduce over b_comm, t_comm and h_comm) and each rank keeps its own rows.
Limits: real coefficients, at most 64 orbitals, fewer than 2^19 beta strings.
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_STAIRCASE_THRUST_H
#define SBD_CHEMISTRY_GDB_MULT_STAIRCASE_THRUST_H

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

#include "sbd/chemistry/tpb/helper.h"
#include "sbd/chemistry/tpb/mult_factored_thrust.h"

namespace sbd {
namespace staircase {

using factored::fm_dmma;
using factored::fm_phase;

/**
   Same-spin products on the staircase: C and sigma hold row a at roff[a] with wid[a] entries
   (wid non-increasing). Tiles as factored::spmm_same_spin (TR x 64, entries by row).
     MODE 0: sigma[I rows, chunk] += H_a[I rows, :] C[:, chunk]   (blockIdx.x = alpha tile row,
             blockIdx.y = 64-wide beta chunk)
     MODE 1: sigma[chunk, I cols] += C[chunk, :] H_b[:, I cols]    (blockIdx.x = beta tile row,
             blockIdx.y = 64-row alpha chunk)
   A 1-D launch over blocks [lo, lo + gridDim.x) of the (alpha, beta) block grid.
   Elements beyond a row's width read as zero and are not written.
*/
template <int MODE>
static __global__ void __launch_bounds__(2 * SBD_FM_TR, 256 / SBD_FM_TR)
spmm_stair(const double* __restrict__ C, double* sigma, const int* __restrict__ trp,
           const int* __restrict__ tj, const int* __restrict__ rp, const unsigned char* __restrict__ ec,
           const double* __restrict__ ev, const long long* __restrict__ roff, const int* __restrict__ wid,
           long long na, long long lo, int ny)
{
    extern __shared__ double smem[];
    // this rank's share: blocks lo + blockIdx.x of the grid, (I, ch) row-major with ny columns (MODE 0)
    // or (ch, I) with ny tile rows (MODE 1)
    const long long lin = lo + blockIdx.x;
    const int I = MODE == 0 ? (int)(lin / ny) : (int)(lin % ny);
    const int ch = MODE == 0 ? (int)(lin % ny) : (int)(lin / ny);
    const int tid = threadIdx.x;
    const int row = tid % SBD_FM_TR, half = tid / SBD_FM_TR;
    const long long y0 = (long long)ch * 64;
    long long cw1;  // H columns used: alpha rows wider than y0 (MODE 0), beta columns below the chunk's width (MODE 1)
    if (MODE == 0) {
        const long long r0 = (long long)I * SBD_FM_TR;
        if (r0 >= na || y0 >= wid[r0]) return;
        long long lo = 0, hi = na;
        while (lo < hi) { const long long m = (lo + hi) / 2; if (wid[m] > y0) lo = m + 1; else hi = m; }
        cw1 = lo;
    } else {
        if (y0 >= na || (long long)I * SBD_FM_TR >= wid[y0]) return;
        cw1 = wid[y0];
    }
    int t0 = trp[I], t1 = trp[I + 1];
    {
        int lo = t0, hi = t1;
        while (lo < hi) { const int m = (lo + hi) / 2; if ((long long)tj[m] * 64 < cw1) lo = m + 1; else hi = m; }
        t1 = lo;
    }
    auto load = [&](double* dst, long long J0) {
        const int u = tid & 63;
        for (int v = tid >> 6; v < 64; v += 2 * SBD_FM_TR / 64) {
            if (MODE == 0) {  // Cs[c][y] = C[J0 + c, y0 + y], c = v, y = u
                const long long x = J0 + v, y = y0 + u;
                const bool ok = x < cw1 && y < wid[x];
                factored::fm_cp_async8(dst + v * SBD_FM_LDC + u, C + (ok ? roff[x] + y : 0), ok);
            } else {          // Cs[c][y] = C[y0 + y, J0 + c], y = v, c = u
                const long long x = y0 + v, y = J0 + u;
                const bool ok = x < na && y < wid[x];
                factored::fm_cp_async8(dst + u * SBD_FM_LDC + v, C + (ok ? roff[x] + y : 0), ok);
            }
        }
        factored::fm_cp_commit();
    };
    double acc[32];
#pragma unroll
    for (int k = 0; k < 32; k++) acc[k] = 0.0;
    if (t0 < t1) load(smem, (long long)tj[t0] * 64);
    for (int t = t0; t < t1; t++) {
        const int st = (t - t0) & 1;
        if (t + 1 < t1) {
            load(smem + (st ^ 1) * 64 * SBD_FM_LDC, (long long)tj[t + 1] * 64);
            factored::fm_cp_wait1();
        }
        else factored::fm_cp_wait0();
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
                if (e + 1 < e1) { c = ec[e + 1]; v = ev[e + 1]; }
#pragma unroll
                for (int k = 0; k < 32; k++) acc[k] += vv * crow[k];
            }
        }
        __syncthreads();
    }
    const long long R = (long long)I * SBD_FM_TR + row;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        const long long y = y0 + half * 32 + k;
        if (MODE == 0) {
            if (R < na && y < wid[R]) sigma[roff[R] + y] += acc[k];
        } else {
            if (y < na && R < wid[y]) sigma[roff[y] + R] += acc[k];
        }
    }
}

/**
   Cross term, bra-driven (factored::fused_cross on the staircase). One block per (segment s, 64-wide tile
   bt of kets' beta columns) from the lists bs / bt; a segment is a bra's kets (sbra[s] = a), or a part of them
   when they exceed the shared memory of a block; its kets (kptr: alpha ranks kket, signed V rows krow, sorted
   by width, padded to a multiple of 4 with ket -1 and the zero row) wider than the tile start are used. Outputs b >= w(a) are dropped. Shared memory: 16 bytes (K_eff) + factored::fm_cross_smem + K ints,
   all dynamic.
*/
static __global__ void __launch_bounds__(128)
cross_bra(const double* __restrict__ C, const double* __restrict__ Vs, int ldv,
          const long long* __restrict__ kptr, const int* __restrict__ kket, const int* __restrict__ krow,
          const int* __restrict__ sbra, const int* __restrict__ bs, const int* __restrict__ btl,
          const long long* __restrict__ goff,
          const unsigned* __restrict__ gent, const int* __restrict__ cbase, const int* __restrict__ rlist,
          const long long* __restrict__ roff, const int* __restrict__ wid, double* sigma)
{
    extern __shared__ double smb[];
    int& keff_s = *(int*)smb;
    double* sm = smb + 2;
    const int sg = bs[blockIdx.x];
    const long long a = sbra[sg];
    const int bt = btl[blockIdx.x];
    const long long kp = kptr[sg];
    const int K = (int)(kptr[sg + 1] - kp);
    const long long b0 = (long long)bt * 64;
    const int c0 = cbase[bt], nch = cbase[bt + 1] - c0;
    double* Bs = sm;
    double* ring = Bs + K * SBD_FM_LD;
    long long* cr_s = (long long*)(ring + 64 * SBD_FM_LD);
    int* vr_s = (int*)(cr_s + K);
    int* cw_s = vr_s + K;
    int* rl_s = cw_s + K;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int r4 = lane & 3, q8 = lane >> 2;
    for (int i = tid; i < K; i += 128) {
        const int k = kket[kp + i];
        cr_s[i] = k >= 0 ? roff[k] : 0;
        cw_s[i] = k >= 0 ? wid[k] : 0;
        vr_s[i] = krow[kp + i] * ldv;
    }
    for (int i = tid; i < nch * 64; i += 128) rl_s[i] = rlist[(long long)c0 * 64 + i];
    __syncthreads();
    if (tid == 0) {  // kets wider than the tile start (sorted by width, padding has width 0)
        int lo = 0, hi = K;
        while (lo < hi) { const int m = (lo + hi) / 2; if (cw_s[m] > b0) lo = m + 1; else hi = m; }
        keff_s = lo;
    }
    __syncthreads();
    const int Keff = keff_s;
    if (Keff == 0 || nch == 0) return;
    const int Ke = (Keff + 3) / 4 * 4;
    const int T = (Ke + 31) / 32;
    const int m = tid & 63, kh = tid >> 6;
    for (int k = kh; k < Ke; k += 2) {
        const bool ok = k < Keff && b0 + m < cw_s[k];
        factored::fm_cp_async8(Bs + k * SBD_FM_LD + m, C + (ok ? cr_s[k] + b0 + m : 0), ok);
    }
    factored::fm_cp_commit();
    factored::fm_load_v_tile(ring, Vs, rl_s, vr_s, Ke, 0, m, kh);

    const long long wa = wid[a];
    double* srow = sigma + roff[a];
    for (int c = 0; c < nch; c++) {
        double acc[4][4][2];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j][0] = acc[i][j][1] = 0.0;
        for (int t = 0; t < T; t++) {
            if (t + 1 < T) {
                factored::fm_load_v_tile(ring + ((t + 1) & 1) * 32 * SBD_FM_LD, Vs, rl_s + c * 64, vr_s, Ke, t + 1, m, kh);
                factored::fm_cp_wait1();
            }
            else factored::fm_cp_wait0();
            __syncthreads();
            const double* As = ring + (t & 1) * 32 * SBD_FM_LD + wm * 32 + q8;
            const double* Bt = Bs + (t * 32) * SBD_FM_LD + wn * 32 + q8;
            const int kc = min(32, Ke - t * 32);
            for (int kk = 0; kk < kc; kk += 4) {
                double fa[4], fb[4];
#pragma unroll
                for (int i = 0; i < 4; i++) fa[i] = As[(kk + r4) * SBD_FM_LD + i * 8];
#pragma unroll
                for (int j = 0; j < 4; j++) fb[j] = Bt[(kk + r4) * SBD_FM_LD + j * 8];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) fm_dmma(acc[i][j][0], acc[i][j][1], fa[i], fb[j]);
            }
            __syncthreads();
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
            const unsigned u = gent[e];
            const long long b = u >> 13;
            if (b < wa) {
                const double v = Xs[(u & 63) * SBD_FM_LD + ((u >> 6) & 63)];
                atomicAdd(srow + b, (u >> 12) & 1 ? -v : v);
            }
        }
        __syncthreads();
        if (c + 1 < nch) factored::fm_load_v_tile(ring, Vs, rl_s + (c + 1) * 64, vr_s, Ke, 0, m, kh);
    }
}

/**
   Cross term, ket-driven. One block per (ket a', group g of 64 of its bras, 64-wide tile bt of output
   columns b) from the lists. The ket's bras (kptr: alpha ranks kbra, signed V rows krow, sorted by
   width) wider than the tile start are used. For each 64-row chunk of the rs rows the output tile
   uses (out tables: entries rs_local 6 | b_local 6 | sign 1 | b' 19, b = E_rs b'):
     T[rs, b] = sgn C[a', b'] (zero for b' >= w(a')),   U[bra, b] += V[bra, rs] T[rs, b]
   by mma.sync; then U is added into the bras' rows.
*/
#define CROSS_KET_SMEM (2 * 64 * SBD_FM_LD * sizeof(double) + 64 * sizeof(long long) + (3 * 64 + 4) * sizeof(int))
static __global__ void __launch_bounds__(128)
cross_ket(const double* __restrict__ C, const double* __restrict__ Vs, int ldv,
          const long long* __restrict__ kptr, const int* __restrict__ kbra, const int* __restrict__ krow,
          const int* __restrict__ bk, const int* __restrict__ bg, const int* __restrict__ btl,
          const long long* __restrict__ goff, const unsigned* __restrict__ gent,
          const int* __restrict__ cbase, const int* __restrict__ rlist, const long long* __restrict__ roff,
          const int* __restrict__ wid, double* sigma)
{
    // all of it dynamic (CROSS_KET_SMEM bytes): static __shared__ arrays next to the dynamic tiles were
    // overwritten by them with nvc++ (job 3454038)
    extern __shared__ double smk[];
    double* Vt = smk;                                       // [rs][bra]
    double* Tt = smk + 64 * SBD_FM_LD;                      // [rs][b]
    long long* ro_s = (long long*)(smk + 2 * 64 * SBD_FM_LD);
    int* w_s = (int*)(ro_s + 64);
    int* vr_s = w_s + 64;
    int* rl_s = vr_s + 64;
    int& keff_s = rl_s[64];
    const long long ak = bk[blockIdx.x];
    const int g = bg[blockIdx.x], bt = btl[blockIdx.x];
    const long long b0 = (long long)bt * 64;
    const long long kp = kptr[ak] + 64LL * g;
    const int K = (int)min(64LL, kptr[ak + 1] - kp);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int r4 = lane & 3, q8 = lane >> 2;
    if (tid < 64) {
        if (tid < K) {
            const int a = kbra[kp + tid];
            ro_s[tid] = roff[a];
            w_s[tid] = wid[a];
            vr_s[tid] = krow[kp + tid] * ldv;
        } else {
            ro_s[tid] = 0;
            w_s[tid] = 0;
            vr_s[tid] = 0;  // never read (k >= Keff), but in bounds if a load is speculated
        }
    }
    __syncthreads();
    if (tid == 0) {
        int lo = 0, hi = K;
        while (lo < hi) { const int m = (lo + hi) / 2; if (w_s[m] > b0) lo = m + 1; else hi = m; }
        keff_s = lo;
    }
    __syncthreads();
    const int Keff = keff_s;
    const int c0 = cbase[bt], nch = cbase[bt + 1] - c0;
    if (Keff == 0 || nch == 0) return;
    const long long wk = wid[ak];
    const double* crow = C + roff[ak];
    double acc[4][4][2];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j][0] = acc[i][j][1] = 0.0;
    for (int c = 0; c < nch; c++) {
        for (int i = tid; i < 64 * SBD_FM_LD; i += 128) Tt[i] = 0.0;
        if (tid < 64) rl_s[tid] = rlist[(long long)(c0 + c) * 64 + tid];
        __syncthreads();
        const long long e1 = goff[c0 + c + 1];
        for (long long e = goff[c0 + c] + tid; e < e1; e += 128) {
            const unsigned u = gent[e];
            const long long bp = u >> 13;
            if (bp < wk) {
                const double v = crow[bp];
                Tt[(u & 63) * SBD_FM_LD + ((u >> 6) & 63)] = (u >> 12) & 1 ? -v : v;
            }
        }
        for (int i = tid; i < 64 * 64; i += 128) {
            const int r = i >> 6, k = i & 63;
            double v = 0.0;
            if (k < Keff) v = Vs[vr_s[k] + rl_s[r]];
            Vt[r * SBD_FM_LD + k] = v;
        }
        __syncthreads();
        const double* As = Vt + wm * 32 + q8;
        const double* Bt = Tt + wn * 32 + q8;
        for (int kk = 0; kk < 64; kk += 4) {
            double fa[4], fb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) fa[i] = As[(kk + r4) * SBD_FM_LD + i * 8];
#pragma unroll
            for (int j = 0; j < 4; j++) fb[j] = Bt[(kk + r4) * SBD_FM_LD + j * 8];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) fm_dmma(acc[i][j][0], acc[i][j][1], fa[i], fb[j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int k = wm * 32 + i * 8 + q8;
        if (k >= Keff) continue;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const long long b = b0 + wn * 32 + j * 8 + 2 * r4 + h;
                if (b < w_s[k]) atomicAdd(sigma + ro_s[k] + b, acc[i][j][h]);
            }
    }
}

// the position of determinant (a, b) of the remainder (sorted by alpha rank, then beta rank), or -1
__device__ __forceinline__ long long rem_find(const long long* __restrict__ rptr, const int* __restrict__ rb,
                                              long long a, long long b)
{
    long long lo = rptr[a], hi = rptr[a + 1];
    while (lo < hi) {
        const long long m = (lo + hi) / 2;
        if (rb[m] < b) lo = m + 1; else hi = m;
    }
    return (lo < rptr[a + 1] && rb[lo] == b) ? lo : -1;
}

/**
   The remainder, per element. One block per remainder determinant r = (a, b), r0 <= r < r1: for every connected ket
   k (a's row of H_a with b, b's row of H_b with a, and the products of a's and b's excitation lists,
   whose occupation entries carry the opposite-spin parts), sigma_r += H_rk C_k, and for a staircase
   ket sigma_k += H_kr C_r.
*/
static __global__ void __launch_bounds__(128)
rem_elem(const double* __restrict__ Cst, const double* __restrict__ Crem, double* Sst, double* __restrict__ Srem,
         const int* __restrict__ ra, const int* __restrict__ rb, const long long* __restrict__ rptr,
         const long long* __restrict__ roff, const int* __restrict__ wid,
         const long long* __restrict__ hap, const int* __restrict__ hac, const double* __restrict__ hav,
         const long long* __restrict__ hbp, const int* __restrict__ hbc, const double* __restrict__ hbv,
         const long long* __restrict__ eap, const int* __restrict__ eak, const int* __restrict__ eaq,
         const double* __restrict__ eas, const long long* __restrict__ ebp, const int* __restrict__ ebk,
         const int* __restrict__ ebq, const double* __restrict__ ebs, const double* __restrict__ Vs, int ldv,
         double I0, long long r0, long long r1)
{
    __shared__ double red[4];
    for (long long r = r0 + blockIdx.x; r < r1; r += gridDim.x) {
        const long long a = ra[r], b = rb[r];
        const double cr = Crem[r];
        const long long nha = hap[a + 1] - hap[a], nhb = hbp[b + 1] - hbp[b];
        const long long nea = eap[a + 1] - eap[a], neb = ebp[b + 1] - ebp[b];
        const long long total = nha + nhb + nea * neb;
        double acc = 0.0;
        for (long long t = threadIdx.x; t < total; t += blockDim.x) {
            long long a2, b2;
            double val;
            if (t < nha) {
                const long long e = hap[a] + t;
                a2 = hac[e];
                b2 = b;
                val = hav[e] + (a2 == a ? I0 : 0.0);
            } else if (t < nha + nhb) {
                const long long e = hbp[b] + (t - nha);
                a2 = a;
                b2 = hbc[e];
                val = hbv[e];
            } else {
                const long long q = t - nha - nhb, i = q / neb, j = q - i * neb;
                const long long ei = eap[a] + i, ej = ebp[b] + j;
                a2 = eak[ei];
                b2 = ebk[ej];
                val = eas[ei] * ebs[ej] * Vs[(long long)eaq[ei] * ldv + ebq[ej]];
            }
            if (b2 < wid[a2]) {
                const long long cell = roff[a2] + b2;
                acc += val * Cst[cell];
                if (cr != 0.0) atomicAdd(Sst + cell, val * cr);
            } else {
                const long long j = rem_find(rptr, rb, a2, b2);
                if (j >= 0) acc += val * Crem[j];
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) Srem[r] = red[0] + red[1] + red[2] + red[3];
        __syncthreads();
    }
}

// C (engine order) from sbd's vector: Cst[cell[i]] or Crem[rem[i]] = sign[i] W[i]
static __global__ void gather_in(const double* __restrict__ W, const long long* __restrict__ where,
                                 const signed char* __restrict__ sgn, long long n, double* Cst, double* Crem)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long w = where[i];
        const double v = sgn[i] * W[i];
        if (w >= 0) Cst[w] = v; else Crem[-w - 1] = v;
    }
}

// W[i] += sign[i] sigma (staircase cell or remainder position)
static __global__ void scatter_out(const double* __restrict__ Sst, const double* __restrict__ Srem,
                                   const long long* __restrict__ where, const signed char* __restrict__ sgn,
                                   long long n, double* __restrict__ W)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long w = where[i];
        W[i] += sgn[i] * (w >= 0 ? Sst[w] : Srem[-w - 1]);
    }
}

// the diagonal: I0 + hdiag_a[a] + hdiag_b[b] + sum over beta electrons r of b of Ja[a, r]
static __global__ void diagonal(const int* __restrict__ da, const int* __restrict__ db,
                                const unsigned long long* __restrict__ xb, const double* __restrict__ ha,
                                const double* __restrict__ hb, const double* __restrict__ Ja, int L, double I0,
                                long long n, double* __restrict__ hii)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long a = da[i], b = db[i];
        double d = I0 + ha[a] + hb[b];
        unsigned long long x = xb[b];
        while (x) {
            const int r = __ffsll((long long)x) - 1;
            x &= x - 1;
            d += Ja[a * L + r];
        }
        hii[i] = d;
    }
}

/**
   Tables of the ket-driven cross term's beta side: per 64-wide tile of output strings b, the rs rows
   that reach it (b = E_rs b' for some b' in the list) in 64-row chunks, with gather entries
   rs_local 6 | b_local 6 | sign 1 | b' 19 (the transpose of factored::BuildCrossTables's beta side).
*/
struct OutTables {
    thrust::device_vector<int> cbase, rlist;
    thrust::device_vector<long long> goff;
    thrust::device_vector<unsigned> gent;
    size_t total_chunks = 0;
};

inline void BuildOutTables(OutTables &x, const factored::SpinStrings &sb, int L)
{
    const int nrs = L * L;
    const size_t nb = sb.n, nbt = (nb + 63) / 64;
    std::vector<int> rank(nbt * nrs, -1);
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++) rank[(b / 64) * nrs + sb.epq[e]] = 0;
    std::vector<int> cbase(nbt + 1, 0);
    for (size_t t = 0; t < nbt; t++) {
        int u = 0;
        for (int rs = 0; rs < nrs; rs++)
            if (rank[t * nrs + rs] == 0) rank[t * nrs + rs] = u++;
        cbase[t + 1] = cbase[t] + (u + 63) / 64;
    }
    std::vector<int> rlist(static_cast<size_t>(cbase[nbt]) * 64, nrs);
    for (size_t t = 0; t < nbt; t++)
        for (int rs = 0; rs < nrs; rs++)
            if (rank[t * nrs + rs] >= 0) rlist[static_cast<size_t>(cbase[t]) * 64 + rank[t * nrs + rs]] = rs;
    std::vector<long long> goff(cbase[nbt] + 1, 0);
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++)
            goff[cbase[b / 64] + rank[(b / 64) * nrs + sb.epq[e]] / 64 + 1]++;
    for (int c = 0; c < cbase[nbt]; c++) goff[c + 1] += goff[c];
    std::vector<long long> fill(goff.begin(), goff.end() - 1);
    std::vector<unsigned> gent(goff.back());
    for (size_t b = 0; b < nb; b++)
        for (int64_t e = sb.eptr[b]; e < sb.eptr[b + 1]; e++) {
            const size_t t = b / 64;
            const int u = rank[t * nrs + sb.epq[e]];
            gent[fill[cbase[t] + u / 64]++] = static_cast<unsigned>(u % 64) | (static_cast<unsigned>(b % 64) << 6)
                                              | (static_cast<unsigned>(sb.esg[e] < 0) << 12)
                                              | (static_cast<unsigned>(sb.eket[e]) << 13);
        }
    x.cbase = cbase;
    x.rlist = rlist;
    x.goff = goff;
    x.gent = gent;
    x.total_chunks = cbase[nbt];
}

}  // namespace staircase

class MultGDBStaircase : public MultBase<double> {
public:
    MultGDBStaircase() {}

    template <typename DetsContainer>
    void Init(const size_t bit_length, const size_t norbs, const DetsContainer &dets, const double &I0,
              const oneInt<double> &I1, const twoInt<double> &I2, MPI_Comm h_comm, MPI_Comm b_comm,
              MPI_Comm t_comm, double base = 1.25, double theta = 0.1)
    {
        SBD_NVTX_RANGE_COLOR("MultGDBStaircase::Init", __LINE__);
        auto t_start = std::chrono::high_resolution_clock::now();
        int sh, sb, st, rh, rb, rt;
        MPI_Comm_size(h_comm, &sh);
        MPI_Comm_size(b_comm, &sb);
        MPI_Comm_size(t_comm, &st);
        MPI_Comm_rank(h_comm, &rh);
        MPI_Comm_rank(b_comm, &rb);
        MPI_Comm_rank(t_comm, &rt);
        nproc_ = sh * sb * st;
        me_ = (rh * sb + rb) * st + rt;
        if (nproc_ > 1) {
            // the gather of C and the sums of sigma pass device memory to MPI
#ifdef SBD_NON_CUDA_AWARE_MPI
            throw std::runtime_error("factored_mult (GDB): several ranks need GPU-aware MPI "
                                     "(SBD_NON_CUDA_AWARE_MPI=OFF)");
#elif defined(CRAY_MPICH_VERSION)
            const char* gpu = std::getenv("MPICH_GPU_SUPPORT_ENABLED");  // Cray's MPIX_Query_cuda_support returns 0
            if (!gpu || std::strcmp(gpu, "1") != 0)
                throw std::runtime_error("factored_mult (GDB): several ranks need MPICH_GPU_SUPPORT_ENABLED=1");
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
            if (!MPIX_Query_cuda_support())
                throw std::runtime_error("factored_mult (GDB): several ranks need MPI with CUDA support enabled");
#endif
        }
        if (norbs > 64) throw std::runtime_error("factored_mult (GDB): at most 64 orbitals");
        this->bit_length_ = bit_length;
        this->norbs_ = norbs;
        this->D_size_ = (2 * norbs + bit_length - 1) / bit_length;
        this->h_comm_ = h_comm;
        this->b_comm_ = b_comm;
        this->t_comm_ = t_comm;
        L_ = static_cast<int>(norbs);
        I0_ = I0;
        nloc_ = dets.size();

        // alpha and beta strings of each determinant (alpha orbital i at bit 2i, beta at 2i + 1): this
        // rank's, then the whole list (b_comm's ranks in order)
        std::vector<uint64_t> xla(nloc_), xlb(nloc_);
        for (size_t i = 0; i < nloc_; i++) {
            uint64_t a = 0, b = 0;
            for (int o = 0; o < L_; o++) {
                const size_t ka = 2 * o, kb = 2 * o + 1;
                if ((dets[i][ka / bit_length] >> (ka % bit_length)) & 1) a |= 1ull << o;
                if ((dets[i][kb / bit_length] >> (kb % bit_length)) & 1) b |= 1ull << o;
            }
            xla[i] = a;
            xlb[i] = b;
        }
        {
            const int nl = static_cast<int>(nloc_);
            std::vector<int> cnt(sb), dsp(sb, 0);
            MPI_Allgather(&nl, 1, MPI_INT, cnt.data(), 1, MPI_INT, b_comm);
            for (int r = 1; r < sb; r++) dsp[r] = dsp[r - 1] + cnt[r - 1];
            gcnt_ = cnt;
            gdsp_ = dsp;
            loff_ = static_cast<size_t>(dsp[rb]);
            n_ = static_cast<size_t>(dsp[sb - 1]) + cnt[sb - 1];
        }
        std::vector<uint64_t> xa(n_), xb(n_);
        MPI_Allgatherv(xla.data(), static_cast<int>(nloc_), MPI_UINT64_T, xa.data(), gcnt_.data(), gdsp_.data(),
                       MPI_UINT64_T, b_comm);
        MPI_Allgatherv(xlb.data(), static_cast<int>(nloc_), MPI_UINT64_T, xb.data(), gcnt_.data(), gdsp_.data(),
                       MPI_UINT64_T, b_comm);
        // rank the strings by determinant count, then bit-reversed value within dyadic buckets
        auto rank = [&](const std::vector<uint64_t> &x, std::vector<uint64_t> &strings, std::vector<int> &of,
                        std::vector<int> &edges) {
            std::vector<uint64_t> u(x);
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            std::vector<long long> cnt(u.size(), 0);
            std::vector<int> idx(n_);
            for (size_t i = 0; i < n_; i++) {
                idx[i] = static_cast<int>(std::lower_bound(u.begin(), u.end(), x[i]) - u.begin());
                cnt[idx[i]]++;
            }
            std::vector<int> order(u.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int p, int q) { return cnt[p] > cnt[q]; });
            edges = BucketEdges(u.size(), base);
            auto rev = [&](uint64_t v) {
                uint64_t r = 0;
                for (int o = 0; o < L_; o++)
                    if ((v >> o) & 1) r |= 1ull << (L_ - 1 - o);
                return r;
            };
            for (size_t k = 0; k + 1 < edges.size(); k++)
                std::stable_sort(order.begin() + edges[k], order.begin() + edges[k + 1],
                                 [&](int p, int q) { return rev(u[p]) < rev(u[q]); });
            std::vector<int> pos(u.size());
            for (size_t r = 0; r < order.size(); r++) pos[order[r]] = static_cast<int>(r);
            strings.resize(u.size());
            for (size_t r = 0; r < order.size(); r++) strings[r] = u[order[r]];
            of.resize(n_);
            for (size_t i = 0; i < n_; i++) of[i] = pos[idx[i]];
        };
        std::vector<int> ea, eb;
        rank(xa, sa_.x, da_, ea);
        rank(xb, sb_.x, db_, eb);
        sa_.n = sa_.x.size();
        sb_.n = sb_.x.size();
        na_ = sa_.n;
        nb_ = sb_.n;
        if (nb_ >= (1u << 19)) throw std::runtime_error("factored_mult (GDB): more than 2^19 beta strings");

        // the staircase: buckets at least theta full, closed toward the top ranks
        const size_t nea = ea.size() - 1, neb = eb.size() - 1;
        std::vector<int> bucket_a(na_), bucket_b(nb_);
        for (size_t k = 0; k < nea; k++)
            for (int r = ea[k]; r < ea[k + 1]; r++) bucket_a[r] = static_cast<int>(k);
        for (size_t k = 0; k < neb; k++)
            for (int r = eb[k]; r < eb[k + 1]; r++) bucket_b[r] = static_cast<int>(k);
        std::vector<long long> cnt(nea * neb, 0);
        for (size_t i = 0; i < n_; i++) cnt[bucket_a[da_[i]] * neb + bucket_b[db_[i]]]++;
        std::vector<char> keep(nea * neb, 0);
        for (size_t i = 0; i < nea; i++)
            for (size_t j = 0; j < neb; j++) {
                const double size = static_cast<double>(ea[i + 1] - ea[i]) * (eb[j + 1] - eb[j]);
                keep[i * neb + j] = cnt[i * neb + j] >= theta * size;
            }
        for (long long i = static_cast<long long>(nea) - 1; i >= 0; i--)
            for (long long j = static_cast<long long>(neb) - 1; j >= 0; j--) {
                char k = keep[i * neb + j];
                if (i + 1 < static_cast<long long>(nea)) k |= keep[(i + 1) * neb + j];
                if (j + 1 < static_cast<long long>(neb)) k |= keep[i * neb + j + 1];
                keep[i * neb + j] = k;
            }
        std::vector<int> wid(na_, 0);
        for (size_t i = 0; i < nea; i++) {
            int w = 0;
            for (size_t j = 0; j < neb; j++)
                if (keep[i * neb + j]) w = eb[j + 1];
            for (int r = ea[i]; r < ea[i + 1]; r++) wid[r] = w;
        }
        std::vector<long long> roff(na_ + 1, 0);
        for (size_t a = 0; a < na_; a++) roff[a + 1] = roff[a] + wid[a];
        ncell_ = static_cast<size_t>(roff[na_]);

        // where each determinant lives: a staircase cell, or the remainder (sorted by alpha, beta rank)
        std::vector<long long> where(n_);
        std::vector<std::pair<int, int>> rem;
        for (size_t i = 0; i < n_; i++)
            if (db_[i] < wid[da_[i]]) where[i] = roff[da_[i]] + db_[i];
            else rem.emplace_back(da_[i], db_[i]);
        std::sort(rem.begin(), rem.end());
        nrem_ = rem.size();
        std::vector<int> rema(nrem_), remb(nrem_);
        std::vector<long long> rptr(na_ + 1, 0);
        for (size_t r = 0; r < nrem_; r++) {
            rema[r] = rem[r].first;
            remb[r] = rem[r].second;
            rptr[rem[r].first + 1]++;
        }
        for (size_t a = 0; a < na_; a++) rptr[a + 1] += rptr[a];
        for (size_t i = 0; i < n_; i++)
            if (db_[i] >= wid[da_[i]]) {
                const auto it = std::lower_bound(rem.begin(), rem.end(), std::make_pair(da_[i], db_[i]));
                where[i] = -1 - static_cast<long long>(it - rem.begin());
            }
        std::vector<signed char> sgn(n_);
        for (size_t i = 0; i < n_; i++) {
            int s = 0;
            uint64_t A = xa[i];
            while (A) {
                const int o = __builtin_ctzll(A);
                A &= A - 1;
                s += __builtin_popcountll(xb[i] & ((1ull << o) - 1));
            }
            sgn[i] = (s & 1) ? -1 : 1;
        }

        // string factors: same-spin Hamiltonians, excitation lists, cross-term tables
        std::vector<double> h1(static_cast<size_t>(L_) * L_);
        std::vector<double> eri(static_cast<size_t>(L_) * L_ * L_ * L_);
        for (int p = 0; p < L_; p++)
            for (int q = 0; q < L_; q++) h1[p * L_ + q] = I1.Value(2 * p, 2 * q);
        for (int p = 0; p < L_; p++)
            for (int q = 0; q < L_; q++)
                for (int r = 0; r < L_; r++)
                    for (int s = 0; s < L_; s++)
                        eri[((static_cast<size_t>(p) * L_ + q) * L_ + r) * L_ + s] = I2.Value(2 * p, 2 * q, 2 * r, 2 * s);
        factored::BuildSpinStrings(sa_, L_, h1, eri);
        factored::BuildSpinStrings(sb_, L_, h1, eri);
        factored::BuildSpinTiles(ta_, sa_, I0);   // the core energy once, on the alpha side
        factored::BuildSpinTiles(tb_, sb_, 0.0);
        int dev = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        size_t smem_limit = static_cast<size_t>(optin);
#if SBD_FM_SMEM_LIMIT > 0
        smem_limit = std::min<size_t>(smem_limit, SBD_FM_SMEM_LIMIT);  // to test the segments on a larger GPU
#endif
        factored::BuildCrossTables(cross_, sa_, sb_, L_, eri, smem_limit);  // V and the bra-driven beta tables
        // the longest ket list a cross_bra block holds in shared memory; longer lists run as segments
        auto smem2 = [&](int k) { return 16 + factored::fm_cross_smem(k, cross_.max_chunks) + static_cast<size_t>(k) * sizeof(int); };
        int kseg = 0;
        while (smem2(kseg + 4) <= smem_limit) kseg += 4;
        if (kseg == 0) throw std::runtime_error("factored_mult (GDB): no excitation list fits in shared memory");
        staircase::BuildOutTables(out_, sb_, L_);                // the ket-driven beta tables

        // the cross term's alpha pairs (bra a = E_pq ket a'): bra-driven if w(a') <= w(a), ket-driven
        // otherwise; lists sorted by width, rows with no staircase cells left out
        const int nrs = L_ * L_;
        auto vrow = [&](int64_t e) { return sa_.esg[e] > 0 ? sa_.epq[e] : nrs + sa_.epq[e]; };
        std::vector<std::vector<std::pair<int, int>>> bra_k(na_), ket_b(na_);  // (ket or bra, V row)
        for (size_t a = 0; a < na_; a++) {
            if (wid[a] == 0) continue;
            for (int64_t e = sa_.eptr[a]; e < sa_.eptr[a + 1]; e++) {
                const int ak = sa_.eket[e];
                if (wid[ak] == 0) continue;
                if (wid[ak] <= wid[a]) bra_k[a].emplace_back(ak, vrow(e));
                else ket_b[ak].emplace_back(static_cast<int>(a), vrow(e));
            }
        }
        auto by_width = [&](const std::pair<int, int> &p, const std::pair<int, int> &q) { return wid[p.first] > wid[q.first]; };
        std::vector<long long> kp2(1, 0), kp1(na_ + 1, 0);
        std::vector<int> kk2, kr2, s2a, kb1, kr1, b2s, b2t, b1k, b1g, b1t;
        int kmax = 0;
        size_t nsplit = 0;
        double flop2 = 0.0, flop1 = 0.0;
        std::vector<double> cost2, cost1;  // estimated cost of each bra-driven and ket-driven block
        std::vector<int> cb(cross_.nbt + 1), cbo(cross_.nbt + 1);
        {
            thrust::host_vector<int> h1v(cross_.cbase), h2v(out_.cbase);
            for (size_t t = 0; t <= cross_.nbt; t++) { cb[t] = h1v[t]; cbo[t] = h2v[t]; }
        }
        for (size_t a = 0; a < na_; a++) {
            auto &l = bra_k[a];
            std::stable_sort(l.begin(), l.end(), by_width);
            const size_t nl = l.size(), nseg = (nl + kseg - 1) / kseg;  // near-equal segments of at most kseg
            nsplit += nseg > 1;
            for (size_t j = 0; j < nseg; j++) {
                const size_t e0 = nl * j / nseg, e1 = nl * (j + 1) / nseg;
                const int K = static_cast<int>((e1 - e0 + 3) / 4 * 4);
                for (size_t e = e0; e < e1; e++) { kk2.push_back(l[e].first); kr2.push_back(l[e].second); }
                for (size_t k = e1 - e0; k < static_cast<size_t>(K); k++) { kk2.push_back(-1); kr2.push_back(2 * nrs); }
                const int sg = static_cast<int>(s2a.size());
                s2a.push_back(static_cast<int>(a));
                kp2.push_back(kp2.back() + K);
                kmax = std::max(kmax, K);
                const int tiles = (wid[l[e0].first] + 63) / 64;
                for (int t = 0; t < tiles; t++) {
                    b2s.push_back(sg);
                    b2t.push_back(t);
                    size_t keff = e0;
                    while (keff < e1 && wid[l[keff].first] > 64 * t) keff++;
                    const double f = 2.0 * 64 * 64 * ((keff - e0 + 3) / 4 * 4) * (cb[t + 1] - cb[t]);
                    flop2 += f;
                    cost2.push_back(f);
                }
            }
            auto &m = ket_b[a];
            std::stable_sort(m.begin(), m.end(), by_width);
            for (auto &p : m) { kb1.push_back(p.first); kr1.push_back(p.second); }
            kp1[a + 1] = kp1[a] + static_cast<long long>(m.size());
            for (size_t g = 0; g * 64 < m.size(); g++) {
                const int tiles = (wid[m[g * 64].first] + 63) / 64;
                for (int t = 0; t < tiles; t++) {
                    b1k.push_back(static_cast<int>(a));
                    b1g.push_back(static_cast<int>(g));
                    b1t.push_back(t);
                    const double f = 2.0 * 64 * 64 * 64 * (cbo[t + 1] - cbo[t]);
                    flop1 += f;
                    cost1.push_back(f);
                }
            }
        }
        // this rank's shares: same-spin blocks by output cells times H entries per row, the cross-term
        // blocks by estimated flops, the remainder by connected kets
        active_rows_ = static_cast<int>(std::lower_bound(wid.begin(), wid.end(), 0, std::greater<int>()) - wid.begin());
        if (active_rows_ > 0) {
            // a block's cost, as spmm_stair runs it: 64 per H tile it loads (a 64 x 64 piece of C) plus the
            // tiles' entries, over the tiles of its tile row in columns below cw1
            const long long nal = static_cast<long long>(na_);
            auto tile_cost = [](const factored::SpinTiles &t) {
                const thrust::host_vector<int> trp = t.trp, tj = t.tj, rp = t.rp;
                std::vector<double> pre(tj.size() + 1, 0.0);  // prefix over tiles of 64 + entries
                for (size_t u = 0; u < tj.size(); u++)
                    pre[u + 1] = pre[u] + 64.0 + (rp[u * (SBD_FM_TR + 1) + SBD_FM_TR] - rp[u * (SBD_FM_TR + 1)]);
                return [trp = std::vector<int>(trp.begin(), trp.end()), tj = std::vector<int>(tj.begin(), tj.end()),
                        pre](int I, long long cw1) {
                    const auto e = std::lower_bound(tj.begin() + trp[I], tj.begin() + trp[I + 1], cw1,
                                                    [](int j, long long c) { return 64LL * j < c; });
                    return pre[e - tj.begin()] - pre[trp[I]];
                };
            };
            const int w0 = wid[0];
            // MODE 0: (alpha tile row I, 64-wide beta chunk ch); its H_a columns are the rows wider than ch*64
            ny0_ = (w0 + 63) / 64;
            const int nI = (active_rows_ + SBD_FM_TR - 1) / SBD_FM_TR;
            std::vector<double> c0(static_cast<size_t>(nI) * ny0_, 0.0);
            {
                const auto cost = tile_cost(ta_);
                for (int ch = 0; ch < ny0_; ch++) {
                    const long long y0 = 64LL * ch;
                    const long long cw1 = std::lower_bound(wid.begin(), wid.begin() + nal, y0,
                                                           std::greater<long long>()) - wid.begin();
                    for (int I = 0; I < nI && wid[static_cast<size_t>(I) * SBD_FM_TR] > y0; I++)
                        c0[static_cast<size_t>(I) * ny0_ + ch] = cost(I, cw1);
                }
            }
            const auto s0 = Share(c0);
            ss0b_ = s0.first; ss0e_ = s0.second;
            // MODE 1: (64-row alpha chunk ch, beta tile row I); its H_b columns are those below the chunk's width
            ny1_ = (w0 + SBD_FM_TR - 1) / SBD_FM_TR;
            const int nc = (active_rows_ + 63) / 64;
            std::vector<double> c1(static_cast<size_t>(nc) * ny1_, 0.0);
            {
                const auto cost = tile_cost(tb_);
                for (int ch = 0; ch < nc; ch++) {
                    const long long cw1 = wid[64LL * ch];
                    for (int I = 0; 1LL * I * SBD_FM_TR < cw1; I++) c1[static_cast<size_t>(ch) * ny1_ + I] = cost(I, cw1);
                }
            }
            const auto s1 = Share(c1);
            ss1b_ = s1.first; ss1e_ = s1.second;
        }
        {
            const auto s2 = Share(cost2), s1 = Share(cost1);
            c2b_ = s2.first; c2e_ = s2.second;
            c1b_ = s1.first; c1e_ = s1.second;
            std::vector<double> cr(nrem_);
            for (size_t r = 0; r < nrem_; r++) {
                const int a = rema[r], b = remb[r];
                cr[r] = static_cast<double>(sa_.hptr[a + 1] - sa_.hptr[a]) + (sb_.hptr[b + 1] - sb_.hptr[b])
                        + static_cast<double>(sa_.eptr[a + 1] - sa_.eptr[a]) * (sb_.eptr[b + 1] - sb_.eptr[b]);
            }
            const auto sr = Share(cr);
            rb_ = sr.first; re_ = sr.second;
        }
        k2ptr_ = kp2; k2ket_ = kk2; k2row_ = kr2; s2a_ = s2a; b2s_ = b2s; b2t_ = b2t;
        k1ptr_ = kp1; k1bra_ = kb1; k1row_ = kr1; b1k_ = b1k; b1g_ = b1g; b1t_ = b1t;
        smem2_ = smem2(kmax);
        if (me_ == 0 && nsplit > 0)
            std::printf(" factored_mult (GDB): %zu excitation lists longer than %d entries (the shared memory of a "
                        "block) run as %zu segments in all\n", nsplit, kseg, s2a.size());
        cudaFuncSetAttribute(staircase::cross_bra, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem2_));
        cudaFuncSetAttribute(staircase::cross_ket, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(CROSS_KET_SMEM));
        const int ss_smem = 2 * 64 * SBD_FM_LDC * static_cast<int>(sizeof(double));
        cudaFuncSetAttribute(staircase::spmm_stair<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, ss_smem);
        cudaFuncSetAttribute(staircase::spmm_stair<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, ss_smem);

        // device copies
        wid_ = wid;
        w0_ = na_ > 0 ? wid[0] : 0;
        roff_ = roff;
        where_ = where;
        sgn_ = sgn;
        rema_ = rema;
        remb_ = remb;
        rptr_ = rptr;
        da_dev_ = da_;
        db_dev_ = db_;
        auto up = [](auto &dst, const auto &src) { dst = src; };
        up(hap_, sa_.hptr); up(hac_, sa_.hcol); up(hav_, sa_.hval);
        up(hbp_, sb_.hptr); up(hbc_, sb_.hcol); up(hbv_, sb_.hval);
        up(eap_, sa_.eptr); up(eak_, sa_.eket); up(eaq_, sa_.epq); up(eas_, sa_.esg);
        up(ebp_, sb_.eptr); up(ebk_, sb_.eket); up(ebq_, sb_.epq); up(ebs_, sb_.esg);
        std::vector<unsigned long long> xbs(sb_.x.begin(), sb_.x.end());
        xb_ = xbs;
        hda_ = sa_.hdiag;
        hdb_ = sb_.hdiag;
        std::vector<double> Ja(na_ * L_, 0.0);
        for (size_t a = 0; a < na_; a++)
            for (int p = 0; p < L_; p++)
                if ((sa_.x[a] >> p) & 1)
                    for (int r = 0; r < L_; r++)
                        Ja[a * L_ + r] += eri[((static_cast<size_t>(p) * L_ + p) * L_ + r) * L_ + r];
        Ja_ = Ja;
        Cst_.assign(ncell_, 0.0);
        Sst_.resize(ncell_);
        Crem_.resize(nrem_);
        Srem_.resize(nrem_);
        if (sb > 1) Wall_.resize(n_);

        size_t in_list = n_ - nrem_;
        auto t_end = std::chrono::high_resolution_clock::now();
        if (me_ == 0) std::printf(" factored_mult (GDB): %zu determinants, %zu alpha and %zu beta strings; staircase %zu cells "
                    "(%.2fx the list) holding %.1f%% of it, remainder %zu; cross-term GEMM %.3e flop bra-driven "
                    "(%zu blocks) + %.3e ket-driven (%zu blocks); %d rank(s); setup %.1f s\n",
                    n_, na_, nb_, ncell_, static_cast<double>(ncell_) / n_, 100.0 * in_list / n_, nrem_, flop2,
                    b2s.size(), flop1, b1k.size(), nproc_,
                    1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());
    }

    void makeQChamDiagTerms(thrust::device_vector<double> &hii)
    {
        hii.resize(nloc_);
        staircase::diagonal<<<1024, 256>>>(ptr(da_dev_) + loff_, ptr(db_dev_) + loff_, ptr(xb_), ptr(hda_), ptr(hdb_),
                                           ptr(Ja_), L_, I0_, static_cast<long long>(nloc_), ptr(hii));
        check("diagonal");
    }

    /// Wb += H Wk (sbd's order of this rank's determinants).
    void run(const thrust::device_vector<double> &hii, const thrust::device_vector<double> &Wk,
             thrust::device_vector<double> &Wb) override
    {
        (void)hii;
        SBD_NVTX_RANGE_COLOR("MultGDBStaircase::run", __LINE__);
        if (Wb.size() != nloc_) Wb.resize(nloc_);
        const long long n = static_cast<long long>(n_), na = static_cast<long long>(na_);
        const double* W = ptr(Wk);
        if (!Wall_.empty()) {  // the whole of C (b_comm's parts in order)
            const auto t0 = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();
            MPI_Allgatherv(W, static_cast<int>(nloc_), MPI_DOUBLE, ptr(Wall_), gcnt_.data(), gdsp_.data(), MPI_DOUBLE,
                           this->b_comm_);
            W = ptr(Wall_);
            tsum_[4] += Seconds(t0);
        }
        staircase::gather_in<<<1024, 256>>>(W, ptr(where_), ptr(sgn_), n, ptr(Cst_), ptr(Crem_));
        cudaMemsetAsync(ptr(Sst_), 0, ncell_ * sizeof(double), 0);
        if (nrem_ > 0) cudaMemsetAsync(ptr(Srem_), 0, nrem_ * sizeof(double), 0);
        check("gather_in");
        timed(0, [&] {
            const int ss_smem = 2 * 64 * SBD_FM_LDC * sizeof(double);
            if (ss0e_ > ss0b_)  // this rank's blocks of each grid
                staircase::spmm_stair<0><<<static_cast<unsigned>(ss0e_ - ss0b_), 2 * SBD_FM_TR, ss_smem>>>(
                    ptr(Cst_), ptr(Sst_), ptr(ta_.trp), ptr(ta_.tj), ptr(ta_.rp), ptr(ta_.ec), ptr(ta_.ev),
                    ptr(roff_), ptr(wid_), na, static_cast<long long>(ss0b_), ny0_);
            if (ss1e_ > ss1b_)
                staircase::spmm_stair<1><<<static_cast<unsigned>(ss1e_ - ss1b_), 2 * SBD_FM_TR, ss_smem>>>(
                    ptr(Cst_), ptr(Sst_), ptr(tb_.trp), ptr(tb_.tj), ptr(tb_.rp), ptr(tb_.ec), ptr(tb_.ev),
                    ptr(roff_), ptr(wid_), na, static_cast<long long>(ss1b_), ny1_);
        });
        timed(1, [&] {
            for (size_t b0 = c2b_; b0 < c2e_; b0 += 1u << 30) {
                const unsigned nbk = static_cast<unsigned>(std::min<size_t>(1u << 30, c2e_ - b0));
                staircase::cross_bra<<<nbk, 128, smem2_>>>(
                    ptr(Cst_), ptr(cross_.Vs), cross_.ldv, ptr(k2ptr_), ptr(k2ket_), ptr(k2row_), ptr(s2a_),
                    ptr(b2s_) + b0, ptr(b2t_) + b0, ptr(cross_.goff), ptr(cross_.gent), ptr(cross_.cbase), ptr(cross_.rlist),
                    ptr(roff_), ptr(wid_), ptr(Sst_));
            }
        });
        timed(2, [&] {
            for (size_t b0 = c1b_; b0 < c1e_; b0 += 1u << 30) {
                const unsigned nbk = static_cast<unsigned>(std::min<size_t>(1u << 30, c1e_ - b0));
                staircase::cross_ket<<<nbk, 128, CROSS_KET_SMEM>>>(
                    ptr(Cst_), ptr(cross_.Vs), cross_.ldv, ptr(k1ptr_), ptr(k1bra_), ptr(k1row_), ptr(b1k_) + b0,
                    ptr(b1g_) + b0, ptr(b1t_) + b0, ptr(out_.goff), ptr(out_.gent), ptr(out_.cbase),
                    ptr(out_.rlist), ptr(roff_), ptr(wid_), ptr(Sst_));
            }
        });
        timed(3, [&] {
            if (re_ > rb_)
                staircase::rem_elem<<<static_cast<unsigned>(std::min<size_t>(re_ - rb_, 1u << 20)), 128>>>(
                    ptr(Cst_), ptr(Crem_), ptr(Sst_), ptr(Srem_), ptr(rema_), ptr(remb_), ptr(rptr_), ptr(roff_),
                    ptr(wid_), ptr(hap_), ptr(hac_), ptr(hav_), ptr(hbp_), ptr(hbc_), ptr(hbv_), ptr(eap_), ptr(eak_),
                    ptr(eaq_), ptr(eas_), ptr(ebp_), ptr(ebk_), ptr(ebq_), ptr(ebs_), ptr(cross_.Vs), cross_.ldv, I0_,
                    static_cast<long long>(rb_), static_cast<long long>(re_));
        });
        if (nproc_ > 1) {  // the sum of the ranks' parts
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (MPI_Comm c : {this->b_comm_, this->t_comm_, this->h_comm_}) {
                int sz;
                MPI_Comm_size(c, &sz);
                if (sz == 1) continue;
                SumInPlace(ptr(Sst_), ncell_, c);
                SumInPlace(ptr(Srem_), nrem_, c);
            }
            tsum_[4] += Seconds(t0);
        }
        staircase::scatter_out<<<1024, 256>>>(ptr(Sst_), ptr(Srem_), ptr(where_) + loff_, ptr(sgn_) + loff_,
                                              static_cast<long long>(nloc_), ptr(Wb));
        check("run");
        nrun_++;
    }

    /// Times of the parts (same-spin, cross bra-driven, cross ket-driven, remainder, communication) over the
    /// runs so far, on rank 0 and the slowest rank of each part. Collective: every rank calls it.
    void PrintTimes() const
    {
        if (nrun_ == 0) return;
        double tmax[5];
        std::copy(tsum_, tsum_ + 5, tmax);
        for (MPI_Comm c : {this->b_comm_, this->t_comm_, this->h_comm_})
            MPI_Allreduce(MPI_IN_PLACE, tmax, 5, MPI_DOUBLE, MPI_MAX, c);
        if (me_ != 0) return;
        std::printf(" factored_mult (GDB) per multiplication over %d: same-spin %.4f s, cross bra-driven %.4f s, "
                    "cross ket-driven %.4f s, remainder %.4f s, communication %.4f s\n", nrun_, tsum_[0] / nrun_,
                    tsum_[1] / nrun_, tsum_[2] / nrun_, tsum_[3] / nrun_, tsum_[4] / nrun_);
        if (nproc_ > 1)
            std::printf(" factored_mult (GDB) slowest rank of each part: same-spin %.4f s, cross bra-driven %.4f s, "
                        "cross ket-driven %.4f s, remainder %.4f s, communication %.4f s\n", tmax[0] / nrun_,
                        tmax[1] / nrun_, tmax[2] / nrun_, tmax[3] / nrun_, tmax[4] / nrun_);
    }

private:
    // this rank's contiguous share [first, second) of items with the given costs, cut at equal cost
    std::pair<size_t, size_t> Share(const std::vector<double> &cost) const
    {
        double total = 0.0;
        for (double c : cost) total += c;
        auto cut = [&](int k) {
            if (k <= 0) return static_cast<size_t>(0);
            if (k >= nproc_) return cost.size();
            const double target = total * k / nproc_;
            double run = 0.0;
            size_t i = 0;
            while (i < cost.size() && run + 0.5 * cost[i] < target) run += cost[i++];
            return i;
        };
        return {cut(me_), cut(me_ + 1)};
    }

    // GPU-aware MPI_Allreduce in place, in pieces of at most 2^30 elements
    static void SumInPlace(double* x, size_t n, MPI_Comm c)
    {
        cudaDeviceSynchronize();
        for (size_t i = 0; i < n; i += 1u << 30)
            MPI_Allreduce(MPI_IN_PLACE, x + i, static_cast<int>(std::min<size_t>(1u << 30, n - i)), MPI_DOUBLE, MPI_SUM, c);
    }

    static double Seconds(std::chrono::high_resolution_clock::time_point t0)
    {
        return 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    static std::vector<int> BucketEdges(size_t m, double base)
    {
        std::vector<int> e = {0};
        double x = 1.0;
        while (static_cast<size_t>(e.back()) < m) {
            e.push_back(std::max(e.back() + 1, static_cast<int>(std::lround(x))));
            x *= base;
        }
        std::vector<int> out;
        for (int v : e)
            if (static_cast<size_t>(v) < m) out.push_back(v);
        out.push_back(static_cast<int>(m));
        return out;
    }

    // runs part k of a multiplication, adding its time to tsum_[k] when timing_; checks it either way
    template <typename F>
    void timed(int k, F f)
    {
        static const char* names[4] = {"same-spin", "cross bra-driven", "cross ket-driven", "remainder"};
        cudaEvent_t a = nullptr, b = nullptr;
        if (timing_) {
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaEventRecord(a, 0);
        }
        f();
        check(names[k]);
        if (timing_) {
            cudaEventRecord(b, 0);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            tsum_[k] += 1e-3 * ms;
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
    }

    template <typename T>
    static T* ptr(thrust::device_vector<T> &v) { return thrust::raw_pointer_cast(v.data()); }
    template <typename T>
    static const T* ptr(const thrust::device_vector<T> &v) { return thrust::raw_pointer_cast(v.data()); }
    static void check(const char* where)
    {
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("factored_mult (GDB): ") + where + ": " + cudaGetErrorString(err));
    }

public:
    bool timing_ = true;

private:
    int L_ = 0;
    double I0_ = 0.0;
    size_t n_ = 0, na_ = 0, nb_ = 0, ncell_ = 0, nrem_ = 0;
    size_t nloc_ = 0, loff_ = 0;                // this rank's determinants: [loff_, loff_ + nloc_) of the list
    std::vector<int> gcnt_, gdsp_;              // b_comm's counts and offsets
    int nproc_ = 1, me_ = 0;                    // ranks over h_comm x b_comm x t_comm, and this one
    size_t ss0b_ = 0, ss0e_ = 0, ss1b_ = 0, ss1e_ = 0;  // this rank's shares: same-spin blocks of each grid,
    int ny0_ = 1, ny1_ = 1;                     // (their grids' second dimensions)
    size_t c2b_ = 0, c2e_ = 0, c1b_ = 0, c1e_ = 0;  // bra- and ket-driven cross blocks,
    size_t rb_ = 0, re_ = 0;                    // remainder determinants
    thrust::device_vector<double> Wall_;        // the whole of C, with b_comm over one rank
    factored::SpinStrings sa_, sb_;
    factored::SpinTiles ta_, tb_;
    factored::CrossTables cross_;
    staircase::OutTables out_;
    std::vector<int> da_, db_;
    thrust::device_vector<int> wid_, rema_, remb_, da_dev_, db_dev_;
    thrust::device_vector<long long> roff_, where_, rptr_;
    thrust::device_vector<signed char> sgn_;
    thrust::device_vector<long long> k2ptr_, k1ptr_;
    thrust::device_vector<int> k2ket_, k2row_, s2a_, b2s_, b2t_, k1bra_, k1row_, b1k_, b1g_, b1t_;
    size_t smem2_ = 0;
    int w0_ = 0, active_rows_ = 0;  // the widest row's width; rows with any staircase cells
    thrust::device_vector<long long> hap_, hbp_, eap_, ebp_;
    thrust::device_vector<int> hac_, hbc_, eak_, eaq_, ebk_, ebq_;
    thrust::device_vector<double> hav_, hbv_, eas_, ebs_;
    thrust::device_vector<unsigned long long> xb_;
    thrust::device_vector<double> hda_, hdb_, Ja_;
    thrust::device_vector<double> Cst_, Sst_, Crem_, Srem_;
    double tsum_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int nrun_ = 0;
};

}  // namespace sbd

#endif
