/**
@file bit_manipulation.h
@brief mathematical tools for bit manipulations
*/

#ifndef SBD_FRAMEWORK_SORT_ARRAY_H
#define SBD_FRAMEWORK_SORT_ARRAY_H

#include "sbd/framework/type_def.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <limits>

namespace sbd {

  namespace detail_mpi_sort_array {
    // ---- small MPI helpers (uint64) ----

    inline std::size_t sbd_mul_div_floor(std::size_t a, std::size_t b, std::size_t den) {
#ifdef SBD_TRADMODE
      long double x = (long double)a * (long double)b / (long double)den;
      
      if (x <= 0.0L) return 0;
      long double maxv = (long double)std::numeric_limits<std::size_t>::max();
      if (x >= maxv) return std::numeric_limits<std::size_t>::max();
      
      return (std::size_t)x; // floor
#else
      // Clang/GCC: exactly for clang type
      return (std::size_t)(((__int128)a * (__int128)b) / (__int128)den);
#endif
    }    
    
    inline uint64_t u64_sum_allreduce(uint64_t x, MPI_Comm comm) {
      uint64_t y = 0;
      MPI_Allreduce(&x, &y, 1, MPI_UINT64_T, MPI_SUM, comm);
      return y;
    }
    
    inline uint64_t u64_exscan(uint64_t x, MPI_Comm comm, int world_rank) {
      uint64_t y = 0;
      MPI_Exscan(&x, &y, 1, MPI_UINT64_T, MPI_SUM, comm);
      return (world_rank == 0) ? 0ULL : y;
    }
    
    // ---- record transported during distributed sort ----
    
    template <typename RealT>
    struct Record {
      RealT    v;
      int32_t  owner;  // original MPI rank
      uint64_t idx;    // original local index
    };
    
    // Comparison: descending by value, then ascending by (owner, idx)
    template <typename RealT>
    inline bool record_less_desc(const Record<RealT>& a, const Record<RealT>& b) {
      if (a.v > b.v) return true;
      if (a.v < b.v) return false;
      if (a.owner < b.owner) return true;
      if (a.owner > b.owner) return false;
      return a.idx < b.idx;
    }
    
    // Bucket id from splitters (descending order).
    // splitters size = P-1, descending.
    // bucket = number of splitters with value >= v  (0..P-1)
    template <typename RealT>
    int bucket_of(const RealT& v, const std::vector<RealT>& splitters) {
      int lo = 0, hi = (int)splitters.size(); // hi = P-1
      while (lo < hi) {
	int mid = (lo + hi) / 2;
	if (splitters[mid] >= v) {
	  lo = mid + 1;
	} else {
	  hi = mid;
	}
      }
      return lo; // 0..P-1
    }
    
    // Create MPI datatype for Record<RealT> using GetMpiType<RealT>::MpiT
    template <typename RealT>
    MPI_Datatype mpi_record_type() {
      MPI_Datatype t;
      
      MPI_Datatype types[3] = {
	GetMpiType<RealT>::MpiT,
	MPI_INT32_T,
	MPI_UINT64_T
      };
      int      bl[3] = { 1, 1, 1 };
      MPI_Aint disp[3];
      
      Record<RealT> dummy{};
      MPI_Aint base = 0;
      MPI_Get_address(&dummy, &base);
      MPI_Get_address(&dummy.v,     &disp[0]);
      MPI_Get_address(&dummy.owner, &disp[1]);
      MPI_Get_address(&dummy.idx,   &disp[2]);
      for (int i = 0; i < 3; ++i) disp[i] -= base;
      
      MPI_Type_create_struct(3, bl, disp, types, &t);
      MPI_Type_commit(&t);
      return t;
    }
    
    // Back message: send global rank back to original owner
    struct Back {
      int32_t  owner;
      uint64_t idx;
      uint64_t grank; // 0-based global rank (0 is largest)
    };
    
    inline MPI_Datatype mpi_back_type() {
      MPI_Datatype t;
      MPI_Datatype types[3] = { MPI_INT32_T, MPI_UINT64_T, MPI_UINT64_T };
      int          bl[3]    = { 1, 1, 1 };
      MPI_Aint     disp[3];
      
      Back dummy{};
      MPI_Aint base = 0;
      MPI_Get_address(&dummy, &base);
      MPI_Get_address(&dummy.owner, &disp[0]);
      MPI_Get_address(&dummy.idx,   &disp[1]);
      MPI_Get_address(&dummy.grank, &disp[2]);
      for (int i = 0; i < 3; ++i) disp[i] -= base;
      
      MPI_Type_create_struct(3, bl, disp, types, &t);
      MPI_Type_commit(&t);
      return t;
    }
    
  } // namespace detail_mpi_sort_array
  
  // -----------------------------------------------------------------------------
  // mpi_sort_array
  //
  // - Does NOT gather all w to one rank.
  // - Returns ranking[i] = global 0-based rank in descending order (0 = largest).
  // - Ties are broken deterministically by (owner_rank, owner_local_index).
  // -----------------------------------------------------------------------------
  template <typename RealT>
  void mpi_find_ranking(const std::vector<RealT> & w,
			std::vector<size_t> & ranking,
			MPI_Comm comm) {
#ifdef SBD_TRADMODE
    static_assert(std::is_floating_point<RealT>::value,
                  "mpi_sort_array: RealT must be a real floating-point type (float/double).");
#else
    static_assert(std::is_floating_point_v<RealT>,
                  "mpi_sort_array: RealT must be a real floating-point type (float/double).");
#endif
    
    using namespace detail_mpi_sort_array;
    
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);
    
    ranking.assign(w.size(), size_t(0));
    
    // Global N
    uint64_t n_local = (uint64_t)w.size();
    uint64_t n_total = u64_sum_allreduce(n_local, comm);
    if (n_total == 0) return;
    
    // Build local records (value + original position)
    std::vector<Record<RealT>> local;
    local.reserve(w.size());
    for (uint64_t i = 0; i < (uint64_t)w.size(); ++i) {
      local.push_back(Record<RealT>{ w[(size_t)i], (int32_t)world_rank, i });
    }
    
    // Local sort descending
    std::sort(local.begin(), local.end(),
              [](const auto& a, const auto& b){ return detail_mpi_sort_array::record_less_desc<RealT>(a, b); });
    
    // ---- Sample sort: choose splitters ----
    const int P = world_size;
    const int oversample = 4;     // modest oversampling; change if desired
    const int s = oversample * P; // samples per rank (approx)
    
    // Take s samples from local sorted data
    std::vector<RealT> samples;
    if (!local.empty()) {
      samples.reserve((size_t)s);
      for (int k = 1; k <= s; ++k) {
	// evenly spaced (avoid endpoints)
	// uint64_t pos = (uint64_t)(((__int128)k * local.size()) / (s + 1));

	auto pos = sbd_mul_div_floor(k, local.size(), s+1);
	if (pos >= (uint64_t)local.size()) pos = (uint64_t)local.size() - 1;
	samples.push_back(local[(size_t)pos].v);
      }
    }
    
    // Allgather sample sizes
    int scount = (int)samples.size();
    std::vector<int> rcounts(P, 0), rdispls(P, 0);
    MPI_Allgather(&scount, 1, MPI_INT, rcounts.data(), 1, MPI_INT, comm);
    
    int total_samples = 0;
    for (int i = 0; i < P; ++i) {
      rdispls[i] = total_samples;
      total_samples += rcounts[i];
    }
    
    std::vector<RealT> all_samples((size_t)total_samples);
    MPI_Allgatherv(samples.data(), scount, GetMpiType<RealT>::MpiT,
                   all_samples.data(), rcounts.data(), rdispls.data(), GetMpiType<RealT>::MpiT,
                   comm);
    
    // Determine splitters (P-1) from globally sorted samples
    std::vector<RealT> splitters;
    splitters.reserve((size_t)std::max(0, P - 1));
    if (P > 1 && !all_samples.empty()) {
      std::sort(all_samples.begin(), all_samples.end(), std::greater<RealT>()); // descending
      for (int i = 1; i < P; ++i) {
	// int idx = (int)(((__int128)i * all_samples.size()) / P);
	const std::size_t idx_u = sbd_mul_div_floor((std::size_t)i,
						    (std::size_t)all_samples.size(),
						    (std::size_t)P);
	const std::size_t idx_clamped = (idx_u < all_samples.size()) ? idx_u : (all_samples.size() ? all_samples.size() - 1 : 0);
	int idx = (int)idx_clamped;
	if (idx < 0) idx = 0;
	if (idx >= (int)all_samples.size()) idx = (int)all_samples.size() - 1;
	splitters.push_back(all_samples[(size_t)idx]);
      }
    } else if (P > 1) {
      // all_samples empty => all ranks had no data (would have returned earlier), but keep safe.
      splitters.assign((size_t)P - 1, RealT{});
    }
    
    // ---- Partition local records into buckets ----
    std::vector<std::vector<Record<RealT>>> buckets((size_t)P);
    for (const auto& rec : local) {
      int b = (P == 1) ? 0 : bucket_of<RealT>(rec.v, splitters);
      buckets[(size_t)b].push_back(rec);
    }
    
    // Prepare Alltoallv counts/displs (Record)
    std::vector<int> send_counts(P, 0), recv_counts(P, 0);
    for (int i = 0; i < P; ++i) {
      // NOTE: counts are int. If you may exceed 2^31 elements per rank, need large-count MPI.
      send_counts[i] = (int)buckets[(size_t)i].size();
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    
    std::vector<int> send_displs(P, 0), recv_displs(P, 0);
    int send_total = 0, recv_total = 0;
    for (int i = 0; i < P; ++i) {
      send_displs[i] = send_total;
      recv_displs[i] = recv_total;
      send_total += send_counts[i];
      recv_total += recv_counts[i];
    }
    
    std::vector<Record<RealT>> sendbuf((size_t)send_total);
    {
      int cursor = 0;
      for (int i = 0; i < P; ++i) {
	auto& b = buckets[(size_t)i];
	if (!b.empty()) {
	  std::memcpy(sendbuf.data() + cursor, b.data(), b.size() * sizeof(Record<RealT>));
	  cursor += (int)b.size();
	}
      }
    }

    std::vector<Record<RealT>> recvbuf((size_t)recv_total);
    
    MPI_Datatype rec_t = mpi_record_type<RealT>();
    MPI_Alltoallv(sendbuf.data(), send_counts.data(), send_displs.data(), rec_t,
                  recvbuf.data(), recv_counts.data(), recv_displs.data(), rec_t,
                  comm);
    
    // Now each rank owns a value-range segment; sort locally
    std::sort(recvbuf.begin(), recvbuf.end(),
              [](const auto& a, const auto& b){ return detail_mpi_sort_array::record_less_desc<RealT>(a, b); });
    
    // ---- Compute global rank offset for this rank’s segment ----
    uint64_t my_segment = (uint64_t)recvbuf.size();
    uint64_t offset = u64_exscan(my_segment, comm, world_rank);

    // Each element in recvbuf gets global rank = offset + local_pos
    // Send back (owner, idx, global rank) to original owners.
    std::vector<std::vector<Back>> back_buckets((size_t)P);
    for (uint64_t i = 0; i < (uint64_t)recvbuf.size(); ++i) {
      const auto& r = recvbuf[(size_t)i];
      Back b;
      b.owner = r.owner;
      b.idx   = r.idx;
      b.grank = offset + i; // 0 is largest
      back_buckets[(size_t)b.owner].push_back(b);
    }
    
    // Alltoallv back
    std::vector<int> bsend_counts(P, 0), brecv_counts(P, 0);
    for (int i = 0; i < P; ++i) {
      bsend_counts[i] = (int)back_buckets[(size_t)i].size();
    }
    MPI_Alltoall(bsend_counts.data(), 1, MPI_INT, brecv_counts.data(), 1, MPI_INT, comm);
    
    std::vector<int> bsend_displs(P, 0), brecv_displs(P, 0);
    int bsend_total = 0, brecv_total = 0;
    for (int i = 0; i < P; ++i) {
      bsend_displs[i] = bsend_total;
      brecv_displs[i] = brecv_total;
      bsend_total += bsend_counts[i];
      brecv_total += brecv_counts[i];
    }
    
    std::vector<Back> bsendbuf((size_t)bsend_total);
    {
        int cursor = 0;
        for (int i = 0; i < P; ++i) {
	  auto& bb = back_buckets[(size_t)i];
	  if (!bb.empty()) {
	    std::memcpy(bsendbuf.data() + cursor, bb.data(), bb.size() * sizeof(Back));
	    cursor += (int)bb.size();
	  }
        }
    }
    std::vector<Back> brecvbuf((size_t)brecv_total);
    
    MPI_Datatype back_t = mpi_back_type();
    MPI_Alltoallv(bsendbuf.data(), bsend_counts.data(), bsend_displs.data(), back_t,
                  brecvbuf.data(), brecv_counts.data(), brecv_displs.data(), back_t,
                  comm);
    
    // Fill ranking: ranking[idx] = global_rank
    ranking.assign(w.size(), size_t(0));
    for (const auto& b : brecvbuf) {
      // owner should match, but keep safe
      if (b.owner != world_rank) continue;
      if (b.idx >= (uint64_t)ranking.size()) continue;
      ranking[(size_t)b.idx] = (size_t)b.grank;
    }
    
    // cleanup
    MPI_Type_free(&rec_t);
    MPI_Type_free(&back_t);
  }
  
}
#endif
