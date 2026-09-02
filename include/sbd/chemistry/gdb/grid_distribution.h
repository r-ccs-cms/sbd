#ifndef SBD_CHEMISTRY_GDB_GRID_DISTRIBUTION_H
#define SBD_CHEMISTRY_GDB_GRID_DISTRIBUTION_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "sbd/chemistry/gdb/helper.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {
namespace gdb {

namespace grid_ab_detail {

template<class KeyA, class KeyB>
inline bool equal_key(const KeyA& a,
                      const KeyB& b) {
  return !sbd::less_from_back(a, b) && !sbd::less_from_back(b, a);
}

template<sbd::det_kind Kind>
inline void erase_cross_rank_duplicates(
    sbd::det_vector<size_t, Kind>& keys, MPI_Comm comm) {
  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  const size_t local_num_words = keys.empty() ? 0 : keys[0].size();
  size_t num_words = 0;
  MPI_Allreduce(&local_num_words, &num_words, 1, SBD_MPI_SIZE_T, MPI_MAX, comm);
  if (num_words == 0) return;

  const int has_local_key = keys.empty() ? 0 : 1;
  std::vector<int> has_key(static_cast<size_t>(mpi_size), 0);
  MPI_Allgather(&has_local_key, 1, MPI_INT,
                has_key.data(), 1, MPI_INT, comm);

  std::vector<size_t> last_local(num_words, 0);
  if (has_local_key) {
    for (size_t k = 0; k < num_words; ++k) last_local[k] = keys[keys.size() - 1][k];
  }

  std::vector<size_t> last_all(static_cast<size_t>(mpi_size) * num_words, 0);
  MPI_Allgather(last_local.data(), static_cast<int>(num_words), SBD_MPI_SIZE_T,
                last_all.data(), static_cast<int>(num_words), SBD_MPI_SIZE_T,
                comm);

  if (!has_local_key) return;

  int prev = mpi_rank - 1;
  while (prev >= 0 && has_key[static_cast<size_t>(prev)] == 0) --prev;
  if (prev < 0) return;

  bool same_as_prev_last = true;
  for (size_t k = 0; k < num_words; ++k) {
    if (keys[0][k] != last_all[static_cast<size_t>(prev) * num_words + k]) {
      same_as_prev_last = false;
      break;
    }
  }

  if (same_as_prev_last) keys.erase(keys.begin());
}

template<sbd::det_kind Kind>
inline sbd::det_vector<size_t, Kind> make_global_unique_spin_keys(
    const sbd::det_vector<size_t, Kind>& local_unique_keys, MPI_Comm comm) {
  sbd::det_vector<size_t, Kind> keys = local_unique_keys;
  sbd::sort_global_bitarray(keys, comm);
  erase_cross_rank_duplicates(keys, comm);
  return keys;
}

inline std::vector<size_t> gather_counts_and_local_begin(const size_t local_n,
                                                         size_t& local_begin,
                                                         size_t& global_n,
                                                         MPI_Comm comm) {
  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  std::vector<size_t> counts(static_cast<size_t>(mpi_size), 0);
  MPI_Allgather(&local_n, 1, SBD_MPI_SIZE_T,
                counts.data(), 1, SBD_MPI_SIZE_T, comm);

  local_begin = 0;
  for (int r = 0; r < mpi_rank; ++r) local_begin += counts[static_cast<size_t>(r)];
  global_n = std::accumulate(counts.begin(), counts.end(), size_t(0));
  return counts;
}

template<sbd::det_kind Kind>
inline std::vector<int> cyclic_blocks_for_local_unique_keys(
    const sbd::det_vector<size_t, Kind>& global_keys,
    const sbd::det_vector<size_t, Kind>& local_unique_queries,
    const size_t n_parts,
    MPI_Comm comm) {
  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  const size_t local_num_words = global_keys.empty() ? 0 : global_keys[0].size();
  size_t num_words = local_num_words;
  MPI_Allreduce(&local_num_words, &num_words, 1, SBD_MPI_SIZE_T, MPI_MAX, comm);

  std::vector<int> blocks(local_unique_queries.size(), 0);
  // Empty ranks must still participate in every collective below. Returning
  // here deadlocks when another rank has queries and enters the Alltoall(v)
  // sequence. Zero-length buffers/counts are valid MPI participants.
  if (num_words == 0) {
    throw std::runtime_error("cyclic_blocks_for_local_unique_keys: no global keys.");
  }

  size_t local_begin = 0;
  size_t global_n = 0;
  const std::vector<size_t> key_counts =
      gather_counts_and_local_begin(global_keys.size(), local_begin, global_n, comm);
  if (global_n == 0) {
    throw std::runtime_error("cyclic_blocks_for_local_unique_keys: empty global key set.");
  }

  std::vector<size_t> last_local(num_words, 0);
  if (!global_keys.empty()) {
    std::memcpy(last_local.data(), global_keys[global_keys.size() - 1].data(),
                num_words * sizeof(size_t));
  }

  std::vector<size_t> last_all(static_cast<size_t>(mpi_size) * num_words, 0);
  MPI_Allgather(last_local.data(), static_cast<int>(num_words), SBD_MPI_SIZE_T,
                last_all.data(), static_cast<int>(num_words), SBD_MPI_SIZE_T,
                comm);

  auto last_less_than_key = [&](const int r, const auto& key) {
    const size_t offset = static_cast<size_t>(r) * num_words;
    for (size_t k = num_words; k > 0; --k) {
      const size_t last_word = last_all[offset + k - 1];
      if (last_word < key[k - 1]) return true;
      if (last_word > key[k - 1]) return false;
    }
    return false;
  };

  auto owner_rank = [&](const auto& key) -> int {
    int last_nonempty = -1;
    for (int r = 0; r < mpi_size; ++r) {
      if (key_counts[static_cast<size_t>(r)] == 0) continue;
      last_nonempty = r;
      if (!last_less_than_key(r, key)) return r;
    }
    return last_nonempty >= 0 ? last_nonempty : 0;
  };

  const size_t query_record_words = num_words + 1;  // local query id + key
  std::vector<int> send_counts(static_cast<size_t>(mpi_size), 0);
  std::vector<int> recv_counts(static_cast<size_t>(mpi_size), 0);

  std::vector<int> query_owner(local_unique_queries.size(), 0);
  for (size_t i = 0; i < local_unique_queries.size(); ++i) {
    const int owner = owner_rank(local_unique_queries[i]);
    query_owner[i] = owner;
    send_counts[static_cast<size_t>(owner)] += static_cast<int>(query_record_words);
  }

  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, comm);

  std::vector<int> send_displs(static_cast<size_t>(mpi_size), 0);
  std::vector<int> recv_displs(static_cast<size_t>(mpi_size), 0);
  for (int r = 1; r < mpi_size; ++r) {
    send_displs[static_cast<size_t>(r)] = send_displs[static_cast<size_t>(r - 1)] +
                                         send_counts[static_cast<size_t>(r - 1)];
    recv_displs[static_cast<size_t>(r)] = recv_displs[static_cast<size_t>(r - 1)] +
                                         recv_counts[static_cast<size_t>(r - 1)];
  }

  const int total_send = std::accumulate(send_counts.begin(), send_counts.end(), 0);
  const int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  std::vector<size_t> sendbuf(static_cast<size_t>(total_send), 0);
  std::vector<size_t> recvbuf(static_cast<size_t>(total_recv), 0);

  std::vector<int> fill = send_displs;
  for (size_t i = 0; i < local_unique_queries.size(); ++i) {
    const int owner = query_owner[i];
    int pos = fill[static_cast<size_t>(owner)];
    sendbuf[static_cast<size_t>(pos++)] = i;
    for (size_t k = 0; k < num_words; ++k) {
      sendbuf[static_cast<size_t>(pos++)] = local_unique_queries[i][k];
    }
    fill[static_cast<size_t>(owner)] = pos;
  }

  MPI_Alltoallv(sendbuf.data(), send_counts.data(), send_displs.data(), SBD_MPI_SIZE_T,
                recvbuf.data(), recv_counts.data(), recv_displs.data(), SBD_MPI_SIZE_T,
                comm);

  const size_t reply_record_words = 2;  // query id + cyclic block
  std::vector<int> reply_send_counts(static_cast<size_t>(mpi_size), 0);
  for (int src = 0; src < mpi_size; ++src) {
    const int n_records = recv_counts[static_cast<size_t>(src)] /
                          static_cast<int>(query_record_words);
    reply_send_counts[static_cast<size_t>(src)] =
        n_records * static_cast<int>(reply_record_words);
  }

  std::vector<int> reply_recv_counts(static_cast<size_t>(mpi_size), 0);
  for (int dest = 0; dest < mpi_size; ++dest) {
    const int n_records = send_counts[static_cast<size_t>(dest)] /
                          static_cast<int>(query_record_words);
    reply_recv_counts[static_cast<size_t>(dest)] =
        n_records * static_cast<int>(reply_record_words);
  }

  std::vector<int> reply_send_displs(static_cast<size_t>(mpi_size), 0);
  std::vector<int> reply_recv_displs(static_cast<size_t>(mpi_size), 0);
  for (int r = 1; r < mpi_size; ++r) {
    reply_send_displs[static_cast<size_t>(r)] =
        reply_send_displs[static_cast<size_t>(r - 1)] +
        reply_send_counts[static_cast<size_t>(r - 1)];
    reply_recv_displs[static_cast<size_t>(r)] =
        reply_recv_displs[static_cast<size_t>(r - 1)] +
        reply_recv_counts[static_cast<size_t>(r - 1)];
  }

  const int total_reply_send =
      std::accumulate(reply_send_counts.begin(), reply_send_counts.end(), 0);
  const int total_reply_recv =
      std::accumulate(reply_recv_counts.begin(), reply_recv_counts.end(), 0);
  std::vector<size_t> reply_sendbuf(static_cast<size_t>(total_reply_send), 0);
  std::vector<size_t> reply_recvbuf(static_cast<size_t>(total_reply_recv), 0);

  std::vector<int> reply_fill = reply_send_displs;
  for (int src = 0; src < mpi_size; ++src) {
    const int begin = recv_displs[static_cast<size_t>(src)];
    const int end = begin + recv_counts[static_cast<size_t>(src)];
    for (int pos = begin; pos < end; pos += static_cast<int>(query_record_words)) {
      const size_t query_id = recvbuf[static_cast<size_t>(pos)];
      std::vector<size_t> key(num_words, 0);
      for (size_t k = 0; k < num_words; ++k) {
        key[k] = recvbuf[static_cast<size_t>(pos) + 1 + k];
      }

      auto it = std::lower_bound(global_keys.begin(), global_keys.end(), key,
                                 [](const auto& a, const auto& b) {
                                   return sbd::less_from_back(a, b);
                                 });
      if (it == global_keys.end() || !equal_key(*it, key)) {
        throw std::runtime_error("cyclic_blocks_for_local_unique_keys: queried key was not found on owner rank.");
      }

      const size_t local_pos = static_cast<size_t>(it - global_keys.begin());
      const size_t global_index = local_begin + local_pos;
      const size_t block = global_index % n_parts;

      int rpos = reply_fill[static_cast<size_t>(src)];
      reply_sendbuf[static_cast<size_t>(rpos++)] = query_id;
      reply_sendbuf[static_cast<size_t>(rpos++)] = block;
      reply_fill[static_cast<size_t>(src)] = rpos;
    }
  }

  MPI_Alltoallv(reply_sendbuf.data(), reply_send_counts.data(), reply_send_displs.data(), SBD_MPI_SIZE_T,
                reply_recvbuf.data(), reply_recv_counts.data(), reply_recv_displs.data(), SBD_MPI_SIZE_T,
                comm);

  for (int pos = 0; pos < total_reply_recv; pos += static_cast<int>(reply_record_words)) {
    const size_t query_id = reply_recvbuf[static_cast<size_t>(pos)];
    const size_t block = reply_recvbuf[static_cast<size_t>(pos + 1)];
    if (query_id >= blocks.size()) {
      throw std::runtime_error("cyclic_blocks_for_local_unique_keys: invalid query id in reply.");
    }
    blocks[query_id] = static_cast<int>(block);
  }

  return blocks;
}

inline void alltoallv_determinants(sbd::det_vector<size_t>& config,
                                   const std::vector<int>& dest_per_det,
                                   const size_t num_words,
                                   MPI_Comm comm) {
  int mpi_size = 1;
  MPI_Comm_size(comm, &mpi_size);

  std::vector<int> send_counts_det(static_cast<size_t>(mpi_size), 0);
  for (const int dest : dest_per_det) {
    ++send_counts_det[static_cast<size_t>(dest)];
  }

  std::vector<int> send_counts_words(static_cast<size_t>(mpi_size), 0);
  std::vector<int> recv_counts_words(static_cast<size_t>(mpi_size), 0);
  std::vector<int> send_displs_words(static_cast<size_t>(mpi_size), 0);
  std::vector<int> recv_displs_words(static_cast<size_t>(mpi_size), 0);

  for (int r = 0; r < mpi_size; ++r) {
    send_counts_words[static_cast<size_t>(r)] =
        send_counts_det[static_cast<size_t>(r)] * static_cast<int>(num_words);
  }

  MPI_Alltoall(send_counts_words.data(), 1, MPI_INT,
               recv_counts_words.data(), 1, MPI_INT, comm);

  for (int r = 1; r < mpi_size; ++r) {
    send_displs_words[static_cast<size_t>(r)] =
        send_displs_words[static_cast<size_t>(r - 1)] +
        send_counts_words[static_cast<size_t>(r - 1)];
    recv_displs_words[static_cast<size_t>(r)] =
        recv_displs_words[static_cast<size_t>(r - 1)] +
        recv_counts_words[static_cast<size_t>(r - 1)];
  }

  const int total_send_words =
      std::accumulate(send_counts_words.begin(), send_counts_words.end(), 0);
  const int total_recv_words =
      std::accumulate(recv_counts_words.begin(), recv_counts_words.end(), 0);

  std::vector<size_t> sendbuf(static_cast<size_t>(total_send_words), 0);
  std::vector<size_t> recvbuf(static_cast<size_t>(total_recv_words), 0);

  std::vector<int> current_displs = send_displs_words;
  for (size_t i = 0; i < config.size(); ++i) {
    const int dest = dest_per_det[i];
    const int pos = current_displs[static_cast<size_t>(dest)];
    std::memcpy(sendbuf.data() + pos, config[i].data(),
                num_words * sizeof(size_t));
    current_displs[static_cast<size_t>(dest)] +=
        static_cast<int>(num_words);
  }

  MPI_Alltoallv(sendbuf.data(), send_counts_words.data(), send_displs_words.data(), SBD_MPI_SIZE_T,
                recvbuf.data(), recv_counts_words.data(), recv_displs_words.data(), SBD_MPI_SIZE_T,
                comm);

  const size_t recv_n = static_cast<size_t>(total_recv_words) / num_words;
  sbd::det_vector<size_t>::init_elem_size(num_words);
  config.resize(recv_n);
  if (!recvbuf.empty()) {
    std::memcpy(config.flat().data(), recvbuf.data(),
                recvbuf.size() * sizeof(size_t));
  }

}

template<sbd::det_kind Kind>
inline int block_for_key_from_unique_list(
    const std::vector<size_t>& key,
    const sbd::det_vector<size_t, Kind>& local_unique_keys,
    const std::vector<int>& local_blocks) {
  auto it = std::lower_bound(local_unique_keys.begin(), local_unique_keys.end(), key,
                             [](const auto& a, const auto& b) {
                               return sbd::less_from_back(a, b);
                             });
  if (it == local_unique_keys.end() || !equal_key(*it, key)) {
    throw std::runtime_error("block_for_key_from_unique_list: key was not found.");
  }
  return local_blocks[static_cast<size_t>(it - local_unique_keys.begin())];
}

} // namespace grid_ab_detail

inline void redistribution_grid_bra_ab_cyclic(
    sbd::det_vector<size_t>& config,
    const size_t bit_length,
    const size_t total_bit_length,
    const size_t N_a,
    const size_t N_b,
    MPI_Comm comm) {
  int mpi_size = 1;
  MPI_Comm_size(comm, &mpi_size);

  if (N_a == 0 || N_b == 0) {
    throw std::runtime_error("redistribution_grid_bra_ab_cyclic: N_a and N_b must be positive.");
  }
  if (N_a * N_b != static_cast<size_t>(mpi_size)) {
    throw std::runtime_error("redistribution_grid_bra_ab_cyclic: N_a * N_b must equal MPI_Comm_size(comm).");
  }
  if (total_bit_length % 2 != 0) {
    throw std::runtime_error("redistribution_grid_bra_ab_cyclic: total bit length must be even.");
  }

  const size_t num_words = (total_bit_length + bit_length - 1) / bit_length;
  const size_t norb = total_bit_length / 2;

#ifndef NDEBUG
  for (const auto& det : config) assert(det.size() == num_words);
#endif

  using namespace grid_ab_detail;

  sbd::det_vector<size_t, sbd::det_kind::half> local_alpha_unique;
  sbd::det_vector<size_t, sbd::det_kind::half> local_beta_unique;
  std::vector<size_t> alpha_counts;
  std::vector<size_t> beta_counts;
  sbd::gdb::getHalfDets(config, bit_length, norb,
                        local_alpha_unique, local_beta_unique,
                        alpha_counts, beta_counts);

  const auto alpha_global_keys =
      make_global_unique_spin_keys(local_alpha_unique, comm);
  const auto beta_global_keys =
      make_global_unique_spin_keys(local_beta_unique, comm);

  std::vector<int> dest_per_det(config.size(), 0);

  const auto local_alpha_blocks = cyclic_blocks_for_local_unique_keys(
      alpha_global_keys, local_alpha_unique, N_a, comm);
  const auto local_beta_blocks = cyclic_blocks_for_local_unique_keys(
      beta_global_keys, local_beta_unique, N_b, comm);

  const size_t half_words = (norb + bit_length - 1) / bit_length;
  std::vector<size_t> alpha_key(half_words, 0);
  std::vector<size_t> beta_key(half_words, 0);
  for (size_t i = 0; i < config.size(); ++i) {
    sbd::getAdet(config[i], bit_length, norb, alpha_key);
    sbd::getBdet(config[i], bit_length, norb, beta_key);
    const int a_block = block_for_key_from_unique_list(
        alpha_key, local_alpha_unique, local_alpha_blocks);
    const int b_block = block_for_key_from_unique_list(
        beta_key, local_beta_unique, local_beta_blocks);
    dest_per_det[i] = a_block * static_cast<int>(N_b) + b_block;
  }

  alltoallv_determinants(config, dest_per_det, num_words, comm);
}

} // namespace gdb
} // namespace sbd

#endif // SBD_CHEMISTRY_GDB_GRID_DISTRIBUTION_H
