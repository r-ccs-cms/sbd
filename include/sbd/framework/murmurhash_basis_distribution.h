#ifndef SBD_MURMURHASH_BASIS_DISTRIBUTION_H
#define SBD_MURMURHASH_BASIS_DISTRIBUTION_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/bit_manipulation.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sbd {
namespace murmur_basis {

struct hash128 {
  std::uint64_t first;
  std::uint64_t second;
};

struct RedistributionTiming {
  double hash_seconds = 0.0;
  double packing_seconds = 0.0;
  double communication_seconds = 0.0;
  double sorting_seconds = 0.0;
};

namespace detail {

inline std::uint64_t rotate_left(std::uint64_t value, int count) {
  return (value << count) | (value >> (64 - count));
}

inline std::uint64_t final_mix(std::uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return value;
}

inline std::uint64_t load_le64(const std::uint8_t* data) {
  return static_cast<std::uint64_t>(data[0]) |
         (static_cast<std::uint64_t>(data[1]) << 8) |
         (static_cast<std::uint64_t>(data[2]) << 16) |
         (static_cast<std::uint64_t>(data[3]) << 24) |
         (static_cast<std::uint64_t>(data[4]) << 32) |
         (static_cast<std::uint64_t>(data[5]) << 40) |
         (static_cast<std::uint64_t>(data[6]) << 48) |
         (static_cast<std::uint64_t>(data[7]) << 56);
}

inline int checked_mpi_count(std::size_t value, const char* description) {
  if(value > static_cast<std::size_t>(INT_MAX))
    throw std::overflow_error(std::string(description) +
                              " exceeds the MPI int count limit");
  return static_cast<int>(value);
}

}  // namespace detail

// MurmurHash3_x64_128, with explicit little-endian loads. This matches the
// public-domain reference algorithm without depending on host byte order.
inline hash128 murmurhash3_x64_128(const void* key, std::size_t length,
                                   std::uint32_t seed = 0) {
  const auto* data = static_cast<const std::uint8_t*>(key);
  const std::size_t block_count = length / 16;
  std::uint64_t h1 = seed;
  std::uint64_t h2 = seed;
  constexpr std::uint64_t c1 = UINT64_C(0x87c37b91114253d5);
  constexpr std::uint64_t c2 = UINT64_C(0x4cf5ad432745937f);

  for(std::size_t block = 0; block < block_count; ++block) {
    const std::uint8_t* bytes = data + 16 * block;
    std::uint64_t k1 = detail::load_le64(bytes);
    std::uint64_t k2 = detail::load_le64(bytes + 8);

    k1 *= c1;
    k1 = detail::rotate_left(k1, 31);
    k1 *= c2;
    h1 ^= k1;
    h1 = detail::rotate_left(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + UINT64_C(0x52dce729);

    k2 *= c2;
    k2 = detail::rotate_left(k2, 33);
    k2 *= c1;
    h2 ^= k2;
    h2 = detail::rotate_left(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + UINT64_C(0x38495ab5);
  }

  const std::uint8_t* tail = data + 16 * block_count;
  std::uint64_t k1 = 0;
  std::uint64_t k2 = 0;
  switch(length & 15) {
    case 15: k2 ^= static_cast<std::uint64_t>(tail[14]) << 48; [[fallthrough]];
    case 14: k2 ^= static_cast<std::uint64_t>(tail[13]) << 40; [[fallthrough]];
    case 13: k2 ^= static_cast<std::uint64_t>(tail[12]) << 32; [[fallthrough]];
    case 12: k2 ^= static_cast<std::uint64_t>(tail[11]) << 24; [[fallthrough]];
    case 11: k2 ^= static_cast<std::uint64_t>(tail[10]) << 16; [[fallthrough]];
    case 10: k2 ^= static_cast<std::uint64_t>(tail[9]) << 8; [[fallthrough]];
    case 9:
      k2 ^= static_cast<std::uint64_t>(tail[8]);
      k2 *= c2;
      k2 = detail::rotate_left(k2, 33);
      k2 *= c1;
      h2 ^= k2;
      [[fallthrough]];
    case 8: k1 ^= static_cast<std::uint64_t>(tail[7]) << 56; [[fallthrough]];
    case 7: k1 ^= static_cast<std::uint64_t>(tail[6]) << 48; [[fallthrough]];
    case 6: k1 ^= static_cast<std::uint64_t>(tail[5]) << 40; [[fallthrough]];
    case 5: k1 ^= static_cast<std::uint64_t>(tail[4]) << 32; [[fallthrough]];
    case 4: k1 ^= static_cast<std::uint64_t>(tail[3]) << 24; [[fallthrough]];
    case 3: k1 ^= static_cast<std::uint64_t>(tail[2]) << 16; [[fallthrough]];
    case 2: k1 ^= static_cast<std::uint64_t>(tail[1]) << 8; [[fallthrough]];
    case 1:
      k1 ^= static_cast<std::uint64_t>(tail[0]);
      k1 *= c1;
      k1 = detail::rotate_left(k1, 31);
      k1 *= c2;
      h1 ^= k1;
      break;
    default: break;
  }

  h1 ^= length;
  h2 ^= length;
  h1 += h2;
  h2 += h1;
  h1 = detail::final_mix(h1);
  h2 = detail::final_mix(h2);
  h1 += h2;
  h2 += h1;
  return {h1, h2};
}

template <typename Row>
hash128 determinant_hash(const Row& determinant, std::uint32_t seed = 0) {
  // SBD stores a determinant as least-significant word first. Encode every
  // word as a fixed-width uint64 in little-endian order, rather than hashing
  // size_t's host representation. Thus the mapping is stable across endian.
  std::vector<std::uint8_t> bytes(determinant.size() * 8);
  for(std::size_t word = 0; word < determinant.size(); ++word) {
    const std::uint64_t value = static_cast<std::uint64_t>(determinant[word]);
    for(unsigned byte = 0; byte < 8; ++byte)
      bytes[8 * word + byte] =
          static_cast<std::uint8_t>(value >> (8 * byte));
  }
  return murmurhash3_x64_128(bytes.data(), bytes.size(), seed);
}

template <typename Row>
int owner_rank(const Row& determinant, int communicator_size,
               std::uint32_t seed = 0) {
  if(communicator_size <= 0)
    throw std::invalid_argument("communicator size must be positive");
  return static_cast<int>(determinant_hash(determinant, seed).first %
                          static_cast<std::uint64_t>(communicator_size));
}

inline void redistribute_unique_determinants_by_hash(
    sbd::det_vector<std::size_t>& determinants, MPI_Comm communicator,
    std::uint32_t seed = 0) {
  int size = 0;
  MPI_Comm_size(communicator, &size);
  if(size <= 0) throw std::runtime_error("invalid MPI communicator size");

  const std::size_t width = determinants.empty() ? 0 : determinants[0].size();
  std::uint64_t local_width = static_cast<std::uint64_t>(width);
  std::uint64_t max_width = 0;
  MPI_Allreduce(&local_width, &max_width, 1, MPI_UINT64_T, MPI_MAX,
                communicator);
  // Empty ranks report zero, so use the maximum fixed row width. All nonempty
  // ranks must agree with it.
  if(width != 0 && local_width != max_width)
    throw std::invalid_argument("inconsistent determinant row widths");
  const std::size_t global_width = static_cast<std::size_t>(max_width);
  if(global_width == 0) {
    determinants.clear();
    return;
  }

  std::vector<int> send_rows(size, 0);
  std::vector<int> destinations(determinants.size());
  for(std::size_t index = 0; index < determinants.size(); ++index) {
    const int destination = owner_rank(determinants[index], size, seed);
    destinations[index] = destination;
    if(send_rows[destination] == INT_MAX)
      throw std::overflow_error("MPI row count exceeds INT_MAX");
    ++send_rows[destination];
  }

  std::vector<int> receive_rows(size, 0);
  MPI_Alltoall(send_rows.data(), 1, MPI_INT, receive_rows.data(), 1, MPI_INT,
               communicator);
  std::vector<int> send_displacements(size, 0);
  std::vector<int> receive_displacements(size, 0);
  std::size_t send_total = 0;
  std::size_t receive_total = 0;
  for(int rank = 0; rank < size; ++rank) {
    send_displacements[rank] = detail::checked_mpi_count(
        send_total * global_width, "send displacement");
    receive_displacements[rank] = detail::checked_mpi_count(
        receive_total * global_width, "receive displacement");
    send_total += static_cast<std::size_t>(send_rows[rank]);
    receive_total += static_cast<std::size_t>(receive_rows[rank]);
  }

  std::vector<std::size_t> send_buffer(send_total * global_width);
  std::vector<std::size_t> receive_buffer(receive_total * global_width);
  std::vector<std::size_t> offsets(size, 0);
  for(int rank = 0; rank < size; ++rank)
    offsets[rank] = static_cast<std::size_t>(send_displacements[rank]);
  for(std::size_t index = 0; index < determinants.size(); ++index) {
    const int destination = destinations[index];
    std::copy(determinants[index].begin(), determinants[index].end(),
              send_buffer.begin() + offsets[destination]);
    offsets[destination] += global_width;
  }

  std::vector<int> send_words(size, 0);
  std::vector<int> receive_words(size, 0);
  for(int rank = 0; rank < size; ++rank) {
    send_words[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(send_rows[rank]) * global_width,
        "send count");
    receive_words[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(receive_rows[rank]) * global_width,
        "receive count");
  }
  MPI_Alltoallv(send_buffer.data(), send_words.data(), send_displacements.data(),
                SBD_MPI_SIZE_T, receive_buffer.data(), receive_words.data(),
                receive_displacements.data(), SBD_MPI_SIZE_T, communicator);

  determinants.clear();
  determinants.reserve(receive_total);
  for(std::size_t index = 0; index < receive_total; ++index)
    determinants.emplace_back(receive_buffer.begin() + index * global_width,
                              receive_buffer.begin() + (index + 1) * global_width);
  sbd::sort_bitarray(determinants);
  determinants.erase(std::unique(determinants.begin(), determinants.end()),
                     determinants.end());
}

// Redistribute determinant/value pairs to their hash owners. Values follow
// their determinant exactly; duplicate determinants are deliberately retained
// because combining their values requires a caller-defined policy.
template <typename Value>
void redistribute_determinant_values_by_hash(
    sbd::det_vector<std::size_t>& determinants,
    std::vector<Value>& values,
    MPI_Comm communicator,
    std::uint32_t seed = 0,
    RedistributionTiming* timing = nullptr) {
  static_assert(std::is_trivially_copyable_v<Value>,
                "redistributed values must be trivially copyable");
  if(determinants.size() != values.size())
    throw std::invalid_argument(
        "determinant and value counts must be identical");
  if(timing != nullptr) *timing = RedistributionTiming{};

  int size = 0;
  MPI_Comm_size(communicator, &size);
  if(size <= 0) throw std::runtime_error("invalid MPI communicator size");

  const std::size_t width = determinants.empty() ? 0 : determinants[0].size();
  std::uint64_t local_width = static_cast<std::uint64_t>(width);
  std::uint64_t max_width = 0;
  double phase_start = MPI_Wtime();
  MPI_Allreduce(&local_width, &max_width, 1, MPI_UINT64_T, MPI_MAX,
                communicator);
  if(timing != nullptr)
    timing->communication_seconds += MPI_Wtime() - phase_start;
  if(width != 0 && local_width != max_width)
    throw std::invalid_argument("inconsistent determinant row widths");
  const std::size_t global_width = static_cast<std::size_t>(max_width);
  if(global_width == 0) {
    determinants.clear();
    values.clear();
    return;
  }

  std::vector<int> destinations(determinants.size());
  std::vector<int> send_rows(size, 0);
  phase_start = MPI_Wtime();
  for(std::size_t index = 0; index < determinants.size(); ++index) {
    const int destination = owner_rank(determinants[index], size, seed);
    destinations[index] = destination;
    if(send_rows[destination] == INT_MAX)
      throw std::overflow_error("MPI row count exceeds INT_MAX");
    ++send_rows[destination];
  }
  if(timing != nullptr) timing->hash_seconds += MPI_Wtime() - phase_start;

  std::vector<int> receive_rows(size, 0);
  phase_start = MPI_Wtime();
  MPI_Alltoall(send_rows.data(), 1, MPI_INT, receive_rows.data(), 1, MPI_INT,
               communicator);
  if(timing != nullptr)
    timing->communication_seconds += MPI_Wtime() - phase_start;

  phase_start = MPI_Wtime();
  std::vector<std::size_t> send_row_offsets(size, 0);
  std::vector<std::size_t> receive_row_offsets(size, 0);
  std::size_t send_total = 0;
  std::size_t receive_total = 0;
  for(int rank = 0; rank < size; ++rank) {
    send_row_offsets[rank] = send_total;
    receive_row_offsets[rank] = receive_total;
    send_total += static_cast<std::size_t>(send_rows[rank]);
    receive_total += static_cast<std::size_t>(receive_rows[rank]);
  }

  std::vector<std::size_t> send_determinants(send_total * global_width);
  std::vector<Value> send_values(send_total);
  std::vector<std::size_t> offsets = send_row_offsets;
  for(std::size_t index = 0; index < determinants.size(); ++index) {
    const int destination = destinations[index];
    const std::size_t position = offsets[destination]++;
    std::copy(determinants[index].begin(), determinants[index].end(),
              send_determinants.begin() + position * global_width);
    send_values[position] = values[index];
  }

  std::vector<std::size_t> receive_determinants(receive_total * global_width);
  std::vector<Value> receive_values(receive_total);
  std::vector<int> send_words(size, 0);
  std::vector<int> receive_words(size, 0);
  std::vector<int> send_word_displacements(size, 0);
  std::vector<int> receive_word_displacements(size, 0);
  std::vector<int> send_value_bytes(size, 0);
  std::vector<int> receive_value_bytes(size, 0);
  std::vector<int> send_value_displacements(size, 0);
  std::vector<int> receive_value_displacements(size, 0);
  for(int rank = 0; rank < size; ++rank) {
    send_words[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(send_rows[rank]) * global_width,
        "send determinant count");
    receive_words[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(receive_rows[rank]) * global_width,
        "receive determinant count");
    send_word_displacements[rank] = detail::checked_mpi_count(
        send_row_offsets[rank] * global_width,
        "send determinant displacement");
    receive_word_displacements[rank] = detail::checked_mpi_count(
        receive_row_offsets[rank] * global_width,
        "receive determinant displacement");
    send_value_bytes[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(send_rows[rank]) * sizeof(Value),
        "send value count");
    receive_value_bytes[rank] = detail::checked_mpi_count(
        static_cast<std::size_t>(receive_rows[rank]) * sizeof(Value),
        "receive value count");
    send_value_displacements[rank] = detail::checked_mpi_count(
        send_row_offsets[rank] * sizeof(Value), "send value displacement");
    receive_value_displacements[rank] = detail::checked_mpi_count(
        receive_row_offsets[rank] * sizeof(Value),
        "receive value displacement");
  }
  if(timing != nullptr) timing->packing_seconds += MPI_Wtime() - phase_start;

  phase_start = MPI_Wtime();
  MPI_Alltoallv(
      send_determinants.data(), send_words.data(),
      send_word_displacements.data(), SBD_MPI_SIZE_T,
      receive_determinants.data(), receive_words.data(),
      receive_word_displacements.data(), SBD_MPI_SIZE_T, communicator);
  MPI_Alltoallv(
      send_values.data(), send_value_bytes.data(),
      send_value_displacements.data(), MPI_BYTE,
      receive_values.data(), receive_value_bytes.data(),
      receive_value_displacements.data(), MPI_BYTE, communicator);
  if(timing != nullptr)
    timing->communication_seconds += MPI_Wtime() - phase_start;

  phase_start = MPI_Wtime();
  sbd::det_vector<std::size_t> received;
  received.reserve(receive_total);
  for(std::size_t index = 0; index < receive_total; ++index)
    received.emplace_back(
        receive_determinants.begin() + index * global_width,
        receive_determinants.begin() + (index + 1) * global_width);

  std::vector<std::size_t> order(receive_total);
  std::iota(order.begin(), order.end(), std::size_t(0));
  std::sort(order.begin(), order.end(),
            [&](std::size_t left, std::size_t right) {
              return received[left] < received[right];
            });
  determinants.clear();
  determinants.reserve(receive_total);
  values.resize(receive_total);
  for(std::size_t output = 0; output < receive_total; ++output) {
    const std::size_t input = order[output];
    determinants.push_back(received[input]);
    values[output] = receive_values[input];
  }
  if(timing != nullptr) timing->sorting_seconds += MPI_Wtime() - phase_start;
}

}  // namespace murmur_basis
}  // namespace sbd

#endif
