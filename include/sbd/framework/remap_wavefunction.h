/**
 * @file framework/remap_wavefunction.h
 * @brief In-memory remapping of distributed wavefunction coefficients.
 *
 * Adapted from the existing ExtSBD GDB/CAOP HCI distributed remap helper.
 * Public argument/return semantics are preserved.
 */
#ifndef SBD_FRAMEWORK_REMAP_WAVEFUNCTION_H
#define SBD_FRAMEWORK_REMAP_WAVEFUNCTION_H

#include <cstddef>
#include <cstdint>
#include "sbd/framework/type_def.h"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sbd {

namespace wavefunction_remap_detail {

struct DeterminantHash {
  std::size_t operator()(const std::vector<std::size_t>& determinant) const
      noexcept {
    // Standard 64-bit FNV-1a, consuming each word least-significant byte first.
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for(const std::size_t word : determinant) {
      std::uint64_t value = static_cast<std::uint64_t>(word);
      for(unsigned int byte = 0; byte < sizeof(std::size_t); ++byte) {
        hash ^= value & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
        value >>= 8;
      }
    }
    return static_cast<std::size_t>(hash);
  }
};

inline std::vector<int> displacements(const std::vector<int>& counts) {
  std::vector<int> result(counts.size(), 0);
  for(std::size_t rank = 1; rank < counts.size(); ++rank) {
    if(counts[rank - 1] > std::numeric_limits<int>::max()
                              - result[rank - 1]) {
      throw std::overflow_error("MPI displacement exceeds INT_MAX");
    }
    result[rank] = result[rank - 1] + counts[rank - 1];
  }
  return result;
}

inline int total_count(const std::vector<int>& counts) {
  int result = 0;
  for(const int count : counts) {
    if(count > std::numeric_limits<int>::max() - result) {
      throw std::overflow_error("MPI element count exceeds INT_MAX");
    }
    result += count;
  }
  return result;
}

inline std::vector<int> word_counts(const std::vector<int>& determinant_counts,
                                    std::size_t words_per_determinant) {
  std::vector<int> result(determinant_counts.size(), 0);
  for(std::size_t rank = 0; rank < determinant_counts.size(); ++rank) {
    const std::size_t count =
        static_cast<std::size_t>(determinant_counts[rank]);
    if(words_per_determinant != 0 &&
       count > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                   words_per_determinant) {
      throw std::overflow_error("MPI determinant word count exceeds INT_MAX");
    }
    result[rank] = static_cast<int>(count * words_per_determinant);
  }
  return result;
}

template <typename DetsContainer>
std::size_t checked_words_per_determinant(const DetsContainer& old_basis,
                                          const DetsContainer& new_basis,
                                          MPI_Comm communicator) {
  std::size_t local_words = 0;
  int local_invalid = 0;
  const auto inspect = [&](const DetsContainer& basis) {
    for(const auto& determinant : basis) {
      if(local_words == 0) local_words = determinant.size();
      if(determinant.size() != local_words) local_invalid = 1;
    }
  };
  inspect(old_basis);
  inspect(new_basis);

  std::size_t global_words = 0;
  MPI_Allreduce(&local_words, &global_words, 1, SBD_MPI_SIZE_T, MPI_MAX,
                communicator);
  if((!old_basis.empty() || !new_basis.empty()) &&
     local_words != global_words) {
    local_invalid = 1;
  }
  int global_invalid = 0;
  MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_MAX,
                communicator);
  if(global_invalid != 0) {
    throw std::invalid_argument(
        "all determinants must have the same positive word count");
  }
  return global_words;
}

template <typename Determinant>
std::vector<std::size_t> make_key(const Determinant& determinant) {
  return std::vector<std::size_t>(determinant.begin(), determinant.end());
}

inline int owner_rank(const std::vector<std::size_t>& determinant,
                      int communicator_size) {
  return static_cast<int>(
      DeterminantHash{}(determinant) %
      static_cast<std::size_t>(communicator_size));
}

}  // namespace wavefunction_remap_detail

/**
 * Copy coefficients from an arbitrarily distributed old determinant basis to
 * an arbitrarily distributed new determinant basis without filesystem I/O.
 * Coefficients of new determinants absent from old_basis are initialized to
 * zero. Both bases must be globally unique.
 * Old determinants absent from new_basis are ignored by default. Set
 * require_old_basis_subset=true when the caller needs to enforce that every
 * old determinant survives in the new basis.
 *
 * The implementation performs a distributed hash join. It does not replicate
 * either global basis on every MPI rank.
 *
 * @return global number of old determinants found in new_basis.
 */
template <typename ElemT, typename DetsContainer>
std::size_t RemapWavefunctionToBasis(
    const DetsContainer& old_basis,
    const std::vector<ElemT>& old_coefficients,
    const DetsContainer& new_basis,
    MPI_Comm communicator,
    std::vector<ElemT>& new_coefficients,
    bool require_old_basis_subset = false) {
  int communicator_size = 1;
  MPI_Comm_size(communicator, &communicator_size);

  int local_size_mismatch = old_basis.size() == old_coefficients.size() ? 0 : 1;
  int global_size_mismatch = 0;
  MPI_Allreduce(&local_size_mismatch, &global_size_mismatch, 1, MPI_INT,
                MPI_MAX, communicator);
  if(global_size_mismatch != 0) {
    throw std::invalid_argument(
        "old determinant and coefficient counts must match on every rank");
  }

  const std::size_t words_per_determinant =
      wavefunction_remap_detail::checked_words_per_determinant(
          old_basis, new_basis, communicator);
  new_coefficients.assign(new_basis.size(), ElemT(0));
  if(words_per_determinant == 0) return 0;

  std::vector<int> old_send_counts(communicator_size, 0);
  for(const auto& determinant : old_basis) {
    const auto key = wavefunction_remap_detail::make_key(determinant);
    ++old_send_counts[wavefunction_remap_detail::owner_rank(key, communicator_size)];
  }
  std::vector<int> old_recv_counts(communicator_size, 0);
  MPI_Alltoall(old_send_counts.data(), 1, MPI_INT,
               old_recv_counts.data(), 1, MPI_INT, communicator);
  const auto old_send_displacements =
      wavefunction_remap_detail::displacements(old_send_counts);
  const auto old_recv_displacements =
      wavefunction_remap_detail::displacements(old_recv_counts);
  const int old_send_total = wavefunction_remap_detail::total_count(old_send_counts);
  const int old_recv_total = wavefunction_remap_detail::total_count(old_recv_counts);
  const auto old_send_word_counts =
      wavefunction_remap_detail::word_counts(old_send_counts, words_per_determinant);
  const auto old_recv_word_counts =
      wavefunction_remap_detail::word_counts(old_recv_counts, words_per_determinant);
  const auto old_send_word_displacements =
      wavefunction_remap_detail::displacements(old_send_word_counts);
  const auto old_recv_word_displacements =
      wavefunction_remap_detail::displacements(old_recv_word_counts);

  std::vector<std::size_t> old_send_words(
      static_cast<std::size_t>(old_send_total) * words_per_determinant);
  std::vector<ElemT> old_send_values(old_send_total);
  std::vector<int> old_offsets = old_send_displacements;
  for(std::size_t index = 0; index < old_basis.size(); ++index) {
    const auto key = wavefunction_remap_detail::make_key(old_basis[index]);
    const int owner = wavefunction_remap_detail::owner_rank(key, communicator_size);
    const int position = old_offsets[owner]++;
    std::copy(key.begin(), key.end(),
              old_send_words.begin() +
                  static_cast<std::size_t>(position) * words_per_determinant);
    old_send_values[position] = old_coefficients[index];
  }

  std::vector<std::size_t> old_recv_words(
      static_cast<std::size_t>(old_recv_total) * words_per_determinant);
  std::vector<ElemT> old_recv_values(old_recv_total);
  MPI_Alltoallv(old_send_words.data(), old_send_word_counts.data(),
                old_send_word_displacements.data(), SBD_MPI_SIZE_T,
                old_recv_words.data(), old_recv_word_counts.data(),
                old_recv_word_displacements.data(), SBD_MPI_SIZE_T,
                communicator);
  MPI_Alltoallv(old_send_values.data(), old_send_counts.data(),
                old_send_displacements.data(), sbd::GetMpiType<ElemT>::MpiT,
                old_recv_values.data(), old_recv_counts.data(),
                old_recv_displacements.data(), sbd::GetMpiType<ElemT>::MpiT,
                communicator);

  struct StoredCoefficient {
    ElemT value;
    bool matched = false;
  };
  std::unordered_map<std::vector<std::size_t>, StoredCoefficient,
                     wavefunction_remap_detail::DeterminantHash> coefficient_by_determinant;
  coefficient_by_determinant.reserve(old_recv_values.size());
  std::size_t local_duplicates = 0;
  for(int index = 0; index < old_recv_total; ++index) {
    const auto begin = old_recv_words.begin() +
        static_cast<std::size_t>(index) * words_per_determinant;
    std::vector<std::size_t> key(begin, begin + words_per_determinant);
    const auto result = coefficient_by_determinant.emplace(
        std::move(key), StoredCoefficient{old_recv_values[index], false});
    if(!result.second) ++local_duplicates;
  }
  std::size_t global_duplicates = 0;
  MPI_Allreduce(&local_duplicates, &global_duplicates, 1, SBD_MPI_SIZE_T,
                MPI_SUM, communicator);
  if(global_duplicates != 0) {
    throw std::invalid_argument("old basis contains global duplicates");
  }

  std::vector<int> query_send_counts(communicator_size, 0);
  for(const auto& determinant : new_basis) {
    const auto key = wavefunction_remap_detail::make_key(determinant);
    ++query_send_counts[wavefunction_remap_detail::owner_rank(key, communicator_size)];
  }
  std::vector<int> query_recv_counts(communicator_size, 0);
  MPI_Alltoall(query_send_counts.data(), 1, MPI_INT,
               query_recv_counts.data(), 1, MPI_INT, communicator);
  const auto query_send_displacements =
      wavefunction_remap_detail::displacements(query_send_counts);
  const auto query_recv_displacements =
      wavefunction_remap_detail::displacements(query_recv_counts);
  const int query_send_total = wavefunction_remap_detail::total_count(query_send_counts);
  const int query_recv_total = wavefunction_remap_detail::total_count(query_recv_counts);
  const auto query_send_word_counts =
      wavefunction_remap_detail::word_counts(query_send_counts, words_per_determinant);
  const auto query_recv_word_counts =
      wavefunction_remap_detail::word_counts(query_recv_counts, words_per_determinant);
  const auto query_send_word_displacements =
      wavefunction_remap_detail::displacements(query_send_word_counts);
  const auto query_recv_word_displacements =
      wavefunction_remap_detail::displacements(query_recv_word_counts);

  std::vector<std::size_t> query_send_words(
      static_cast<std::size_t>(query_send_total) * words_per_determinant);
  std::vector<std::size_t> query_local_positions(query_send_total);
  std::vector<int> query_offsets = query_send_displacements;
  for(std::size_t index = 0; index < new_basis.size(); ++index) {
    const auto key = wavefunction_remap_detail::make_key(new_basis[index]);
    const int owner = wavefunction_remap_detail::owner_rank(key, communicator_size);
    const int position = query_offsets[owner]++;
    std::copy(key.begin(), key.end(),
              query_send_words.begin() +
                  static_cast<std::size_t>(position) * words_per_determinant);
    query_local_positions[position] = index;
  }

  std::vector<std::size_t> query_recv_words(
      static_cast<std::size_t>(query_recv_total) * words_per_determinant);
  MPI_Alltoallv(query_send_words.data(), query_send_word_counts.data(),
                query_send_word_displacements.data(), SBD_MPI_SIZE_T,
                query_recv_words.data(), query_recv_word_counts.data(),
                query_recv_word_displacements.data(), SBD_MPI_SIZE_T,
                communicator);

  std::vector<ElemT> response_send_values(query_recv_total, ElemT(0));
  for(int index = 0; index < query_recv_total; ++index) {
    const auto begin = query_recv_words.begin() +
        static_cast<std::size_t>(index) * words_per_determinant;
    const std::vector<std::size_t> key(
        begin, begin + words_per_determinant);
    const auto found = coefficient_by_determinant.find(key);
    if(found != coefficient_by_determinant.end()) {
      response_send_values[index] = found->second.value;
      found->second.matched = true;
    }
  }

  std::vector<ElemT> response_recv_values(query_send_total, ElemT(0));
  MPI_Alltoallv(response_send_values.data(), query_recv_counts.data(),
                query_recv_displacements.data(), sbd::GetMpiType<ElemT>::MpiT,
                response_recv_values.data(), query_send_counts.data(),
                query_send_displacements.data(), sbd::GetMpiType<ElemT>::MpiT,
                communicator);
  for(int index = 0; index < query_send_total; ++index) {
    new_coefficients[query_local_positions[index]] =
        response_recv_values[index];
  }

  std::size_t local_matched = 0;
  for(const auto& item : coefficient_by_determinant) {
    if(item.second.matched) ++local_matched;
  }
  std::size_t global_matched = 0;
  std::size_t global_old_size = 0;
  const std::size_t local_old_size = old_basis.size();
  MPI_Allreduce(&local_matched, &global_matched, 1, SBD_MPI_SIZE_T, MPI_SUM,
                communicator);
  MPI_Allreduce(&local_old_size, &global_old_size, 1, SBD_MPI_SIZE_T, MPI_SUM,
                communicator);
  if(require_old_basis_subset && global_matched != global_old_size) {
    throw std::invalid_argument("old basis is not a subset of new basis");
  }
  return global_matched;
}

}  // namespace sbd

#endif
