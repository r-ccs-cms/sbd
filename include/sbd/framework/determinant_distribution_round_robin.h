/**
 * @file sbd/framework/determinant_distribution_round_robin.h
 * @brief Coefficient-weighted round-robin determinant distribution.
 */
#ifndef SBD_FRAMEWORK_DETERMINANT_DISTRIBUTION_ROUND_ROBIN_H
#define SBD_FRAMEWORK_DETERMINANT_DISTRIBUTION_ROUND_ROBIN_H

#include "sbd/framework/sort_array.h"
#include "sbd/framework/type_def.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd {

/**
 * Redistribute determinants and their coefficients by descending |c_j| rank
 * in round-robin order. The input need not already be weight-sorted, and the
 * output is not globally bitstring-sorted. Replicated communicator groups
 * must call this independently with identical initial determinant shards.
 */
template <typename ElemT, typename DetsContainer>
void redistribute_determinants_weight_round_robin(
    DetsContainer& determinants,
    std::vector<ElemT>& coefficients,
    MPI_Comm comm) {
  if(determinants.size() != coefficients.size())
    throw std::invalid_argument("determinant and coefficient counts differ");

  int comm_size = 1;
  MPI_Comm_size(comm, &comm_size);
  if(comm_size == 1) return;

  std::vector<double> weights(coefficients.size());
  for(std::size_t index = 0; index < coefficients.size(); ++index)
    weights[index] = static_cast<double>(std::abs(coefficients[index]));
  std::vector<std::size_t> ranking;
  mpi_find_ranking(weights, ranking, comm);

  std::size_t local_words =
      determinants.empty() ? 0 : determinants[0].size();
  std::size_t determinant_words = 0;
  MPI_Allreduce(&local_words, &determinant_words, 1,
                SBD_MPI_SIZE_T, MPI_MAX, comm);
  if(determinant_words == 0) {
    determinants.clear();
    coefficients.clear();
    return;
  }

  std::vector<std::vector<std::size_t>> indices_by_destination(
      static_cast<std::size_t>(comm_size));
  for(std::size_t index = 0; index < ranking.size(); ++index)
    indices_by_destination[
        ranking[index] % static_cast<std::size_t>(comm_size)].push_back(index);

  std::vector<int> send_counts(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> receive_counts(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> send_displacements(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> receive_displacements(
      static_cast<std::size_t>(comm_size), 0);
  std::size_t total_send = 0;
  for(int destination = 0; destination < comm_size; ++destination) {
    const std::size_t count =
        indices_by_destination[static_cast<std::size_t>(destination)].size();
    if(count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       total_send >
           static_cast<std::size_t>(std::numeric_limits<int>::max()) - count)
      throw std::overflow_error(
          "round-robin determinant MPI count exceeds INT_MAX");
    send_counts[static_cast<std::size_t>(destination)] =
        static_cast<int>(count);
    send_displacements[static_cast<std::size_t>(destination)] =
        static_cast<int>(total_send);
    total_send += count;
  }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               receive_counts.data(), 1, MPI_INT, comm);
  std::size_t total_receive = 0;
  for(int source = 0; source < comm_size; ++source) {
    const int count = receive_counts[static_cast<std::size_t>(source)];
    if(count < 0 ||
       total_receive >
           static_cast<std::size_t>(std::numeric_limits<int>::max()) -
               static_cast<std::size_t>(count))
      throw std::overflow_error(
          "round-robin determinant MPI receive exceeds INT_MAX");
    receive_displacements[static_cast<std::size_t>(source)] =
        static_cast<int>(total_receive);
    total_receive += static_cast<std::size_t>(count);
  }

  std::vector<std::size_t> send_determinants(
      total_send * determinant_words);
  std::vector<ElemT> send_coefficients(total_send);
  for(int destination = 0; destination < comm_size; ++destination) {
    std::size_t output = static_cast<std::size_t>(
        send_displacements[static_cast<std::size_t>(destination)]);
    for(const std::size_t input :
        indices_by_destination[static_cast<std::size_t>(destination)]) {
      std::copy(determinants[input].begin(), determinants[input].end(),
                send_determinants.begin() + output * determinant_words);
      send_coefficients[output] = coefficients[input];
      ++output;
    }
  }

  std::vector<int> send_word_counts(static_cast<std::size_t>(comm_size), 0);
  std::vector<int> receive_word_counts(
      static_cast<std::size_t>(comm_size), 0);
  std::vector<int> send_word_displacements(
      static_cast<std::size_t>(comm_size), 0);
  std::vector<int> receive_word_displacements(
      static_cast<std::size_t>(comm_size), 0);
  for(int rank = 0; rank < comm_size; ++rank) {
    const std::size_t send_words =
        static_cast<std::size_t>(send_counts[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t receive_words =
        static_cast<std::size_t>(
            receive_counts[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t send_displacement =
        static_cast<std::size_t>(
            send_displacements[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t receive_displacement =
        static_cast<std::size_t>(
            receive_displacements[static_cast<std::size_t>(rank)]) *
        determinant_words;
    if(send_words > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       receive_words >
           static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       send_displacement >
           static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       receive_displacement >
           static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error(
          "round-robin determinant MPI word count exceeds INT_MAX");
    send_word_counts[static_cast<std::size_t>(rank)] =
        static_cast<int>(send_words);
    receive_word_counts[static_cast<std::size_t>(rank)] =
        static_cast<int>(receive_words);
    send_word_displacements[static_cast<std::size_t>(rank)] =
        static_cast<int>(send_displacement);
    receive_word_displacements[static_cast<std::size_t>(rank)] =
        static_cast<int>(receive_displacement);
  }

  std::vector<std::size_t> receive_determinants(
      total_receive * determinant_words);
  std::vector<ElemT> receive_coefficients(total_receive);
  MPI_Alltoallv(send_determinants.data(), send_word_counts.data(),
                send_word_displacements.data(), SBD_MPI_SIZE_T,
                receive_determinants.data(), receive_word_counts.data(),
                receive_word_displacements.data(), SBD_MPI_SIZE_T, comm);
  MPI_Alltoallv(send_coefficients.data(), send_counts.data(),
                send_displacements.data(), GetMpiType<ElemT>::MpiT,
                receive_coefficients.data(), receive_counts.data(),
                receive_displacements.data(), GetMpiType<ElemT>::MpiT, comm);

  DetsContainer redistributed(total_receive);
  for(std::size_t index = 0; index < total_receive; ++index)
    std::copy(receive_determinants.begin() + index * determinant_words,
              receive_determinants.begin() +
                  (index + 1) * determinant_words,
              redistributed[index].begin());
  determinants = std::move(redistributed);
  coefficients = std::move(receive_coefficients);
}

} // namespace sbd

#endif
