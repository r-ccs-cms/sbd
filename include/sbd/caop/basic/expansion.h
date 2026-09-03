/**
 * @file sbd/caop/basic/expansion.h
 * @brief Deterministic Heatbath expansion for CAOP Hamiltonians.
 */
#ifndef SBD_CAOP_BASIC_EXPANSION_H
#define SBD_CAOP_BASIC_EXPANSION_H

#include "sbd/caop/basic/basis.h"
#include "sbd/caop/basic/generalop.h"
#include "sbd/framework/sort_array.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd {
namespace caop {

namespace detail {
template <typename DetsContainer>
void append_unique(DetsContainer& destination, DetsContainer source) {
  if(source.empty()) return;
  ::sbd::sort_unique_local_bitarray(source);
  if(destination.empty()) {
    destination = std::move(source);
    return;
  }
  ::sbd::sort_unique_local_bitarray(destination);
  const auto less = [](const auto& lhs, const auto& rhs) {
    return ::sbd::less_from_back(lhs, rhs);
  };
  DetsContainer merged;
  merged.resize(destination.size() + source.size());
  std::merge(std::make_move_iterator(destination.begin()),
             std::make_move_iterator(destination.end()),
             std::make_move_iterator(source.begin()),
             std::make_move_iterator(source.end()), merged.begin(), less);
  merged.resize(std::unique(merged.begin(), merged.end()) - merged.begin());
  destination = std::move(merged);
}

template <typename DetsContainer>
void finalize_distributed_layout(DetsContainer& candidates, MPI_Comm comm) {
  ::sbd::sort_global_bitarray(candidates, comm);
  ::sbd::redistribution_bitarray(candidates, comm);
}

} // namespace detail


/** Retain local parents satisfying |c_j|^2 > threshold. */
template <typename ElemT, typename RealT, typename DetsContainer>
void TruncateHeatbathParents(DetsContainer& parents,
                             std::vector<ElemT>& coefficients,
                             RealT threshold) {
  if(parents.size() != coefficients.size())
    throw std::invalid_argument("parent and coefficient counts differ");
  if(threshold < RealT(0))
    throw std::invalid_argument("heatbath truncation must be non-negative");
  std::size_t output = 0;
  for(std::size_t input = 0; input < coefficients.size(); ++input) {
    const RealT magnitude = std::abs(coefficients[input]);
    if(!(threshold < magnitude * magnitude)) continue;
    if(output != input) {
      parents[output] = std::move(parents[input]);
      coefficients[output] = std::move(coefficients[input]);
    }
    ++output;
  }
  parents.resize(output);
  coefficients.resize(output);
}


/**
 * Redistribute a native CAOP parent shard over b_comm by descending |c_j|
 * rank in round-robin order.  Determinants and coefficients move together.
 * The input need not be weight-sorted; mpi_find_ranking performs a distributed
 * ranking without gathering the full basis.  Call this independently in every
 * h/t replica group, whose initial b shards must be identical.
 *
 * This layout is intended for expansion only.  It is not the globally
 * bitstring-sorted native layout required by CAOP diagonalization.
 */
template <typename ElemT, typename DetsContainer>
void RedistributeHeatbathParents(
    DetsContainer& parents,
    std::vector<ElemT>& coefficients,
    MPI_Comm b_comm) {
  if(parents.size() != coefficients.size())
    throw std::invalid_argument("parent and coefficient counts differ");

  int b_size = 1;
  MPI_Comm_size(b_comm, &b_size);
  if(b_size == 1) return;

  std::vector<double> weights(coefficients.size());
  for(std::size_t index = 0; index < coefficients.size(); ++index)
    weights[index] = static_cast<double>(std::abs(coefficients[index]));
  std::vector<std::size_t> ranking;
  ::sbd::mpi_find_ranking(weights, ranking, b_comm);

  std::size_t local_words = parents.empty() ? 0 : parents[0].size();
  std::size_t determinant_words = 0;
  MPI_Allreduce(&local_words, &determinant_words, 1,
                SBD_MPI_SIZE_T, MPI_MAX, b_comm);
  if(determinant_words == 0) {
    parents.clear();
    coefficients.clear();
    return;
  }

  std::vector<std::vector<std::size_t>> indices_by_destination(
      static_cast<std::size_t>(b_size));
  for(std::size_t index = 0; index < ranking.size(); ++index)
    indices_by_destination[ranking[index] % static_cast<std::size_t>(b_size)]
        .push_back(index);

  std::vector<int> send_counts(static_cast<std::size_t>(b_size), 0);
  std::vector<int> receive_counts(static_cast<std::size_t>(b_size), 0);
  std::vector<int> send_displacements(static_cast<std::size_t>(b_size), 0);
  std::vector<int> receive_displacements(static_cast<std::size_t>(b_size), 0);
  std::size_t total_send = 0;
  for(int destination = 0; destination < b_size; ++destination) {
    const std::size_t count =
        indices_by_destination[static_cast<std::size_t>(destination)].size();
    if(count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       total_send > static_cast<std::size_t>(std::numeric_limits<int>::max()) - count)
      throw std::overflow_error("parent round-robin MPI count exceeds INT_MAX");
    send_counts[static_cast<std::size_t>(destination)] =
        static_cast<int>(count);
    send_displacements[static_cast<std::size_t>(destination)] =
        static_cast<int>(total_send);
    total_send += count;
  }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               receive_counts.data(), 1, MPI_INT, b_comm);
  std::size_t total_receive = 0;
  for(int source = 0; source < b_size; ++source) {
    const int count = receive_counts[static_cast<std::size_t>(source)];
    if(count < 0 ||
       total_receive > static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                           static_cast<std::size_t>(count))
      throw std::overflow_error("parent round-robin MPI receive exceeds INT_MAX");
    receive_displacements[static_cast<std::size_t>(source)] =
        static_cast<int>(total_receive);
    total_receive += static_cast<std::size_t>(count);
  }

  std::vector<std::size_t> send_determinants(total_send * determinant_words);
  std::vector<ElemT> send_coefficients(total_send);
  for(int destination = 0; destination < b_size; ++destination) {
    std::size_t output = static_cast<std::size_t>(
        send_displacements[static_cast<std::size_t>(destination)]);
    for(const std::size_t input :
        indices_by_destination[static_cast<std::size_t>(destination)]) {
      std::copy(parents[input].begin(), parents[input].end(),
                send_determinants.begin() + output * determinant_words);
      send_coefficients[output] = coefficients[input];
      ++output;
    }
  }

  std::vector<int> send_word_counts(static_cast<std::size_t>(b_size), 0);
  std::vector<int> receive_word_counts(static_cast<std::size_t>(b_size), 0);
  std::vector<int> send_word_displacements(static_cast<std::size_t>(b_size), 0);
  std::vector<int> receive_word_displacements(static_cast<std::size_t>(b_size), 0);
  for(int rank = 0; rank < b_size; ++rank) {
    const std::size_t send_words =
        static_cast<std::size_t>(send_counts[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t receive_words =
        static_cast<std::size_t>(receive_counts[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t send_displacement =
        static_cast<std::size_t>(send_displacements[static_cast<std::size_t>(rank)]) *
        determinant_words;
    const std::size_t receive_displacement =
        static_cast<std::size_t>(receive_displacements[static_cast<std::size_t>(rank)]) *
        determinant_words;
    if(send_words > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       receive_words > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       send_displacement > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       receive_displacement > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("parent determinant MPI word count exceeds INT_MAX");
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
                receive_word_displacements.data(), SBD_MPI_SIZE_T, b_comm);
  MPI_Alltoallv(send_coefficients.data(), send_counts.data(),
                send_displacements.data(), ::sbd::GetMpiType<ElemT>::MpiT,
                receive_coefficients.data(), receive_counts.data(),
                receive_displacements.data(), ::sbd::GetMpiType<ElemT>::MpiT,
                b_comm);

  DetsContainer redistributed(total_receive);
  for(std::size_t index = 0; index < total_receive; ++index)
    std::copy(receive_determinants.begin() + index * determinant_words,
              receive_determinants.begin() + (index + 1) * determinant_words,
              redistributed[index].begin());
  parents = std::move(redistributed);
  coefficients = std::move(receive_coefficients);
}

namespace detail {
template <typename ElemT>
using magnitude_type = decltype(std::abs(ElemT{}));
} // namespace detail

/**
 * Flat data for GeneralOp's non-diagonal terms, ordered by decreasing
 * magnitude of the original Hamiltonian coefficient.
 *
 * CoeffT is deliberately selected by the caller.  Deterministic Heatbath
 * stores coefficient magnitudes as RealT, while later PT2/variance code may
 * use the same layout with the signed or complex ElemT coefficients intact.
 */
template <typename CoeffT>
struct HeatbathLookup {
  std::vector<CoeffT> coefficients;
  ::sbd::det_vector<std::size_t> creation_masks;
  ::sbd::det_vector<std::size_t> annihilation_masks;
  std::vector<int> orbitals;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> creation_counts;
};

template <typename LookupCoeffT, typename ElemT, typename Transform>
HeatbathLookup<LookupCoeffT> MakeHeatbathLookup(
    const ::sbd::GeneralOp<ElemT>& hamiltonian,
    std::size_t determinant_words,
    std::size_t bit_length,
    Transform transform_coefficient) {
  using MagnitudeT = detail::magnitude_type<ElemT>;
  if(determinant_words == 0 || bit_length == 0 ||
     bit_length > 8 * sizeof(std::size_t))
    throw std::invalid_argument("invalid determinant dimensions");

  std::vector<std::size_t> order;
  order.reserve(hamiltonian.NumOpTerms());
  for(std::size_t index = 0; index < hamiltonian.NumOpTerms(); ++index)
    if(std::abs(hamiltonian.OpCoef(static_cast<int>(index))) >
       MagnitudeT(0))
      order.push_back(index);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                            std::size_t rhs) {
    const auto lhs_coefficient =
        std::abs(hamiltonian.OpCoef(static_cast<int>(lhs)));
    const auto rhs_coefficient =
        std::abs(hamiltonian.OpCoef(static_cast<int>(rhs)));
    if(lhs_coefficient != rhs_coefficient)
      return lhs_coefficient > rhs_coefficient;
    return lhs < rhs;
  });

  HeatbathLookup<LookupCoeffT> lookup;
  lookup.coefficients.reserve(order.size());
  const std::vector<std::size_t> zero_mask(determinant_words, 0);
  lookup.creation_masks =
      ::sbd::det_vector<std::size_t>(order.size(), zero_mask);
  lookup.annihilation_masks =
      ::sbd::det_vector<std::size_t>(order.size(), zero_mask);
  lookup.offsets.reserve(order.size() + 1);
  lookup.creation_counts.reserve(order.size());
  lookup.offsets.push_back(0);

  for(std::size_t output = 0; output < order.size(); ++output) {
    const std::size_t input = order[output];
    const ::sbd::ProductOp product =
        hamiltonian.OpTerm(static_cast<int>(input));
    const std::size_t creation_count =
        static_cast<std::size_t>(product.n_dag());
    for(std::size_t op_index = 0; op_index < product.size(); ++op_index) {
      const ::sbd::CAOp op = product.FOp(op_index);
      if(op.d() != (op_index < creation_count))
        throw std::invalid_argument(
            "CAOP Heat-Bath lookup requires normal-ordered ProductOp terms");
      if(op.q() < 0) throw std::invalid_argument("negative CAOP site index");
      const std::size_t site = static_cast<std::size_t>(op.q());
      const std::size_t word = site / bit_length;
      if(word >= determinant_words)
        throw std::invalid_argument("CAOP site exceeds determinant storage");
      const std::size_t mask = std::size_t(1) << (site % bit_length);
      (op.d() ? lookup.creation_masks[output][word]
              : lookup.annihilation_masks[output][word]) |= mask;
      lookup.orbitals.push_back(op.q());
    }
    lookup.coefficients.push_back(transform_coefficient(
        hamiltonian.OpCoef(static_cast<int>(input))));
    lookup.creation_counts.push_back(creation_count);
    lookup.offsets.push_back(lookup.orbitals.size());
  }
  return lookup;
}

template <typename Determinant, typename LookupCoeffT>
bool ApplyHeatbathTerm(const Determinant& parent,
                       const HeatbathLookup<LookupCoeffT>& lookup,
                       std::size_t term,
                       std::vector<std::size_t>& child) {
  const auto& creation = lookup.creation_masks[term];
  const auto& annihilation = lookup.annihilation_masks[term];
  if(parent.size() != creation.size() || parent.size() != annihilation.size())
    throw std::invalid_argument(
        "parent determinant word count differs from lookup");
  for(std::size_t word = 0; word < parent.size(); ++word) {
    if((parent[word] & annihilation[word]) != annihilation[word]) return false;
    if((parent[word] & (creation[word] & ~annihilation[word])) != 0)
      return false;
  }
  child.assign(parent.begin(), parent.end());
  for(std::size_t word = 0; word < child.size(); ++word) {
    child[word] &= ~annihilation[word];
    child[word] |= creation[word];
  }
  return true;
}

/**
 * Generate deterministic CAOP Heatbath carryover candidates.
 *
 * Every supplied parent is retained independently of the cutoff.  New
 * candidates are generated only from GeneralOp's non-diagonal terms and are
 * accepted when |a_k c_j| is strictly greater than cutoff.  h_comm prevents
 * replicated parents from being inserted once per Hamiltonian shard, while
 * t_comm partitions the local parent shard.  The result is globally unique
 * and distributed over comm.
 *
 * lookup.coefficients must contain non-negative coefficient magnitudes in
 * descending order.  A lookup retaining signed or complex coefficients for
 * other algorithms must not be passed to this function.
 */
template <typename ElemT, typename RealT, typename DetsContainer>
void HeatbathExpansion(
    const DetsContainer& parents,
    const std::vector<ElemT>& coefficients,
    const HeatbathLookup<RealT>& lookup,
    RealT cutoff, std::size_t max_batch_size,
    DetsContainer& candidates,
    MPI_Comm h_comm,
    MPI_Comm t_comm,
    MPI_Comm comm) {
  if(parents.size() != coefficients.size())
    throw std::invalid_argument("parent and coefficient counts differ");
  if(lookup.creation_masks.size() != lookup.coefficients.size() ||
     lookup.annihilation_masks.size() != lookup.coefficients.size())
    throw std::invalid_argument("heatbath lookup array sizes differ");
  if(cutoff < RealT(0)) throw std::invalid_argument("negative D-HB cutoff");
  if(max_batch_size == 0)
    throw std::invalid_argument("D-HB batch size must be nonzero");

  int h_rank = 0;
  int t_rank = 0;
  int t_size = 1;
  MPI_Comm_rank(h_comm, &h_rank);
  MPI_Comm_rank(t_comm, &t_rank);
  MPI_Comm_size(t_comm, &t_size);

  std::size_t begin = 0;
  std::size_t end = parents.size();
  ::sbd::get_mpi_range(t_size, t_rank, begin, end);

  candidates.clear();
  if(h_rank == 0)
    candidates.insert(candidates.end(), parents.begin() + begin,
                      parents.begin() + end);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    DetsContainer thread_candidates;
    std::vector<std::size_t> child;
    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_num_threads();
#endif
    const std::size_t local_batch_size = std::max<std::size_t>(
        1, max_batch_size / static_cast<std::size_t>(thread_count));
    thread_candidates.reserve(local_batch_size);
    auto flush = [&]() {
      if(thread_candidates.empty()) return;
#ifdef _OPENMP
#pragma omp critical(sbd_caop_heatbath_merge)
#endif
      { detail::append_unique(candidates, std::move(thread_candidates)); }
      thread_candidates.clear();
      thread_candidates.reserve(local_batch_size);
    };
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(std::size_t parent_index = begin; parent_index < end; ++parent_index) {
      const RealT absolute_parent = std::abs(coefficients[parent_index]);
      if(!(absolute_parent > RealT(0))) continue;
      const RealT threshold = cutoff / absolute_parent;
      for(std::size_t term_index = 0;
          term_index < lookup.coefficients.size(); ++term_index) {
        if(!(lookup.coefficients[term_index] > threshold)) break;
        if(ApplyHeatbathTerm(
               parents[parent_index],lookup,term_index,child)) {
          thread_candidates.push_back(child);
          if(thread_candidates.size() == local_batch_size) flush();
        }
      }
    }
    flush();
  }

  detail::finalize_distributed_layout(candidates, comm);
}


} // namespace caop
} // namespace sbd

#endif
