/**
 * @file sbd/caop/basic/expansion.h
 * @brief Deterministic Heatbath expansion for CAOP Hamiltonians.
 */
#ifndef SBD_CAOP_BASIC_EXPANSION_H
#define SBD_CAOP_BASIC_EXPANSION_H

#include "sbd/caop/basic/basis.h"
#include "sbd/caop/basic/generalop_flatdata.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
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

struct HeatbathExpansionStats {
  std::size_t parents = 0;
  std::size_t terms_visited = 0;
  std::size_t mask_rejections = 0;
  std::size_t accepted_before_unique = 0;
};

struct HeatbathExpansionProfile {
  double generation_seconds = 0.0;
  double global_sort_unique_seconds = 0.0;
  double redistribution_seconds = 0.0;
};


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
 * flat_data.coefficients must contain non-negative coefficient magnitudes in
 * descending order. Flat data retaining signed or complex coefficients for
 * other algorithms must not be passed to this function.
 */
template <typename ElemT, typename RealT, typename DetsContainer>
void HeatbathExpansion(
    const DetsContainer& parents,
    const std::vector<ElemT>& coefficients,
    const ::sbd::GeneralOpFlatData<RealT>& flat_data,
    RealT cutoff, std::size_t max_batch_size,
    DetsContainer& candidates,
    MPI_Comm h_comm,
    MPI_Comm t_comm,
    MPI_Comm comm,
    HeatbathExpansionStats* stats = nullptr,
    HeatbathExpansionProfile* profile = nullptr) {
  if(parents.size() != coefficients.size())
    throw std::invalid_argument("parent and coefficient counts differ");
  if(flat_data.creation_masks.size() != flat_data.coefficients.size() ||
     flat_data.annihilation_masks.size() != flat_data.coefficients.size())
    throw std::invalid_argument("GeneralOp flat data array sizes differ");
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

  HeatbathExpansionStats local_stats;
  if(h_rank == 0) local_stats.parents = end - begin;
  const double generation_start = profile == nullptr ? 0.0 : MPI_Wtime();

  candidates.clear();
  if(h_rank == 0)
    candidates.insert(candidates.end(), parents.begin() + begin,
                      parents.begin() + end);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    DetsContainer thread_candidates;
    HeatbathExpansionStats thread_stats;
    std::vector<std::size_t> bra;
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
          term_index < flat_data.coefficients.size(); ++term_index) {
        if(!(flat_data.coefficients[term_index] > threshold)) break;
        if(stats != nullptr) ++thread_stats.terms_visited;
        if(::sbd::TryGenerateBraDetFromGeneralOpFlatTerm(
               parents[parent_index],flat_data,term_index,bra)) {
          thread_candidates.push_back(bra);
          if(stats != nullptr) ++thread_stats.accepted_before_unique;
          if(thread_candidates.size() == local_batch_size) flush();
        } else if(stats != nullptr) {
          ++thread_stats.mask_rejections;
        }
      }
    }
    flush();
    if(stats != nullptr) {
#ifdef _OPENMP
#pragma omp critical(sbd_caop_heatbath_stats)
#endif
      {
        local_stats.terms_visited += thread_stats.terms_visited;
        local_stats.mask_rejections += thread_stats.mask_rejections;
        local_stats.accepted_before_unique +=
            thread_stats.accepted_before_unique;
      }
    }
  }

  double local_profile[3] = {};
  if(profile != nullptr)
    local_profile[0] = MPI_Wtime() - generation_start;
  const double sort_start = profile == nullptr ? 0.0 : MPI_Wtime();
  ::sbd::sort_global_bitarray(candidates, comm);
  if(profile != nullptr)
    local_profile[1] = MPI_Wtime() - sort_start;
  const double redistribution_start =
      profile == nullptr ? 0.0 : MPI_Wtime();
  ::sbd::redistribution_bitarray(candidates, comm);
  if(profile != nullptr) {
    local_profile[2] = MPI_Wtime() - redistribution_start;
    double maximum_profile[3] = {};
    MPI_Allreduce(local_profile, maximum_profile, 3, MPI_DOUBLE, MPI_MAX, comm);
    profile->generation_seconds = maximum_profile[0];
    profile->global_sort_unique_seconds = maximum_profile[1];
    profile->redistribution_seconds = maximum_profile[2];
  }
  if(stats != nullptr) {
    const unsigned long long local_counts[4] = {
        static_cast<unsigned long long>(local_stats.parents),
        static_cast<unsigned long long>(local_stats.terms_visited),
        static_cast<unsigned long long>(local_stats.mask_rejections),
        static_cast<unsigned long long>(local_stats.accepted_before_unique)};
    unsigned long long global_counts[4] = {};
    MPI_Allreduce(local_counts, global_counts, 4, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);
    stats->parents = static_cast<std::size_t>(global_counts[0]);
    stats->terms_visited = static_cast<std::size_t>(global_counts[1]);
    stats->mask_rejections = static_cast<std::size_t>(global_counts[2]);
    stats->accepted_before_unique =
        static_cast<std::size_t>(global_counts[3]);
  }
}


} // namespace caop
} // namespace sbd

#endif
