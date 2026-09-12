// Adapted from public SBD 93ebabec6c84; CAOP diagonal adapter.
#ifndef SBD_CAOP_STAT_HASH_OWNED_EXTERNAL_OBSERVABLE_H
#define SBD_CAOP_STAT_HASH_OWNED_EXTERNAL_OBSERVABLE_H

#include "sbd/caop/stat/u_statistic_external_observable.h"
#include "sbd/framework/murmurhash_basis_distribution.h"
#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"


#include "sbd/framework/mpi_utility.h"
#include <mpi.h>
#include <algorithm>
#include <map>
#include <utility>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <cmath>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sbd::caop::stat {

struct HashObservableCounts {
  std::size_t received_records = 0;
  std::size_t unique_children = 0;
  std::size_t external_children = 0;
};

struct HashObservableTiming {
  murmur_basis::RedistributionTiming redistribution;
  LocalObservableTiming local_observable;
  double denominator_allreduce_seconds = 0.0;
  double final_allreduce_seconds = 0.0;
};

inline void make_hash_owned_membership_basis(
    sbd::det_vector<std::size_t>& variational_basis,
    MPI_Comm communicator,
    std::uint32_t hash_seed = 0) {
  murmur_basis::redistribute_unique_determinants_by_hash(
      variational_basis, communicator, hash_seed);
}

// Ring the owner-local external queries through Hamiltonian shards. At most one
// other owner's query block is held at a time; no global child gather or full H.
template<class ElemT>
std::map<std::vector<std::size_t>,ElemT> distributed_diagonals(
    const sbd::det_vector<std::size_t>& children,
    const sbd::det_vector<std::size_t>& membership,
    const sbd::GeneralOp<ElemT>& h, std::size_t bits, MPI_Comm h_comm) {
  auto query = children;
  sbd::sort_bitarray(query);
  sbd::det_vector<std::size_t> external;
  for(const auto& child: query)
    if(!std::binary_search(membership.begin(),membership.end(),child,
        [](const auto& a,const auto& b){return sbd::less_from_back(a,b);}))
      external.push_back(child);
  query = std::move(external);
  std::vector<ElemT> values(query.size(), ElemT{});
  int h_size; MPI_Comm_size(h_comm,&h_size);
  for(int step=0; step<h_size; ++step) {
#pragma omp parallel for schedule(static)
    for(std::size_t i=0;i<query.size();++i)
      values[i] += diagonal_element(query[i],bits,h);
    if(h_size>1) {
      sbd::det_vector<std::size_t> next;
      std::vector<ElemT> next_values;
      sbd::MpiSlide(query,next,1,h_comm);
      sbd::MpiSlide(values,next_values,1,h_comm);
      query=std::move(next); values=std::move(next_values);
    }
  }
  std::map<std::vector<std::size_t>,ElemT> result;
  for(std::size_t i=0;i<query.size();++i)
    result.emplace(std::vector<std::size_t>(query[i].begin(),query[i].end()),values[i]);
  return result;
}

// Move sampled child records to the same MurmurHash3 owners as the
// variational basis, reduce/exclude locally, then sum the observable globally.
template <typename ElemT, typename RealT, typename RecordT>
UStatisticExternalObservable<RealT>
distributed_u_statistic_external_observable(
    sbd::det_vector<std::size_t>& children,
    std::vector<RecordT>& contributions,
    const sbd::det_vector<std::size_t>& hash_owned_variational_basis,
    std::size_t sample_count,
    std::size_t bit_length,
    const sbd::GeneralOp<ElemT>& hamiltonian,
    RealT reference_energy,
    MPI_Comm communicator,
    std::uint32_t hash_seed = 0,
    RealT minimum_abs_denominator = RealT(0),
    bool compute_variance = true,
    bool compute_pt2 = true,
    HashObservableTiming* timing = nullptr,
    HashObservableCounts* counts = nullptr,
    MPI_Comm h_comm = MPI_COMM_SELF, RealT cutoff = RealT(0)) {
  if(counts != nullptr) *counts = HashObservableCounts{};
  if(timing != nullptr) *timing = HashObservableTiming{};
  murmur_basis::redistribute_determinant_values_by_hash(
      children, contributions, communicator, hash_seed,
      timing == nullptr ? nullptr : &timing->redistribution);
  const std::size_t received_records = children.size();
  const double merge_start = MPI_Wtime();
  std::vector<Contribution<ElemT,RealT>> merged;
  const std::vector<Contribution<ElemT,RealT>>* final_records = nullptr;
  if constexpr(std::is_same_v<RecordT,Contribution<ElemT,RealT>>) {
    final_records = &contributions;
  } else {
    // Sorting by child AND parent preserves same-parent cross-shard interference.
    std::vector<std::size_t> order(children.size());
    std::iota(order.begin(),order.end(),std::size_t(0));
    std::sort(order.begin(),order.end(),[&](auto a,auto b) {
      if(children[a]!=children[b]) return sbd::less_from_back(children[a],children[b]);
      return std::tie(contributions[a].parent_rank,contributions[a].parent_index) <
             std::tie(contributions[b].parent_rank,contributions[b].parent_index);
    });
    sbd::det_vector<std::size_t> reduced;
    for(std::size_t start=0; start<order.size();) {
      const auto idx=order[start]; const auto& first=contributions[idx];
      ElemT x{}; std::size_t end=start;
      while(end<order.size()) {
        const auto j=order[end]; const auto& r=contributions[j];
        if(children[j]!=children[idx] || r.parent_rank!=first.parent_rank ||
           r.parent_index!=first.parent_index) break;
        x+=r.weighted_amplitude; ++end;
      }
      // x = w*c*H/p, now summed over ALL Hamiltonian shards.
      if(std::abs(x)>0 && std::abs(x)*first.probability/first.count>=cutoff) {
        reduced.push_back(children[idx]);
        merged.push_back({x,static_cast<RealT>(sbd::SquaredNorm(x))/first.count});
      }
      start=end;
    }
    children=std::move(reduced);
    // Release raw shard records before PT2 queries and local evaluation.
    std::vector<RecordT>().swap(contributions);
    final_records=&merged;
  }
  const double merge_seconds = MPI_Wtime()-merge_start;
  int h_size; MPI_Comm_size(h_comm,&h_size);
  const double diagonal_start = MPI_Wtime();
  std::map<std::vector<std::size_t>,ElemT> diagonals;
  if(compute_pt2 && h_size>1)
    diagonals=distributed_diagonals(children,hash_owned_variational_basis,
                                   hamiltonian,bit_length,h_comm);
  const double diagonal_seconds = MPI_Wtime()-diagonal_start;
  bool local_denominator_error = false;
  const auto local =
      u_statistic_external_observable_sorted_basis<ElemT, RealT>(
      children, *final_records, hash_owned_variational_basis,
      sample_count, bit_length, hamiltonian, reference_energy,
      minimum_abs_denominator, compute_variance, compute_pt2,
      &local_denominator_error,
      timing == nullptr ? nullptr : &timing->local_observable,
      compute_pt2 && h_size>1 ? &diagonals : nullptr);
  if(timing) timing->local_observable.reduction_observable_seconds +=
      merge_seconds + diagonal_seconds;

  if(counts != nullptr) {
    counts->received_records = received_records;
    counts->unique_children = local.unique_children;
    counts->external_children = local.external_determinants;
  }

  int local_error = local_denominator_error ? 1 : 0;
  int global_error = 0;
  double phase_start = MPI_Wtime();
  MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX,
                communicator);
  if(timing != nullptr)
    timing->denominator_allreduce_seconds = MPI_Wtime() - phase_start;
  if(global_error != 0)
    throw std::domain_error("external PT2 denominator is too small");

  UStatisticExternalObservable<RealT> global;
  phase_start = MPI_Wtime();
  MPI_Allreduce(&local.variance, &global.variance, 1,
                sbd::GetMpiType<RealT>::MpiT, MPI_SUM, communicator);
  MPI_Allreduce(&local.pt2, &global.pt2, 1,
                sbd::GetMpiType<RealT>::MpiT, MPI_SUM, communicator);
  MPI_Allreduce(&local.external_determinants,
                &global.external_determinants, 1,
                SBD_MPI_SIZE_T, MPI_SUM, communicator);
  MPI_Allreduce(&local.unique_children, &global.unique_children, 1,
                SBD_MPI_SIZE_T, MPI_SUM, communicator);
  if(timing != nullptr)
    timing->final_allreduce_seconds = MPI_Wtime() - phase_start;
  return global;
}

}  // namespace sbd::caop::stat

#endif
