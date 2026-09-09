#ifndef SBD_GDB_PARENT_SAMPLING_H
#define SBD_GDB_PARENT_SAMPLING_H

#include "sbd/framework/distributed_alias_sampling.h"
#include "sbd/framework/type_def.h"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sbd {
namespace gdb {

template <typename RealT>
struct ParentSample {
  std::vector<RealT> probabilities;
  std::vector<std::size_t> counts;
  RealT global_weight_sum{};
};

template <typename RealT>
struct PreparedParentSampling {
  std::vector<RealT> probabilities;
  distributed_sampling::DistributedAliasTables alias_tables;
  RealT global_weight_sum{};
};

template <typename ElemT, typename RealT>
std::vector<RealT> coefficient_sampling_probabilities(
    const std::vector<ElemT>& coefficients,
    MPI_Comm communicator,
    RealT* global_weight_sum = nullptr) {
  std::vector<RealT> weights(coefficients.size());
  RealT local_sum{};
  int local_invalid = 0;
  for(std::size_t index = 0; index < coefficients.size(); ++index) {
    const RealT weight = static_cast<RealT>(std::abs(coefficients[index]));
    if(weight < RealT(0) || !std::isfinite(weight)) local_invalid = 1;
    weights[index] = weight;
    local_sum += weight;
  }
  if(!std::isfinite(local_sum)) local_invalid = 1;
  int any_invalid = 0;
  MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
                communicator);
  if(any_invalid)
    throw std::invalid_argument("coefficient sampling weights are invalid");

  RealT total{};
  MPI_Allreduce(&local_sum, &total, 1, sbd::GetMpiType<RealT>::MpiT,
                MPI_SUM, communicator);
  if(!(total > RealT(0)) || !std::isfinite(total))
    throw std::invalid_argument(
        "coefficient sampling requires positive global weight");
  for(RealT& weight : weights) weight /= total;
  if(global_weight_sum != nullptr) *global_weight_sum = total;
  return weights;
}

template <typename ElemT, typename RealT>
PreparedParentSampling<RealT>
prepare_parent_sampling_by_coefficient_magnitude(
    const std::vector<ElemT>& coefficients,
    MPI_Comm communicator,
    int root = 0) {
  PreparedParentSampling<RealT> result;
  result.probabilities = coefficient_sampling_probabilities<ElemT, RealT>(
      coefficients, communicator, &result.global_weight_sum);
  const std::vector<double> alias_weights(result.probabilities.begin(),
                                          result.probabilities.end());
  result.alias_tables =
      distributed_sampling::prepare_distributed_alias_tables(
          alias_weights, communicator, root);
  return result;
}

template <typename RealT>
std::vector<std::size_t> sample_prepared_parents(
    const PreparedParentSampling<RealT>& prepared,
    std::size_t global_draw_count,
    std::uint64_t base_seed,
    std::uint64_t batch_id,
    MPI_Comm communicator) {
  return distributed_sampling::distributed_alias_sample_counts(
      prepared.alias_tables, global_draw_count, base_seed, batch_id,
      communicator);
}

template <typename ElemT, typename RealT>
ParentSample<RealT> sample_parents_by_coefficient_magnitude(
    const std::vector<ElemT>& coefficients,
    std::size_t global_draw_count,
    std::uint64_t base_seed,
    std::uint64_t batch_id,
    MPI_Comm communicator,
    int root = 0) {
  const auto prepared =
      prepare_parent_sampling_by_coefficient_magnitude<ElemT, RealT>(
          coefficients, communicator, root);
  ParentSample<RealT> result;
  result.probabilities = prepared.probabilities;
  result.global_weight_sum = prepared.global_weight_sum;
  result.counts = sample_prepared_parents(
      prepared, global_draw_count, base_seed, batch_id, communicator);
  return result;
}

}  // namespace gdb
}  // namespace sbd

#endif
