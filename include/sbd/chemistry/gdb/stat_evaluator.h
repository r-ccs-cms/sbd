#ifndef SBD_GDB_STAT_EVALUATOR_H
#define SBD_GDB_STAT_EVALUATOR_H

#include "sbd/chemistry/gdb/hash_owned_external_observable.h"
#include "sbd/chemistry/gdb/parent_sampling.h"
#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"
#include "sbd/framework/stat/stat_evaluator_batch.h"
#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/chemistry/basic/integrals.h"
#include "sbd/chemistry/gdb/heatbath_lookup.h"
#include "sbd/framework/determinant_distribution_round_robin.h"

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sbd {
namespace gdb {

struct EvaluatedBatch {
  stat_evaluator::BatchRecord record;
  stat_evaluator::BatchTiming timing;
  stat_evaluator::BatchProfile profile;
};

template <typename RealT>
struct StatEvaluatorOptions {
  stat_evaluator::Observable observable = stat_evaluator::Observable::both;
  std::size_t bit_length = 64;
  std::size_t norb = 0;
  std::size_t sample_count = 20000;
  RealT heatbath_cutoff = RealT(0);
  RealT reference_energy = RealT(0);
  std::uint64_t base_seed = 0;
  std::size_t expansion_batch_size = 1000000;
  RealT minimum_abs_denominator = RealT(0);
};

struct BatchRequest {
  std::string calculation_id;
  std::uint64_t batch_id = 0;
};

template <typename ElemT, typename RealT, typename BatchConsumer>
void evaluate_statistical_batches(
    sbd::det_vector<std::size_t> parents,
    std::vector<ElemT> coefficients,
    const ElemT& scalar_integral,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    const StatEvaluatorOptions<RealT>& options,
    const std::vector<BatchRequest>& batches,
    MPI_Comm communicator,
    BatchConsumer consume_batch) {
  if(batches.empty()) return;
  if(options.sample_count < 2)
    throw std::invalid_argument("statistical batch requires at least two samples");
  if(options.norb == 0 || options.bit_length == 0)
    throw std::invalid_argument("invalid statistical evaluator dimensions");
  for(const auto& batch : batches)
    if(batch.calculation_id.empty())
      throw std::invalid_argument("calculation ID is empty");
  const bool compute_variance =
      options.observable != stat_evaluator::Observable::pt2;
  const bool compute_pt2 =
      options.observable != stat_evaluator::Observable::variance;
  if(compute_pt2 && !std::isfinite(options.reference_energy))
    throw std::invalid_argument("PT2 reference energy must be finite");
  const double total_start = MPI_Wtime();

  sbd::det_vector<std::size_t> membership_basis(parents);
  const double distribution_start = MPI_Wtime();
  make_hash_owned_membership_basis(membership_basis, communicator);
  sbd::redistribute_determinants_weight_round_robin(
      parents, coefficients, communicator);
  const double preparation_distribution_seconds =
      MPI_Wtime() - distribution_start;

  const double sampling_start = MPI_Wtime();
  const auto prepared_sampling =
      prepare_parent_sampling_by_coefficient_magnitude<ElemT, RealT>(
          coefficients, communicator);
  const double preparation_sampling_seconds = MPI_Wtime() - sampling_start;

  const double lookup_start = MPI_Wtime();
  RealT max_abs_coefficient = RealT(0);
  for(const auto& coefficient : coefficients)
    max_abs_coefficient = std::max(
        max_abs_coefficient,
        static_cast<RealT>(std::abs(coefficient)));
  const detail::HeatbathLookup<ElemT, RealT> lookup(
      options.norb, two_integrals, options.heatbath_cutoff,
      max_abs_coefficient);
  const double preparation_lookup_seconds = MPI_Wtime() - lookup_start;

  int mpi_size = 1;
  MPI_Comm_size(communicator, &mpi_size);
  for(std::size_t batch_index = 0; batch_index < batches.size();
      ++batch_index) {
    const auto& batch = batches[batch_index];
    const double batch_start = batch_index == 0 ? total_start : MPI_Wtime();
    EvaluatedBatch output;
    output.profile.parents = parents.size();
    output.profile.membership = membership_basis.size();
    output.profile.lookup_entries = lookup.entry_count();
    output.profile.lookup_bytes = lookup.storage_bytes();
    if(batch_index == 0) {
      output.timing.distribution_seconds =
          preparation_distribution_seconds;
      output.timing.sampling_seconds = preparation_sampling_seconds;
      output.timing.expansion_seconds = preparation_lookup_seconds;
    }

    const double batch_sampling_start = MPI_Wtime();
    const auto counts = sample_prepared_parents(
        prepared_sampling, options.sample_count, options.base_seed,
        batch.batch_id, communicator);
    output.timing.sampling_seconds +=
        MPI_Wtime() - batch_sampling_start;

    for(const auto count : counts) {
      output.profile.draws += count;
      if(count != 0) ++output.profile.sampled_parents;
    }
    const double expansion_start = MPI_Wtime();
    sbd::det_vector<std::size_t> children;
    std::vector<Contribution<ElemT, RealT>> contributions;
    local_integral_driven_contribution_expansion_with_lookup(
        parents, coefficients, counts, prepared_sampling.probabilities,
        options.bit_length, options.norb, one_integrals, two_integrals,
        lookup, options.heatbath_cutoff, options.expansion_batch_size,
        children, contributions);
    output.timing.expansion_seconds += MPI_Wtime() - expansion_start;

    output.profile.generated_records = children.size();
    const double observable_start = MPI_Wtime();
    HashObservableTiming hash_timing;
    HashObservableCounts hash_counts;
    const auto result =
        distributed_u_statistic_external_observable<ElemT, RealT>(
            children, contributions, membership_basis,
            options.sample_count, options.bit_length, options.norb,
            scalar_integral, one_integrals, two_integrals,
            options.reference_energy, communicator, 0,
            options.minimum_abs_denominator, compute_variance,
            compute_pt2, &hash_timing, &hash_counts);
    output.profile.received_records = hash_counts.received_records;
    output.profile.unique_children = hash_counts.unique_children;
    output.profile.external_children = hash_counts.external_children;
    output.timing.hash_observable_seconds =
        MPI_Wtime() - observable_start;
    const std::array<double, 7> local_detail_times{
        hash_timing.redistribution.hash_seconds,
        hash_timing.redistribution.packing_seconds,
        hash_timing.redistribution.communication_seconds,
        hash_timing.redistribution.sorting_seconds,
        hash_timing.local_observable.child_sort_seconds,
        hash_timing.local_observable.reduction_observable_seconds,
        hash_timing.denominator_allreduce_seconds +
            hash_timing.final_allreduce_seconds};
    output.timing.hash_seconds = local_detail_times[0];
    output.timing.hash_packing_seconds = local_detail_times[1];
    output.timing.hash_communication_seconds = local_detail_times[2];
    output.timing.hash_receive_sort_seconds = local_detail_times[3];
    output.timing.child_sort_seconds = local_detail_times[4];
    output.timing.local_observable_seconds = local_detail_times[5];
    output.timing.observable_allreduce_seconds = local_detail_times[6];
    output.timing.total_seconds = MPI_Wtime() - batch_start;

    auto& record = output.record;
    record.calculation_id = batch.calculation_id;
    record.observable = options.observable;
    record.base_seed = options.base_seed;
    record.batch_id = batch.batch_id;
    record.sample_count = options.sample_count;
    record.heatbath_cutoff =
        static_cast<double>(options.heatbath_cutoff);
    record.has_reference_energy = compute_pt2;
    record.reference_energy =
        static_cast<double>(options.reference_energy);
    record.has_variance = compute_variance;
    record.variance_estimate = static_cast<double>(result.variance);
    record.has_pt2 = compute_pt2;
    record.pt2_estimate = static_cast<double>(result.pt2);
    record.mpi_size = mpi_size;
#ifdef _OPENMP
    record.omp_threads = omp_get_max_threads();
#else
    record.omp_threads = 1;
#endif
    record.elapsed_seconds = output.timing.total_seconds;
    consume_batch(output);
  }
}

template <typename ElemT, typename RealT>
EvaluatedBatch evaluate_statistical_batch(
    sbd::det_vector<std::size_t> parents,
    std::vector<ElemT> coefficients,
    const ElemT& scalar_integral,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    const StatEvaluatorOptions<RealT>& options,
    const BatchRequest& batch,
    MPI_Comm communicator) {
  EvaluatedBatch output;
  evaluate_statistical_batches(
      std::move(parents), std::move(coefficients), scalar_integral,
      one_integrals, two_integrals, options, std::vector<BatchRequest>{batch},
      communicator,
      [&](const EvaluatedBatch& evaluated) { output = evaluated; });
  return output;
}

}  // namespace gdb
}  // namespace sbd

#endif
