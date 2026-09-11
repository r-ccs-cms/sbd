#ifndef SBD_DISTRIBUTED_ALIAS_SAMPLING_H
#define SBD_DISTRIBUTED_ALIAS_SAMPLING_H

#include "sbd/framework/type_def.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sbd {
namespace distributed_sampling {

namespace detail {

inline std::uint64_t mix64(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

inline double unit_random(std::uint64_t seed, std::uint64_t stream) {
  const std::uint64_t bits = mix64(seed ^ mix64(stream + 1));
  return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

}  // namespace detail

class AliasTable {
 public:
  AliasTable() = default;
  explicit AliasTable(const std::vector<double>& weights) { build(weights); }

  void build(const std::vector<double>& weights) {
    if(weights.empty())
      throw std::invalid_argument("empty Alias weights");
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if(!(total > 0.0) || !std::isfinite(total))
      throw std::invalid_argument("Alias weights must have positive finite sum");

    const std::size_t size = weights.size();
    probability_.assign(size, 1.0);
    alias_.resize(size);
    std::vector<double> scaled(size);
    std::vector<std::size_t> small;
    std::vector<std::size_t> large;
    small.reserve(size);
    large.reserve(size);
    for(std::size_t index = 0; index < size; ++index) {
      if(weights[index] < 0.0 || !std::isfinite(weights[index]))
        throw std::invalid_argument("Alias weights must be finite and nonnegative");
      scaled[index] = weights[index] * static_cast<double>(size) / total;
      (scaled[index] < 1.0 ? small : large).push_back(index);
      alias_[index] = index;
    }
    while(!small.empty() && !large.empty()) {
      const std::size_t low = small.back();
      small.pop_back();
      const std::size_t high = large.back();
      large.pop_back();
      probability_[low] = scaled[low];
      alias_[low] = high;
      scaled[high] -= 1.0 - scaled[low];
      (scaled[high] < 1.0 ? small : large).push_back(high);
    }
  }

  std::size_t sample(double column_random, double coin_random) const {
    if(probability_.empty())
      throw std::runtime_error("sampling an empty Alias table");
    const std::size_t column = std::min(
        static_cast<std::size_t>(column_random * probability_.size()),
        probability_.size() - 1);
    return coin_random < probability_[column] ? column : alias_[column];
  }

 private:
  std::vector<double> probability_;
  std::vector<std::size_t> alias_;
};

struct DistributedAliasTables {
  AliasTable rank_table;
  AliasTable local_table;
  std::size_t local_item_count = 0;
  bool has_global_weight = false;
  bool has_local_weight = false;
  int communicator_size = 0;
  int root = 0;
};

// Collectively prepare the rank and local Alias tables for a fixed distributed
// weight distribution. The returned tables can be reused for many batches.
inline DistributedAliasTables prepare_distributed_alias_tables(
    const std::vector<double>& local_weights,
    MPI_Comm communicator,
    int root = 0) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);
  if(root < 0 || root >= size)
    throw std::invalid_argument("Alias sampling root is outside communicator");

  int local_invalid = 0;
  double local_sum = 0.0;
  for(const double weight : local_weights) {
    if(weight < 0.0 || !std::isfinite(weight)) local_invalid = 1;
    local_sum += weight;
  }
  if(!std::isfinite(local_sum)) local_invalid = 1;
  int any_invalid = 0;
  MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
                communicator);
  if(any_invalid)
    throw std::invalid_argument("distributed Alias weights are invalid");

  std::vector<double> rank_weights;
  if(rank == root) rank_weights.resize(static_cast<std::size_t>(size));
  MPI_Gather(&local_sum, 1, MPI_DOUBLE,
             rank == root ? rank_weights.data() : nullptr, 1, MPI_DOUBLE,
             root, communicator);

  DistributedAliasTables tables;
  tables.local_item_count = local_weights.size();
  tables.has_local_weight = local_sum > 0.0;
  tables.communicator_size = size;
  tables.root = root;
  if(tables.has_local_weight) tables.local_table.build(local_weights);

  int has_global_weight = 0;
  if(rank == root) {
    const double global_sum =
        std::accumulate(rank_weights.begin(), rank_weights.end(), 0.0);
    has_global_weight = global_sum > 0.0 && std::isfinite(global_sum);
    if(has_global_weight) tables.rank_table.build(rank_weights);
  }
  MPI_Bcast(&has_global_weight, 1, MPI_INT, root, communicator);
  tables.has_global_weight = has_global_weight != 0;
  return tables;
}

// Root samples how many draws belong to each rank from a prepared table. Each
// rank then samples its local item indices conditionally and returns counts.
inline std::vector<std::size_t> distributed_alias_sample_counts(
    const DistributedAliasTables& tables,
    std::size_t global_draw_count,
    std::uint64_t base_seed,
    std::uint64_t batch_id,
    MPI_Comm communicator) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);
  if(size != tables.communicator_size)
    throw std::invalid_argument(
        "prepared Alias tables do not match communicator size");

  std::vector<std::size_t> rank_draw_counts;
  std::size_t local_draw_count = 0;
  if(rank == tables.root) {
    rank_draw_counts.assign(static_cast<std::size_t>(size), 0);
    if(global_draw_count != 0 && tables.has_global_weight) {
      const std::uint64_t rank_seed = detail::mix64(
          base_seed ^ detail::mix64(batch_id) ^ UINT64_C(0x72616e6b));
      for(std::size_t draw = 0; draw < global_draw_count; ++draw) {
        const std::uint64_t stream = 2 * static_cast<std::uint64_t>(draw);
        const std::size_t selected = tables.rank_table.sample(
            detail::unit_random(rank_seed, stream),
            detail::unit_random(rank_seed, stream + 1));
        ++rank_draw_counts[selected];
      }
    }
  }
  if(global_draw_count != 0 && !tables.has_global_weight)
    throw std::invalid_argument(
        "positive draw count requires positive global weight");
  MPI_Scatter(rank == tables.root ? rank_draw_counts.data() : nullptr, 1,
              SBD_MPI_SIZE_T, &local_draw_count, 1, SBD_MPI_SIZE_T,
              tables.root, communicator);

  std::vector<std::size_t> counts(tables.local_item_count, 0);
  if(local_draw_count == 0) return counts;
  if(!tables.has_local_weight)
    throw std::runtime_error("draw assigned to an empty local Alias table");
  const std::uint64_t local_seed = detail::mix64(
      base_seed ^ detail::mix64(batch_id) ^
      detail::mix64(static_cast<std::uint64_t>(rank)) ^
      UINT64_C(0x6c6f63616c));
  for(std::size_t draw = 0; draw < local_draw_count; ++draw) {
    const std::uint64_t stream = 2 * static_cast<std::uint64_t>(draw);
    const std::size_t selected = tables.local_table.sample(
        detail::unit_random(local_seed, stream),
        detail::unit_random(local_seed, stream + 1));
    ++counts[selected];
  }
  return counts;
}

// Convenience interface for one-off sampling. Repeated callers should prepare
// the tables once with prepare_distributed_alias_tables() and use the overload
// above for each batch.
inline std::vector<std::size_t> distributed_alias_sample_counts(
    const std::vector<double>& local_weights,
    std::size_t global_draw_count,
    std::uint64_t base_seed,
    std::uint64_t batch_id,
    MPI_Comm communicator,
    int root = 0) {
  const auto tables =
      prepare_distributed_alias_tables(local_weights, communicator, root);
  return distributed_alias_sample_counts(
      tables, global_draw_count, base_seed, batch_id, communicator);
}

}  // namespace distributed_sampling
}  // namespace sbd

#endif
