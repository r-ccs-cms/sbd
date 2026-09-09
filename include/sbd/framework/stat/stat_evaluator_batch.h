#ifndef SBD_STAT_EVALUATOR_BATCH_H
#define SBD_STAT_EVALUATOR_BATCH_H

#include "sbd/framework/timestamp.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sbd {
namespace stat_evaluator {

enum class Observable { both, variance, pt2 };

inline const char* observable_name(Observable observable) {
  switch(observable) {
    case Observable::both: return "both";
    case Observable::variance: return "variance";
    case Observable::pt2: return "pt2";
  }
  throw std::invalid_argument("unknown observable");
}

inline Observable parse_observable(const std::string& value) {
  if(value == "both") return Observable::both;
  if(value == "variance") return Observable::variance;
  if(value == "pt2") return Observable::pt2;
  throw std::invalid_argument("observable must be both, variance, or pt2");
}

struct BatchTiming {
  double distribution_seconds = 0.0;
  double sampling_seconds = 0.0;
  double expansion_seconds = 0.0;
  double hash_observable_seconds = 0.0;
  double hash_seconds = 0.0;
  double hash_packing_seconds = 0.0;
  double hash_communication_seconds = 0.0;
  double hash_receive_sort_seconds = 0.0;
  double child_sort_seconds = 0.0;
  double local_observable_seconds = 0.0;
  double observable_allreduce_seconds = 0.0;
  double total_seconds = 0.0;

  std::array<double, 12> values() const {
    return {distribution_seconds, sampling_seconds, expansion_seconds,
            hash_observable_seconds, hash_seconds, hash_packing_seconds,
            hash_communication_seconds, hash_receive_sort_seconds,
            child_sort_seconds, local_observable_seconds,
            observable_allreduce_seconds, total_seconds};
  }
};

// Rank-local counts at well-defined pipeline boundaries. Byte counts describe
// lookup storage, not RSS or the process peak allocation.
struct BatchProfile {
  std::uint64_t parents = 0;
  std::uint64_t membership = 0;
  std::uint64_t lookup_entries = 0;
  std::uint64_t lookup_bytes = 0;
  std::uint64_t sampled_parents = 0;
  std::uint64_t draws = 0;
  std::uint64_t generated_records = 0;
  std::uint64_t received_records = 0;
  std::uint64_t unique_children = 0;
  std::uint64_t external_children = 0;

  std::array<std::uint64_t, 10> values() const {
    return {parents, membership, lookup_entries, lookup_bytes, sampled_parents,
            draws, generated_records, received_records, unique_children,
            external_children};
  }
};

// Collective writer: gather small profiles, then emit complete lines on root
// in rank order, avoiding interleaved output from separate MPI processes.
inline void write_stats_profiles(std::ostream& output,
                                 std::uint64_t batch_id,
                                 const BatchProfile& profile,
                                 MPI_Comm communicator,
                                 int root = 0) {
  int rank = 0, size = 1;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);
  if(root < 0 || root >= size)
    throw std::invalid_argument("profile root is outside communicator");
  const auto local = profile.values();
  constexpr std::size_t fields = 10;
  const char* names[fields] = {
      "parents", "membership", "lookup_entries", "lookup_bytes",
      "sampled_parents", "draws", "generated_records", "received_records",
      "unique_children", "external_children"};
  std::vector<std::uint64_t> gathered(rank == root ? size * fields : 0);
  MPI_Gather(local.data(), static_cast<int>(fields), MPI_UINT64_T,
             gathered.data(), static_cast<int>(fields), MPI_UINT64_T,
             root, communicator);
  std::array<char, MPI_MAX_PROCESSOR_NAME + 1> host{};
  int host_length = 0;
  MPI_Get_processor_name(host.data(), &host_length);
  std::vector<char> hosts(rank == root ? size * host.size() : 0);
  MPI_Gather(host.data(), static_cast<int>(host.size()), MPI_CHAR,
             hosts.data(), static_cast<int>(host.size()), MPI_CHAR,
             root, communicator);
  if(rank != root) return;
  const auto timestamp = sbd::make_timestamp();
  const auto old_precision = output.precision();
  const auto old_flags = output.flags();
  output << std::defaultfloat << std::setprecision(17);
  for(int source = 0; source < size; ++source) {
    output << ' ' << timestamp << " sbd::stats: profile batch_id=" << batch_id
           << " rank=" << source << " host="
           << std::quoted(std::string(hosts.data() + source * host.size()));
    for(std::size_t field = 0; field < fields; ++field)
      output << ' ' << names[field] << '=' << gathered[source * fields + field];
    output << '\n';
  }
  for(std::size_t field = 0; field < fields; ++field) {
    auto minimum = gathered[field];
    auto maximum = minimum;
    long double total = 0;
    for(int source = 0; source < size; ++source) {
      const auto value = gathered[source * fields + field];
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      total += value;
    }
    output << ' ' << timestamp << " sbd::stats: profile_summary batch_id="
           << batch_id << " metric=" << names[field]
           << " min=" << minimum << " mean=" << total / size
           << " max=" << maximum << '\n';
  }
  output.precision(old_precision);
  output.flags(old_flags);
  output.flush();
}

struct BatchRecord {
  static constexpr int format_version = 1;
  std::string calculation_id;
  Observable observable = Observable::both;
  std::uint64_t base_seed = 0;
  std::uint64_t batch_id = 0;
  std::size_t sample_count = 0;
  double heatbath_cutoff = 0.0;
  bool has_reference_energy = false;
  double reference_energy = 0.0;
  bool has_variance = false;
  double variance_estimate = 0.0;
  bool has_pt2 = false;
  double pt2_estimate = 0.0;
  int mpi_size = 1;
  int omp_threads = 1;
  double elapsed_seconds = 0.0;
};

inline std::string make_default_calculation_id(MPI_Comm communicator,
                                                int root = 0) {
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  std::uint64_t timestamp = rank == root ? sbd::make_timestamp() : 0;
  MPI_Bcast(&timestamp, 1, MPI_UINT64_T, root, communicator);
  return std::to_string(timestamp);
}

inline void write_csv_header(std::ostream& output) {
  output << "format_version,calculation_id,observable,base_seed,batch_id,"
            "sample_count,heatbath_cutoff,reference_energy,variance_estimate,"
            "pt2_estimate,mpi_size,omp_threads,elapsed_seconds\n";
}

inline void write_csv_record(std::ostream& output,
                             const BatchRecord& record) {
  if(record.calculation_id.empty())
    throw std::invalid_argument("calculation ID is empty");
  const std::streamsize old_precision = output.precision();
  const std::ios::fmtflags old_flags = output.flags();
  output << BatchRecord::format_version << ',' << record.calculation_id << ','
         << observable_name(record.observable) << ',' << record.base_seed << ','
         << record.batch_id << ',' << record.sample_count << ','
         << std::setprecision(17) << record.heatbath_cutoff << ',';
  if(record.has_reference_energy) output << record.reference_energy;
  output << ',';
  if(record.has_variance) output << record.variance_estimate;
  output << ',';
  if(record.has_pt2) output << record.pt2_estimate;
  output << ',' << record.mpi_size << ',' << record.omp_threads << ','
         << record.elapsed_seconds << '\n';
  output.precision(old_precision);
  output.flags(old_flags);
  output.flush();
}

// Collective, like write_stats_profiles(). Timings remain rank-local until
// this output boundary; only root emits rank rows and cross-rank summaries.
inline void write_stats_timing(std::ostream& output,
                               std::uint64_t batch_id,
                               const BatchTiming& timing,
                               MPI_Comm communicator,
                               int root = 0) {
  int rank = 0, size = 1;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);
  if(root < 0 || root >= size)
    throw std::invalid_argument("timing root is outside communicator");
  const auto local = timing.values();
  constexpr std::size_t fields = 12;
  const char* names[fields] = {
      "distribution", "sampling", "expansion", "hash_observable", "hash",
      "hash_pack", "hash_mpi", "hash_receive_sort", "child_sort",
      "local_observable", "observable_allreduce", "total"};
  std::vector<double> gathered(rank == root ? size * fields : 0);
  MPI_Gather(local.data(), static_cast<int>(fields), MPI_DOUBLE,
             gathered.data(), static_cast<int>(fields), MPI_DOUBLE,
             root, communicator);
  if(rank != root) return;
  const auto timestamp = sbd::make_timestamp();
  const auto old_precision = output.precision();
  const auto old_flags = output.flags();
  output << std::defaultfloat << std::setprecision(17);
  for(int source = 0; source < size; ++source) {
    output << ' ' << timestamp << " sbd::stats: timing batch_id=" << batch_id
           << " rank=" << source;
    for(std::size_t field = 0; field < fields; ++field)
      output << ' ' << names[field] << '=' << gathered[source * fields + field];
    output << '\n';
  }
  for(std::size_t field = 0; field < fields; ++field) {
    auto minimum = gathered[field];
    auto maximum = minimum;
    long double total = 0;
    for(int source = 0; source < size; ++source) {
      const auto value = gathered[source * fields + field];
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      total += value;
    }
    output << ' ' << timestamp << " sbd::stats: timing_summary batch_id="
           << batch_id << " metric=" << names[field]
           << " min=" << minimum << " mean=" << total / size
           << " max=" << maximum << '\n';
  }
  output.precision(old_precision);
  output.flags(old_flags);
  output.flush();
}

}  // namespace stat_evaluator
}  // namespace sbd

#endif
