#include "sbd/framework/stat/stat_evaluator_batch.h"

#include <mpi.h>

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const std::string automatic_id =
      sbd::stat_evaluator::make_default_calculation_id(MPI_COMM_WORLD);
  std::uint64_t local_id = std::stoull(automatic_id);
  std::uint64_t minimum_id = 0;
  std::uint64_t maximum_id = 0;
  MPI_Allreduce(&local_id, &minimum_id, 1, MPI_UINT64_T, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_id, &maximum_id, 1, MPI_UINT64_T, MPI_MAX,
                MPI_COMM_WORLD);

  sbd::stat_evaluator::BatchRecord record;
  record.calculation_id = "test";
  record.observable = sbd::stat_evaluator::Observable::variance;
  record.base_seed = 17;
  record.batch_id = 3;
  record.sample_count = 20000;
  record.heatbath_cutoff = 1.0e-8;
  record.has_variance = true;
  record.variance_estimate = 2.5;
  record.mpi_size = 4;
  record.omp_threads = 2;
  record.elapsed_seconds = 1.25;
  std::ostringstream csv;
  sbd::stat_evaluator::write_csv_header(csv);
  sbd::stat_evaluator::write_csv_record(csv, record);
  const std::string text = csv.str();
  const bool empty_columns = text.find(",1e-08,,2.5,,4,2,1.25\n") !=
                             std::string::npos;
  bool ok = !automatic_id.empty() && minimum_id == maximum_id &&
            empty_columns &&
            sbd::stat_evaluator::parse_observable("both") ==
                sbd::stat_evaluator::Observable::both;
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  sbd::stat_evaluator::BatchProfile profile;
  profile.draws = rank + 1;
  std::ostringstream profiles;
  profiles << std::scientific << std::setprecision(3);
  const auto flags = profiles.flags();
  sbd::stat_evaluator::write_stats_profiles(
      profiles, 3, profile, MPI_COMM_WORLD, size - 1);
  ok = ok && profiles.flags() == flags && profiles.precision() == 3;
  if(rank == size - 1) {
    const auto result = profiles.str();
    for(int source = 0; source < size; ++source) {
      ok = ok && result.find("profile batch_id=3 rank=" +
          std::to_string(source) + " host=\"") != std::string::npos;
      ok = ok && result.find(" draws=" + std::to_string(source + 1) +
          " generated_records=") != std::string::npos;
    }
    std::ostringstream mean;
    mean << (size + 1) / 2.0;
    ok = ok && result.find("metric=draws min=1 mean=" + mean.str() +
        " max=" + std::to_string(size)) != std::string::npos;
  } else {
    ok = ok && profiles.str().empty();
  }
  sbd::stat_evaluator::BatchTiming timing;
  timing.expansion_seconds = rank + 1;
  timing.total_seconds = 2 * (rank + 1);
  std::ostringstream timings;
  timings << std::scientific << std::setprecision(3);
  const auto timing_flags = timings.flags();
  sbd::stat_evaluator::write_stats_timing(
      timings, 3, timing, MPI_COMM_WORLD, size - 1);
  ok = ok && timings.flags() == timing_flags && timings.precision() == 3;
  if(rank == size - 1) {
    const auto result = timings.str();
    for(int source = 0; source < size; ++source) {
      ok = ok && result.find("timing batch_id=3 rank=" +
          std::to_string(source) + " distribution=") != std::string::npos;
      ok = ok && result.find(" expansion=" + std::to_string(source + 1) +
          " hash_observable=") != std::string::npos;
    }
    std::ostringstream mean;
    mean << (size + 1) / 2.0;
    ok = ok && result.find("metric=expansion min=1 mean=" + mean.str() +
        " max=" + std::to_string(size)) != std::string::npos;
    ok = ok && result.find("metric=total min=2 mean=" + std::to_string(size + 1) +
        " max=" + std::to_string(2 * size)) != std::string::npos;
  } else {
    ok = ok && timings.str().empty();
  }
  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << (global_ok ? "stat evaluator batch record: PASS\n"
                            : "stat evaluator batch record: FAIL\n");
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
