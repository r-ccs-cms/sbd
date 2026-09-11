#include "sbd/framework/distributed_alias_sampling.h"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool ok = true;
  constexpr std::size_t draws = 120000;
  std::vector<double> weights;
  if(rank == 0) weights = {1.0, 0.0, 3.0};
  const auto tables =
      sbd::distributed_sampling::prepare_distributed_alias_tables(
          weights, MPI_COMM_WORLD);
  const auto counts =
      sbd::distributed_sampling::distributed_alias_sample_counts(
          tables, draws, 12345, 7, MPI_COMM_WORLD);
  const auto repeated =
      sbd::distributed_sampling::distributed_alias_sample_counts(
          tables, draws, 12345, 7, MPI_COMM_WORLD);
  const auto one_off =
      sbd::distributed_sampling::distributed_alias_sample_counts(
          weights, draws, 12345, 7, MPI_COMM_WORLD);
  ok = ok && counts == repeated && counts == one_off &&
       counts.size() == weights.size();
  if(rank == 0) {
    ok = ok && counts[1] == 0 &&
         std::abs(static_cast<double>(counts[0]) / draws - 0.25) < 0.01 &&
         std::abs(static_cast<double>(counts[2]) / draws - 0.75) < 0.01;
  } else {
    ok = ok && counts.empty();
  }
  const std::size_t local_total =
      std::accumulate(counts.begin(), counts.end(), std::size_t(0));
  std::size_t global_total = 0;
  MPI_Allreduce(&local_total, &global_total, 1, SBD_MPI_SIZE_T,
                MPI_SUM, MPI_COMM_WORLD);
  ok = ok && global_total == draws;

  const std::vector<double> zero_weights(weights.size(), 0.0);
  const auto zero_draw_counts =
      sbd::distributed_sampling::distributed_alias_sample_counts(
          zero_weights, 0, 12345, 8, MPI_COMM_WORLD);
  ok = ok && std::accumulate(zero_draw_counts.begin(),
                             zero_draw_counts.end(), std::size_t(0)) == 0;

  bool rejected_zero_weight = false;
  try {
    (void)sbd::distributed_sampling::distributed_alias_sample_counts(
        zero_weights, 1, 12345, 9, MPI_COMM_WORLD);
  } catch(const std::invalid_argument&) {
    rejected_zero_weight = true;
  }
  ok = ok && rejected_zero_weight;

  std::vector<double> invalid_weights = weights;
  if(rank == size - 1) {
    if(invalid_weights.empty()) invalid_weights.push_back(-1.0);
    else invalid_weights[0] = -1.0;
  }
  bool rejected_invalid = false;
  try {
    (void)sbd::distributed_sampling::distributed_alias_sample_counts(
        invalid_weights, 1, 12345, 10, MPI_COMM_WORLD);
  } catch(const std::invalid_argument&) {
    rejected_invalid = true;
  }
  ok = ok && rejected_invalid;

  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if(rank == 0) {
    if(global_ok)
      std::cout << "distributed Alias sampling: PASS\n";
    else
      std::cerr << "distributed Alias sampling: FAIL\n";
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
