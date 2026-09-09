#include "sbd/chemistry/gdb/stat_evaluator.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sbd::det_vector<std::size_t>::init_elem_size(1);
  sbd::det_vector<std::size_t> parents;
  std::vector<double> coefficients;
  if(rank == 0) {
    std::vector<std::size_t> parent(1, 0);
    sbd::setocc(parent, 64, 0, true);
    parents.push_back(parent);
    coefficients.push_back(1.0);
  }
  sbd::oneInt<double> one;
  one.norbs = 4;
  one.store.assign(16, 0.0);
  one(2, 0) = 0.5;
  sbd::twoInt<double> two;
  two.norbs = 2;
  two.store.assign(6, 0.0);
  two.DirectMat.assign(16, 0.0);
  two.ExchangeMat.assign(16, 0.0);

  using Observable = sbd::stat_evaluator::Observable;
  sbd::gdb::StatEvaluatorOptions<double> options;
  options.norb = 2;
  options.sample_count = 2;
  options.reference_energy = -2.0;
  options.base_seed = 11;
  options.expansion_batch_size = 4;
  sbd::gdb::BatchRequest batch{"driver-test", 0};
  const auto both = sbd::gdb::evaluate_statistical_batch<double, double>(
      parents, coefficients, 0.0, one, two, options, batch, MPI_COMM_WORLD);
  options.observable = Observable::variance;
  options.reference_energy = std::numeric_limits<double>::quiet_NaN();
  batch.batch_id = 1;
  const auto variance = sbd::gdb::evaluate_statistical_batch<double, double>(
      parents, coefficients, 0.0, one, two, options, batch, MPI_COMM_WORLD);
  options.observable = Observable::pt2;
  options.reference_energy = -2.0;
  batch.batch_id = 2;
  const auto pt2 = sbd::gdb::evaluate_statistical_batch<double, double>(
      parents, coefficients, 0.0, one, two, options, batch, MPI_COMM_WORLD);
  options.observable = Observable::both;
  const std::vector<sbd::gdb::BatchRequest> batches{
      {"multi-driver-test", 10}, {"multi-driver-test", 11}};
  std::vector<sbd::gdb::EvaluatedBatch> multiple;
  sbd::gdb::evaluate_statistical_batches<double, double>(
      parents, coefficients, 0.0, one, two, options, batches,
      MPI_COMM_WORLD,
      [&](const sbd::gdb::EvaluatedBatch& evaluated) {
        multiple.push_back(evaluated);
      });
  bool profile_ok = true;
  for(const auto& evaluated : multiple) {
    const auto& profile = evaluated.profile;
    profile_ok = profile_ok && profile.sampled_parents <= profile.parents &&
        profile.sampled_parents <= profile.draws &&
        profile.external_children <= profile.unique_children &&
        profile.unique_children <= profile.received_records &&
        profile.lookup_entries == 0 && profile.lookup_bytes > 0;
    const auto local = profile.values();
    std::array<std::uint64_t, 10> totals{};
    MPI_Allreduce(local.data(), totals.data(), 10, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    profile_ok = profile_ok && totals[0] == 1 && totals[1] == 1 &&
        totals[4] == 1 && totals[5] == options.sample_count &&
        totals[6] == 1 && totals[7] == 1 && totals[8] == 1 && totals[9] == 1;
  }
  const bool ok = profile_ok && both.record.has_variance && both.record.has_pt2 &&
      variance.record.has_variance && !variance.record.has_pt2 &&
      !variance.record.has_reference_energy &&
      !pt2.record.has_variance && pt2.record.has_pt2 &&
      std::abs(both.record.variance_estimate - 0.25) < 1.0e-14 &&
      std::abs(both.record.pt2_estimate + 0.125) < 1.0e-14 &&
      std::abs(variance.record.variance_estimate - 0.25) < 1.0e-14 &&
      std::abs(pt2.record.pt2_estimate + 0.125) < 1.0e-14 &&
      multiple.size() == 2 && multiple[0].record.batch_id == 10 &&
      multiple[1].record.batch_id == 11 &&
      std::abs(multiple[0].record.variance_estimate - 0.25) < 1.0e-14 &&
      std::abs(multiple[1].record.pt2_estimate + 0.125) < 1.0e-14;
  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << (global_ok ? "gdb stat evaluator driver: PASS\n"
                            : "gdb stat evaluator driver: FAIL\n");
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
