#include "sbd/chemistry/gdb/hash_owned_external_observable.h"
#include "sbd/chemistry/gdb/parent_sampling.h"
#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"
#include "sbd/framework/determinant_distribution_round_robin.h"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

std::vector<std::size_t> determinant(int occupied) {
  std::vector<std::size_t> value(1, 0);
  sbd::setocc(value, 64, occupied, true);
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  constexpr std::size_t bit_length = 64;
  constexpr std::size_t norb = 3;
  constexpr std::size_t draws = 20000;
  sbd::det_vector<std::size_t>::init_elem_size(1);

  sbd::det_vector<std::size_t> all_parents;
  all_parents.push_back(determinant(0));
  all_parents.push_back(determinant(2));
  const std::vector<double> all_coefficients{1.0, 1.0};
  std::size_t begin = 0;
  std::size_t end = all_parents.size();
  sbd::get_mpi_range(size, rank, begin, end);
  sbd::det_vector<std::size_t> parents(end - begin);
  std::vector<double> coefficients(end - begin);
  for(std::size_t index = begin; index < end; ++index) {
    parents[index - begin] = all_parents[index];
    coefficients[index - begin] = all_coefficients[index];
  }

  sbd::det_vector<std::size_t> membership_basis(parents);
  sbd::gdb::make_hash_owned_membership_basis(
      membership_basis, MPI_COMM_WORLD);
  sbd::redistribute_determinants_weight_round_robin(
      parents, coefficients, MPI_COMM_WORLD);

  const auto sample =
      sbd::gdb::sample_parents_by_coefficient_magnitude<
          double, double>(coefficients, draws, 987654321, 4,
                          MPI_COMM_WORLD);
  const auto repeated =
      sbd::gdb::sample_parents_by_coefficient_magnitude<
          double, double>(coefficients, draws, 987654321, 4,
                          MPI_COMM_WORLD);
  bool ok = sample.counts == repeated.counts &&
            sample.probabilities == repeated.probabilities &&
            std::abs(sample.global_weight_sum - 2.0) < 1.0e-14;
  for(const double probability : sample.probabilities)
    ok = ok && std::abs(probability - 0.5) < 1.0e-14;

  sbd::oneInt<double> one_integrals;
  one_integrals.norbs = 6;
  one_integrals.store.assign(36, 0.0);
  one_integrals(4, 0) = 2.0;
  one_integrals(4, 2) = -0.5;
  sbd::twoInt<double> two_integrals;
  two_integrals.norbs = static_cast<int>(norb);
  two_integrals.store.assign(21, 0.0);
  two_integrals.DirectMat.assign(36, 0.0);
  two_integrals.ExchangeMat.assign(36, 0.0);

  sbd::det_vector<std::size_t> children;
  std::vector<sbd::gdb::Contribution<double, double>> values;
  sbd::gdb::local_integral_driven_contribution_expansion(
      parents, coefficients, sample.counts, sample.probabilities,
      bit_length, norb, one_integrals, two_integrals,
      0.0, 16, children, values);
  const auto estimate =
      sbd::gdb::distributed_u_statistic_external_observable<
          double, double>(
          children, values, membership_basis, draws, bit_length, norb,
          0.0, one_integrals, two_integrals, -2.0, MPI_COMM_WORLD);

  std::size_t local_draws = 0;
  for(const std::size_t count : sample.counts) local_draws += count;
  std::size_t global_draws = 0;
  MPI_Allreduce(&local_draws, &global_draws, 1, SBD_MPI_SIZE_T,
                MPI_SUM, MPI_COMM_WORLD);
  ok = ok && global_draws == draws &&
       estimate.external_determinants == 1 &&
       std::abs(estimate.variance - 2.25) < 0.12 &&
       std::abs(estimate.pt2 + 1.125) < 0.06;

  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if(rank == 0) {
    if(global_ok)
      std::cout << "gdb sampled pipeline: PASS"
                << " variance=" << estimate.variance
                << " pt2=" << estimate.pt2 << '\n';
    else
      std::cerr << "gdb sampled pipeline: FAIL"
                << " variance=" << estimate.variance
                << " pt2=" << estimate.pt2
                << " draws=" << global_draws << '\n';
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
