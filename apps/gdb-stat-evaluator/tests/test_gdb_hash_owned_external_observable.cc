#include "sbd/chemistry/gdb/hash_owned_external_observable.h"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

std::vector<std::size_t> determinant(std::initializer_list<int> occupied) {
  std::vector<std::size_t> value(1, 0);
  for(const int orbital : occupied)
    sbd::setocc(value, 64, orbital, true);
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
  constexpr std::size_t norb = 2;
  constexpr std::size_t draws = 3;
  sbd::det_vector<std::size_t>::init_elem_size(1);

  sbd::oneInt<double> one_integrals;
  one_integrals.norbs = 4;
  one_integrals.store.assign(16, 0.0);
  sbd::twoInt<double> two_integrals;
  two_integrals.norbs = static_cast<int>(norb);
  two_integrals.store.assign(6, 0.0);
  two_integrals.DirectMat.assign(16, 0.0);
  two_integrals.ExchangeMat.assign(16, 0.0);

  sbd::det_vector<std::size_t> full_basis;
  full_basis.push_back(determinant({0, 1}));
  full_basis.push_back(determinant({2, 3}));
  const auto external = determinant({1, 2});

  sbd::det_vector<std::size_t> full_children;
  full_children.push_back(external);
  full_children.push_back(external);
  full_children.push_back(full_basis[0]);
  using Contribution =
      sbd::gdb::Contribution<double, double>;
  const std::vector<Contribution> full_values{
      {8.0, 64.0}, {-4.0 / 3.0, 8.0 / 9.0}, {100.0, 10000.0}};

  const auto reference =
      sbd::gdb::u_statistic_external_observable<double, double>(
          full_children, full_values, full_basis, draws, bit_length, norb,
          0.0, one_integrals, two_integrals, -2.0);

  sbd::det_vector<std::size_t> local_basis;
  for(std::size_t index = 0; index < full_basis.size(); ++index)
    if(static_cast<int>(index % static_cast<std::size_t>(size)) == rank)
      local_basis.push_back(full_basis[index]);
  sbd::gdb::make_hash_owned_membership_basis(
      local_basis, MPI_COMM_WORLD);

  sbd::det_vector<std::size_t> local_children;
  std::vector<Contribution> local_values;
  for(std::size_t index = 0; index < full_children.size(); ++index) {
    if(static_cast<int>(index % static_cast<std::size_t>(size)) != rank)
      continue;
    local_children.push_back(full_children[index]);
    local_values.push_back(full_values[index]);
  }
  sbd::gdb::HashObservableCounts profile;
  const auto distributed =
      sbd::gdb::distributed_u_statistic_external_observable<
          double, double>(
          local_children, local_values, local_basis, draws,
          bit_length, norb, 0.0, one_integrals, two_integrals,
          -2.0, MPI_COMM_WORLD, 0, 0.0, true, true, nullptr, &profile);

  const bool owns_external = sbd::murmur_basis::owner_rank(external, size) == rank;
  const bool owns_internal = sbd::murmur_basis::owner_rank(full_basis[0], size) == rank;
  const bool profile_ok =
      profile.received_records == 2 * owns_external + owns_internal &&
      profile.unique_children == owns_external + owns_internal &&
      profile.external_children == static_cast<std::size_t>(owns_external) &&
      distributed.unique_children == 2 && distributed.external_determinants == 1;
  bool owner_ok = profile_ok;
  for(const auto& child : local_children)
    owner_ok = owner_ok &&
        sbd::murmur_basis::owner_rank(child, size) == rank;
  for(const auto& determinant_value : local_basis)
    owner_ok = owner_ok &&
        sbd::murmur_basis::owner_rank(determinant_value, size) == rank;
  bool collective_denominator_error = false;
  try {
    sbd::det_vector<std::size_t> zero_children;
    std::vector<Contribution> zero_values;
    if(rank == 0) {
      zero_children.push_back(external);
      zero_values.push_back({1.0, 0.0});
    }
    (void)sbd::gdb::distributed_u_statistic_external_observable<
        double, double>(
        zero_children, zero_values, local_basis, draws,
        bit_length, norb, 0.0, one_integrals, two_integrals,
        0.0, MPI_COMM_WORLD);
  } catch(const std::domain_error&) {
    collective_denominator_error = true;
  }

  const bool local_ok = owner_ok && collective_denominator_error &&
      distributed.external_determinants == reference.external_determinants &&
      std::abs(distributed.variance - reference.variance) < 1.0e-13 &&
      std::abs(distributed.pt2 - reference.pt2) < 1.0e-13;
  int ok = local_ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if(rank == 0) {
    if(global_ok)
      std::cout << "gdb hash-owned external observable: PASS\n";
    else
      std::cerr << "gdb hash-owned external observable: FAIL"
                << " variance=" << distributed.variance
                << " reference_variance=" << reference.variance
                << " pt2=" << distributed.pt2
                << " reference_pt2=" << reference.pt2 << '\n';
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
