#include "sbd/sbd.h"
#include "sbd/framework/determinant_distribution_round_robin.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

template <typename ElemT>
bool test_distribution(int rank, int size, MPI_Comm comm) {
  const std::vector<ElemT> global_coefficients = {
      ElemT(0.1), ElemT(-0.7), ElemT(0.3), ElemT(-0.6),
      ElemT(0.2), ElemT(-0.5), ElemT(0.4)};
  std::size_t begin = 0;
  std::size_t end = global_coefficients.size();
  sbd::get_mpi_range(size, rank, begin, end);

  sbd::det_vector<std::size_t> determinants(end - begin);
  std::vector<ElemT> coefficients(end - begin);
  for(std::size_t global = begin; global < end; ++global) {
    determinants[global - begin][0] = global + 1;
    coefficients[global - begin] = global_coefficients[global];
  }

  sbd::redistribute_determinants_weight_round_robin(
      determinants, coefficients, comm);

  std::vector<std::size_t> order(global_coefficients.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                             std::size_t rhs) {
    return std::abs(global_coefficients[lhs]) >
           std::abs(global_coefficients[rhs]);
  });

  bool ok = determinants.size() == coefficients.size();
  for(std::size_t local = 0; local < determinants.size(); ++local) {
    const std::size_t original = determinants[local][0] - 1;
    if(original >= global_coefficients.size()) {
      ok = false;
      continue;
    }
    const auto position = std::find(order.begin(), order.end(), original);
    ok = ok && position != order.end() &&
         coefficients[local] == global_coefficients[original] &&
         static_cast<int>((position - order.begin()) % size) == rank;
  }

  std::size_t local_count = determinants.size();
  std::size_t global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, SBD_MPI_SIZE_T,
                MPI_SUM, comm);
  return ok && global_count == global_coefficients.size();
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  sbd::det_vector<std::size_t>::init_elem_size(1);

  bool ok = test_distribution<double>(rank, size, MPI_COMM_WORLD);
  ok = ok && test_distribution<std::complex<double>>(
                 rank, size, MPI_COMM_WORLD);
  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if(rank == 0) {
    if(global_ok)
      std::cout << "generator round-robin: PASS\n";
    else
      std::cerr << "generator round-robin: FAIL\n";
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
