#include "sbd/sbd.h"

#include <algorithm>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  bool ok = (size == 4);

  try {
    const std::vector<size_t> alpha{1, 4, 16, 64};
    const std::vector<size_t> beta{2, 8, 32, 128};
    sbd::det_vector<size_t> basis;
    if(rank == 0) {
      for(const size_t a : alpha) {
        for(const size_t b : beta) basis.emplace_back(std::vector<size_t>{a | b});
      }
    }

    sbd::gdb::redistribution_grid_bra_ab_cyclic(
        basis, 8, 8, 2, 2, MPI_COMM_WORLD);

    for(const auto & determinant : basis) {
      const size_t value = determinant[0];
      const auto a_it = std::find(alpha.begin(), alpha.end(), value & size_t(85));
      const auto b_it = std::find(beta.begin(), beta.end(), value & size_t(170));
      ok = ok && a_it != alpha.end() && b_it != beta.end();
      if(a_it != alpha.end() && b_it != beta.end()) {
        const size_t ai = static_cast<size_t>(a_it - alpha.begin());
        const size_t bi = static_cast<size_t>(b_it - beta.begin());
        ok = ok && rank == static_cast<int>((ai % 2) * 2 + (bi % 2));
      }
    }

    sbd::redistribution_bitarray(basis, MPI_COMM_WORLD);
    sbd::sort_bitarray(basis);

    const size_t local_count = basis.size();
    size_t global_count = 0, min_count = 0, max_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, SBD_MPI_SIZE_T, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &min_count, 1, SBD_MPI_SIZE_T, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &max_count, 1, SBD_MPI_SIZE_T, MPI_MAX,
                  MPI_COMM_WORLD);
    ok = ok && global_count == 16 && max_count - min_count <= 1;
    for(size_t i = 1; i < basis.size(); ++i) {
      ok = ok && sbd::less_from_back(basis[i-1], basis[i]);
    }

    size_t local_seen = 0;
    for(const auto & determinant : basis) {
      const size_t value = determinant[0];
      const size_t ai = static_cast<size_t>(
          std::find(alpha.begin(), alpha.end(), value & size_t(85)) - alpha.begin());
      const size_t bi = static_cast<size_t>(
          std::find(beta.begin(), beta.end(), value & size_t(170)) - beta.begin());
      if(ai < 4 && bi < 4) local_seen |= size_t(1) << (4 * ai + bi);
    }
    size_t global_seen = 0;
    MPI_Allreduce(&local_seen, &global_seen, 1, SBD_MPI_SIZE_T, MPI_BOR,
                  MPI_COMM_WORLD);
    ok = ok && global_seen == (size_t(1) << 16) - 1;
  } catch(const std::exception & error) {
    std::cerr << "rank " << rank << ": " << error.what() << std::endl;
    ok = false;
  }

  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "GDB grid-cyclic and count redistribution: "
              << (global_ok ? "PASS" : "FAIL") << std::endl;
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
