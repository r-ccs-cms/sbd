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
    constexpr size_t bit_length = 64;
    constexpr size_t total_bit_length = 136;
    const std::vector<size_t> alpha_orbitals{0, 1, 64, 65};
    const std::vector<size_t> beta_orbitals{0, 1, 64, 65};

    auto make_determinant = [](const size_t alpha_orbital,
                               const size_t beta_orbital) {
      std::vector<size_t> determinant(3, 0);
      const size_t alpha_position = 2 * alpha_orbital;
      const size_t beta_position = 2 * beta_orbital + 1;
      determinant[alpha_position / bit_length] |=
          size_t(1) << (alpha_position % bit_length);
      determinant[beta_position / bit_length] |=
          size_t(1) << (beta_position % bit_length);
      return determinant;
    };

    std::vector<std::vector<size_t>> alpha_keys, beta_keys;
    for(const size_t orbital : alpha_orbitals) {
      std::vector<size_t> key(2, 0);
      sbd::getAdet(make_determinant(orbital, 0), bit_length,
                   total_bit_length / 2, key);
      alpha_keys.push_back(std::move(key));
    }
    for(const size_t orbital : beta_orbitals) {
      std::vector<size_t> key(2, 0);
      sbd::getBdet(make_determinant(0, orbital), bit_length,
                   total_bit_length / 2, key);
      beta_keys.push_back(std::move(key));
    }

    std::vector<std::vector<size_t>> expected_determinants;
    for(const size_t alpha_orbital : alpha_orbitals) {
      for(const size_t beta_orbital : beta_orbitals) {
        expected_determinants.push_back(
            make_determinant(alpha_orbital, beta_orbital));
      }
    }

    auto run_case = [&](const bool distributed_input) {
      sbd::det_vector<size_t> basis;
      for(size_t ai = 0; ai < alpha_orbitals.size(); ++ai) {
        for(size_t bi = 0; bi < beta_orbitals.size(); ++bi) {
          const size_t determinant_id = 4 * ai + bi;
          const int input_rank = distributed_input
              ? static_cast<int>(determinant_id % 3)
              : 0;
          if(rank == input_rank) {
            basis.emplace_back(make_determinant(alpha_orbitals[ai],
                                                beta_orbitals[bi]));
          }
        }
      }

      sbd::gdb::redistribution_grid_bra_ab_cyclic(
          basis, bit_length, total_bit_length, 2, 2, MPI_COMM_WORLD);

      for(const auto & determinant : basis) {
        std::vector<size_t> alpha_key(2, 0), beta_key(2, 0);
        sbd::getAdet(determinant, bit_length, total_bit_length / 2, alpha_key);
        sbd::getBdet(determinant, bit_length, total_bit_length / 2, beta_key);
        const auto alpha_it = std::find(alpha_keys.begin(), alpha_keys.end(), alpha_key);
        const auto beta_it = std::find(beta_keys.begin(), beta_keys.end(), beta_key);
        ok = ok && alpha_it != alpha_keys.end() && beta_it != beta_keys.end();
        if(alpha_it != alpha_keys.end() && beta_it != beta_keys.end()) {
          const size_t ai = static_cast<size_t>(alpha_it - alpha_keys.begin());
          const size_t bi = static_cast<size_t>(beta_it - beta_keys.begin());
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
        const auto it = std::find(expected_determinants.begin(),
                                  expected_determinants.end(), determinant);
        if(it != expected_determinants.end()) {
          local_seen |= size_t(1) << static_cast<size_t>(it - expected_determinants.begin());
        }
      }
      size_t global_seen = 0;
      MPI_Allreduce(&local_seen, &global_seen, 1, SBD_MPI_SIZE_T, MPI_BOR,
                    MPI_COMM_WORLD);
      ok = ok && global_seen == (size_t(1) << 16) - 1;
    };

    run_case(false);
    run_case(true);
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
