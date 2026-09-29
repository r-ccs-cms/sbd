// Adapted from the ExtSBD GDB warm-start remap tests.
#include "sbd/framework/remap_wavefunction.h"

#include <mpi.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

bool run_hash_test() {
  const sbd::wavefunction_remap_detail::DeterminantHash hash;
  if(hash({}) != static_cast<std::size_t>(UINT64_C(0xcbf29ce484222325)))
    return false;
  // FNV-1a reference result for eight zero bytes on a 64-bit size_t host.
  if constexpr(sizeof(std::size_t) == 8)
    return hash({0}) == UINT64_C(0xa8c7f832281a39c5);
  return true;
}

std::vector<std::size_t> determinant(std::size_t identifier) {
  return {identifier + 1};
}

template <typename ElemT>
ElemT coefficient(std::size_t identifier) {
  if(identifier == 2) return ElemT(0);
  const double real = (identifier % 2 == 0 ? 1.0 : -1.0) *
                      static_cast<double>(identifier + 1) / 10.0;
  if constexpr(std::is_same_v<ElemT, std::complex<double>>) {
    return ElemT(real, static_cast<double>(identifier + 1) / 100.0);
  } else {
    return ElemT(real);
  }
}

template <typename ElemT>
bool close(ElemT lhs, ElemT rhs) {
  return std::abs(lhs - rhs) <= 1.0e-14;
}

template <typename ElemT>
bool run_value_test(MPI_Comm communicator) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  std::vector<std::vector<std::size_t>> old_basis;
  std::vector<ElemT> old_coefficients;
  for(std::size_t identifier = 0; identifier < 6; ++identifier) {
    if(static_cast<int>(identifier % static_cast<std::size_t>(size)) == rank) {
      old_basis.push_back(determinant(identifier));
      old_coefficients.push_back(coefficient<ElemT>(identifier));
    }
  }

  std::vector<std::vector<std::size_t>> new_basis;
  for(std::size_t offset = 0; offset < 9; ++offset) {
    const std::size_t identifier = 8 - offset;
    const int destination = static_cast<int>(
        (identifier * 3 + 1) % static_cast<std::size_t>(size));
    if(destination == rank) new_basis.push_back(determinant(identifier));
  }

  std::vector<ElemT> new_coefficients;
  const std::size_t matched = sbd::RemapWavefunctionToBasis(
      old_basis, old_coefficients, new_basis, communicator, new_coefficients);
  bool local_ok = matched == 6 && new_coefficients.size() == new_basis.size();
  for(std::size_t index = 0; index < new_basis.size(); ++index) {
    const std::size_t identifier = new_basis[index][0] - 1;
    const ElemT expected = identifier < 6
        ? coefficient<ElemT>(identifier) : ElemT(0);
    local_ok = local_ok && close(new_coefficients[index], expected);
  }
  int local_result = local_ok ? 1 : 0;
  int global_result = 0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                communicator);
  return global_result != 0;
}

bool run_projection_test(MPI_Comm communicator, bool strict) {
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  std::vector<std::vector<std::size_t>> old_basis;
  std::vector<double> old_coefficients;
  if(rank == 0) {
    old_basis.push_back(determinant(99));
    old_coefficients.push_back(1.0);
  }
  std::vector<std::vector<std::size_t>> new_basis;
  if(rank == 0) new_basis.push_back(determinant(0));
  std::vector<double> new_coefficients;
  bool threw = false;
  std::size_t matched = 999;
  try {
    matched = sbd::RemapWavefunctionToBasis(
        old_basis, old_coefficients, new_basis, communicator,
        new_coefficients, strict);
  } catch(const std::invalid_argument&) {
    threw = true;
  }
  const bool expected = strict
      ? threw
      : (!threw && matched == 0 && new_coefficients.size() == new_basis.size()
         && (new_coefficients.empty() || new_coefficients[0] == 0.0));
  int local_result = expected ? 1 : 0;
  int global_result = 0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                communicator);
  return global_result != 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm communicator = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);

  const bool hash_ok = run_hash_test();
  const bool real_ok = run_value_test<double>(communicator);
  const bool complex_ok = run_value_test<std::complex<double>>(communicator);
  const bool projection_ok = run_projection_test(communicator, false);
  const bool strict_subset_ok = run_projection_test(communicator, true);
  const bool all_ok = hash_ok && real_ok && complex_ok && projection_ok &&
                      strict_subset_ok;
  if(rank == 0) {
    std::cout << "FNV-1a reference: " << (hash_ok ? "pass" : "FAIL") << '\n';
    std::cout << "real remap: " << (real_ok ? "pass" : "FAIL") << '\n';
    std::cout << "complex remap: " << (complex_ok ? "pass" : "FAIL") << '\n';
    std::cout << "old-only determinant ignored: "
              << (projection_ok ? "pass" : "FAIL") << '\n';
    std::cout << "strict subset rejection: "
              << (strict_subset_ok ? "pass" : "FAIL") << '\n';
  }
  MPI_Finalize();
  return all_ok ? 0 : 1;
}
