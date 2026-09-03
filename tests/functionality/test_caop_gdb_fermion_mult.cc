#include "sbd/sbd.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t bit_length = 64;
constexpr int sites = 4;

sbd::det_vector<std::size_t> make_basis(bool interleaved) {
  sbd::det_vector<std::size_t> basis;
  for(int left = 0; left < sites; ++left) {
    for(int right = left + 1; right < sites; ++right) {
      const int left_bit = interleaved ? 2 * left : left;
      const int right_bit = interleaved ? 2 * right : right;
      basis.push_back(std::vector<std::size_t>{
          (std::size_t{1} << left_bit) |
          (std::size_t{1} << right_bit)});
    }
  }
  return basis;
}

void test_caop_matches_gdb(MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  if(size != 1) return;

  sbd::det_vector<std::size_t>::init_elem_size(1);
  sbd::det_vector<std::size_t, sbd::det_kind::half>::init_elem_size(1);
  const auto gdb_basis = make_basis(true);
  const auto ca_op_basis = make_basis(false);
  const std::vector<double> input{0.13, -0.27, 0.41, 0.19, -0.31, 0.53};

  const auto fcidump = sbd::LoadFCIDump("caop_gdb_hubbard4_fcidump.txt");
  int norb = 0;
  int nelec = 0;
  double scalar = 0.0;
  sbd::oneInt<double> one_body;
  sbd::twoInt<double> two_body;
  sbd::SetupIntegrals(fcidump, norb, nelec, scalar, one_body, two_body);

  sbd::gdb::DetIndexMap index_map;
  std::vector<sbd::gdb::ExcitationLookup> excitation_lookup;
  sbd::gdb::MakeHelpers(gdb_basis, bit_length, norb, index_map,
                        excitation_lookup, comm, comm, comm);
  std::vector<double> gdb_diagonal;
  sbd::gdb::makeQChamDiagTerms(
      gdb_basis, bit_length, norb, index_map, excitation_lookup,
      scalar, one_body, two_body, gdb_diagonal, comm, comm, comm);
  std::vector<double> gdb_result(input.size(), 0.0);
  sbd::gdb::mult(gdb_diagonal, input, gdb_result, bit_length, norb,
                 gdb_basis, index_map, excitation_lookup,
                 scalar, one_body, two_body, comm, comm, comm);

  int ca_op_norb = 0;
  int ca_op_nelec = 0;
  sbd::GeneralOp<double> hamiltonian;
  sbd::GeneralOp_From_FCIDump(
      "caop_gdb_hubbard4_fcidump.txt", comm, comm,
      ca_op_norb, ca_op_nelec, hamiltonian);
  const std::vector<int> slide{0};
  std::vector<double> ca_op_diagonal;
  sbd::makeCAOpHamDiagTerms(
      ca_op_basis, bit_length, slide, hamiltonian, ca_op_diagonal);
  std::vector<double> ca_op_result(input.size(), 0.0);
  sbd::mult(ca_op_diagonal, input, ca_op_result, ca_op_basis,
            bit_length, slide, hamiltonian, true, comm, comm, comm);
  std::vector<double> ca_op_without_sign(input.size(), 0.0);
  sbd::mult(ca_op_diagonal, input, ca_op_without_sign, ca_op_basis,
            bit_length, slide, hamiltonian, false, comm, comm, comm);

  double maximum_error = 0.0;
  double missing_sign_error = 0.0;
  for(std::size_t index = 0; index < input.size(); ++index) {
    maximum_error = std::max(
        maximum_error, std::abs(ca_op_result[index] - gdb_result[index]));
    missing_sign_error = std::max(
        missing_sign_error,
        std::abs(ca_op_without_sign[index] - gdb_result[index]));
  }
  if(missing_sign_error < 1.0e-6 || maximum_error > 1.0e-12)
    throw std::runtime_error("CAOP and GDB fermionic mult results differ");
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = 0;
  try {
    test_caop_matches_gdb(MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
      std::cout << "CAOP_GDB_HUBBARD_FERMION_MULT_TEST PASS" << std::endl;
  } catch(const std::exception& error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) std::cerr << error.what() << std::endl;
    result = 1;
  }
  MPI_Finalize();
  return result;
}
