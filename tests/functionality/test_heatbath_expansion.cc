/**
 * @file test_heatbath_expansion.cc
 * @brief MPI regression for determinant/coefficient slicing in HeatbathExpansion.
 */

#include "sbd/sbd.h"

#include <mpi.h>

#include <complex>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> make_determinant(
    const std::vector<int>& occupied,
    std::size_t bit_length,
    std::size_t spin_orbitals) {
  std::vector<std::size_t> determinant(
      (spin_orbitals + bit_length - 1) / bit_length, 0);
  for(const int orbital : occupied) {
    sbd::setocc(determinant, bit_length, orbital, true);
  }
  return determinant;
}

bool same_determinants(const sbd::det_vector<std::size_t>& lhs,
                       const sbd::det_vector<std::size_t>& rhs) {
  if(lhs.size() != rhs.size()) return false;
  for(std::size_t i = 0; i < lhs.size(); ++i) {
    if(std::vector<std::size_t>(lhs[i]) !=
       std::vector<std::size_t>(rhs[i])) {
      return false;
    }
  }
  return true;
}

template <typename ElemT>
void aligned_reference(
    const sbd::det_vector<std::size_t>& determinants,
    const std::vector<ElemT>& coefficients,
    std::size_t bit_length,
    std::size_t norb,
    const ElemT& scalar_integral,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    double cutoff,
    sbd::det_vector<std::size_t>& expanded,
    MPI_Comm b_comm,
    MPI_Comm comm) {
  int world_rank = 0;
  int b_rank = 0;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_rank(b_comm, &b_rank);

  MPI_Comm x_comm = MPI_COMM_NULL;
  MPI_Comm_split(comm, b_rank, world_rank, &x_comm);
  int x_rank = 0;
  int x_size = 1;
  MPI_Comm_rank(x_comm, &x_rank);
  MPI_Comm_size(x_comm, &x_size);

  std::size_t begin = 0;
  std::size_t end = determinants.size();
  sbd::get_mpi_range(x_size, x_rank, begin, end);
  sbd::det_vector<std::size_t> local_determinants(
      determinants.begin() + begin, determinants.begin() + end);
  std::vector<ElemT> local_coefficients(
      coefficients.begin() + begin, coefficients.begin() + end);

  expanded.clear();
  sbd::gdb::local_heatbath_expansion(
      local_determinants, local_coefficients, bit_length, norb,
      scalar_integral, one_integrals, two_integrals, cutoff, 0, expanded);

  sbd::sort_global_bitarray(expanded, comm);
  sbd::redistribution_bitarray(expanded, comm);
  MPI_Comm_free(&x_comm);
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if(argc != 3 || size != 4) {
    if(rank == 0) {
      std::cerr << "usage: test_heatbath_expansion FCIDUMP h|t"
                << " (exactly four MPI ranks required)\n";
    }
    MPI_Finalize();
    return 1;
  }

  const std::string replicated_dimension(argv[2]);
  if(replicated_dimension != "h" && replicated_dimension != "t") {
    if(rank == 0) std::cerr << "replicated dimension must be h or t\n";
    MPI_Finalize();
    return 1;
  }

  const int h_size = replicated_dimension == "h" ? 2 : 1;
  const int b_size = 2;
  const int t_size = replicated_dimension == "t" ? 2 : 1;
  MPI_Comm h_comm = MPI_COMM_NULL;
  MPI_Comm b_comm = MPI_COMM_NULL;
  MPI_Comm t_comm = MPI_COMM_NULL;
  sbd::gdb::DetBasisCommunicator(
      comm, h_size, b_size, t_size, h_comm, b_comm, t_comm);

  sbd::FCIDump fcidump;
  if(rank == 0) fcidump = sbd::LoadFCIDump(argv[1]);
  sbd::MpiBcast(fcidump, 0, comm);
  int norb = 0;
  int nelec = 0;
  double scalar_integral = 0.0;
  sbd::oneInt<double> one_integrals;
  sbd::twoInt<double> two_integrals;
  sbd::SetupIntegrals(fcidump, norb, nelec, scalar_integral,
                      one_integrals, two_integrals);

  constexpr std::size_t bit_length = 64;
  const std::size_t spin_orbitals = 2 * static_cast<std::size_t>(norb);
  sbd::det_vector<std::size_t>::init_elem_size(
      (spin_orbitals + bit_length - 1) / bit_length);
  sbd::det_vector<std::size_t, sbd::det_kind::half>::init_elem_size(
      (static_cast<std::size_t>(norb) + bit_length - 1) / bit_length);

  int b_rank = 0;
  MPI_Comm_rank(b_comm, &b_rank);
  sbd::det_vector<std::size_t> determinants;
  std::vector<double> coefficients;
  if(b_rank == 0) {
    determinants.push_back(make_determinant(
        {0,1,2,3,4,5,6,7,8,9}, bit_length, spin_orbitals));
    determinants.push_back(make_determinant(
        {0,1,2,3,4,5,6,7,8,11}, bit_length, spin_orbitals));
    coefficients = {0.91, -0.21};
  } else {
    determinants.push_back(make_determinant(
        {0,1,2,3,4,5,6,7,10,11}, bit_length, spin_orbitals));
    determinants.push_back(make_determinant(
        {0,1,2,3,4,5,6,9,10,11}, bit_length, spin_orbitals));
    coefficients = {0.31, 0.17};
  }

  bool all_modes_match = true;
  const std::vector<double> cutoffs{0.1, 0.01, 0.001};
  const std::vector<std::size_t> batch_sizes{1, 1000000};
  for(int truncated = 0; truncated <= 1; ++truncated) {
    sbd::det_vector<std::size_t> input_determinants = determinants;
    std::vector<double> input_coefficients = coefficients;
    if(truncated != 0) {
      input_determinants.resize(1);
      input_coefficients.resize(1);
    }
    for(const double cutoff : cutoffs) {
      sbd::det_vector<std::size_t> expected;
      aligned_reference(
          input_determinants, input_coefficients, bit_length,
          static_cast<std::size_t>(norb), scalar_integral, one_integrals,
          two_integrals, cutoff, expected, b_comm, comm);

      for(const std::size_t batch_size : batch_sizes) {
        for(int type = 0; type <= 1; ++type) {
          sbd::det_vector<std::size_t> actual;
          sbd::gdb::HeatbathExpansion(
              input_determinants, input_coefficients, bit_length,
              static_cast<std::size_t>(norb), scalar_integral, one_integrals,
              two_integrals, type, cutoff, batch_size,
              actual, b_comm, comm);

          const int local_match = same_determinants(actual, expected) ? 1 : 0;
          int global_match = 0;
          MPI_Allreduce(&local_match, &global_match, 1, MPI_INT, MPI_MIN, comm);
          all_modes_match = all_modes_match && global_match;
          if(rank == 0) {
            std::cout << "layout=" << replicated_dimension
                      << " truncated=" << truncated
                      << " cutoff=" << cutoff
                      << " batch=" << batch_size
                      << " type=" << type << " match: "
                      << (global_match ? "yes" : "NO") << '\n';
          }
        }
      }
    }
  }

  std::complex<double> complex_scalar_integral;
  sbd::oneInt<std::complex<double>> complex_one_integrals;
  sbd::twoInt<std::complex<double>> complex_two_integrals;
  int complex_norb = 0;
  int complex_nelec = 0;
  sbd::SetupIntegrals(
      fcidump, complex_norb, complex_nelec, complex_scalar_integral,
      complex_one_integrals, complex_two_integrals);
  std::vector<std::complex<double>> complex_coefficients;
  for(std::size_t i = 0; i < coefficients.size(); ++i) {
    const double imaginary_part = (i % 2 == 0) ? 0.13 : -0.07;
    complex_coefficients.emplace_back(coefficients[i], imaginary_part);
  }
  for(const double cutoff : cutoffs) {
    sbd::det_vector<std::size_t> expected;
    aligned_reference(
        determinants, complex_coefficients, bit_length,
        static_cast<std::size_t>(complex_norb), complex_scalar_integral,
        complex_one_integrals, complex_two_integrals, cutoff,
        expected, b_comm, comm);
    for(const std::size_t batch_size : batch_sizes) {
      sbd::det_vector<std::size_t> actual;
      sbd::gdb::HeatbathExpansion(
          determinants, complex_coefficients, bit_length,
          static_cast<std::size_t>(complex_norb), complex_scalar_integral,
          complex_one_integrals, complex_two_integrals, 1, cutoff, batch_size,
          actual, b_comm, comm);
      const int local_match = same_determinants(actual, expected) ? 1 : 0;
      int global_match = 0;
      MPI_Allreduce(&local_match, &global_match, 1, MPI_INT, MPI_MIN, comm);
      all_modes_match = all_modes_match && global_match;
      if(rank == 0) {
        std::cout << "layout=" << replicated_dimension
                  << " complex=1 cutoff=" << cutoff
                  << " batch=" << batch_size
                  << " type=1 match: "
                  << (global_match ? "yes" : "NO") << '\n';
      }
    }
  }

  MPI_Comm_free(&h_comm);
  MPI_Comm_free(&b_comm);
  MPI_Comm_free(&t_comm);
  MPI_Finalize();
  return all_modes_match ? 0 : 2;
}
