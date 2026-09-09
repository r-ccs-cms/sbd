#include "sbd/sbd.h"
#include "sbd/chemistry/gdb/exact_external_observable.h"
#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"

#include <mpi.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int result = 0;
  try {
    if(argc != 6 || size != 1)
      throw std::invalid_argument(
          "usage: check_checkpoint_against_exact FCIDUMP DETFILE LOADNAME E0 BIT_LENGTH");
    sbd::FCIDump fcidump = sbd::LoadFCIDump(argv[1]);
    int norb_i = 0;
    int nelec = 0;
    double scalar = 0.0;
    sbd::oneInt<double> one;
    sbd::twoInt<double> two;
    sbd::SetupIntegrals(fcidump, norb_i, nelec, scalar, one, two);
    const std::size_t norb = static_cast<std::size_t>(norb_i);
    const std::size_t bit_length = std::stoull(argv[5]);
    sbd::det_vector<std::size_t>::init_elem_size(
        (2 * norb + bit_length - 1) / bit_length);
    sbd::det_vector<std::size_t> basis;
    sbd::load_basis_from_files({argv[2]}, basis, bit_length, 2 * norb,
                               MPI_COMM_WORLD);
    std::vector<double> coefficients;
    sbd::LoadWavefunction(argv[3], basis, MPI_COMM_SELF, MPI_COMM_WORLD,
                          MPI_COMM_SELF, coefficients);

    std::vector<std::size_t> counts(basis.size(), 1);
    std::vector<double> probabilities(basis.size(), 1.0);
    sbd::det_vector<std::size_t> children;
    std::vector<sbd::gdb::Contribution<double, double>> records;
    sbd::gdb::local_integral_driven_contribution_expansion(
        basis, coefficients, counts, probabilities, bit_length, norb,
        one, two, 0.0, 1000000, children, records);
    std::vector<double> amplitudes(records.size());
    for(std::size_t i = 0; i < records.size(); ++i)
      amplitudes[i] = records[i].weighted_amplitude;
    const auto exact =
        sbd::gdb::exact_external_observable<double, double>(
            children, amplitudes, basis, bit_length, norb, scalar, one, two,
            std::stod(argv[4]));
    std::cout << std::setprecision(17)
              << "exact_external_determinants=" << exact.external_determinants
              << " exact_variance=" << exact.variance
              << " exact_pt2=" << exact.pt2 << '\n';
  } catch(const std::exception& error) {
    if(rank == 0) std::cerr << "error: " << error.what() << '\n';
    result = 1;
  }
  MPI_Finalize();
  return result;
}
