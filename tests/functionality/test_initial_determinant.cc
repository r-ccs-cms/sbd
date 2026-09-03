#include "sbd/sbd.h"
#include "sbd/framework/determinant_initialization.h"

#include <cmath>
#include <complex>
#include <iostream>

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  bool ok = (size == 4);

  try {
    const auto parsed = sbd::from_string_checked("10000001", 4, 8);
    ok = ok && parsed.size() == 2 && parsed[0] == 1 && parsed[1] == 8;
    try {
      static_cast<void>(sbd::from_string_checked("1000000", 4, 8));
      ok = false;
    } catch(const std::invalid_argument &) {}
    try {
      static_cast<void>(sbd::from_string_checked("1000000x", 4, 8));
      ok = false;
    } catch(const std::invalid_argument &) {}

    sbd::det_vector<size_t> distributed_basis;
    distributed_basis.emplace_back(std::vector<size_t>{static_cast<size_t>(rank)});
    std::vector<double> coefficients;
    sbd::BasisInitVectorFromDeterminant(
        coefficients, distributed_basis, std::vector<size_t>{2},
        MPI_COMM_WORLD);
    double local_weight = coefficients.empty() ? 0.0 : coefficients[0];
    double global_weight = 0.0;
    MPI_Allreduce(&local_weight, &global_weight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    ok = ok && std::abs(global_weight - 1.0) < 1.0e-14;
    ok = ok && ((rank == 2 && local_weight == 1.0) ||
                (rank != 2 && local_weight == 0.0));
    std::vector<std::complex<double>> complex_coefficients;
    sbd::BasisInitVectorFromDeterminant(
        complex_coefficients, distributed_basis, std::vector<size_t>{2},
        MPI_COMM_WORLD);
    ok = ok && complex_coefficients.size() == 1 &&
         complex_coefficients[0] == std::complex<double>(local_weight, 0.0);

    try {
      sbd::BasisInitVectorFromDeterminant(
          coefficients, distributed_basis, std::vector<size_t>{size_t(8)},
          MPI_COMM_WORLD);
      ok = false;
    } catch(const std::invalid_argument &) {}

    sbd::det_vector<size_t> duplicate_basis;
    duplicate_basis.emplace_back(std::vector<size_t>{
        static_cast<size_t>(rank == 1 || rank == 2 ? 7 : rank)});
    try {
      sbd::BasisInitVectorFromDeterminant(
          coefficients, duplicate_basis, std::vector<size_t>{size_t(7)},
          MPI_COMM_WORLD);
      ok = false;
    } catch(const std::invalid_argument &) {}

    sbd::det_vector<size_t, sbd::det_kind::half> adets;
    sbd::det_vector<size_t, sbd::det_kind::half> bdets;
    for(size_t value = 0; value < 4; ++value) {
      adets.emplace_back(std::vector<size_t>{value});
      bdets.emplace_back(std::vector<size_t>{value});
    }
    std::vector<double> product_coefficients;
    sbd::BasisInitVectorFromDeterminants(
        product_coefficients, adets, bdets, 2, 2,
        std::vector<size_t>{2}, std::vector<size_t>{1}, MPI_COMM_WORLD);
    double local_product_weight = 0.0;
    for(double value : product_coefficients) local_product_weight += value;
    double global_product_weight = 0.0;
    MPI_Allreduce(&local_product_weight, &global_product_weight, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ok = ok && std::abs(global_product_weight - 1.0) < 1.0e-14;
    ok = ok && ((rank == 2 && product_coefficients.size() == 4 &&
                 product_coefficients[1] == 1.0) ||
                (rank != 2 && local_product_weight == 0.0));

    sbd::det_vector<size_t, sbd::det_kind::half> duplicate_adets = adets;
    duplicate_adets.emplace_back(std::vector<size_t>{size_t(2)});
    try {
      sbd::BasisInitVectorFromDeterminants(
          product_coefficients, duplicate_adets, bdets, 2, 2,
          std::vector<size_t>{2}, std::vector<size_t>{1}, MPI_COMM_WORLD);
      ok = false;
    } catch(const std::invalid_argument &) {}

  } catch(const std::exception & error) {
    std::cerr << "rank " << rank << ": " << error.what() << std::endl;
    ok = false;
  }

  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "initial determinant: "
              << (global_ok ? "PASS" : "FAIL") << std::endl;
  }
  MPI_Finalize();
  return global_ok ? 0 : 1;
}
