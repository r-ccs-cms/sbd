#include "sbd/chemistry/gdb/exact_external_observable.h"
#include "sbd/chemistry/gdb/u_statistic_external_observable.h"

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

std::size_t binomial(std::size_t n, std::size_t k) {
  if(k > n) return 0;
  k = std::min(k, n - k);
  std::size_t value = 1;
  for(std::size_t i = 1; i <= k; ++i)
    value = value * (n - k + i) / i;
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
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

  sbd::det_vector<std::size_t> basis;
  basis.push_back(determinant({2, 3}));
  basis.push_back(determinant({0, 1}));
  basis.push_back(determinant({2, 3}));
  sbd::det_vector<std::size_t> sorted_basis(basis);
  sbd::sort_bitarray(sorted_basis);
  const auto external = determinant({1, 2});
  sbd::det_vector<std::size_t> exact_children;
  exact_children.push_back(external);
  exact_children.push_back(external);
  const std::vector<double> exact_amplitudes{2.0, -0.5};
  const auto exact =
      sbd::gdb::exact_external_observable<double, double>(
          exact_children, exact_amplitudes, basis, bit_length, norb,
          0.0, one_integrals, two_integrals, -2.0);

  constexpr double p0 = 0.25;
  constexpr double p1 = 0.75;
  double expected_variance = 0.0;
  double expected_pt2 = 0.0;
  bool saw_negative_batch = false;
  bool sorted_core_matches_wrapper = true;
  for(std::size_t count0 = 0; count0 <= draws; ++count0) {
    const std::size_t count1 = draws - count0;
    sbd::det_vector<std::size_t> children;
    std::vector<sbd::gdb::Contribution<double, double>> values;
    if(count0 != 0) {
      children.push_back(external);
      values.push_back({static_cast<double>(count0) * 2.0 / p0,
                        static_cast<double>(count0) * 4.0 / (p0 * p0)});
    }
    if(count1 != 0) {
      children.push_back(external);
      values.push_back({static_cast<double>(count1) * -0.5 / p1,
                        static_cast<double>(count1) * 0.25 / (p1 * p1)});
    }
    const auto batch =
        sbd::gdb::u_statistic_external_observable<double, double>(
            children, values, basis, draws, bit_length, norb,
            0.0, one_integrals, two_integrals, -2.0);
    const auto sorted_batch = sbd::gdb::
        u_statistic_external_observable_sorted_basis<double, double>(
            children, values, sorted_basis, draws, bit_length, norb,
            0.0, one_integrals, two_integrals, -2.0);
    sorted_core_matches_wrapper = sorted_core_matches_wrapper &&
        batch.variance == sorted_batch.variance &&
        batch.pt2 == sorted_batch.pt2 &&
        batch.external_determinants == sorted_batch.external_determinants;
    const double probability = static_cast<double>(binomial(draws, count0)) *
                               std::pow(p0, static_cast<int>(count0)) *
                               std::pow(p1, static_cast<int>(count1));
    expected_variance += probability * batch.variance;
    expected_pt2 += probability * batch.pt2;
    saw_negative_batch = saw_negative_batch || batch.variance < 0.0;
  }

  bool rejected_small_batch = false;
  try {
    sbd::det_vector<std::size_t> no_children;
    std::vector<sbd::gdb::Contribution<double, double>> no_values;
    (void)sbd::gdb::u_statistic_external_observable<double, double>(
        no_children, no_values, basis, 1, bit_length, norb,
        0.0, one_integrals, two_integrals, -2.0);
  } catch(const std::invalid_argument&) {
    rejected_small_batch = true;
  }

  const bool ok = exact.external_determinants == 1 && saw_negative_batch &&
                  rejected_small_batch && sorted_core_matches_wrapper &&
                  std::abs(expected_variance - exact.variance) < 1.0e-13 &&
                  std::abs(expected_pt2 - exact.pt2) < 1.0e-13;
  if(ok)
    std::cout << "gdb U-statistic finite-batch expectation: PASS\n";
  else
    std::cerr << "gdb U-statistic finite-batch expectation: FAIL"
              << " exact_variance=" << exact.variance
              << " expected_variance=" << expected_variance
              << " exact_pt2=" << exact.pt2
              << " expected_pt2=" << expected_pt2
              << " saw_negative=" << saw_negative_batch
              << " sorted_core_matches_wrapper=" << sorted_core_matches_wrapper
              << " rejected_N1=" << rejected_small_batch << '\n';
  MPI_Finalize();
  return ok ? 0 : 1;
}
