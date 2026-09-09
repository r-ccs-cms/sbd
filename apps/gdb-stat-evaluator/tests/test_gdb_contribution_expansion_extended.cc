#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

std::vector<std::size_t> determinant(std::initializer_list<int> occupied) {
  std::vector<std::size_t> value(1, 0);
  for(const int orbital : occupied) sbd::setocc(value, 64, orbital, true);
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  constexpr std::size_t bit_length = 64;
  constexpr std::size_t norb = 3;
  sbd::det_vector<std::size_t>::init_elem_size(1);

  sbd::oneInt<double> one;
  one.norbs = 6;
  one.store.assign(36, 0.0);
  sbd::twoInt<double> two;
  two.norbs = 3;
  two.store.assign(21, 0.0);
  two.DirectMat.assign(36, 0.0);
  two.ExchangeMat.assign(36, 0.0);

  // A genuine double excitation {0,1}->{2,3}.
  two(2, 0, 3, 1) = 0.3;
  sbd::det_vector<std::size_t> parents;
  parents.push_back(determinant({0, 1}));
  std::vector<double> coefficients{0.5};
  std::vector<std::size_t> counts{2};
  std::vector<double> probabilities{0.25};
  sbd::det_vector<std::size_t> children;
  std::vector<sbd::gdb::Contribution<double, double>> records;
  sbd::gdb::local_integral_driven_contribution_expansion(
      parents, coefficients, counts, probabilities, bit_length, norb,
      one, two, 0.0, 1, children, records);

  int annihilated_first = 0;
  int annihilated_second = 1;
  int created_first = 2;
  int created_second = 3;
  const double hij = sbd::TwoExcite(
      parents[0], bit_length, annihilated_first, annihilated_second,
      created_first, created_second, one, two);
  const auto double_child = determinant({2, 3});
  bool ok = children.size() == 1 && records.size() == 1 &&
      children[0] == double_child &&
      std::abs(records[0].weighted_amplitude - 4.0 * hij) < 1.0e-14 &&
      std::abs(records[0].self_product - 8.0 * hij * hij) < 1.0e-14;

  // Two distinct parents emit the same child. The kernel must retain both
  // records because the membership owner performs the later sum and square.
  two.store.assign(21, 0.0);
  one(2, 0) = 0.5;
  one(2, 4) = 0.25;
  parents.clear();
  parents.push_back(determinant({0, 1}));
  parents.push_back(determinant({1, 4}));
  coefficients = {1.0, -2.0};
  counts = {1, 1};
  probabilities = {0.5, 0.5};
  sbd::gdb::local_integral_driven_contribution_expansion(
      parents, coefficients, counts, probabilities, bit_length, norb,
      one, two, 0.0, 2, children, records);
  const auto common_child = determinant({1, 2});
  std::vector<double> common_amplitudes;
  for(std::size_t i = 0; i < children.size(); ++i) {
    if(children[i] == common_child)
      common_amplitudes.push_back(records[i].weighted_amplitude);
  }
  std::sort(common_amplitudes.begin(), common_amplitudes.end());
  int from0 = 0;
  int to2 = 2;
  int from4 = 4;
  const double h0 = sbd::OneExcite(
      parents[0], bit_length, from0, to2, one, two);
  const double h1 = sbd::OneExcite(
      parents[1], bit_length, from4, to2, one, two);
  std::vector<double> expected{2.0 * coefficients[0] * h0,
                               2.0 * coefficients[1] * h1};
  std::sort(expected.begin(), expected.end());
  ok = ok && common_amplitudes.size() == 2 &&
      std::abs(common_amplitudes[0] - expected[0]) < 1.0e-14 &&
      std::abs(common_amplitudes[1] - expected[1]) < 1.0e-14;

  std::cout << (ok ? "gdb contribution expansion extended: PASS\n"
                   : "gdb contribution expansion extended: FAIL\n");
  MPI_Finalize();
  return ok ? 0 : 1;
}
