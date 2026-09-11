#include "sbd/chemistry/gdb/exact_external_observable.h"
#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"

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

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  constexpr std::size_t bit_length = 64;
  constexpr std::size_t norb = 2;
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
  basis.push_back(determinant({0, 1}));
  sbd::det_vector<std::size_t> manual_children;
  manual_children.push_back(determinant({1, 2}));
  manual_children.push_back(determinant({1, 2}));
  manual_children.push_back(determinant({0, 1}));
  const std::vector<double> manual_amplitudes{2.0, -0.5, 10.0};
  const auto manual =
      sbd::gdb::exact_external_observable<double, double>(
          manual_children, manual_amplitudes, basis, bit_length, norb,
          0.0, one_integrals, two_integrals, -2.0);
  bool ok = manual.external_determinants == 1 &&
            std::abs(manual.variance - 2.25) < 1.0e-14 &&
            std::abs(manual.pt2 + 1.125) < 1.0e-14;

  one_integrals(2, 0) = 0.5;
  one_integrals(3, 1) = 0.4;
  const std::vector<double> coefficients{1.0};
  const std::vector<std::size_t> counts{1};
  const std::vector<double> exhaustive_weights{1.0};
  sbd::det_vector<std::size_t> generated_children;
  std::vector<sbd::gdb::Contribution<double, double>> generated;
  sbd::gdb::local_integral_driven_contribution_expansion(
      basis, coefficients, counts, exhaustive_weights, bit_length, norb,
      one_integrals, two_integrals, 0.0, 4,
      generated_children, generated);
  std::vector<double> amplitudes(generated.size());
  for(std::size_t index = 0; index < generated.size(); ++index)
    amplitudes[index] = generated[index].weighted_amplitude;
  const auto integrated =
      sbd::gdb::exact_external_observable<double, double>(
          generated_children, amplitudes, basis, bit_length, norb,
          0.0, one_integrals, two_integrals, -1.0);
  ok = ok && integrated.external_determinants == 2 &&
       std::abs(integrated.variance - 0.41) < 1.0e-14 &&
       std::abs(integrated.pt2 + 0.41) < 1.0e-14;

  if(ok)
    std::cout << "gdb exact external observable: PASS\n";
  else
    std::cerr << "gdb exact external observable: FAIL"
              << " manual_variance=" << manual.variance
              << " manual_pt2=" << manual.pt2
              << " generated=" << generated_children.size()
              << " integrated_variance=" << integrated.variance
              << " integrated_pt2=" << integrated.pt2 << '\n';
  MPI_Finalize();
  return ok ? 0 : 1;
}
