#include "sbd/chemistry/gdb/statistical_contribution_expansion.h"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  constexpr std::size_t bit_length = 64;
  constexpr std::size_t norb = 2;
  sbd::det_vector<std::size_t>::init_elem_size(1);

  sbd::det_vector<std::size_t> parents;
  std::vector<std::size_t> parent(1, 0);
  sbd::setocc(parent, bit_length, 0, true);
  sbd::setocc(parent, bit_length, 1, true);
  parents.push_back(parent);

  sbd::oneInt<double> one_integrals;
  one_integrals.norbs = 4;
  one_integrals.store.assign(16, 0.0);
  one_integrals(2, 0) = 0.5;
  one_integrals(3, 1) = 0.4;

  sbd::twoInt<double> two_integrals;
  two_integrals.norbs = static_cast<int>(norb);
  two_integrals.store.assign(6, 0.0);
  two_integrals.DirectMat.assign(16, 0.0);
  two_integrals.ExchangeMat.assign(16, 0.0);

  const std::vector<double> coefficients{0.5};
  const std::vector<std::size_t> counts{2};
  const std::vector<double> probabilities{0.25};
  sbd::det_vector<std::size_t> children;
  std::vector<sbd::gdb::Contribution<double, double>> values;
  sbd::gdb::local_integral_driven_contribution_expansion(
      parents, coefficients, counts, probabilities, bit_length, norb,
      one_integrals, two_integrals, 0.21, 1, children, values);

  const std::size_t generated_children = children.size();
  const std::size_t generated_values = values.size();
  double generated_x = 0.0;
  double generated_z = 0.0;
  bool ok = generated_children == 1 && generated_values == 1;
  if(ok) {
    generated_x = values[0].weighted_amplitude;
    generated_z = values[0].self_product;
    ok = !sbd::getocc(children[0], bit_length, 0) &&
         sbd::getocc(children[0], bit_length, 1) &&
         sbd::getocc(children[0], bit_length, 2) &&
         !sbd::getocc(children[0], bit_length, 3) &&
         std::abs(generated_x + 2.0) < 1.0e-14 &&
         std::abs(generated_z - 2.0) < 1.0e-14;
  }

  std::vector<std::size_t> zero_counts{0};
  sbd::gdb::local_integral_driven_contribution_expansion(
      parents, coefficients, zero_counts, probabilities, bit_length, norb,
      one_integrals, two_integrals, 0.0, 0, children, values);
  ok = ok && children.empty() && values.empty();

  if(ok) {
    std::cout << "gdb statistical contribution expansion: PASS\n";
  } else {
    std::cerr << "gdb statistical contribution expansion: FAIL"
              << " generated_children=" << generated_children
              << " generated_values=" << generated_values
              << " x=" << generated_x << " z=" << generated_z
              << " zero_children=" << children.size() << '\n';
  }
  MPI_Finalize();
  return ok ? 0 : 1;
}
