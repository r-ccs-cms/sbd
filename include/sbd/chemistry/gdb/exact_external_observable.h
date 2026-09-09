#ifndef SBD_GDB_EXACT_EXTERNAL_OBSERVABLE_H
#define SBD_GDB_EXACT_EXTERNAL_OBSERVABLE_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/bit_manipulation.h"
#include "sbd/chemistry/basic/integrals.h"
#include "sbd/chemistry/basic/determinants.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sbd {
namespace gdb {

template <typename RealT>
struct ExactExternalObservable {
  RealT variance{};
  RealT pt2{};
  std::size_t external_determinants = 0;
};

// One-rank exhaustive reference. Contributions must be the unscaled c_i H_ai
// amplitudes. Equal children are summed before squaring, and determinants in
// the variational space are excluded.
template <typename ElemT, typename RealT>
ExactExternalObservable<RealT> exact_external_observable(
    const sbd::det_vector<std::size_t>& children,
    const std::vector<ElemT>& amplitudes,
    const sbd::det_vector<std::size_t>& variational_basis,
    std::size_t bit_length,
    std::size_t norb,
    const ElemT& scalar_integral,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    RealT reference_energy,
    RealT minimum_abs_denominator = RealT(0)) {
  if(children.size() != amplitudes.size())
    throw std::invalid_argument("child and amplitude counts differ");
  if(minimum_abs_denominator < RealT(0))
    throw std::invalid_argument("minimum denominator magnitude is negative");

  sbd::det_vector<std::size_t> sorted_basis(variational_basis);
  sbd::sort_bitarray(sorted_basis);
  sorted_basis.erase(std::unique(sorted_basis.begin(), sorted_basis.end()),
                     sorted_basis.end());

  std::vector<std::size_t> order(children.size());
  std::iota(order.begin(), order.end(), std::size_t(0));
  std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                             std::size_t rhs) {
    return sbd::less_from_back(children[lhs], children[rhs]);
  });

  ExactExternalObservable<RealT> result;
  std::size_t begin = 0;
  while(begin < order.size()) {
    const auto& child = children[order[begin]];
    ElemT residual{};
    std::size_t end = begin;
    while(end < order.size() && children[order[end]] == child) {
      residual += amplitudes[order[end]];
      ++end;
    }

    if(!std::binary_search(sorted_basis.begin(), sorted_basis.end(), child)) {
      const RealT squared_residual =
          static_cast<RealT>(sbd::SquaredNorm(residual));
      const ElemT diagonal_value = sbd::ZeroExcite(
          child, bit_length, norb, scalar_integral,
          one_integrals, two_integrals);
      const RealT diagonal = static_cast<RealT>(std::real(diagonal_value));
      const RealT denominator = reference_energy - diagonal;
      if(std::abs(denominator) <= minimum_abs_denominator)
        throw std::domain_error("external PT2 denominator is too small");
      result.variance += squared_residual;
      result.pt2 += squared_residual / denominator;
      ++result.external_determinants;
    }
    begin = end;
  }
  return result;
}

}  // namespace gdb
}  // namespace sbd

#endif
