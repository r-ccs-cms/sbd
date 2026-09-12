// Adapted from public SBD 93ebabec6c84; CAOP diagonal adapter.
#ifndef SBD_CAOP_STAT_U_STATISTIC_EXTERNAL_OBSERVABLE_H
#define SBD_CAOP_STAT_U_STATISTIC_EXTERNAL_OBSERVABLE_H

#include "sbd/caop/stat/contribution_expansion.h"
#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/bit_manipulation.h"



#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <map>
#include <stdexcept>
#include <vector>

namespace sbd {
namespace ca_stat {

template <typename RealT>
struct UStatisticExternalObservable {
  RealT variance{};
  RealT pt2{};
  std::size_t external_determinants = 0;
  std::size_t unique_children = 0;
};

struct LocalObservableTiming {
  double child_sort_seconds = 0.0;
  double reduction_observable_seconds = 0.0;
};

// Core evaluator for a variational basis already sorted and uniqued with
// sbd::sort_bitarray(). Equal children are reduced before applying the square.
template <typename ElemT, typename RealT>
UStatisticExternalObservable<RealT>
u_statistic_external_observable_sorted_basis(
    const sbd::det_vector<std::size_t>& children,
    const std::vector<Contribution<ElemT, RealT>>& contributions,
    const sbd::det_vector<std::size_t>& variational_basis,
    std::size_t sample_count,
    std::size_t bit_length,
    const sbd::GeneralOp<ElemT>& hamiltonian,
    RealT reference_energy,
    RealT minimum_abs_denominator = RealT(0),
    bool compute_variance = true,
    bool compute_pt2 = true,
    bool* denominator_too_small = nullptr,
    LocalObservableTiming* timing = nullptr,
    const std::map<std::vector<std::size_t>, ElemT>* diagonals = nullptr) {
  if(children.size() != contributions.size())
    throw std::invalid_argument("child and contribution counts differ");
  if(sample_count < 2)
    throw std::invalid_argument("U-statistic requires at least two samples");
  if(minimum_abs_denominator < RealT(0))
    throw std::invalid_argument("minimum denominator magnitude is negative");
  if(timing != nullptr) *timing = LocalObservableTiming{};

  const double sort_start = MPI_Wtime();
  std::vector<std::size_t> order(children.size());
  std::iota(order.begin(), order.end(), std::size_t(0));
  std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                             std::size_t rhs) {
    return sbd::less_from_back(children[lhs], children[rhs]);
  });
  if(timing != nullptr)
    timing->child_sort_seconds = MPI_Wtime() - sort_start;

  const double observable_start = MPI_Wtime();
  const RealT normalization = static_cast<RealT>(sample_count) *
                              static_cast<RealT>(sample_count - 1);
  UStatisticExternalObservable<RealT> result;
  std::size_t begin = 0;
  while(begin < order.size()) {
    const auto& child = children[order[begin]];
    ElemT weighted_sum{};
    RealT self_product_sum{};
    std::size_t end = begin;
    while(end < order.size() && children[order[end]] == child) {
      const auto& contribution = contributions[order[end]];
      weighted_sum += contribution.weighted_amplitude;
      self_product_sum += contribution.self_product;
      ++end;
    }

    ++result.unique_children;
    if(!std::binary_search(variational_basis.begin(), variational_basis.end(),
                           child, [](const auto& a, const auto& b) {
                             return sbd::less_from_back(a, b);
                           })) {
      const RealT numerator =
          static_cast<RealT>(sbd::SquaredNorm(weighted_sum)) -
          self_product_sum;
      const RealT estimate = numerator / normalization;
      if(compute_variance) result.variance += estimate;
      if(compute_pt2) {
        const ElemT diagonal_value = diagonals ? diagonals->at(std::vector<std::size_t>(child.begin(), child.end()))
            : diagonal_element(child, bit_length, hamiltonian);
        const RealT diagonal = static_cast<RealT>(std::real(diagonal_value));
        const RealT denominator = reference_energy - diagonal;
        if(!std::isfinite(denominator) ||
           std::abs(denominator) <= minimum_abs_denominator) {
          if(denominator_too_small != nullptr)
            *denominator_too_small = true;
          else
            throw std::domain_error("external PT2 denominator is too small");
        } else {
          result.pt2 += estimate / denominator;
        }
      }
      ++result.external_determinants;
    }
    begin = end;
  }
  if(timing != nullptr)
    timing->reduction_observable_seconds = MPI_Wtime() - observable_start;
  return result;
}

// Convenient one-off interface accepting an unsorted variational basis.
template <typename ElemT, typename RealT>
UStatisticExternalObservable<RealT> u_statistic_external_observable(
    const sbd::det_vector<std::size_t>& children,
    const std::vector<Contribution<ElemT, RealT>>& contributions,
    const sbd::det_vector<std::size_t>& variational_basis,
    std::size_t sample_count,
    std::size_t bit_length,
    const sbd::GeneralOp<ElemT>& hamiltonian,
    RealT reference_energy,
    RealT minimum_abs_denominator = RealT(0),
    bool compute_variance = true,
    bool compute_pt2 = true,
    bool* denominator_too_small = nullptr) {
  sbd::det_vector<std::size_t> sorted_basis(variational_basis);
  sbd::sort_bitarray(sorted_basis);
  return u_statistic_external_observable_sorted_basis<ElemT, RealT>(
      children, contributions, sorted_basis, sample_count, bit_length, hamiltonian, reference_energy,
      minimum_abs_denominator, compute_variance, compute_pt2,
      denominator_too_small);
}

}  // namespace ca_stat
}  // namespace sbd

#endif
