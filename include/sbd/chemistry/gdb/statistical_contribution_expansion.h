#ifndef SBD_GDB_STATISTICAL_CONTRIBUTION_EXPANSION_H
#define SBD_GDB_STATISTICAL_CONTRIBUTION_EXPANSION_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/chemistry/basic/integrals.h"
#include "sbd/chemistry/basic/determinants.h"
#include "sbd/chemistry/gdb/heatbath_lookup.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sbd {
namespace gdb {

template <typename ElemT, typename RealT>
struct Contribution {
  ElemT weighted_amplitude{};
  RealT self_product{};
};

// Emit the two per-parent quantities needed by the U-statistic estimator:
//   x_ai = count_i * c_i * H_ai / p_i
//   z_ai = count_i * |c_i * H_ai|^2 / p_i^2
// Duplicate children are intentionally retained for later hash-owner reduce.
template <typename ElemT, typename RealT>
void local_integral_driven_contribution_expansion_with_lookup(
    const sbd::det_vector<std::size_t>& parents,
    const std::vector<ElemT>& coefficients,
    const std::vector<std::size_t>& counts,
    const std::vector<RealT>& probabilities,
    std::size_t bit_length,
    std::size_t norb,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    const detail::HeatbathLookup<ElemT, RealT>& lookup,
    RealT cutoff,
    std::size_t max_batch_size,
    sbd::det_vector<std::size_t>& children,
    std::vector<Contribution<ElemT, RealT>>& contributions) {
  const std::size_t parent_count = parents.size();
  if(coefficients.size() != parent_count || counts.size() != parent_count ||
     probabilities.size() != parent_count)
    throw std::invalid_argument("parent arrays must have identical sizes");
  if(cutoff < RealT(0))
    throw std::invalid_argument("heatbath cutoff must be non-negative");
  children.clear();
  contributions.clear();
  if(parent_count == 0) return;

  for(std::size_t i = 0; i < parent_count; ++i) {
    if(counts[i] == 0) continue;
    if(!(probabilities[i] > RealT(0)) || !std::isfinite(probabilities[i]))
      throw std::invalid_argument("sampled parent probability must be positive and finite");
  }
  constexpr std::size_t automatic_batch_size = 1000000;
  const std::size_t aggregate_batch_size =
      max_batch_size == 0 ? automatic_batch_size : max_batch_size;
  const std::size_t spin_orbitals = 2 * norb;

#pragma omp parallel
  {
#ifdef _OPENMP
    const std::size_t thread_count = static_cast<std::size_t>(omp_get_num_threads());
#else
    const std::size_t thread_count = 1;
#endif
    const std::size_t local_batch_size =
        std::max<std::size_t>(1, aggregate_batch_size / thread_count);
    sbd::det_vector<std::size_t> local_children;
    std::vector<Contribution<ElemT, RealT>> local_contributions;
    local_children.reserve(local_batch_size);
    local_contributions.reserve(local_batch_size);
    std::vector<std::size_t> child(parents[0].begin(), parents[0].end());
    std::vector<int> occupied(spin_orbitals);
    std::vector<int> unoccupied(spin_orbitals);

    auto flush = [&]() {
      if(local_children.empty()) return;
#pragma omp critical(sbd_gdb_statistical_contribution_append)
      {
        for(std::size_t i = 0; i < local_children.size(); ++i) {
          children.push_back(local_children[i]);
          contributions.push_back(local_contributions[i]);
        }
      }
      local_children.clear();
      local_contributions.clear();
    };

    auto emit = [&](std::size_t parent, const ElemT& hij) {
      const ElemT hc = coefficients[parent] * hij;
      if(std::abs(hc) <= cutoff) return;
      const RealT probability = probabilities[parent];
      const RealT count = static_cast<RealT>(counts[parent]);
      local_children.push_back(child);
      local_contributions.push_back({
          hc * (count / probability),
          count * static_cast<RealT>(sbd::SquaredNorm(hc)) /
              (probability * probability)});
      if(local_children.size() == local_batch_size) flush();
    };

#pragma omp for schedule(static)
    for(std::size_t parent = 0; parent < parent_count; ++parent) {
      if(counts[parent] == 0 || coefficients[parent] == ElemT(0)) continue;
      std::size_t occupied_count = 0;
      std::size_t unoccupied_count = 0;
      for(std::size_t orbital = 0; orbital < spin_orbitals; ++orbital) {
        if(sbd::getocc(parents[parent], bit_length, static_cast<int>(orbital)))
          occupied[occupied_count++] = static_cast<int>(orbital);
        else
          unoccupied[unoccupied_count++] = static_cast<int>(orbital);
      }

      for(std::size_t oi = 0; oi < occupied_count; ++oi) {
        int annihilated = occupied[oi];
        for(std::size_t ua = 0; ua < unoccupied_count; ++ua) {
          int created = unoccupied[ua];
          if((annihilated % 2) != (created % 2)) continue;
          const ElemT hij = sbd::OneExcite(parents[parent], bit_length,
                                           annihilated, created,
                                           one_integrals, two_integrals);
          if(std::abs(coefficients[parent] * hij) <= cutoff) continue;
          sbd::assign_det(child, parents[parent]);
          sbd::setocc(child, bit_length, annihilated, false);
          sbd::setocc(child, bit_length, created, true);
          emit(parent, hij);
        }
      }

      for(std::size_t oi = 0; oi < occupied_count; ++oi) {
        for(std::size_t oj = oi + 1; oj < occupied_count; ++oj) {
          int first = occupied[oi];
          int second = occupied[oj];
          for(auto entry = lookup.begin(first, second);
              entry != lookup.end(first, second); ++entry) {
            if(entry->abs_integral * std::abs(coefficients[parent]) <= cutoff) break;
            if(sbd::getocc(parents[parent], bit_length, entry->created_first) ||
               sbd::getocc(parents[parent], bit_length, entry->created_second)) continue;
            int created_first = entry->created_first;
            int created_second = entry->created_second;
            const ElemT hij = sbd::TwoExcite(
                parents[parent], bit_length, first, second,
                created_first, created_second, one_integrals, two_integrals);
            sbd::assign_det(child, parents[parent]);
            sbd::setocc(child, bit_length, first, false);
            sbd::setocc(child, bit_length, second, false);
            sbd::setocc(child, bit_length, created_first, true);
            sbd::setocc(child, bit_length, created_second, true);
            emit(parent, hij);
          }
        }
      }
    }
    flush();
  }
}

template <typename ElemT, typename RealT>
void local_integral_driven_contribution_expansion(
    const sbd::det_vector<std::size_t>& parents,
    const std::vector<ElemT>& coefficients,
    const std::vector<std::size_t>& counts,
    const std::vector<RealT>& probabilities,
    std::size_t bit_length,
    std::size_t norb,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    RealT cutoff,
    std::size_t max_batch_size,
    sbd::det_vector<std::size_t>& children,
    std::vector<Contribution<ElemT, RealT>>& contributions) {
  const std::size_t parent_count = parents.size();
  if(coefficients.size() != parent_count || counts.size() != parent_count ||
     probabilities.size() != parent_count)
    throw std::invalid_argument("parent arrays must have identical sizes");

  RealT max_abs_coefficient = RealT(0);
  for(std::size_t i = 0; i < parent_count; ++i) {
    if(counts[i] == 0) continue;
    max_abs_coefficient = std::max(
        max_abs_coefficient,
        static_cast<RealT>(std::abs(coefficients[i])));
  }
  detail::HeatbathLookup<ElemT, RealT> lookup(
      norb, two_integrals, cutoff, max_abs_coefficient);
  local_integral_driven_contribution_expansion_with_lookup(
      parents, coefficients, counts, probabilities, bit_length, norb,
      one_integrals, two_integrals, lookup, cutoff, max_batch_size,
      children, contributions);
}

}  // namespace gdb
}  // namespace sbd

#endif
