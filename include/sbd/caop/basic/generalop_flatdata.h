/**
 * @file sbd/caop/basic/generalop_flatdata.h
 * @brief Flat data representation of normal-ordered GeneralOp terms.
 */
#ifndef SBD_CAOP_BASIC_GENERALOP_FLATDATA_H
#define SBD_CAOP_BASIC_GENERALOP_FLATDATA_H

#include <mpi.h>

#include "sbd/framework/det_vector.h"
#include "sbd/framework/type_def.h"
#include "sbd/caop/basic/generalop.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace sbd {

/**
 * Flat data for GeneralOp's non-diagonal terms, ordered by decreasing
 * magnitude of the original coefficient.
 *
 * CoeffT is deliberately selected by the caller. A magnitude-based algorithm
 * can store real magnitudes, while variance or PT2 code can retain signed or
 * complex coefficients in the same layout.
 */
template <typename CoeffT>
struct GeneralOpFlatData {
  std::vector<CoeffT> coefficients;
  det_vector<std::size_t> creation_masks;
  det_vector<std::size_t> annihilation_masks;
  std::vector<int> orbitals;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> creation_counts;
};

/** Build ket-side masks and flat operator data from a normal-ordered GeneralOp. */
template <typename FlatCoeffT, typename ElemT, typename Transform>
GeneralOpFlatData<FlatCoeffT> MakeKetSideGeneralOpFlatData(
    const GeneralOp<ElemT>& op,
    std::size_t determinant_words,
    std::size_t bit_length,
    Transform transform_coefficient) {
  using MagnitudeT = decltype(std::abs(ElemT{}));
  if(determinant_words == 0 || bit_length == 0 ||
     bit_length > 8 * sizeof(std::size_t))
    throw std::invalid_argument("invalid determinant dimensions");

  std::vector<std::size_t> order;
  order.reserve(op.NumOpTerms());
  for(std::size_t index = 0; index < op.NumOpTerms(); ++index)
    if(std::abs(op.OpCoef(static_cast<int>(index))) > MagnitudeT(0))
      order.push_back(index);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                            std::size_t rhs) {
    const auto lhs_coefficient =
        std::abs(op.OpCoef(static_cast<int>(lhs)));
    const auto rhs_coefficient =
        std::abs(op.OpCoef(static_cast<int>(rhs)));
    if(lhs_coefficient != rhs_coefficient)
      return lhs_coefficient > rhs_coefficient;
    return lhs < rhs;
  });

  GeneralOpFlatData<FlatCoeffT> flat_data;
  flat_data.coefficients.reserve(order.size());
  const std::vector<std::size_t> zero_mask(determinant_words, 0);
  flat_data.creation_masks =
      det_vector<std::size_t>(order.size(), zero_mask);
  flat_data.annihilation_masks =
      det_vector<std::size_t>(order.size(), zero_mask);
  flat_data.offsets.reserve(order.size() + 1);
  flat_data.creation_counts.reserve(order.size());
  flat_data.offsets.push_back(0);

  for(std::size_t output = 0; output < order.size(); ++output) {
    const std::size_t input = order[output];
    const ProductOp product = op.OpTerm(static_cast<int>(input));
    const std::size_t creation_count =
        static_cast<std::size_t>(product.n_dag());
    for(std::size_t op_index = 0; op_index < product.size(); ++op_index) {
      const CAOp factor = product.FOp(op_index);
      if(factor.d() != (op_index < creation_count))
        throw std::invalid_argument(
            "ket-side GeneralOp flat data requires normal-ordered terms");
      if(factor.q() < 0)
        throw std::invalid_argument("negative CAOP site index");
      const std::size_t site = static_cast<std::size_t>(factor.q());
      const std::size_t word = site / bit_length;
      if(word >= determinant_words)
        throw std::invalid_argument("CAOP site exceeds determinant storage");
      const std::size_t mask = std::size_t{1} << (site % bit_length);
      (factor.d() ? flat_data.creation_masks[output]
                  : flat_data.annihilation_masks[output])[word] |= mask;
      flat_data.orbitals.push_back(factor.q());
    }
    flat_data.coefficients.push_back(
        transform_coefficient(op.OpCoef(static_cast<int>(input))));
    flat_data.creation_counts.push_back(creation_count);
    flat_data.offsets.push_back(flat_data.orbitals.size());
  }
  return flat_data;
}

/**
 * Check ket-side occupation masks and generate the connected bra determinant.
 * The coefficient and fermionic sign are not applied.
 */
template <typename Determinant, typename FlatCoeffT>
bool TryGenerateBraDetFromGeneralOpFlatTerm(
    const Determinant& ket,
    const GeneralOpFlatData<FlatCoeffT>& flat_data,
    std::size_t term,
    std::vector<std::size_t>& bra) {
  const auto& creation = flat_data.creation_masks[term];
  const auto& annihilation = flat_data.annihilation_masks[term];
  if(ket.size() != creation.size() || ket.size() != annihilation.size())
    throw std::invalid_argument(
        "ket determinant word count differs from GeneralOp flat data");
  for(std::size_t word = 0; word < ket.size(); ++word) {
    if((ket[word] & annihilation[word]) != annihilation[word]) return false;
    if((ket[word] & (creation[word] & ~annihilation[word])) != 0)
      return false;
  }
  bra.assign(ket.begin(), ket.end());
  for(std::size_t word = 0; word < bra.size(); ++word) {
    bra[word] &= ~annihilation[word];
    bra[word] |= creation[word];
  }
  return true;
}

} // namespace sbd

#endif
