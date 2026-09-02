#ifndef SBD_FRAMEWORK_DETERMINANT_INITIALIZATION_H
#define SBD_FRAMEWORK_DETERMINANT_INITIALIZATION_H

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {

inline std::vector<size_t> from_string_checked(
    const std::string & bitstring, const size_t bit_length,
    const size_t total_bits) {
  if( bit_length == 0 || bit_length > 8 * sizeof(size_t) ) {
    throw std::invalid_argument("invalid determinant bit length");
  }
  if( bitstring.size() != total_bits ) {
    throw std::invalid_argument(
        "determinant bitstring length does not match the basis");
  }
  for(const char value : bitstring) {
    if( value != '0' && value != '1' ) {
      throw std::invalid_argument(
          "determinant bitstring must contain only 0 and 1");
    }
  }
  return from_string(bitstring, bit_length, total_bits);
}

template <typename ElemT, typename DetsContainer>
void BasisInitVectorFromDeterminant(
    std::vector<ElemT> & coefficients, const DetsContainer & basis,
    const std::vector<size_t> & requested_determinant, MPI_Comm b_comm) {
  coefficients.assign(basis.size(), ElemT(0));
  const auto match = std::find(
      basis.begin(), basis.end(), requested_determinant);
  size_t local_matches = match == basis.end() ? 0 : 1;
  if(match != basis.end() &&
     std::find(std::next(match), basis.end(), requested_determinant) !=
         basis.end()) {
    local_matches = 2;
  }
  size_t global_matches = 0;
  MPI_Allreduce(&local_matches, &global_matches, 1, SBD_MPI_SIZE_T,
                MPI_SUM, b_comm);
  if( global_matches != 1 ) {
    throw std::invalid_argument(
        global_matches == 0
            ? "initial determinant is not present in the distributed basis"
            : "initial determinant occurs more than once in the distributed basis");
  }
  if(match != basis.end()) {
    coefficients[static_cast<size_t>(std::distance(basis.begin(), match))] =
        ElemT(1);
  }
}

} // namespace sbd

#endif
