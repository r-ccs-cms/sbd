#ifndef SBD_CAOP_SPECTRUM_RESULTS_H
#define SBD_CAOP_SPECTRUM_RESULTS_H
#include "sbd/framework/sparse_solver/block_lanczos_coefficients.h"

#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace sbd { namespace caop { namespace spectrum {
namespace ss=sbd::sparse_solver;
template<class ElemT=double> struct SpectrumResult {
  ss::LanczosCoefficients<ElemT> coefficients;
  std::vector<ElemT> c0;
  std::vector<std::size_t> orbitals;
  std::vector<std::string> operator_files; // Empty for single-particle response.
  bool addition=true;
  bool fermion=true;
  std::size_t sites=0,bit_length=64;
  double reference_energy=0,seed_discarded_norm=0;
  std::size_t basis_size=0;
};
// G^+_ab=<c_a (w+E0-H+i eta)^-1 c_b^dag>.
// G^-_ab=<c_b^dag (w-E0+H+i eta)^-1 c_a>: transpose the seed resolvent.
template<class ElemT>
inline std::vector<ss::Complex> green_function(const SpectrumResult<ElemT>& r,double omega,double eta) {
  auto g=ss::green_function(r.coefficients,r.c0,r.orbitals.size(),{omega+(r.addition?r.reference_energy:-r.reference_energy),eta},r.addition?1:-1);
  const auto s=r.orbitals.size();
  if(!r.addition) for(std::size_t i=0;i<s;++i) for(std::size_t j=0;j<i;++j) std::swap(g[i*s+j],g[j*s+i]);
  return g;
}
} } } // namespace
#endif
