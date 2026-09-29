#ifndef SBD_CAOP_BLOCK_LANCZOS_COEFFICIENT_IO_H
#define SBD_CAOP_BLOCK_LANCZOS_COEFFICIENT_IO_H
#include "sbd/caop/spectrum/results.h"
#include <iomanip>
#include <fstream>
#include <type_traits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbd { namespace caop { namespace spectrum {
template<class ElemT>
inline void write_matrix(std::ostream& out,const std::vector<ElemT>& m,std::size_t rows,std::size_t cols) {
  if(m.size()!=ss::matrix_size(rows,cols)) throw std::invalid_argument("coefficient matrix shape");
  out<<rows<<' '<<cols<<'\n';
  for(auto x:m) out<<x<<' ';
  out<<'\n';
}
template<class ElemT>
inline std::vector<ElemT> read_matrix(std::istream& in,std::size_t& r,std::size_t& c) {
  if(!(in>>r>>c)||r>100000||c>100000||(c&&r>10000000/c)) throw std::runtime_error("invalid coefficient matrix size");
  std::vector<ElemT> m(r*c);
  for(auto& x:m) if(!(in>>x)||!std::isfinite(std::abs(x))) throw std::runtime_error("invalid coefficient value");
  return m;
}
template<class ElemT>
inline void save_coefficients(const std::string& path,const SpectrumResult<ElemT>& r,const ss::LanczosOptions& options) {
  ss::validate_coefficients(r.coefficients);
  std::ofstream out(path); if(!out) throw std::runtime_error("cannot write coefficients: "+path);
  constexpr bool complex=std::is_same_v<ElemT,std::complex<double>>;
  out<<std::setprecision(17)<<"EXTSBD_CAOP_SPECTRUM "<<(complex?"complex":"real")<<'\n';
  out<<r.addition<<' '<<r.reference_energy<<' '<<r.basis_size<<'\n';
  out<<r.fermion<<' '<<r.sites<<' '<<r.bit_length<<'\n';
  out<<options.max_steps<<' '<<options.rank_atol<<' '<<options.rank_rtol<<' '<<options.orthogonality_tolerance<<'\n';
  out<<r.orbitals.size()<<'\n'; for(auto o:r.orbitals) out<<o<<' '; out<<'\n';
  out<<r.operator_files.size()<<'\n';
  if(!r.operator_files.empty()) {
    if(r.operator_files.size()!=r.orbitals.size()||!r.addition) throw std::invalid_argument("general operator metadata");
    for(const auto& file:r.operator_files) out<<std::quoted(file)<<'\n';
  }
  out<<r.seed_discarded_norm<<' '<<r.coefficients.max_local_correction<<' '<<r.coefficients.stop_reason<<'\n';
  write_matrix(out,r.c0,r.coefficients.ranks.front(),r.orbitals.size());
  out<<r.coefficients.A.size()<<'\n';
  const auto& c=r.coefficients;
  for(std::size_t j=0;j<c.A.size();++j) write_matrix(out,c.A[j],c.ranks[j],c.ranks[j]);
  for(std::size_t j=0;j<c.B.size();++j) write_matrix(out,c.B[j],c.ranks[j+1],c.ranks[j]);
  write_matrix(out,c.terminal_B,c.ranks.back(),c.A.empty()?0:c.ranks[c.A.size()-1]);
  for(auto d:r.coefficients.discarded_norms) out<<d<<' '; out<<'\n';
  out.close();if(!out) throw std::runtime_error("coefficient write failed");
}
// The header identifies only the scalar representation, not an operator kind
// or a format version. Operator labels are a separately counted metadata list.
inline bool read_coefficient_scalar(std::istream& in) {
  std::string magic,scalar;
  if(!(in>>magic>>scalar)||magic!="EXTSBD_CAOP_SPECTRUM"||
     (scalar!="real" && scalar!="complex"))
    throw std::runtime_error("expected coefficient header: EXTSBD_CAOP_SPECTRUM real|complex");
  return scalar=="complex";
}
inline bool complex_coefficient_file(const std::string& path) {
  std::ifstream in(path);
  return read_coefficient_scalar(in);
}
template<class ElemT=double>
inline SpectrumResult<ElemT> load_coefficients(const std::string& path) {
  std::ifstream in(path);
  const bool complex=read_coefficient_scalar(in);
  if(complex && !std::is_same_v<ElemT,std::complex<double>>)
    throw std::runtime_error("coefficient scalar type mismatch");
  SpectrumResult<ElemT> r;ss::LanczosOptions o;std::size_t count=0;
  if(!(in>>r.addition>>r.reference_energy>>r.basis_size>>r.fermion>>r.sites>>r.bit_length>>o.max_steps>>o.rank_atol>>o.rank_rtol>>o.orthogonality_tolerance>>count)||!count||count>100000||!std::isfinite(r.reference_energy)||!r.sites||!r.bit_length||r.bit_length>64||!o.max_steps||!std::isfinite(o.rank_atol)||!std::isfinite(o.rank_rtol)||o.rank_atol<0||o.rank_rtol<0||!(o.orthogonality_tolerance>0)||!std::isfinite(o.orthogonality_tolerance)) throw std::runtime_error("invalid coefficient metadata");
  r.orbitals.resize(count);for(auto& v:r.orbitals) if(!(in>>v)) throw std::runtime_error("invalid orbital metadata");
  std::size_t label_count=0;
  if(!(in>>label_count)||(label_count!=0 && label_count!=count))
    throw std::runtime_error("invalid operator label count");
  if(label_count) {
    if(!r.addition) throw std::runtime_error("general operator resolvent sign");
    r.operator_files.resize(count);
    for(auto& file:r.operator_files) if(!(in>>std::quoted(file))||file.empty()) throw std::runtime_error("invalid operator label");
  }
  if(!(in>>r.seed_discarded_norm>>r.coefficients.max_local_correction>>r.coefficients.stop_reason)) throw std::runtime_error("invalid diagnostics");
  if(!std::isfinite(r.seed_discarded_norm)||r.seed_discarded_norm<0||!std::isfinite(r.coefficients.max_local_correction)||r.coefficients.max_local_correction<0) throw std::runtime_error("invalid diagnostics");
  for(auto orbital:r.orbitals) if(orbital>=(label_count?count:r.sites)) throw std::runtime_error("invalid orbital index");
  std::size_t rows=0,cols=0;
  r.c0=read_matrix<ElemT>(in,rows,cols);
  if(cols!=count||rows>count) throw std::runtime_error("C0 observable count mismatch");
  const auto seed_rank=rows;
  std::size_t steps=0;
  if(!(in>>steps)||steps>100000) throw std::runtime_error("invalid step count");
  auto& c=r.coefficients;
  c.ranks.resize(steps+1);
  for(std::size_t j=0;j<steps;++j) {
    c.A.push_back(read_matrix<ElemT>(in,rows,cols));
    if(rows!=cols) throw std::runtime_error("A shape");
    c.ranks[j]=rows;
  }
  for(std::size_t j=1;j<steps;++j) {
    c.B.push_back(read_matrix<ElemT>(in,rows,cols));
    if(rows!=c.ranks[j]||cols!=c.ranks[j-1]) throw std::runtime_error("B shape");
  }
  c.terminal_B=read_matrix<ElemT>(in,rows,cols);
  c.ranks.back()=rows;
  if(cols!=(steps?c.ranks[steps-1]:0)) throw std::runtime_error("terminal B shape");
  if(seed_rank!=c.ranks[0]) throw std::runtime_error("C0 rank mismatch");
  r.coefficients.discarded_norms.resize(steps);
  for(auto& d:r.coefficients.discarded_norms) if(!(in>>d)||!std::isfinite(d)||d<0) throw std::runtime_error("invalid discarded norm");
  std::string trailing;if(in>>trailing) throw std::runtime_error("trailing coefficient data");
  // Shape and physical-domain validation, independent of a Hamiltonian.
  green_function(r,0,1);
  return r;
}
} } } // namespace
#endif
