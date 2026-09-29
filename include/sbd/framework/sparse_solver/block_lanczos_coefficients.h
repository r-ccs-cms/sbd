#ifndef SBD_BLOCK_LANCZOS_COEFFICIENTS_H
#define SBD_BLOCK_LANCZOS_COEFFICIENTS_H
#include "small_matrix.h"
#include <string>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd { namespace sparse_solver {
struct LanczosOptions {
  std::size_t max_steps=100;
  double rank_atol=1e-13, rank_rtol=1e-12, orthogonality_tolerance=1e-9;
};
template<class T> struct LanczosCoefficients {
  std::vector<std::vector<T>> A,B;
  std::vector<T> terminal_B;
  // r_j: ranks[j] for A[j]; ranks.back() is the terminal residual rank.
  // A[j]: r_j*r_j, B[j]: r_{j+1}*r_j, terminal_B: r_m*r_{m-1}.
  std::vector<std::size_t> ranks{0};
  std::vector<double> discarded_norms;
  double max_local_correction=0;
  std::string stop_reason="zero_seed";
};
using Complex=std::complex<double>;
template<class T> std::vector<Complex> complex_matrix(const std::vector<T>& a) {
  return std::vector<Complex>(a.begin(),a.end());
}
// Partial pivoting, multiple RHS; no explicit inverse in continued fraction.
inline std::vector<Complex> solve(std::vector<Complex> a,std::vector<Complex> b,
    std::size_t n,std::size_t nrhs) {
  if(a.size()!=matrix_size(n,n)||b.size()!=matrix_size(n,nrhs)) throw std::invalid_argument("linear solve shape");
  for(std::size_t k=0;k<n;++k) {
    std::size_t p=k;
    for(std::size_t i=k+1;i<n;++i) if(std::abs(a[i*n+k])>std::abs(a[p*n+k])) p=i;
    if(!(std::abs(a[p*n+k])>0) || !std::isfinite(std::abs(a[p*n+k]))) throw std::runtime_error("singular/non-finite continued fraction");
    for(std::size_t j=0;j<n;++j) std::swap(a[k*n+j],a[p*n+j]);
    for(std::size_t j=0;j<nrhs;++j) std::swap(b[k*nrhs+j],b[p*nrhs+j]);
    for(std::size_t i=k+1;i<n;++i) {
      const auto f=a[i*n+k]/a[k*n+k];
      for(std::size_t j=k+1;j<n;++j) a[i*n+j]-=f*a[k*n+j];
      for(std::size_t j=0;j<nrhs;++j) b[i*nrhs+j]-=f*b[k*nrhs+j];
    }
  }
  for(std::size_t i=n;i-->0;) for(std::size_t j=0;j<nrhs;++j) {
    for(std::size_t k=i+1;k<n;++k) b[i*nrhs+j]-=a[i*n+k]*b[k*nrhs+j];
    b[i*nrhs+j]/=a[i*n+i];
  }
  return b;
}
template<class T> void validate_coefficients(const LanczosCoefficients<T>& c) {
  const auto steps=c.A.size();
  if(c.ranks.size()!=steps+1 || c.B.size()!=(steps?steps-1:0))
    throw std::invalid_argument("coefficient chain shape");
  if(!steps) {
    if(c.ranks[0]||!c.terminal_B.empty()) throw std::invalid_argument("empty chain rank");
    return;
  }
  for(std::size_t j=0;j<steps;++j) {
    if(!c.ranks[j]||c.ranks[j+1]>c.ranks[j]||c.A[j].size()!=matrix_size(c.ranks[j],c.ranks[j]))
      throw std::invalid_argument("A/rank shape");
    const auto& b=j+1==steps?c.terminal_B:c.B[j];
    if(b.size()!=matrix_size(c.ranks[j+1],c.ranks[j])) throw std::invalid_argument("B shape");
  }
}
template<class T> std::vector<Complex> green_function(const LanczosCoefficients<T>& c,
    const std::vector<T>& c0,std::size_t orbitals,Complex z,double h_sign=1) {
  if(!(z.imag()>0)||!std::isfinite(z.real())||!std::isfinite(z.imag())||std::abs(h_sign)!=1)
    throw std::invalid_argument("resolvent requires finite z, eta>0 and sign +/-1");
  validate_coefficients(c);
  if(c.ranks[0]>orbitals||c0.size()!=matrix_size(c.ranks[0],orbitals)) throw std::invalid_argument("C0 shape");
  if(c.A.empty()) return std::vector<Complex>(matrix_size(orbitals,orbitals));
  const auto last=c.ranks[c.A.size()-1];
  std::vector<Complex> self(matrix_size(last,last));
  for(std::size_t j=c.A.size();j-->0;) {
    const auto n=c.ranks[j];
    auto d=complex_matrix(c.A[j]);
    for(std::size_t i=0;i<n;++i) for(std::size_t k=0;k<n;++k)
      d[i*n+k]=(i==k?z:Complex{})-h_sign*d[i*n+k]-self[i*n+k];
    if(!j) {
      auto f=complex_matrix(c0);
      return multiply(adjoint(f,n,orbitals),solve(std::move(d),f,n,orbitals),orbitals,n,orbitals);
    }
    const auto prev=c.ranks[j-1];
    auto b=complex_matrix(c.B[j-1]);
    self=multiply(adjoint(b,n,prev),solve(std::move(d),b,n,prev),prev,n,prev);
  }
  throw std::logic_error("unreachable");
}
inline std::vector<Complex> spectral_matrix(const std::vector<Complex>& g,std::size_t n) {
  if(g.size()!=matrix_size(n,n)) throw std::invalid_argument("spectral matrix shape");
  std::vector<Complex> a(g.size());
  const Complex factor(0,1/(2*std::acos(-1.0)));
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) a[i*n+j]=factor*(g[i*n+j]-std::conj(g[j*n+i]));
  return a;
}
}} // namespace
#endif
