#ifndef SBD_SMALL_MATRIX_H
#define SBD_SMALL_MATRIX_H
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>


namespace sbd { namespace sparse_solver {
template<class T> T conjugate(T x) { return x; }
template<class T> std::complex<T> conjugate(std::complex<T> x) { return std::conj(x); }
// Small dense arrays are row-major; dimensions are explicit arguments.
inline std::size_t matrix_size(std::size_t rows,std::size_t cols) {
  if(cols && rows>std::numeric_limits<std::size_t>::max()/cols)
    throw std::overflow_error("matrix size overflow");
  return rows*cols;
}
template<class T> std::vector<T> adjoint(const std::vector<T>& a,std::size_t rows,std::size_t cols) {
  if(a.size()!=matrix_size(rows,cols)) throw std::invalid_argument("adjoint shape");
  std::vector<T> b(a.size());
  for(std::size_t i=0;i<rows;++i) for(std::size_t j=0;j<cols;++j) b[j*rows+i]=conjugate(a[i*cols+j]);
  return b;
}
template<class T> std::vector<T> multiply(const std::vector<T>& a,const std::vector<T>& b,
    std::size_t rows,std::size_t inner,std::size_t cols) {
  if(a.size()!=matrix_size(rows,inner)||b.size()!=matrix_size(inner,cols))
    throw std::invalid_argument("matrix product shape");
  std::vector<T> c(matrix_size(rows,cols));
  for(std::size_t i=0;i<rows;++i) for(std::size_t k=0;k<inner;++k)
    for(std::size_t j=0;j<cols;++j) c[i*cols+j]+=a[i*inner+k]*b[k*cols+j];
  return c;
}
}} // namespace
#endif
