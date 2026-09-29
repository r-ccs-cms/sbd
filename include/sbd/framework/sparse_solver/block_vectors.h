#ifndef SBD_BLOCK_VECTORS_H
#define SBD_BLOCK_VECTORS_H
#include "small_matrix.h"
#include <omp.h>
#include "sbd/framework/type_def.h"
#include "sbd/framework/dm_vector.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd { namespace sparse_solver {
template<class T> struct QRResult { std::vector<T> factor; std::size_t rank=0; double discarded_norm=0; };
// V[v][i]: one contiguous local vector per outer entry. Small coefficients
// remain flat row-major arrays. No transpose/packing is used for arithmetic.
template<class T> void check_block_rows(const std::vector<std::vector<T>>& x,std::size_t rows) {
  for(const auto& v:x) if(v.size()!=rows) throw std::invalid_argument("ragged vector block");
}
template<class T>
std::vector<T> gram(const std::vector<std::vector<T>>& x,
    const std::vector<std::vector<T>>& y,MPI_Comm comm) {
  const auto rows=x.empty()?(y.empty()?0:y.front().size()):x.front().size();
  check_block_rows(x,rows);check_block_rows(y,rows);
  std::vector<T> a(matrix_size(x.size(),y.size()));
  for(std::size_t j=0;j<x.size();++j) for(std::size_t k=0;k<y.size();++k)
    sbd::InnerProduct(x[j],y[k],a[j*y.size()+k],comm);
  return a;
}
// Right transform of the logical (basis row, vector) block; shrinking only.
template<class T> void transform_in_place(std::vector<std::vector<T>>& x,
    const std::vector<T>& a,std::size_t cols,T scale=T(1)) {
  const auto width=x.size(),rows=width?x.front().size():0;
  check_block_rows(x,rows);
  if(cols>width||a.size()!=matrix_size(width,cols)) throw std::invalid_argument("block transform shape");
  std::vector<T> row(cols);
  for(std::size_t i=0;i<rows;++i) {
    std::fill(row.begin(),row.end(),T{});
    for(std::size_t j=0;j<cols;++j) for(std::size_t k=0;k<width;++k) row[j]+=scale*x[k][i]*a[k*cols+j];
    for(std::size_t j=0;j<cols;++j) x[j][i]=row[j];
  }
  x.resize(cols);
}
template<class T> void subtract_product(std::vector<std::vector<T>>& y,
    const std::vector<std::vector<T>>& x,const std::vector<T>& a) {
  const auto rows=y.empty()?0:y.front().size();
  check_block_rows(x,rows);check_block_rows(y,rows);
  if(a.size()!=matrix_size(x.size(),y.size())) throw std::invalid_argument("block subtract shape");
  for(std::size_t j=0;j<y.size();++j) for(std::size_t k=0;k<x.size();++k)
    for(std::size_t i=0;i<rows;++i) y[j][i]-=x[k][i]*a[k*y.size()+j];
}
// Two-pass MGS. Overwrite with independent Q vectors and shrink outer size.
template<class T>
QRResult<T> orthogonalize(std::vector<std::vector<T>>& x,MPI_Comm comm,
    double atol=1e-13,double rtol=1e-12) {
  if(!std::isfinite(atol)||!std::isfinite(rtol)||atol<0||rtol<0) throw std::invalid_argument("QR tolerances");
  const auto width=x.size(),rows=width?x.front().size():0;
  auto g=gram(x,x,comm);
  for(const auto& v:g) if(!std::isfinite(std::abs(v))) throw std::runtime_error("non-finite block Gram matrix");
  double scale=0,lost=0;
  for(std::size_t j=0;j<width;++j) scale=std::max(scale,std::sqrt(std::max(0.0,double(std::real(g[j*width+j])))));
  std::vector<T> c(matrix_size(width,width));std::size_t rank=0;
  std::vector<T> projection;projection.reserve(width);
  for(std::size_t j=0;j<width;++j) {
    // MGS returns the projection coefficients for one pass. Accumulate both
    // passes into the QR factor; the source vectors x[0..rank) are unchanged.
    for(int pass=0;pass<2;++pass) {
      sbd::MGS(x,rank,x[j],projection,comm);
      for(std::size_t k=0;k<rank;++k) c[k*width+j]+=projection[k];
    }
    T norm2{};
    sbd::InnerProduct(x[j],x[j],norm2,comm);
    if(!std::isfinite(std::abs(norm2))) throw std::runtime_error("non-finite QR norm");
    const double norm=std::sqrt(std::max(0.0,double(std::real(norm2))));
    if(norm<=atol+rtol*scale) {lost+=norm*norm;continue;}
    c[rank*width+j]=T(norm);
    for(std::size_t i=0;i<rows;++i) x[rank][i]=x[j][i]/T(norm);
    ++rank;
  }
  x.resize(rank);c.resize(rank*width);
  return {std::move(c),rank,std::sqrt(lost)};
}
}} // namespace
#endif
