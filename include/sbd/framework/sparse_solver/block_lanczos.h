#ifndef SBD_BLOCK_LANCZOS_H
#define SBD_BLOCK_LANCZOS_H
#include "block_lanczos_coefficients.h"
#include "block_vectors.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd { namespace sparse_solver {
/** Progress after one completed recurrence step; ranks are block dimensions. */
struct LanczosProgress {
  std::size_t step=0, current_rank=0, next_rank=0;
  double elapsed_seconds=0;
};
using LanczosProgressCallback=std::function<void(const LanczosProgress&)>;

/**
 * @brief Construct a block Lanczos basis by recurrence and return its coefficients.
 *
 * For a Hermitian operator H, advances the block three-term recurrence and
 * records the diagonal blocks A and coupling blocks B. Numerically dependent
 * residual directions are discarded, so the active block rank may decrease.
 *
 * @tparam T Real or complex scalar supported by the SBD distributed inner product.
 * @tparam ApplyH Callable implementing `apply_h(const Block& X, Block& Y)`,
 *         where Block is `std::vector<std::vector<T>>`.
 * @param apply_h Adds H*X to Y; it must preserve Y's existing contents rather
 *        than overwrite them. X and Y are distinct objects. The callable must
 *        leave X unchanged and preserve both block shapes. Each outer entry
 *        is one vector, and each inner entry is a locally owned basis element.
 *        H must act consistently with the vector distribution on comm; any
 *        additional communication required for its application is the
 *        callable's responsibility.
 * @param comm Communicator over which vector inner products are summed.
 *        All ranks must participate with matching block counts, column order,
 *        options, and a consistent global operator. Local row counts may differ
 *        and may be zero.
 * @param q Globally orthonormal initial block, checked before iteration.
 *        Passed by value as working storage; use std::move to transfer ownership.
 * @param rows Number of locally owned elements in each vector of q.
 * @param opt Iteration limit, rank thresholds, and orthogonality tolerance.
 * @param on_iteration Optional synchronous callback after each completed step,
 *        invoked on every participating rank. Elapsed time is local to that rank.
 *        No logging or additional MPI communication is performed by the callback
 *        mechanism. Observers must not alter solver state; exceptions propagate.
 * @return Coefficient blocks, ranks, and diagnostics. The final residual factor
 *         is stored in terminal_B. Stops on zero residual rank or max_steps;
 *         an empty initial block returns an empty coefficient chain.
 *
 * @note Retains only two large vector blocks. It reprojects against the current
 *       block and uses two-pass MGS within the residual block, but does not
 *       reorthogonalize against the full Krylov history. Neither the basis
 *       history nor eigenvectors are returned; this is not an eigensolver
 *       with an eigenvalue convergence criterion.
 * @throws std::invalid_argument If local shapes, options, or initial
 *         orthonormality are invalid.
 * @throws std::runtime_error If non-finite arithmetic or an inconsistent
 *         Hermitian projection is detected. Exceptions from apply_h propagate.
 */
template<class T,class ApplyH>
LanczosCoefficients<T> block_lanczos(
    ApplyH&& apply_h,MPI_Comm comm,std::vector<std::vector<T>> q,std::size_t rows,
    const LanczosOptions& opt={}, const LanczosProgressCallback& on_iteration={}) {
  const double started=on_iteration?MPI_Wtime():0;
  const auto width0=q.size();
  std::size_t width=width0;
  check_block_rows(q,rows);
  if(!opt.max_steps || opt.rank_atol<0 || opt.rank_rtol<0 ||
     !std::isfinite(opt.rank_atol) || !std::isfinite(opt.rank_rtol) ||
     !(opt.orthogonality_tolerance>0) || !std::isfinite(opt.orthogonality_tolerance))
    throw std::invalid_argument("Lanczos dimensions/options");
  LanczosCoefficients<T> out;
  out.ranks[0]=width;
  if(!width) return out;
  auto gg=gram(q,q,comm);
  for(std::size_t i=0;i<width;++i) for(std::size_t j=0;j<width;++j)
    if(!std::isfinite(std::abs(gg[i*width+j])) || std::abs(gg[i*width+j]-T(i==j))>opt.orthogonality_tolerance)
      throw std::invalid_argument("initial Lanczos block is not orthonormal");
  std::vector<std::vector<T>> previous(width,std::vector<T>(rows));
  std::size_t old_width=0;
  for(std::size_t step=0;step<opt.max_steps;++step) {
    const auto& x=q;
    auto& w=previous;
    if(step) transform_in_place(previous,adjoint(out.B.back(),width,old_width),width,T(-1));
    apply_h(x,w); // Y += H X; nonaliasing input and output.
    const auto work_gram=gram(w,w,comm);
    for(const auto& value:work_gram)
      if(!std::isfinite(std::abs(value))) throw std::runtime_error("non-finite Hamiltonian action");
    double work_scale=0;
    for(std::size_t j=0;j<width;++j)
      work_scale=std::max(work_scale,std::sqrt(std::max(0.0,double(std::real(work_gram[j*width+j])))));
    auto a=gram(x,w,comm);
    double hermitian_error=0, a_scale=0;
    for(std::size_t i=0;i<width;++i) for(std::size_t j=0;j<width;++j) {
      hermitian_error=std::max(hermitian_error,double(std::abs(a[i*width+j]-conjugate(a[j*width+i]))));
      a_scale=std::max(a_scale,double(std::abs(a[i*width+j])));
    }
    if(hermitian_error>opt.orthogonality_tolerance*(1+a_scale)) throw std::runtime_error("non-Hermitian projection or unstable recurrence");
    for(std::size_t i=0;i<width;++i) {
      a[i*width+i]=T(std::real(a[i*width+i]));
      for(std::size_t j=0;j<i;++j) { const T v=(a[i*width+j]+conjugate(a[j*width+i]))/T(2); a[i*width+j]=v; a[j*width+i]=conjugate(v); }
    }
    subtract_product(w,x,a);
    // Local second pass only; no old Krylov history is retained.
    auto correction=gram(x,w,comm);
    double cn=0;
    for(const auto& v:correction) cn+=std::norm(std::complex<double>(v));
    out.max_local_correction=std::max(out.max_local_correction,std::sqrt(cn));
    subtract_product(w,x,correction);
    for(std::size_t i=0;i<a.size();++i) a[i]+=correction[i];
    auto qr=orthogonalize(w,comm,opt.rank_atol+opt.rank_rtol*work_scale,opt.rank_rtol);
    out.ranks.push_back(qr.rank);
    out.A.push_back(std::move(a)); out.discarded_norms.push_back(qr.discarded_norm);
    if(on_iteration) on_iteration({step+1,width,qr.rank,MPI_Wtime()-started});
    if(!qr.rank || step+1==opt.max_steps) {
      out.terminal_B=std::move(qr.factor);
      out.stop_reason=qr.rank?"max_steps":"rank_threshold"; break;
    }
    out.B.push_back(std::move(qr.factor));
    old_width=width; width=qr.rank;
    q.swap(previous);
  }
  return out;
}

}} // namespace
#endif
