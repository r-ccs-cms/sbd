#ifndef SBD_CAOP_SPECTRUM_CHECKPOINT_IO_H
#define SBD_CAOP_SPECTRUM_CHECKPOINT_IO_H
#include <omp.h>
#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/bit_manipulation.h"
#include "sbd/framework/dm_vector.h"
#include "sbd/caop/basic/restart.h"
#include "sbd/framework/remap_wavefunction.h"
#include "sbd/framework/murmurhash_basis_distribution.h"
#include <fstream>
#include <limits>
#include <cmath>

// CAOP checkpoint support. Generic matching stays in framework.
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sbd { namespace caop { namespace spectrum {
using Det=std::vector<std::size_t>;
using Basis=sbd::det_vector<std::size_t>;
struct DetLess {
  template<class A,class B> bool operator()(const A& a,const B& b) const {
    return sbd::less_from_back(a,b);
  }
};
template<class ElemT=double> struct Wavefunction {
  Basis basis;                       // Only this rank's determinants.
  std::vector<ElemT> coefficients;  // Exactly basis.size() entries.
};
template<class Row> inline void validate_det(const Row& d,std::size_t sites,std::size_t bits) {
  if(!sites||!bits||bits>8*sizeof(std::size_t)||d.size()!=(sites+bits-1)/bits) throw std::invalid_argument("determinant dimensions");
  for(std::size_t k=0;k<d.size();++k) {
    const auto used=std::min(bits,sites-k*bits);
    if(used<8*sizeof(std::size_t) && (d[k]>>used)) throw std::invalid_argument("determinant has out-of-range bits");
  }
}
template<class ElemT>
inline typename sbd::GetRealType<ElemT>::RealT global_norm2(const Wavefunction<ElemT>& state,MPI_Comm comm) {
  int bad=state.basis.size()!=state.coefficients.size(),any=0;
  using RealT=typename sbd::GetRealType<ElemT>::RealT;
  RealT local=0,total=0;
  for(const auto& x:state.coefficients) { if(!std::isfinite(std::abs(x))) bad=1;local+=sbd::SquaredNorm(x); }
  MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
  if(any) throw std::invalid_argument("invalid local wavefunction shape/value");
  MPI_Allreduce(&local,&total,1,sbd::GetMpiType<RealT>::MpiT,MPI_SUM,comm);
  if(!std::isfinite(total)) throw std::invalid_argument("non-finite wavefunction norm");
  return total;
}
inline std::pair<std::size_t,std::size_t> row_range(std::size_t n,int rank,int size) {
  return {n/size*rank+std::min(std::size_t(rank),n%size),n/size*(rank+1)+std::min(std::size_t(rank+1),n%size)};
}
/**
 * @brief Read the saved determinant basis and amplitudes from native CAOP checkpoints.
 *
 * Unlike sbd::LoadWavefunction, this function reconstructs the saved basis;
 * it does not project onto a caller-supplied basis or normalize amplitudes.
 * Each participating rank reads its row interval of every saved shard, then
 * redistributes the records to unique hash owners. Saved rank ownership and
 * local record order are not preserved. Use RemapWavefunctionToBasis (or the
 * norm-accounting RemapWavefunction adapter) to project onto another basis.
 *
 * @tparam ElemT Stored amplitude type: double or std::complex<double>.
 * @param prefix File prefix; six-digit shard indices and .bin are appended.
 * @param shards Number of saved shards, independent of the size of comm.
 * @param sites Number of CAOP modes in each determinant.
 * @param bits Number of used bits per native size_t word.
 * @param comm Communicator distributing checkpoint rows and returned records.
 * @return Rank-local, sorted, unique determinants and their matching amplitudes.
 *         Amplitudes retain their saved values without normalization.
 * @pre All ranks of comm participate with the same arguments, including ranks
 *      with empty read intervals. ElemT, sites and bits match the saved data.
 * @note Files use native size_t and ElemT representations, with no endian or
 *       scalar-type metadata. This is not a portable interchange format.
 * @note Initializes the process-wide det_vector<size_t> word width to
 *       ceil(sites / bits). If already initialized, it must have that same
 *       width; a different width raises std::length_error.
 * @throws std::invalid_argument Invalid dimensions or duplicate source records.
 * @throws std::runtime_error Missing, malformed, incompatible or non-finite data.
 * @throws std::overflow_error Redistribution exceeds MPI count limits.
 * @warning Rank-local failures are not collectively recoverable. MPI callers
 *          must terminate the affected computation, as the CLI does via MPI_Abort.
 * @see sbd::SaveWavefunction, sbd::LoadWavefunction, sbd::RemapWavefunctionToBasis
 */
template<class ElemT=double>
inline Wavefunction<ElemT> ReadWavefunctionCheckpoint(const std::string& prefix,std::size_t shards,
    std::size_t sites,std::size_t bits,MPI_Comm comm) {
  if(!shards||!sites||!bits||bits>8*sizeof(std::size_t)) throw std::invalid_argument("checkpoint dimensions");
  int rank=0,size=1;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
  Basis::init_elem_size((sites+bits-1)/bits);
  Wavefunction<ElemT> input;
  for(std::size_t shard=0;shard<shards;++shard) {
    if(shard>std::size_t(std::numeric_limits<int>::max())) throw std::invalid_argument("too many shards");
    const auto path=sbd::statefilename(prefix,int(shard));
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in) throw std::runtime_error("cannot read checkpoint: "+path);
    const auto bytes=in.tellg();in.seekg(0);
    std::size_t n=0,words=0;
    in.read(reinterpret_cast<char*>(&n),sizeof n);in.read(reinterpret_cast<char*>(&words),sizeof words);
    if(!in||words!=(sites+bits-1)/bits||bytes<std::streamoff(2*sizeof(std::size_t))) throw std::runtime_error("checkpoint header: "+path);
    const auto record=words*sizeof(std::size_t)+sizeof(ElemT);
    const auto payload=std::size_t(bytes)-2*sizeof(std::size_t);
    if(payload%record||n!=payload/record) throw std::runtime_error("checkpoint size/type mismatch: "+path);
    const auto range=row_range(n,rank,size);
    in.seekg(2*sizeof(std::size_t)+range.first*words*sizeof(std::size_t));
    Det d(words);
    for(std::size_t i=range.first;i<range.second;++i) {
      in.read(reinterpret_cast<char*>(d.data()),words*sizeof(std::size_t));
      if(!in) throw std::runtime_error("checkpoint determinant read failed");
      validate_det(d,sites,bits);input.basis.push_back(d);
    }
    in.seekg(2*sizeof(std::size_t)+n*words*sizeof(std::size_t)+range.first*sizeof(ElemT));
    for(std::size_t i=range.first;i<range.second;++i) {
      ElemT c=0;in.read(reinterpret_cast<char*>(&c),sizeof c);
      if(!in||!std::isfinite(std::abs(c))) throw std::runtime_error("invalid checkpoint amplitude");
      input.coefficients.push_back(c);
    }
  }
  Wavefunction<ElemT> result;
  result.basis=input.basis;
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(result.basis,comm);
  // Also rejects duplicate source determinants across different saved shards.
  sbd::RemapWavefunctionToBasis(input.basis,input.coefficients,result.basis,
      comm,result.coefficients,true);
  return result;
}
// One reader per text file, followed by the existing distributed hash union.
// Files may be unsorted, overlap or include repeated determinants.
inline Basis LoadBasis(const std::vector<std::string>& paths,std::size_t sites,
    std::size_t bits,MPI_Comm comm) {
  if(!sites||!bits||bits>8*sizeof(std::size_t)) throw std::invalid_argument("basis dimensions");
  Basis::init_elem_size((sites+bits-1)/bits);
  int rank=0,size=1;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
  Basis basis;
  for(std::size_t file=0;file<paths.size();++file) {
    if(file%size!=std::size_t(rank)) continue;
    std::ifstream in(paths[file]);if(!in) throw std::runtime_error("cannot read basis: "+paths[file]);
    std::string line;
    while(std::getline(in,line)) {
      if(!line.empty()&&line.back()=='\r') line.pop_back();
      if(line.empty()) continue;
      if(line.size()!=sites||line.find_first_not_of("01")!=std::string::npos) throw std::runtime_error("basis must contain exactly sites binary digits per line");
      Det d((sites+bits-1)/bits);
      for(std::size_t i=0;i<sites;++i) if(line[sites-1-i]=='1') d[i/bits]|=std::size_t(1)<<(i%bits);
      basis.push_back(d);
    }
    if(in.bad()) throw std::runtime_error("basis read failed");
  }
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(basis,comm);
  return basis;
}
// Wavefunction adapter with norm accounting; distributed matching is in framework.
template<class ElemT=double> struct RemappedWavefunction {
  Wavefunction<ElemT> state;
  typename sbd::GetRealType<ElemT>::RealT retained_norm2=0,discarded_norm2=0;
};
// All input/output bases are distributed, not replicated. The target may have
// overlaps before union. Normalization is exclusively the caller's decision.
template<class ElemT=double>
inline RemappedWavefunction<ElemT> RemapWavefunction(const Wavefunction<ElemT>& source,
    Basis target,MPI_Comm comm) {
  global_norm2(source,comm);
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(target,comm);
  using RealT=typename sbd::GetRealType<ElemT>::RealT;
  RemappedWavefunction<ElemT> out;out.state.basis=std::move(target);
  sbd::RemapWavefunctionToBasis(source.basis,source.coefficients,out.state.basis,
      comm,out.state.coefficients);
  out.retained_norm2=global_norm2(out.state,comm);
  // Query membership back onto source rows, so tiny discarded weights are not
  // obtained by subtracting two nearly equal global norms.
  std::vector<double> present(out.state.basis.size(),1),source_present;
  sbd::RemapWavefunctionToBasis(out.state.basis,present,source.basis,comm,source_present);
  RealT lost=0;
  for(std::size_t i=0;i<source.coefficients.size();++i)
    if(source_present[i]==0) lost+=sbd::SquaredNorm(source.coefficients[i]);
  MPI_Allreduce(&lost,&out.discarded_norm2,1,sbd::GetMpiType<RealT>::MpiT,MPI_SUM,comm);
  return out;
}
} } } // namespace
#endif
