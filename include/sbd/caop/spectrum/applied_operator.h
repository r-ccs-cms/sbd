#ifndef SBD_CAOP_APPLIED_OPERATOR_H
#define SBD_CAOP_APPLIED_OPERATOR_H
#include "sbd/caop/spectrum/checkpoint_io.h"
#include "sbd/framework/remap_wavefunction.h"
#include "sbd/framework/murmurhash_basis_distribution.h"
#include "sbd/framework/sparse_solver/small_matrix.h"

#include "sbd/caop/basic/generalop.h"
#include "sbd/caop/basic/generalop_flatdata.h"
#include "sbd/caop/basic/mkham.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sbd { namespace caop { namespace spectrum {
// CAOP-specific A|psi0> construction. GDB/TPB belong in their own model layer.
template<class Row>
inline bool single_particle_excite(const Row& source,std::size_t orbital,bool addition,bool fermion,
    std::size_t bits,Det& target,double& phase) {
  target.assign(source.begin(),source.end());
  const auto mask=std::size_t(1)<<(orbital%bits);
  if(bool(target.at(orbital/bits)&mask)==addition) return false;
  phase=fermion?sbd::bit_string_sign_factor(target,bits,orbital%bits,orbital/bits):1;
  target[orbital/bits]^=mask;return true;
}
template<class ElemT=double> struct SeedSpace {
  Basis basis;                      // Only this rank's excitation determinants.
  std::size_t generated_basis_size=0; // Global unique generated rows, before extra.
  std::vector<ElemT> local_seeds;   // Exactly basis.size() * number of seeds.
};
// Count generated determinants before union with extra. Zero amplitudes and
// cancelled contributions still count as generated rows, not nonzero support.
template<class ElemT>
inline void complete_seed_basis(SeedSpace<ElemT>& out,const Basis& extra,MPI_Comm comm) {
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(out.basis,comm);
  const std::size_t local=out.basis.size(),extra_local=extra.size();
  std::size_t extra_global=0;
  MPI_Allreduce(&local,&out.generated_basis_size,1,SBD_MPI_SIZE_T,MPI_SUM,comm);
  MPI_Allreduce(&extra_local,&extra_global,1,SBD_MPI_SIZE_T,MPI_SUM,comm);
  if(extra_global) {
    const auto words=out.basis.cflat().size();
    if(extra.size()>out.basis.cflat().max_size()/out.basis.elem_size()-out.basis.size())
      throw std::length_error("extra basis capacity overflow");
    out.basis.resize(out.basis.size()+extra.size());
    std::copy(extra.cflat().begin(),extra.cflat().end(),out.basis.flat().begin()+words);
    sbd::murmur_basis::redistribute_unique_determinants_by_hash(out.basis,comm);
  }
}
namespace detail {
// Split one b shard without replicating the full parent wavefunction over h/t.
// All ranks participate; only rank zero supplies input. Empty shards are valid.
template<class ElemT>
Wavefunction<ElemT> scatter_parents(const Wavefunction<ElemT>& input,MPI_Comm comm) {
  int rank=0,size=1;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
  Wavefunction<ElemT> out;
  if(rank==0) {
    for(int dest=size-1;dest>=0;--dest) {
      const auto range=row_range(input.basis.size(),dest,size);
      Wavefunction<ElemT> part;
      const auto width=input.basis.elem_size();
      part.basis.resize(range.second-range.first);
      std::copy(input.basis.cflat().begin()+range.first*width,
          input.basis.cflat().begin()+range.second*width,part.basis.flat().begin());
      part.coefficients.assign(input.coefficients.begin()+range.first,input.coefficients.begin()+range.second);
      if(dest==0) out=std::move(part);
      else {sbd::MpiSend(part.basis,dest,comm);sbd::MpiSend(part.coefficients,dest,comm);}
    }
  } else {sbd::MpiRecv(out.basis,0,comm);sbd::MpiRecv(out.coefficients,0,comm);}
  return out;
}
// Contributions already have their final b hash owner. Merge the h/t shards
// at that same b coordinate, retaining signed/complex amplitudes per column.
template<class ElemT>
void gather_contributions(Basis& basis,std::vector<ElemT>& values,MPI_Comm comm) {
  int rank=0,size=1;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
  if(rank==0) {
    for(int source=1;source<size;++source) {
      Basis incoming;std::vector<ElemT> amplitudes;
      sbd::MpiRecv(incoming,source,comm);sbd::MpiRecv(amplitudes,source,comm);
      const auto rows=basis.size(),words=basis.cflat().size();
      const auto capacity=std::min(basis.cflat().max_size()/basis.elem_size(),values.max_size());
      if(incoming.size()>capacity-rows) throw std::length_error("gathered seed capacity overflow");
      basis.resize(rows+incoming.size());values.resize(rows+incoming.size());
      std::copy(incoming.cflat().begin(),incoming.cflat().end(),basis.flat().begin()+words);
      std::copy(amplitudes.begin(),amplitudes.end(),values.begin()+rows);
    }
  } else {
    sbd::MpiSend(basis,0,comm);sbd::MpiSend(values,0,comm);
    basis=Basis{};values.clear();
  }
}
template<class ElemT>
void sum_contributions(Basis& basis,std::vector<ElemT>& values) {
  std::vector<std::size_t> order(basis.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return DetLess{}(basis[a],basis[b]);});
  Basis unique;std::vector<ElemT> summed;unique.reserve(basis.size());summed.reserve(values.size());
  for(auto i:order) {
    if(!unique.empty() && std::equal(basis[i].begin(),basis[i].end(),unique[unique.size()-1].begin()))
      summed.back()+=values[i];
    else {unique.push_back(basis[i]);summed.push_back(values[i]);}
  }
  basis=std::move(unique);values=std::move(summed);
}
template<class ElemT>
void collect_column(Basis& basis,std::vector<ElemT>& values,MPI_Comm b_comm,MPI_Comm h_comm,MPI_Comm t_comm) {
  sbd::murmur_basis::redistribute_determinant_values_by_hash(basis,values,b_comm);
  sum_contributions(basis,values);
  gather_contributions(basis,values,t_comm);
  int t_rank=0;MPI_Comm_rank(t_comm,&t_rank);
  if(t_rank==0) {
    gather_contributions(basis,values,h_comm);
    sum_contributions(basis,values);
  }
}
template<class ElemT>
SeedSpace<ElemT> assemble_seeds(std::vector<Basis>& generated,std::vector<std::vector<ElemT>>& amplitudes,
    const Basis& extra,MPI_Comm b_comm,MPI_Comm h_comm,MPI_Comm t_comm) {
  int h_rank=0,t_rank=0;MPI_Comm_rank(h_comm,&h_rank);MPI_Comm_rank(t_comm,&t_rank);
  SeedSpace<ElemT> out;
  if(h_rank!=0 || t_rank!=0) return out;
  std::size_t rows=0;
  const auto capacity=out.basis.cflat().max_size()/out.basis.elem_size();
  for(const auto& basis:generated) {
    if(basis.size()>capacity-rows) throw std::length_error("seed union capacity overflow");
    rows+=basis.size();
  }
  out.basis.resize(rows);
  auto destination=out.basis.flat().begin();
  for(const auto& basis:generated)
    destination=std::copy(basis.cflat().begin(),basis.cflat().end(),destination);
  complete_seed_basis(out,extra,b_comm);
  const auto count=generated.size();
  out.local_seeds.assign(sbd::sparse_solver::matrix_size(out.basis.size(),count),0);
  for(std::size_t v=0;v<count;++v) {
    std::vector<ElemT> mapped;
    sbd::RemapWavefunctionToBasis(generated[v],amplitudes[v],out.basis,b_comm,mapped,true);
    for(std::size_t i=0;i<out.basis.size();++i) out.local_seeds[i*count+v]=mapped[i];
  }
  return out;
}
} // namespace detail
// With h/t communicators, state is already split into disjoint parent shards.
// Every rank holds all operators; only h=t=0 supplies extra and receives seeds.
template<class ElemT>
inline SeedSpace<ElemT> single_particle_seeds(const Wavefunction<ElemT>& state,
    const std::vector<std::size_t>& orbitals,bool addition,bool fermion,
    std::size_t sites,std::size_t bits,const Basis& extra,MPI_Comm comm,MPI_Comm h_comm=MPI_COMM_SELF,MPI_Comm t_comm=MPI_COMM_SELF) {
  if(orbitals.empty()) throw std::invalid_argument("no excitation orbitals");
  for(auto o:orbitals) if(o>=sites) throw std::invalid_argument("excitation orbital out of range");
  global_norm2(state,comm); // Collective shape/finite checks, no normalization.
  for(const auto& d:extra) validate_det(d,sites,bits);
  for(const auto& d:state.basis) validate_det(d,sites,bits);
  const auto count=orbitals.size();
  std::vector<Basis> generated(count);
  std::vector<std::vector<ElemT>> amplitudes(count);
  Det d;double phase=0;
  for(std::size_t v=0;v<count;++v) {
    generated[v].reserve(state.basis.size());amplitudes[v].reserve(state.basis.size());
    for(std::size_t i=0;i<state.basis.size();++i) {
      if(!single_particle_excite(state.basis[i],orbitals[v],addition,fermion,bits,d,phase)) continue;
      generated[v].push_back(d);amplitudes[v].push_back(phase*state.coefficients[i]);
    }
    detail::collect_column(generated[v],amplitudes[v],comm,h_comm,t_comm);
  }
  return detail::assemble_seeds(generated,amplitudes,extra,comm,h_comm,t_comm);
}
// One complete CAOP operator per column on every rank. Parent shards are disjoint.
template<class ElemT>
inline SeedSpace<ElemT> general_operator_seeds(const Wavefunction<ElemT>& state,
    const std::vector<sbd::GeneralOp<ElemT>>& operators,bool fermion,
    std::size_t sites,std::size_t bits,const Basis& extra,MPI_Comm comm,MPI_Comm h_comm=MPI_COMM_SELF,MPI_Comm t_comm=MPI_COMM_SELF) {
  if(operators.empty()) throw std::invalid_argument("no applied operators");
  global_norm2(state,comm);
  for(const auto& d:state.basis) validate_det(d,sites,bits);
  for(const auto& d:extra) validate_det(d,sites,bits);
  const auto count=operators.size();
  std::vector<Basis> generated(count);
  std::vector<std::vector<ElemT>> amplitudes(count);
  for(std::size_t v=0;v<count;++v) {
    const auto& op=operators[v];
    if((op.NumOpTerms()||op.NumNcTerms()) && std::size_t(op.max_index())>=sites)
      throw std::invalid_argument("applied operator index out of range");
    const auto flat=sbd::MakeKetSideGeneralOpFlatData<ElemT>(op,state.basis.elem_size(),bits,
        [](ElemT c){return c;});
    std::vector<ElemT> diagonal;
    sbd::makeCAOpHamDiagTerms(state.basis,bits,{},op,diagonal);
    auto& basis=generated[v];auto& values=amplitudes[v];
    // Count first, then assign each parent a disjoint output interval.
    // Both passes preserve parent/term order independently of the thread count.
    std::vector<std::size_t> offsets(state.basis.size()+1,0);
#pragma omp parallel
    {
      Det bra;
#pragma omp for schedule(static)
      for(std::size_t i=0;i<state.basis.size();++i) {
        std::size_t nout=diagonal[i]!=ElemT(0);
        for(std::size_t n=0;n<flat.coefficients.size();++n)
          if(sbd::TryGenerateBraDetFromGeneralOpFlatTerm(state.basis[i],flat,n,bra)) ++nout;
        offsets[i+1]=nout;
      }
    }
    const auto capacity=std::min(basis.flat().max_size()/basis.elem_size(),values.max_size());
    for(std::size_t i=0;i<state.basis.size();++i) {
      if(offsets[i+1]>capacity-offsets[i]) throw std::length_error("operator seed capacity overflow");
      offsets[i+1]+=offsets[i];
    }
    basis.resize(offsets.back());values.resize(offsets.back());
#pragma omp parallel
    {
      Det bra,work;
#pragma omp for schedule(static)
      for(std::size_t i=0;i<state.basis.size();++i) {
        auto pos=offsets[i];
        if(diagonal[i]!=ElemT(0)) {
          std::copy(state.basis[i].begin(),state.basis[i].end(),basis[pos].begin());
          values[pos++]=diagonal[i]*state.coefficients[i];
        }
        for(std::size_t n=0;n<flat.coefficients.size();++n) {
          if(!sbd::TryGenerateBraDetFromGeneralOpFlatTerm(state.basis[i],flat,n,bra)) continue;
          double phase=1;
          if(fermion) {
            work.assign(state.basis[i].begin(),state.basis[i].end());
            for(auto k=flat.offsets[n+1];k>flat.offsets[n];) {
              const auto q=std::size_t(flat.orbitals[--k]);
              phase*=sbd::bit_string_sign_factor(work,bits,q%bits,q/bits);
              work[q/bits]^=std::size_t(1)<<(q%bits);
            }
          }
          std::copy(bra.begin(),bra.end(),basis[pos].begin());
          values[pos++]=phase*flat.coefficients[n]*state.coefficients[i];
        }
      }
    }
    detail::collect_column(basis,values,comm,h_comm,t_comm);
  }
  return detail::assemble_seeds(generated,amplitudes,extra,comm,h_comm,t_comm);
}
} } } // namespace
#endif
