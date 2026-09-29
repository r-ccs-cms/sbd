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
  std::vector<ElemT> local_seeds;   // Exactly basis.size() * number of seeds.
};
template<class ElemT>
inline SeedSpace<ElemT> single_particle_seeds(const Wavefunction<ElemT>& state,
    const std::vector<std::size_t>& orbitals,bool addition,bool fermion,
    std::size_t sites,std::size_t bits,const Basis& extra,MPI_Comm comm) {
  if(orbitals.empty()) throw std::invalid_argument("no excitation orbitals");
  for(auto o:orbitals) if(o>=sites) throw std::invalid_argument("excitation orbital out of range");
  global_norm2(state,comm); // Collective shape/finite checks, no normalization.
  SeedSpace<ElemT> out;out.basis=extra;
  for(const auto& d:extra) validate_det(d,sites,bits);
  const auto s=orbitals.size();
  const auto max_rows=out.basis.flat().max_size()/std::max(std::size_t(1),out.basis.elem_size());
  if(extra.size()>max_rows || state.basis.size()>(max_rows-extra.size())/s)
    throw std::length_error("seed basis capacity overflow");
  // At most one candidate per parent and orbital, before deduplication.
  out.basis.reserve(extra.size()+state.basis.size()*s);
  Det d;double phase=0;
  for(const auto& parent:state.basis) {
    validate_det(parent,sites,bits);
    for(auto o:orbitals)
      if(single_particle_excite(parent,o,addition,fermion,bits,d,phase)) out.basis.push_back(d);
  }
  // Existing SBD distributed union: each row has one owner and is locally sorted.
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(out.basis,comm);
  if(out.basis.size()>std::numeric_limits<std::size_t>::max()/s)
    throw std::overflow_error("seed block size overflow");
  out.local_seeds.assign(out.basis.size()*s,0);
  for(std::size_t v=0;v<s;++v) {
    Basis generated;std::vector<ElemT> amplitudes,mapped;
    generated.reserve(state.basis.size());
    amplitudes.reserve(state.basis.size());
    for(std::size_t i=0;i<state.basis.size();++i) {
      if(!single_particle_excite(state.basis[i],orbitals[v],addition,fermion,bits,d,phase)) continue;
      generated.push_back(d);amplitudes.push_back(phase*state.coefficients[i]);
    }
    // Single-mode c/cdag is injective on its allowed parent configurations.
    // Thus each column has a unique distributed source basis for the remapper.
    sbd::RemapWavefunctionToBasis(generated,amplitudes,out.basis,comm,mapped,true);
    for(std::size_t i=0;i<out.basis.size();++i) out.local_seeds[i*s+v]=mapped[i];
  }
  return out;
}
// One complete CAOP operator per column. Only the root b plane calls this.
template<class ElemT>
inline SeedSpace<ElemT> general_operator_seeds(const Wavefunction<ElemT>& state,
    const std::vector<sbd::GeneralOp<ElemT>>& operators,bool fermion,
    std::size_t sites,std::size_t bits,const Basis& extra,MPI_Comm comm) {
  if(operators.empty()) throw std::invalid_argument("no applied operators");
  global_norm2(state,comm);
  for(const auto& d:state.basis) validate_det(d,sites,bits);
  SeedSpace<ElemT> out;out.basis=extra;
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
    // Equal targets (including contributions from different parent ranks) meet
    // at their hash owner. Remap requires one source row per determinant.
    sbd::murmur_basis::redistribute_determinant_values_by_hash(basis,values,comm);
    Basis unique;std::vector<ElemT> summed;
    unique.reserve(basis.size());summed.reserve(values.size());
    for(std::size_t i=0;i<basis.size();++i) {
      if(!unique.empty() && std::equal(basis[i].begin(),basis[i].end(),unique[unique.size()-1].begin()))
        summed.back()+=values[i];
      else {unique.push_back(basis[i]);summed.push_back(values[i]);}
    }
    basis=std::move(unique);values=std::move(summed);
    for(const auto& d:basis) out.basis.push_back(d);
  }
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(out.basis,comm);
  out.local_seeds.assign(sbd::sparse_solver::matrix_size(out.basis.size(),count),0);
  for(std::size_t v=0;v<count;++v) {
    std::vector<ElemT> mapped;
    sbd::RemapWavefunctionToBasis(generated[v],amplitudes[v],out.basis,comm,mapped,true);
    for(std::size_t i=0;i<out.basis.size();++i) out.local_seeds[i*count+v]=mapped[i];
  }
  return out;
}
} } } // namespace
#endif
