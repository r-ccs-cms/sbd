#ifndef SBD_CAOP_SPECTRUM_H
#define SBD_CAOP_SPECTRUM_H
#include "sbd/caop/spectrum/results.h"
#include "sbd/caop/spectrum/applied_operator.h"
#include "sbd/framework/sparse_solver/block_lanczos.h"
#include "sbd/caop/basic/helper.h"
#include "sbd/caop/basic/mult.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include "sbd/caop/basic/mkham.h"
#include "sbd/framework/mpi_utility.h"

namespace sbd { namespace caop { namespace spectrum {
// CAOP response using the existing single-vector Hamiltonian operations.
// Convert seeds once after distribution; iteration operates directly on V[v][i].
template<class ElemT>
inline SpectrumResult<ElemT> spectrum(const Wavefunction<ElemT>& state,const sbd::GeneralOp<ElemT>& h,bool fermion,
    const std::vector<std::size_t>& orbitals,bool addition,std::size_t sites,std::size_t bits,
    const Basis& extra,double reference_energy,
    MPI_Comm h_comm,MPI_Comm b_comm,MPI_Comm t_comm,const ss::LanczosOptions& options={},
    const std::vector<sbd::GeneralOp<ElemT>>* applied_operators=nullptr,bool store_hamiltonian=true) {
  // Parent/extra input exists only on h_rank=t_rank=0.
  // Only completed excitation bases and seed blocks are replicated over h/t.
  int h_rank=0,t_rank=0;
  MPI_Comm_rank(h_comm,&h_rank);MPI_Comm_rank(t_comm,&t_rank);
  double norm=0;
  if(h_rank==0 && t_rank==0) norm=global_norm2(state,b_comm);
  if(t_rank==0) MPI_Bcast(&norm,1,MPI_DOUBLE,0,h_comm);
  MPI_Bcast(&norm,1,MPI_DOUBLE,0,t_comm);
  if(!std::isfinite(norm)||std::abs(norm-1)>1e-8||!std::isfinite(reference_energy)) throw std::invalid_argument("normalized reference state and finite energy required");
  if(!sites||!bits||bits>8*sizeof(std::size_t)||((h.NumOpTerms()||h.NumNcTerms()) && std::size_t(h.max_index())>=sites))
    throw std::invalid_argument("Hamiltonian/bit dimensions");
  SeedSpace<ElemT> seeds;
  if(h_rank==0 && t_rank==0)
    seeds=applied_operators
      ?general_operator_seeds(state,*applied_operators,fermion,sites,bits,extra,b_comm)
      :single_particle_seeds(state,orbitals,addition,fermion,sites,bits,extra,b_comm);
  if(applied_operators && !addition) throw std::invalid_argument("general operators require positive resolvent");
  if(t_rank==0) {
    sbd::MpiBcast(seeds.basis,0,h_comm);
    sbd::MpiBcast(seeds.local_seeds,0,h_comm);
  }
  sbd::MpiBcast(seeds.basis,0,t_comm);
  sbd::MpiBcast(seeds.local_seeds,0,t_comm);
  if(seeds.local_seeds.size()!=ss::matrix_size(seeds.basis.size(),orbitals.size()))
    throw std::invalid_argument("seed observable count mismatch");
  SpectrumResult<ElemT> out;out.orbitals=orbitals;out.addition=addition;out.reference_energy=reference_energy;
  const auto local_size=seeds.basis.size();
  MPI_Allreduce(&local_size,&out.basis_size,1,SBD_MPI_SIZE_T,MPI_SUM,b_comm);
  out.fermion=fermion;out.sites=sites;out.bit_length=bits;
  const auto rows=seeds.basis.size(),p=orbitals.size();
  std::vector<std::vector<ElemT>> vectors(p,std::vector<ElemT>(rows));
  for(std::size_t v=0;v<p;++v) for(std::size_t i=0;i<rows;++i)
    vectors[v][i]=seeds.local_seeds[i*p+v];
  std::vector<ElemT>().swap(seeds.local_seeds);
  auto qr=ss::orthogonalize(vectors,b_comm,options.rank_atol,options.rank_rtol);
  out.c0=std::move(qr.factor);out.seed_discarded_norm=qr.discarded_norm;
  if(!qr.rank) return out;
  std::vector<int> slide;
  sbd::make_slide(slide,b_comm,t_comm);
  std::vector<ElemT> hii;
  std::vector<std::vector<std::vector<std::size_t>>> ih,jh;
  std::vector<std::vector<std::vector<ElemT>>> hij;
  if(store_hamiltonian)
    sbd::makeCAOpHam(seeds.basis,bits,slide,h,fermion,hii,ih,jh,hij,
        h_comm,b_comm,t_comm);
  else
    sbd::makeCAOpHamDiagTerms(seeds.basis,bits,slide,h,hii);
  auto apply_h=[&](const auto& x,auto& y) {
    // Each existing mult owns MPI/OpenMP and adds to its output vector.
    for(std::size_t v=0;v<x.size();++v) {
      if(store_hamiltonian)
        sbd::mult(hii,ih,jh,hij,x[v],y[v],slide,h_comm,b_comm,t_comm);
      else
        sbd::mult(hii,x[v],y[v],seeds.basis,int(bits),slide,h,fermion,
            h_comm,b_comm,t_comm);
    }
  };
  out.coefficients=ss::block_lanczos<ElemT>(apply_h,b_comm,std::move(vectors),rows,options);
  return out;
}
} } } // namespace
#endif
