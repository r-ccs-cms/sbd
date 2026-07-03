/**
@file sbd/caop/basic/carryover.h
@brief functions for carryover bitstrings for creation/annihilation operator
*/
#ifndef SBD_CAOP_BASIC_CARRYOVER_H
#define SBD_CAOP_BASIC_CARRYOVER_H

#include "sbd/framework/sort_array.h"

namespace sbd {

  std::string carryoverfilename(const std::string & carryovername,
				int index) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename = carryovername + tag + ".txt";
    return filename;
  }

  template <typename ElemT, typename RealT>
  void CarryOverBasis(const std::vector<ElemT> & w,
		      const std::vector<std::vector<size_t>> & bs,
		      MPI_Comm b_comm,
		      size_t kept,
		      std::vector<std::vector<size_t>> & rbs,
		      RealT & discarted_weight) {
    // using RealT = typename GetRealType<ElemT>::RealT;
    std::vector<RealT> r(w.size());
#pragma omp parallel for
    for(size_t i=0; i < w.size(); i++) {
      r[i] = SquaredNorm(w[i]);
    }
    std::vector<size_t> ranking(r.size());
    mpi_find_ranking(r,ranking,b_comm);
    size_t rbs_size = 0;
    size_t num_threads = omp_get_max_threads();
    std::vector<size_t> local_size(num_threads,0);
#pragma omp parallel
    {
      size_t thread_id = omp_get_thread_num();
      for(size_t k=thread_id; k < bs.size(); k+=num_threads) {
	if( ranking[k] < kept ) {
	  local_size[thread_id]++;
	}
      }
    }
    std::vector<size_t> offset(num_threads,0);
    for(size_t tid=0; tid < num_threads; tid++) {
      rbs_size += local_size[tid];
      if( tid < num_threads-1 ) {
	offset[tid+1] = rbs_size;
      }
    }
    rbs.resize(rbs_size);
    std::vector<RealT> keep_weight_local(num_threads,0.0);
#pragma omp parallel
    {
      size_t thread_id = omp_get_thread_num();
      size_t local_addr = offset[thread_id];
      for(size_t k=thread_id; k < bs.size(); k+=num_threads) {
	if( ranking[k] < kept ) {
	  rbs[local_addr++] = bs[k];
	  keep_weight_local[thread_id] += r[k];
	}
      }
    }
    RealT keep_weight = 0.0;
    for(size_t tid=0; tid < num_threads; tid++) {
      keep_weight += keep_weight_local[tid];
    }
    RealT keep_weight_global = 0.0;
    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    MPI_Allreduce(&keep_weight,&keep_weight_global,
		  1,DataT,MPI_SUM,b_comm);
    discarted_weight = 1.0-keep_weight_global;
    sort_bitarray(rbs);
  }
}

#endif
