/**
@file sbd/chemistry/tpb/occupation.h
@brief function to evaluate occupation density (one-particle reduced density matrix in quantum chemistry)
 */
#ifndef SBD_CHEMISTRY_TPB_OCCUPATION_H
#define SBD_CHEMISTRY_TPB_OCCUPATION_H

namespace sbd {
    
  template <typename ElemT>
  void OccupationDensity(const std::vector<int> & oIdx,
			 const std::vector<ElemT> & W,
			 const std::vector<std::vector<size_t>> & adet,
			 const std::vector<std::vector<size_t>> & bdet,
			 const size_t bit_length,
			 size_t adet_comm_size,
			 size_t bdet_comm_size,
			 MPI_Comm b_comm,
			 std::vector<ElemT> & res) {
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    if( static_cast<size_t>(mpi_size_b) != adet_comm_size*bdet_comm_size ) {
      throw std::invalid_argument("OccupationDensity: MPI Size is not appropriate");
    }

    int adet_rank = mpi_rank_b / bdet_comm_size;
    int bdet_rank = mpi_rank_b % bdet_comm_size;
    size_t ia_start = 0;
    size_t ia_end   = adet.size();
    size_t ib_start = 0;
    size_t ib_end   = bdet.size();
    get_mpi_range(adet_comm_size,adet_rank,ia_start,ia_end);
    get_mpi_range(bdet_comm_size,bdet_rank,ib_start,ib_end);

    size_t ia_size = ia_end-ia_start;
    size_t ib_size = ib_end-ib_start;
    

    std::vector<std::vector<ElemT>> daop(oIdx.size(),std::vector<ElemT>(ia_size,ElemT(0.0)));
    std::vector<std::vector<ElemT>> dbop(oIdx.size(),std::vector<ElemT>(ib_size,ElemT(0.0)));

#pragma omp parallel for
    for(size_t ia=ia_start; ia < ia_end; ia++) {
      for(size_t io=0; io < oIdx.size(); io++) {
	if ( getocc(adet[ia],bit_length,oIdx[io]) ) {
	  daop[io][ia-ia_start] = 1.0;
	}
      }
    }

#pragma omp parallel for
    for(size_t ib=ib_start; ib < ib_end; ib++) {
      for(size_t io=0; io < oIdx.size(); io++) {
	if( getocc(bdet[ib],bit_length,oIdx[io]) ) {
	  dbop[io][ib-ib_start] = 1.0;
	}
      }
    }

    res.resize(2*oIdx.size(),ElemT(0.0));
    
#pragma omp parallel
    {
      size_t num_threads = omp_get_num_threads();
      size_t thread_id = omp_get_thread_num();
      size_t chunk_size = ia_size / num_threads;
      size_t ja_start = thread_id * chunk_size + ia_start;
      size_t ja_end   = (thread_id+1) * chunk_size + ia_start;
      if( thread_id == num_threads - 1 ) {
	ja_end = ia_end;
      }
      std::vector<ElemT> local_res(2*oIdx.size(),ElemT(0.0));
      
      for(size_t ia = ja_start; ia < ja_end; ia++) {
	for(size_t ib = ib_start; ib < ib_end; ib++) {
	  size_t idx = (ia-ia_start) * ib_size + (ib-ib_start);
	  for(size_t io=0; io < oIdx.size(); io++) {
	    local_res[2*io]   += SquaredNorm(W[idx]) * daop[io][ia-ia_start];
	    local_res[2*io+1] += SquaredNorm(W[idx]) * dbop[io][ib-ib_start];
	  }
	}
      }

#pragma omp critical
      {
	for(size_t io=0; io < 2*oIdx.size(); io++) {
	  res[io] += local_res[io];
	}
      }
      
    }
    MpiAllreduce(res,MPI_SUM,b_comm);
    
  }

  
}
#endif
