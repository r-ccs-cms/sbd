/**
@file sbd/chemistry/gdb/occupation.h
@brief function to evaluate occupation density (one-particle reduced density matrix in quantum chemistry)
 */
#ifndef SBD_CHEMISTRY_GDB_OCCUPATION_H
#define SBD_CHEMISTRY_GDB_OCCUPATION_H

namespace sbd {
  namespace gdb {

    template <typename ElemT, typename DetsContainer>
    void OccupationDensity(const std::vector<int> & oidx,
			   const std::vector<ElemT> & w,
			   const DetsContainer & det,
			   size_t bit_length,
			   MPI_Comm b_comm,
			   std::vector<double> & res) {
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      res.resize(2*oidx.size(),0.0);
#pragma omp parallel
      {
	size_t num_threads = omp_get_num_threads();
	size_t thread_id = omp_get_thread_num();
	size_t i_begin = thread_id;
	size_t i_end   = det.size();
	std::vector<double> local_res(2*oidx.size(),ElemT(0.0));
	for(size_t i=i_begin; i < i_end; i+=num_threads) {
	  for(size_t io=0; io < oidx.size(); io++) {
	    double weight = GetReal(Conjugate(w[i]) * w[i]);
	    if( getocc(det[i],bit_length,2*oidx[io]) ) {
	      local_res[2*io] += weight;
	    }
	    if( getocc(det[i],bit_length,2*oidx[io]+1) ) {
	      local_res[2*io+1] += weight;
	    }
	  }
	}

#pragma omp critical
	{
	  for(size_t io=0; io < 2*oidx.size(); io++) {
	    res[io] += local_res[io];
	  }
	}
      }
      MpiAllreduce(res,MPI_SUM,b_comm);
    }

  } // end namespace gdb
} // end namespace sbd
#endif
