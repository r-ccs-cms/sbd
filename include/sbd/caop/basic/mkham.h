/**
   @file sbd/caop/basic/mkham.h
   @brief Function to make Hamiltonian for GeneralOp
 */
#ifndef SBD_CAOP_BASIC_MKHAM_H
#define SBD_CAOP_BASIC_MKHAM_H

#include <omp.h>

namespace sbd {

  template <typename ElemT>
  void makeCAOpHamDiagTerms(const det_vector<size_t> & bs,
			       const size_t bit_length,
			       const std::vector<int> & slide,
			       const GeneralOp<ElemT> & H,
			       std::vector<ElemT> & hii) {

    hii.resize(bs.size(),ElemT(0.0));
#pragma omp parallel
    {
      std::vector<size_t> v;
      size_t size_t_one = static_cast<size_t>(1);
      bool check;
#pragma omp for
      for(size_t ib=0; ib < bs.size(); ib++) {
	v = bs[ib];
	for(size_t n=0; n < H.d_.size(); n++) {
	  check = false;
	  for(int k=0; k < H.d_[n].n_dag_; k++) {
	    size_t q = static_cast<size_t>(H.d_[n].fops_[k].q_);
	    size_t r = q / bit_length;
	    size_t x = q % bit_length;
	    if( ( v[r] & ( size_t_one << x ) ) == 0 ) {
	      check = true;
	      break;
	    }
	  }
	  if( check ) continue;
	  hii[ib] += H.e_[n];
	}
      }
    }
  }
  
  template <typename ElemT>
  void makeCAOpHam(const det_vector<size_t> & bs,
		   const size_t bit_length,
		   const std::vector<int> & slide,
		   const GeneralOp<ElemT> & H,
		   const bool sign,
		   std::vector<ElemT> & hii,
		   std::vector<std::vector<std::vector<size_t>>> & ih,
		   std::vector<std::vector<std::vector<size_t>>> & jh,
		   std::vector<std::vector<std::vector<ElemT>>> & hij,
		   MPI_Comm h_comm,
		   MPI_Comm b_comm,
		   MPI_Comm t_comm) {
    
    makeCAOpHamDiagTerms(bs,bit_length,slide,H,hii);

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    size_t brasize = bs.size();
    size_t num_thread;
    size_t num_task = slide.size();
    size_t num_terms = H.NumOpTerms();

    det_vector<size_t> tbs;
    det_vector<size_t> rbs;

    if( slide.size() != 0 ) {
      if( slide[0] != 0 ) {
	MpiSlide(bs,tbs,-slide[0],b_comm);
      } else {
	tbs = bs;
      }
    }

#pragma omp parallel
    {
      num_thread = omp_get_num_threads();
    }

    ih.resize(num_task);
    jh.resize(num_task);
    hij.resize(num_task);
    
    for(size_t task=0; task < slide.size(); task++) {
      ih[task].resize(num_thread);
      jh[task].resize(num_thread);
      hij[task].resize(num_thread);

#pragma omp parallel
      {
	// Round-robin assignment of work to threads.
	// When num_thread > 1, thread 0 is reserved for MPI overlap in mult()
	// and receives no COO rows.  Threads 1..N-1 cover all brasize rows using
	// a stride of num_compute = num_thread - 1.
	// When num_thread == 1, thread 0 covers all rows (unchanged behaviour).
	size_t thread_id  = omp_get_thread_num();
	size_t ib_end     = brasize;
	size_t num_compute = (num_thread > 1) ? (num_thread - 1) : size_t(1);
	// ib_start: thread 0 gets ib_end (empty loop) when multi-threaded;
	//           thread k>0 starts at k-1 so threads cover {0, 1, ..., N-2, N-1, ...}
	size_t ib_start   = (num_thread > 1)
	                      ? (thread_id > 0 ? thread_id - 1 : ib_end)
	                      : size_t(0);
	size_t reserve_size = (thread_id == 0 && num_thread > 1)
	                      ? size_t(0) : brasize * num_terms / num_compute;
	ih[task][thread_id].reserve(reserve_size);
	jh[task][thread_id].reserve(reserve_size);
	hij[task][thread_id].reserve(reserve_size);

	std::vector<size_t> vb;
	std::vector<size_t> vk;
	size_t size_t_one = static_cast<size_t>(1);
	int sign_count;
	bool check;
	
	for(size_t ib=ib_start; ib < ib_end; ib+=num_compute) {

	  vb = bs[ib];
	  for(size_t n=0; n < H.o_.size(); n++) {
	    sign_count = 1;
	    vk = vb;
	    check = false;
	    for(int k=0; k < H.o_[n].n_dag_; k++) {
	      size_t q = static_cast<size_t>(H.o_[n].fops_[k].q_);
	      size_t r = q / bit_length;
	      size_t x = q % bit_length;
	      if ( ( vk[r] & ( size_t_one << x ) ) != 0 ) {
		vk[r] = vk[r] ^ ( size_t_one << x );
		if ( sign ) {
		  sign_count *= bit_string_sign_factor(vk,bit_length,x,r);
		}
	      } else {
		check = true;
		break;
	      }
	    }
	    if( check ) continue;
	    for(int k = H.o_[n].n_dag_; k < H.o_[n].fops_.size(); k++) {
	      size_t q = static_cast<size_t>(H.o_[n].fops_[k].q_);
	      size_t r = q / bit_length;
	      size_t x = q % bit_length;
	      if( ( vk[r] & ( size_t_one << x ) ) == 0 ) {
		vk[r] = vk[r] | ( size_t_one << x );
		if( sign ) {
		  sign_count *= bit_string_sign_factor(vk,bit_length,x,r);
		}
	      } else {
		check = true;
		break;
	      }
	    }
	    if( check ) continue;
	    /*
	    auto itik = std::lower_bound(tbs.begin(),tbs.end(),vk,
					 [](const std::vector<size_t> & x,
					    const std::vector<size_t> & y) {
					   return x < y;
					 });
	    */
	    auto itik = std::lower_bound(tbs.begin(),tbs.end(),vk,
					 [](const auto & x,
					    const auto & y) {
					   return sbd::less_from_back(x,y);
					 });
	    if( itik == tbs.end() ) continue;
	    if( *itik == vk ) {
	      auto ik = static_cast<size_t>(itik - tbs.begin());
	      ih[task][thread_id].push_back(ib);
	      jh[task][thread_id].push_back(ik);
	      hij[task][thread_id].push_back(H.c_[n] * ElemT(sign_count));
	    }
	  }
	}
      }
      
      if( task != slide.size()-1 ) {
	int bslide = slide[task]-slide[task+1];
	rbs = tbs;
	MpiSlide(rbs,tbs,bslide,b_comm);
      }
    }
  }

  
}

#endif
