/**
@file sbd/caop/basic/mult.h
@brief Function to perform Hamiltonian operation for general field operators
 */
#ifndef SBD_CAOP_BASIC_MULT_H
#define SBD_CAOP_BASIC_MULT_H

#ifdef SBD_THRUST
#include "sbd/caop/basic/mult_thrust.h"
#endif

namespace sbd {

#ifndef SBD_THRUST
  template <typename ElemT>
  void mult(const std::vector<ElemT> & hd,
	    const std::vector<ElemT> & wk,
	    std::vector<ElemT> & wb,
	    const det_vector<size_t> & bs,
	    const int bit_length,
	    const std::vector<int> & slide,
	    const GeneralOp<ElemT> & H,
	    bool sign,
	    MPI_Comm h_comm,
	    MPI_Comm b_comm,
	    MPI_Comm t_comm) {

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    std::vector<ElemT> twk;
    std::vector<ElemT> rwk;
    det_vector<size_t> tbs;
    det_vector<size_t> rbs;

    if( slide.size() != 0 ) {
      if( slide[0] != 0 ) {
	MpiSlide(wk,twk,slide[0],b_comm);
	MpiSlide(bs,tbs,slide[0],b_comm);
      } else {
	twk = wk;
	tbs = bs;
      }
    }

    det_vector<size_t> m1, m2;
    H.PrecomputeMasks(bit_length, m1, m2);

    ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma omp parallel for
    for(size_t i=0; i < wb.size(); i++) {
      wb[i] *= volp;
    }

    if( mpi_rank_t == 0 ) {
#pragma omp parallel for
      for(size_t i=0; i < twk.size(); i++) {
	wb[i] += hd[i] * twk[i];
      }
    }

    for(size_t task=0; task < slide.size(); task++) {
#pragma omp parallel
      {
	size_t ib_start = 0;
	size_t ib_end   = bs.size();
	std::vector<size_t> vk;
	int sign_count;
	size_t size_t_one = static_cast<size_t>(1);

#pragma omp for schedule(dynamic)
	for(size_t ib = ib_start; ib < ib_end; ib++) {

	  const auto& vb = bs[ib];
	  for(size_t n=0; n < H.o_.size(); n++) {
	    // fast reject: required bits absent or forbidden bits present
	    {
	      bool reject = false;
	      for(size_t w=0; w < m1[n].size(); w++) {
		if( (~vb[w] & m1[n][w]) || (vb[w] & m2[n][w]) ) {
		  reject = true; break;
		}
	      }
	      if( reject ) continue;
	    }
	    // survivor: construct ket and accumulate sign
	    sign_count = 1;
	    assign_det(vk, vb);
	    for(int k=0; k < H.o_[n].n_dag_; k++) {
	      int q = H.o_[n].fops_[k].q_;
	      int r = q / bit_length;
	      int x = q % bit_length;
	      vk[r] ^= ( size_t_one << x );
	      if( sign ) {
		sign_count *= bit_string_sign_factor(vk,bit_length,x,r);
	      }
	    }
	    for(int k = H.o_[n].n_dag_; k < (int)H.o_[n].fops_.size(); k++) {
	      int q = H.o_[n].fops_[k].q_;
	      int r = q / bit_length;
	      int x = q % bit_length;
	      vk[r] |= ( size_t_one << x );
	      if( sign ) {
		sign_count *= bit_string_sign_factor(vk,bit_length,x,r);
	      }
	    }

	    // we assume that tbs is aligned in ascending order
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
	      wb[ib] += H.c_[n] * ElemT(sign_count) * twk[ik];
	    }
	  }
	}
      }

      if( task != slide.size()-1 ) {
	int bslide = slide[task]-slide[task+1];
	rwk.resize(twk.size());
	std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	rbs = tbs;
	MpiSlide(rwk,twk,bslide,b_comm);
	MpiSlide(rbs,tbs,bslide,b_comm);
      }
    }
    MpiAllreduce(wb,MPI_SUM,t_comm);
    MpiAllreduce(wb,MPI_SUM,h_comm);

  }
#endif // !SBD_THRUST

  template <typename ElemT>
  void mult(const std::vector<ElemT> & hii,
	    const std::vector<std::vector<std::vector<size_t>>> & ih,
	    const std::vector<std::vector<std::vector<size_t>>> & jh,
	    const std::vector<std::vector<std::vector<ElemT>>> & hij,
	    const std::vector<ElemT> & w,
	    std::vector<ElemT> & hw,
	    const std::vector<int> & slide,
	    MPI_Comm h_comm,
	    MPI_Comm b_comm,
	    MPI_Comm t_comm) {

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    // Ping-pong buffers for the ring-shifted weight vector.
    // Thread 0 writes *twk_next via MpiSlide while compute threads
    // read *twk_cur — both are separate buffers so there is no data race.
    std::vector<ElemT> buf_A, buf_B;
    std::vector<ElemT>* twk_cur  = &buf_A;
    std::vector<ElemT>* twk_next = &buf_B;

    if( slide.size() != 0 ) {
      if( slide[0] != 0 )
	MpiSlide(w, *twk_cur, -slide[0], b_comm);
      else
	*twk_cur = w;
    }

    ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma omp parallel for
    for(size_t i=0; i < hw.size(); i++) hw[i] *= volp;

    if( mpi_rank_t == 0 ) {
#pragma omp parallel for
      for(size_t i=0; i < twk_cur->size(); i++)
	hw[i] += hii[i] * (*twk_cur)[i];
    }

    // Ring-slide loop with overlap.
    // Each task uses a per-task #pragma omp parallel (matching original structure).
    // Inside the parallel region: thread 0 calls MpiSlide into *twk_next while
    // all threads (including thread 0 after MPI completes) compute from *twk_cur.
    // The implicit barrier at the end of the parallel region ensures the slide
    // finishes before the pointer swap.  hij[task] is indexed by thread_id so
    // it matches makeCAOpHam's original num_thread slot layout exactly.
    for(size_t task=0; task < slide.size(); task++) {
      bool last_task = (task == slide.size() - 1);
#pragma omp parallel
      {
	size_t thread_id   = omp_get_thread_num();

	// Thread 0 starts the ring-shift into the next ping-pong buffer.
	// MpiSlide blocks until data transfer completes.
	// Threads 1..N-1 proceed directly to compute — their work overlaps
	// with thread 0's MPI blocking call.
	// makeCAOpHam assigns no COO rows to thread 0 when num_thread > 1,
	// so thread 0's compute loop below is a no-op in that case.
	// When num_thread == 1, thread 0 does MPI then handles all rows.
	if( thread_id == 0 && !last_task ) {
	  int bslide = slide[task]-slide[task+1];
	  MpiSlide(*twk_cur, *twk_next, bslide, b_comm);
	}

	for(size_t k=0; k < hij[task][thread_id].size(); k++)
	  hw[ih[task][thread_id][k]] += hij[task][thread_id][k]
	    * (*twk_cur)[jh[task][thread_id][k]];
	// Implicit barrier at end of omp parallel
      }
      // Swap pointers so the next task reads the freshly received data.
      if( !last_task ) std::swap(twk_cur, twk_next);
    }
    MpiAllreduce(hw,MPI_SUM,t_comm);
    MpiAllreduce(hw,MPI_SUM,h_comm);
  }
	    
	    
  
}

#endif
