/**
@file sbd/caop/basic/mult.h
@brief Function to perform Hamiltonian operation for general field operators
 */
#ifndef SBD_CAOP_BASIC_MULT_H
#define SBD_CAOP_BASIC_MULT_H

namespace sbd {

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

    if( H.m1_.empty() ) H.PrecomputeMasks(bit_length);

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
	std::vector<size_t> vb;
	std::vector<size_t> vk;
	int sign_count;
	size_t size_t_one = static_cast<size_t>(1);

#pragma omp for schedule(dynamic)
	for(size_t ib = ib_start; ib < ib_end; ib++) {

	  vb = bs[ib];
	  for(size_t n=0; n < H.o_.size(); n++) {
	    // fast reject: required bits absent or forbidden bits present
	    {
	      bool reject = false;
	      for(size_t w=0; w < H.m1_[n].size(); w++) {
		if( (~vb[w] & H.m1_[n][w]) || (vb[w] & H.m2_[n][w]) ) {
		  reject = true; break;
		}
	      }
	      if( reject ) continue;
	    }
	    // survivor: construct ket and accumulate sign
	    sign_count = 1;
	    vk = vb;
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

    std::vector<ElemT> twk;
    std::vector<ElemT> rwk;

    if( slide.size() != 0 ) {
      if( slide[0] != 0 ) {
	MpiSlide(w,twk,-slide[0],b_comm);
      } else {
	twk = w;
      }
    }

    ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma omp parallel for
    for(size_t i=0; i < hw.size(); i++) {
      hw[i] *= volp;
    }

    if( mpi_rank_t == 0 ) {
#pragma omp parallel for
      for(size_t i=0; i < twk.size(); i++) {
	hw[i] += hii[i] * twk[i];
      }
    }

    for(size_t task=0; task < slide.size(); task++) {
#pragma omp parallel
      {
	size_t thread_id = omp_get_thread_num();
	size_t num_threads = omp_get_num_threads();
	for(size_t k=0; k < hij[task][thread_id].size(); k++) {
	  hw[ih[task][thread_id][k]] += hij[task][thread_id][k]
	    * twk[jh[task][thread_id][k]];
	}
      }
      if( task != slide.size()-1 ) {
	int bslide = slide[task]-slide[task+1];
	rwk.resize(twk.size());
	std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	MpiSlide(rwk,twk,bslide,b_comm);
      }
    }
    MpiAllreduce(hw,MPI_SUM,t_comm);
    MpiAllreduce(hw,MPI_SUM,h_comm);
  }
	    
	    
  
}

#endif
