/**
@file sbd/chemistry/tpb/restart.h
@brief restart from itersectional bitstring
 */
#ifndef SBD_CHEMISTRY_TPB_RESTART_H
#define SBD_CHEMISTRY_TPB_RESTART_H

#include <iomanip>
#include <sstream>
#include <fstream>

namespace sbd {

  std::string to_padded_string(size_t n, size_t d) {
    std::ostringstream oss;
    oss << std::setw(d) << std::setfill('0') << n;
    return oss.str();
  }

  template <typename ElemT>
  void SaveWavefunction(const std::string file,
			const det_vector<size_t, det_kind::half> & adet,
			const det_vector<size_t, det_kind::half> & bdet,
			size_t adet_comm_size,
			size_t bdet_comm_size,
			MPI_Comm h_comm,
			MPI_Comm b_comm,
			MPI_Comm t_comm,
			std::vector<ElemT> & W) {
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

    size_t det_length = adet.elem_size();
    int adet_rank = mpi_rank_b / bdet_comm_size;
    int bdet_rank = mpi_rank_b % bdet_comm_size;

    size_t adet_start = 0;
    size_t adet_end   = adet.size();
    size_t bdet_start = 0;
    size_t bdet_end   = bdet.size();
    get_mpi_range(adet_comm_size,adet_rank,adet_start,adet_end);
    get_mpi_range(bdet_comm_size,bdet_rank,bdet_start,bdet_end);
    size_t adet_range = adet_end - adet_start;
    size_t bdet_range = bdet_end - bdet_start;

    det_vector<size_t, det_kind::half> adet_copy(adet);
    det_vector<size_t, det_kind::half> bdet_copy(bdet);

    if( mpi_rank_h == 0 && mpi_rank_t == 0 ) {
      std::string tag = to_padded_string(mpi_rank_b,6);
      std::string filename = file + tag;
      std::ofstream ofile(filename,std::ios::binary);
      ofile.write(reinterpret_cast<char *>(&adet_range),sizeof(size_t));
      ofile.write(reinterpret_cast<char *>(&bdet_range),sizeof(size_t));
      ofile.write(reinterpret_cast<char *>(&det_length),sizeof(size_t));
      for(size_t i=adet_start; i < adet_end; i++) {
	ofile.write(reinterpret_cast<char *>(adet_copy[i].data()),sizeof(size_t)*det_length);
      }
      for(size_t i=bdet_start; i < bdet_end; i++) {
	ofile.write(reinterpret_cast<char *>(bdet_copy[i].data()),sizeof(size_t)*det_length);
      }
      ofile.write(reinterpret_cast<char *>(W.data()),sizeof(ElemT)*adet_range*bdet_range);
      ofile.close();
    }
  }

#ifdef SBD_RESTART_OLD

  template <typename ElemT>
  void LoadWavefunction(const std::string & file,
			const det_vector<size_t, det_kind::half> & adet,
			const det_vector<size_t, det_kind::half> & bdet,
			const size_t adet_comm_size,
			const size_t bdet_comm_size,
			const MPI_Comm h_comm,
			const MPI_Comm b_comm,
			const MPI_Comm t_comm,
			std::vector<ElemT> & W) {

    using RealT = typename GetRealType<ElemT>::RealT;

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    // information of current basis
    std::vector<size_t> adet_start(adet_comm_size);
    std::vector<size_t> adet_end(adet_comm_size);
    std::vector<size_t> bdet_start(bdet_comm_size);
    std::vector<size_t> bdet_end(bdet_comm_size);

    for(int rank=0; rank < adet_comm_size; rank++) {
      adet_start[rank] = 0;
      adet_end[rank]   = adet.size();
      get_mpi_range(adet_comm_size,rank,adet_start[rank],adet_end[rank]);
    }
    for(int rank=0; rank < bdet_comm_size; rank++) {
      bdet_start[rank] = 0;
      bdet_end[rank]   = bdet.size();
      get_mpi_range(bdet_comm_size,rank,bdet_start[rank],bdet_end[rank]);
    }

#ifdef SBD_DEBUG_RESTART
    if( mpi_rank_h == 0 ) {
      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
	if( mpi_rank_t == rank_t ) {
	  for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	    if( mpi_rank_b == rank_b ) {
	      std::cout << " adet ranges at (" << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "):";
	      for(int rank=0; rank < adet_comm_size; rank++) {
		std::cout << " [" << adet_start[rank] << "," << adet_end[rank] << ")";
	      }
	      std::cout << std::endl;
	      std::cout << " bdet ranges at (" << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "):";
	      for(int rank=0; rank < bdet_comm_size; rank++) {
		std::cout << " [" << bdet_start[rank] << "," << bdet_end[rank] << ")";
	      }
	      std::cout << std::endl;
	    }
	    MPI_Barrier(b_comm);
	  }
	}
	MPI_Barrier(t_comm);
      }
    }
#endif
    
    size_t this_adet_rank = mpi_rank_b / bdet_comm_size;
    size_t this_bdet_rank = mpi_rank_b % bdet_comm_size;
    size_t this_adet_range = adet_end[this_adet_rank]-adet_start[this_adet_rank];
    size_t this_bdet_range = bdet_end[this_bdet_rank]-bdet_start[this_bdet_rank];
    W.resize(this_adet_range*this_bdet_range,ElemT(0.0));

    if( mpi_rank_h == 0 ) {
      
      size_t load_adet_size;
      size_t load_bdet_size;
      size_t load_det_length;
      std::vector<std::vector<size_t>> load_adet;
      std::vector<std::vector<size_t>> load_bdet;
      std::vector<ElemT> load_W;
      std::vector<size_t> load_det_size(2,0);
      if( mpi_rank_t == 0 ) {
	std::string tag = to_padded_string(mpi_rank_b,6);
	std::string filename = file + tag;
	std::ifstream ifile(filename,std::ios::binary);

	if( ifile.is_open() ) {
	  ifile.read(reinterpret_cast<char *>(&load_adet_size),sizeof(size_t));
	  ifile.read(reinterpret_cast<char *>(&load_bdet_size),sizeof(size_t));
	  ifile.read(reinterpret_cast<char *>(&load_det_length),sizeof(size_t));
	  load_adet.resize(load_adet_size,std::vector<size_t>(load_det_length));
	  load_bdet.resize(load_bdet_size,std::vector<size_t>(load_det_length));
	  for(size_t i=0; i < load_adet_size; i++) {
	    ifile.read(reinterpret_cast<char *>(load_adet[i].data()),sizeof(size_t)*load_det_length);
	  }
	  for(size_t i=0; i < load_bdet_size; i++) {
	    ifile.read(reinterpret_cast<char *>(load_bdet[i].data()),sizeof(size_t)*load_det_length);
	    
	  }
	  load_W.resize(load_adet_size*load_bdet_size,ElemT(0.0));
	  ifile.read(reinterpret_cast<char *>(load_W.data()),sizeof(ElemT)*load_adet_size*load_bdet_size);
	  ifile.close();
	  load_det_size[0] = load_adet_size;
	  load_det_size[1] = load_bdet_size;
	} else {
          std::cerr << "failed to open input file " << filename << ", exiting ..." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 0);
	}
      } // end read-in from file

      MPI_Bcast(load_det_size.data(),2,SBD_MPI_SIZE_T,0,t_comm);

      if( mpi_rank_t != 0 ) {
	load_adet_size = load_det_size[0];
	load_bdet_size = load_det_size[1];
      }

      size_t load_det_count = 0;
      if( load_det_size[0]*load_det_size[1] != 0 ) {
	MpiBcast(load_adet,0,t_comm);
	MpiBcast(load_bdet,0,t_comm);
	MpiBcast(load_W,0,t_comm);
	load_det_count = 1;
      } else {
	load_adet.resize(0);
	load_bdet.resize(0);
	load_W.resize(0);
	load_det_count = 0;
      }
      size_t load_mpi_size_b = 0;
      MPI_Allreduce(&load_det_count,&load_mpi_size_b,1,SBD_MPI_SIZE_T,MPI_SUM,b_comm);

#ifdef SBD_DEBUG_RESTART
      std::cout << " number of loaded det file = " << load_mpi_size_b << std::endl;
      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
	if( rank_t == mpi_rank_t ) {
	  for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	    if( rank_b == mpi_rank_b ) {
	      std::cout << " LoadWavefunction at (h,b,t) = (" << mpi_rank_h
			<< "," << mpi_rank_b << "," << mpi_rank_t << ") ";
	      std::cout << ": size(adet) = " << load_det_size[0];
	      for(int k=0; k < std::min(load_adet.size(),static_cast<size_t>(4)); k++) {
		std::cout << " (" << makestring(load_adet[k],20,36) << ")";
	      }
	      std::cout << std::endl;
	      std::cout << " LoadWavefunction at (h,b,t) = (" << mpi_rank_h
			<< "," << mpi_rank_b << "," << mpi_rank_t << ") ";
	      std::cout << ": size(bdet) = " << load_det_size[1];
	      for(int k=0; k < std::min(load_bdet.size(),static_cast<size_t>(4)); k++) {
		std::cout << " (" << makestring(load_bdet[k],20,36) << ")";
	      }
	      std::cout << std::endl;
	    }
	    MPI_Barrier(b_comm);
	  }
	}
	MPI_Barrier(t_comm);
      }
#endif
      
      size_t load_rank_start = 0;
      size_t load_rank_end   = load_mpi_size_b;
      get_mpi_range(mpi_size_t,mpi_rank_t,load_rank_start,load_rank_end);

#ifdef SBD_DEBUG_RESTART
      std::cout << " LoadWavefunction at (" << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t
		<< "): load_rank_start, load_rank_end = " << load_rank_start << ", " << load_rank_end << std::endl;
#endif

      std::vector<std::vector<size_t>> send_I(mpi_size_b);
      std::vector<std::vector<ElemT>>  send_W(mpi_size_b);

      std::vector<int> send_adet_rank(load_adet_size,-1);
      std::vector<size_t> send_adet_idx(load_adet_size);
      std::vector<int> send_bdet_rank(load_bdet_size,-1);
      std::vector<size_t> send_bdet_idx(load_bdet_size);

      for(int load_rank=load_rank_start; load_rank < load_rank_end; load_rank++) {
	if( mpi_rank_b == load_rank ) {
	  for(size_t ia = 0; ia < load_adet_size; ia++) {
	    auto itja = std::find(adet.begin(),adet.end(),load_adet[ia]);
	    if( itja != adet.end() ) {
	      size_t ja = std::distance(adet.begin(),itja);
	      int adet_rank = 0;
	      for(int rank=0; rank < adet_comm_size; rank++) {
		if( ( adet_start[rank] <= ja ) && ( ja < adet_end[rank] ) ) {
		  adet_rank = rank;
		  break;
		}
	      }
	      send_adet_rank[ia] = adet_rank;
	      send_adet_idx[ia]  = ja;
	    }
	  }
	  for(size_t ib=0; ib < load_bdet_size; ib++) {
	    auto itjb = std::find(bdet.begin(),bdet.end(),load_bdet[ib]);
	    if( itjb != bdet.end() ) {
	      size_t jb = std::distance(bdet.begin(),itjb);
	      int bdet_rank = 0;
	      for(int rank=0; rank < bdet_comm_size; rank++) {
		if( ( bdet_start[rank] <= jb ) && ( jb < bdet_end[rank] ) ) {
		  bdet_rank = rank;
		  break;
		}
	      }
	      send_bdet_rank[ib] = bdet_rank;
	      send_bdet_idx[ib]  = jb;
	    }
	  }
	}
      }

      std::vector<size_t> send_buffer_size(mpi_size_b,0);
      
      for(int load_rank=load_rank_start; load_rank < load_rank_end; load_rank++) {
	if( mpi_rank_b == load_rank ) {
	  for(size_t ia=0; ia < load_adet_size; ia++) {
	    if( send_adet_rank[ia] > -1 ) {
	      for(size_t ib=0; ib < load_bdet_size; ib++) {
		if( send_bdet_rank[ib] > -1 ) {
		  int target_rank = send_adet_rank[ia] * bdet_comm_size + send_bdet_rank[ib];
		  send_buffer_size[target_rank] += 1;
		}
	      }
	    }
	  }
	  for(int rank=0; rank < mpi_size_b; rank++) {
	    send_W[rank].reserve(send_buffer_size[rank]);
	    send_I[rank].reserve(send_buffer_size[rank]);
	  }
	  for(size_t ia=0; ia < load_adet_size; ia++) {
	    if( send_adet_rank[ia] > -1 ) {
	      for(size_t ib=0; ib < load_bdet_size; ib++) {
		if( send_bdet_rank[ib] > -1 ) {
		  int target_rank = send_adet_rank[ia] * bdet_comm_size + send_bdet_rank[ib];
		  size_t target_I = (send_adet_idx[ia]-adet_start[send_adet_rank[ia]])
		    * ( bdet_end[send_bdet_rank[ib]]-bdet_start[send_bdet_rank[ib]] )
		    + send_bdet_idx[ib] - bdet_start[send_bdet_rank[ib]];
		  send_I[target_rank].push_back(target_I);
		  send_W[target_rank].push_back(load_W[ia*load_bdet_size+ib]);
		}
	      }
	    }
	  }
	}
      }


#ifdef SBD_DEBUG_RESTART
      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
	if( rank_t == mpi_rank_t ) {
	  for(int rank_b=load_rank_start; rank_b < load_rank_end; rank_b++) {
	    if( rank_b == mpi_rank_b ) {
	      std::cout << " LoadWavefunction at (h,b,t) = (" << mpi_rank_h
			<< "," << mpi_rank_b << "," << mpi_rank_t << "): ";
	      for(int rank_s=0; rank_s < send_I.size(); rank_s++) {
		std::cout << " send_I[" << rank_s << "] =";
		for(int k=0; k < std::min(send_I[rank_s].size(),static_cast<size_t>(4)); k++) {
		  std::cout << " " << send_I[rank_s][k];
		}
		std::cout << " ... (size " << send_I[rank_s].size() << ")";
	      }
	      std::cout << std::endl;
	    }
	    MPI_Barrier(b_comm);
	  }
	}
	MPI_Barrier(t_comm);
      }
#endif

      std::vector<std::vector<size_t>> recv_I(load_rank_end-load_rank_start);
      std::vector<std::vector<ElemT>>  recv_W(load_rank_end-load_rank_start);

      MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;

      for(int send_rank=load_rank_start; send_rank < load_rank_end; send_rank++) {
	for(int recv_rank=0; recv_rank < mpi_size_b; recv_rank++) {
	  if( send_rank != recv_rank ) {
	    if( mpi_rank_b == send_rank ) {
	      size_t send_size = send_I[recv_rank].size();
	      MPI_Send(&send_size,1,SBD_MPI_SIZE_T,recv_rank,0,b_comm);
	      if( send_size != 0 ) {
		MPI_Send(send_I[recv_rank].data(),send_size,SBD_MPI_SIZE_T,recv_rank,1,b_comm);
		MPI_Send(send_W[recv_rank].data(),send_size,DataT,recv_rank,2,b_comm);
	      }
	    } else if ( mpi_rank_b == recv_rank ) {
	      size_t recv_size = 0;
	      MPI_Status status;
	      MPI_Recv(&recv_size,1,SBD_MPI_SIZE_T,send_rank,0,b_comm,&status);
	      recv_I[send_rank-load_rank_start].resize(recv_size);
	      recv_W[send_rank-load_rank_start].resize(recv_size);
	      if( recv_size != 0 ) {
		MPI_Recv(recv_I[send_rank-load_rank_start].data(),recv_size,SBD_MPI_SIZE_T,send_rank,1,b_comm,&status);
		MPI_Recv(recv_W[send_rank-load_rank_start].data(),recv_size,DataT,send_rank,2,b_comm,&status);
	      }
	    }
	  } else {
	    if( mpi_rank_b == send_rank ) {
	      recv_I[send_rank-load_rank_start] = send_I[recv_rank];
	      recv_W[send_rank-load_rank_start] = send_W[recv_rank];
	    }
	  }
	}
      }

#ifdef SBD_DEBUG_RESTART
      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
	if( rank_t == mpi_rank_t ) {
	  for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	    if( rank_b == mpi_rank_b ) {
	      std::cout << " LoadWavefunction at (" << mpi_rank_h
			<< "," << mpi_rank_b << "," << mpi_rank_t
			<< "):";
	      for(int rank_r=0; rank_r < recv_I.size(); rank_r++) {
		std::cout << " recv_I[" << rank_r+load_rank_start << "] =";
		for(size_t k=0;  k < std::min(recv_I[rank_r].size(),static_cast<size_t>(4)); k++) {
		  std::cout << " " << recv_I[rank_r][k];
		}
		std::cout << " ... (size " << recv_I[rank_r].size() << ")";
	      }
	      std::cout << std::endl;
	    }
	    MPI_Barrier(b_comm);
	  }
	}
	MPI_Barrier(t_comm);
      }
#endif
      
      for(int rank=0; rank < recv_I.size(); rank++) {
	for(size_t k=0; k < recv_I[rank].size(); k++) {
	  W[recv_I[rank][k]] += recv_W[rank][k];
	}
      }

      MpiAllreduce(W,MPI_SUM,t_comm);
      
    } // end read-in and distribution of wave function within h_comm_rank == 0
    MpiAllreduce(W,MPI_SUM,h_comm);

    RealT norm_W;
    Normalize(W,norm_W,b_comm);
    
  }
  
#else

  template <typename ElemT>
  void LoadWavefunction(const std::string & file,
			const det_vector<size_t, det_kind::half> & adet,
			const det_vector<size_t, det_kind::half> & bdet,
			const size_t adet_comm_size,
			const size_t bdet_comm_size,
			const MPI_Comm h_comm,
			const MPI_Comm b_comm,
			const MPI_Comm t_comm,
			std::vector<ElemT> & W) {

    using RealT = typename GetRealType<ElemT>::RealT;

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    // information of current basis
    std::vector<size_t> adet_start(adet_comm_size);
    std::vector<size_t> adet_end(adet_comm_size);
    std::vector<size_t> bdet_start(bdet_comm_size);
    std::vector<size_t> bdet_end(bdet_comm_size);

    for(int rank=0; rank < adet_comm_size; rank++) {
      adet_start[rank] = 0;
      adet_end[rank]   = adet.size();
      get_mpi_range(adet_comm_size,rank,adet_start[rank],adet_end[rank]);
    }
    for(int rank=0; rank < bdet_comm_size; rank++) {
      bdet_start[rank] = 0;
      bdet_end[rank]   = bdet.size();
      get_mpi_range(bdet_comm_size,rank,bdet_start[rank],bdet_end[rank]);
    }

    size_t this_adet_rank = mpi_rank_b / bdet_comm_size;
    size_t this_bdet_rank = mpi_rank_b % bdet_comm_size;
    size_t this_adet_range = adet_end[this_adet_rank]-adet_start[this_adet_rank];
    size_t this_bdet_range = bdet_end[this_bdet_rank]-bdet_start[this_bdet_rank];
    W.resize(this_adet_range*this_bdet_range,ElemT(0.0));

    size_t load_adet_size;
    size_t load_bdet_size;
    size_t load_det_length;
    std::vector<std::vector<size_t>> load_adet;
    std::vector<std::vector<size_t>> load_bdet;
    std::vector<ElemT> load_W;
    std::vector<size_t> load_det_size(2,0);
    
    if( mpi_rank_h == 0 ) {
      
      if( mpi_rank_t == 0 ) {
	std::string tag = to_padded_string(mpi_rank_b,6);
	std::string filename = file + tag;
	std::ifstream ifile(filename,std::ios::binary);

	if( ifile.is_open() ) {
	  ifile.read(reinterpret_cast<char *>(&load_adet_size),sizeof(size_t));
	  ifile.read(reinterpret_cast<char *>(&load_bdet_size),sizeof(size_t));
	  ifile.read(reinterpret_cast<char *>(&load_det_length),sizeof(size_t));
	  load_adet.resize(load_adet_size,std::vector<size_t>(load_det_length));
	  load_bdet.resize(load_bdet_size,std::vector<size_t>(load_det_length));
	  for(size_t i=0; i < load_adet_size; i++) {
	    ifile.read(reinterpret_cast<char *>(load_adet[i].data()),sizeof(size_t)*load_det_length);
	  }
	  for(size_t i=0; i < load_bdet_size; i++) {
	    ifile.read(reinterpret_cast<char *>(load_bdet[i].data()),sizeof(size_t)*load_det_length);
	    
	  }
	  load_W.resize(load_adet_size*load_bdet_size,ElemT(0.0));
	  ifile.read(reinterpret_cast<char *>(load_W.data()),sizeof(ElemT)*load_adet_size*load_bdet_size);
	  ifile.close();
	  load_det_size[0] = load_adet_size;
	  load_det_size[1] = load_bdet_size;
	} else {
          std::cerr << "failed to open input file " << filename << ", exiting ..." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 0);
	}
      } // end read-in from file

      MPI_Bcast(load_det_size.data(),2,SBD_MPI_SIZE_T,0,t_comm);
    }

    MPI_Bcast(load_det_size.data(),2,SBD_MPI_SIZE_T,0,h_comm);

    load_adet_size = load_det_size[0];
    load_bdet_size = load_det_size[1];
      
    size_t load_det_count = 0;
    if( load_det_size[0]*load_det_size[1] != 0 ) {
      if( mpi_rank_h == 0 ) {
	MpiBcast(load_adet,0,t_comm);
	MpiBcast(load_bdet,0,t_comm);
	MpiBcast(load_W,0,t_comm);
	load_det_count = 1;
      }
      MpiBcast(load_adet,0,h_comm);
      MpiBcast(load_bdet,0,h_comm);
      MpiBcast(load_W,0,h_comm);
      load_det_count = 1;
    } else {
      load_adet.resize(0);
      load_bdet.resize(0);
      load_W.resize(0);
      load_det_count = 0;
    }
    
    size_t load_mpi_size_b = 0;
    MPI_Allreduce(&load_det_count,&load_mpi_size_b,1,SBD_MPI_SIZE_T,MPI_SUM,b_comm);
      
    size_t load_rank_start = 0;
    size_t load_rank_end   = load_mpi_size_b;
    int mpi_size_d = mpi_size_t*mpi_size_h;
    int mpi_rank_d = mpi_size_t*mpi_rank_h + mpi_rank_t;
    get_mpi_range(mpi_size_d,mpi_rank_d,load_rank_start,load_rank_end);

    std::vector<std::vector<size_t>> send_I(mpi_size_b);
    std::vector<std::vector<ElemT>>  send_W(mpi_size_b);

    std::vector<int> send_adet_rank(load_adet_size,-1);
    std::vector<size_t> send_adet_idx(load_adet_size);
    std::vector<int> send_bdet_rank(load_bdet_size,-1);
    std::vector<size_t> send_bdet_idx(load_bdet_size);

    for(int load_rank=load_rank_start; load_rank < load_rank_end; load_rank++) {
      if( mpi_rank_b == load_rank ) {
	for(size_t ia = 0; ia < load_adet_size; ia++) {
	  auto itja = std::find(adet.begin(),adet.end(),load_adet[ia]);
	  if( itja != adet.end() ) {
	    size_t ja = std::distance(adet.begin(),itja);
	    int adet_rank = 0;
	    for(int rank=0; rank < adet_comm_size; rank++) {
	      if( ( adet_start[rank] <= ja ) && ( ja < adet_end[rank] ) ) {
		adet_rank = rank;
		break;
	      }
	    }
	    send_adet_rank[ia] = adet_rank;
	    send_adet_idx[ia]  = ja;
	  }
	}
	for(size_t ib=0; ib < load_bdet_size; ib++) {
	  auto itjb = std::find(bdet.begin(),bdet.end(),load_bdet[ib]);
	  if( itjb != bdet.end() ) {
	    size_t jb = std::distance(bdet.begin(),itjb);
	    int bdet_rank = 0;
	    for(int rank=0; rank < bdet_comm_size; rank++) {
	      if( ( bdet_start[rank] <= jb ) && ( jb < bdet_end[rank] ) ) {
		bdet_rank = rank;
		break;
	      }
	    }
	    send_bdet_rank[ib] = bdet_rank;
	    send_bdet_idx[ib]  = jb;
	  }
	}
      }
    }

    std::vector<size_t> send_buffer_size(mpi_size_b,0);
    
    for(int load_rank=load_rank_start; load_rank < load_rank_end; load_rank++) {
      if( mpi_rank_b == load_rank ) {
	for(size_t ia=0; ia < load_adet_size; ia++) {
	  if( send_adet_rank[ia] > -1 ) {
	    for(size_t ib=0; ib < load_bdet_size; ib++) {
	      if( send_bdet_rank[ib] > -1 ) {
		int target_rank = send_adet_rank[ia] * bdet_comm_size + send_bdet_rank[ib];
		send_buffer_size[target_rank] += 1;
	      }
	    }
	  }
	}
	for(int rank=0; rank < mpi_size_b; rank++) {
	  send_W[rank].reserve(send_buffer_size[rank]);
	  send_I[rank].reserve(send_buffer_size[rank]);
	}
	for(size_t ia=0; ia < load_adet_size; ia++) {
	  if( send_adet_rank[ia] > -1 ) {
	    for(size_t ib=0; ib < load_bdet_size; ib++) {
	      if( send_bdet_rank[ib] > -1 ) {
		int target_rank = send_adet_rank[ia] * bdet_comm_size + send_bdet_rank[ib];
		size_t target_I = (send_adet_idx[ia]-adet_start[send_adet_rank[ia]])
		  * ( bdet_end[send_bdet_rank[ib]]-bdet_start[send_bdet_rank[ib]] )
		  + send_bdet_idx[ib] - bdet_start[send_bdet_rank[ib]];
		send_I[target_rank].push_back(target_I);
		send_W[target_rank].push_back(load_W[ia*load_bdet_size+ib]);
	      }
	    }
	  }
	}
      }
    }
    
    std::vector<std::vector<size_t>> recv_I(load_rank_end-load_rank_start);
    std::vector<std::vector<ElemT>>  recv_W(load_rank_end-load_rank_start);
    
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    
    for(int send_rank=load_rank_start; send_rank < load_rank_end; send_rank++) {
      for(int recv_rank=0; recv_rank < mpi_size_b; recv_rank++) {
	if( send_rank != recv_rank ) {
	  if( mpi_rank_b == send_rank ) {
	    size_t send_size = send_I[recv_rank].size();
	    MPI_Send(&send_size,1,SBD_MPI_SIZE_T,recv_rank,0,b_comm);
	    if( send_size != 0 ) {
	      MPI_Send(send_I[recv_rank].data(),send_size,SBD_MPI_SIZE_T,recv_rank,1,b_comm);
	      MPI_Send(send_W[recv_rank].data(),send_size,DataT,recv_rank,2,b_comm);
	    }
	  } else if ( mpi_rank_b == recv_rank ) {
	    size_t recv_size = 0;
	    MPI_Status status;
	    MPI_Recv(&recv_size,1,SBD_MPI_SIZE_T,send_rank,0,b_comm,&status);
	    recv_I[send_rank-load_rank_start].resize(recv_size);
	    recv_W[send_rank-load_rank_start].resize(recv_size);
	    if( recv_size != 0 ) {
	      MPI_Recv(recv_I[send_rank-load_rank_start].data(),recv_size,SBD_MPI_SIZE_T,send_rank,1,b_comm,&status);
	      MPI_Recv(recv_W[send_rank-load_rank_start].data(),recv_size,DataT,send_rank,2,b_comm,&status);
	    }
	  }
	} else {
	  if( mpi_rank_b == send_rank ) {
	    recv_I[send_rank-load_rank_start] = send_I[recv_rank];
	    recv_W[send_rank-load_rank_start] = send_W[recv_rank];
	  }
	}
      }
    }

    for(int rank=0; rank < recv_I.size(); rank++) {
      for(size_t k=0; k < recv_I[rank].size(); k++) {
	W[recv_I[rank][k]] += recv_W[rank][k];
      }
    }

    MpiAllreduce(W,MPI_SUM,t_comm);
    MpiAllreduce(W,MPI_SUM,h_comm);

    RealT norm_W;
    Normalize(W,norm_W,b_comm);
    
  }
  
  
#endif

  template <typename ElemT>
  void SaveMatrixFormWF(const std::string filename,
			size_t bit_length, size_t norb,
			const det_vector<size_t, det_kind::half> & adet,
			const det_vector<size_t, det_kind::half> & bdet,
			size_t adet_comm_size,
			size_t bdet_comm_size,
			MPI_Comm h_comm,
			MPI_Comm b_comm,
			MPI_Comm t_comm,
			std::vector<ElemT> & W) {
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

    std::vector<size_t> adet_start(adet_comm_size);
    std::vector<size_t> adet_end(adet_comm_size);
    std::vector<size_t> bdet_start(bdet_comm_size);
    std::vector<size_t> bdet_end(bdet_comm_size);
    
    for(int rank=0; rank < adet_comm_size; rank++) {
      adet_start[rank] = 0;
      adet_end[rank]   = adet.size();
      get_mpi_range(adet_comm_size,rank,adet_start[rank],adet_end[rank]);
    }
    for(int rank=0; rank < bdet_comm_size; rank++) {
      bdet_start[rank] = 0;
      bdet_end[rank]   = bdet.size();
      get_mpi_range(bdet_comm_size,rank,bdet_start[rank],bdet_end[rank]);
    }

    std::cout.precision(16);
    
    if( mpi_rank_h == 0 && mpi_rank_t == 0 ) {

      if( get_extension(filename) == std::string("bin") ) {
	if( mpi_rank_b == 0 ) {
	  
	  std::vector<std::vector<ElemT>> Vb(bdet_comm_size,std::vector<ElemT>(W));

	  size_t total_size = 0;
	  std::ofstream obin(filename,std::ios::binary);
	  for(int rank_adet=0; rank_adet < adet_comm_size; rank_adet++) {
	    for(int rank_bdet=0; rank_bdet < bdet_comm_size; rank_bdet++) {
	      int src_rank = rank_adet * bdet_comm_size + rank_bdet;
	      if( src_rank == 0 ) {
		Vb[rank_bdet] = W;
		total_size += W.size();
	      } else {
		MpiRecv(Vb[rank_bdet],src_rank,b_comm);
		total_size += Vb[rank_bdet].size();
	      }
	    }
	    for(size_t ia = adet_start[rank_adet]; ia < adet_end[rank_adet]; ia++) {
	      for(int rank_bdet=0; rank_bdet < bdet_comm_size; rank_bdet++) {
		for(size_t ib = bdet_start[rank_bdet]; ib < bdet_end[rank_bdet]; ib++) {
		  size_t idx = ( ia - adet_start[rank_adet] )
		             * ( bdet_end[rank_bdet]-bdet_start[rank_bdet] )
		             + ib - bdet_start[rank_bdet];
		  obin.write(reinterpret_cast<char *>(&Vb[rank_bdet][idx]),sizeof(ElemT));
		}
	      }
	    }
	  }
	  obin.close();
	  if ( total_size != adet.size()*bdet.size() ) {
	    std::cout << " SaveMatrixFormWF: total size is not compatible " << std::endl;
	  }
	} else {
	  std::vector<ElemT> V(W);
	  MpiSend(V,0,b_comm);
	}
      } else if ( get_extension(filename) == std::string("dat") ||
		  get_extension(filename) == std::string("txt") ) {
	
	if( mpi_rank_b == 0 ) {

	  std::vector<std::vector<ElemT>> Vb(bdet_comm_size,std::vector<ElemT>(W));

	  size_t total_size = 0;
	  std::ofstream ofile(filename);
	  for(int rank_adet=0; rank_adet < adet_comm_size; rank_adet++) {
	    for(int rank_bdet=0; rank_bdet < bdet_comm_size; rank_bdet++) {
	      int src_rank = rank_adet * bdet_comm_size + rank_bdet;
	      if( src_rank == 0 ) {
		Vb[rank_bdet] = W;
		total_size += W.size();
	      } else {
		MpiRecv(Vb[rank_bdet],src_rank,b_comm);
		total_size += Vb[rank_bdet].size();
	      }
	    }
	    for(size_t ia = adet_start[rank_adet]; ia < adet_end[rank_adet]; ia++) {
	      for(int rank_bdet=0; rank_bdet < bdet_comm_size; rank_bdet++) {
		for(size_t ib = bdet_start[rank_bdet]; ib < bdet_end[rank_bdet]; ib++) {
		  size_t idx = ( ia - adet_start[rank_adet] )
		             * ( bdet_end[rank_bdet]-bdet_start[rank_bdet] )
		             + ib - bdet_start[rank_bdet];
		  ofile << Vb[rank_bdet][idx] << " # " << ia << ": "
			<< makestring(adet[ia],bit_length,norb) << " "
			<< ib << ": " << makestring(bdet[ib],bit_length,norb)
			<< std::endl;
		}
	      }
	    }
	  }
	  ofile.close();
	  if ( total_size != adet.size()*bdet.size() ) {
	    std::cout << " SaveMatrixFormWF: total size is not compatible " << std::endl;
	  }
	} else {
	  std::vector<ElemT> V(W);
	  MpiSend(V,0,b_comm);
	}
      }
	
    }

  }

  
} // end namespace sbd

#endif // end ifndef SBD_CHEMISTRY_PTMB_RESTART_H
