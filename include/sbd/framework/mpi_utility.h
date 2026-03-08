/// This is a part of qscd
/**
@file mpi_utility.h
@brief tools for mpi parallelization
 */

#ifndef SBD_FRAMEWORK_MPI_UTILITY_H
#define SBD_FRAMEWORK_MPI_UTILITY_H

#include <type_traits>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mpi.h"

namespace sbd {

  void get_mpi_range(int mpi_size, int mpi_rank, size_t & i_begin, size_t & i_end)
  {
    size_t i_all = i_end - i_begin;
    
    size_t i_div = i_all / mpi_size;
    size_t i_res = i_all % mpi_size;
  
    size_t j = i_begin;
    size_t j_begin;
    size_t j_end;
    if( mpi_rank < i_res )
      {
	j_begin = i_begin + ( i_div + 1 ) * mpi_rank;
	j_end = i_begin + ( i_div + 1 ) * ( mpi_rank + 1 );
      }
    else
      {
	j_begin = i_begin + ( i_div + 1 ) * i_res + i_div * ( mpi_rank - i_res );
	j_end = i_begin + ( i_div + 1 ) * i_res + i_div * ( mpi_rank + 1 - i_res );
      }
    i_begin = j_begin;
    i_end = j_end;
  }

  template <typename ElemT>
  void MpiSend(const std::vector<ElemT> & data, int dest, MPI_Comm comm) {
    size_t d_size = data.size();
    MPI_Send(&d_size,1,SBD_MPI_SIZE_T,dest,0,comm);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( d_size != 0 ) {
      MPI_Send(data.data(),d_size,DataT,dest,1,comm);
    }
  }

  template <typename ElemT>
  void MpiRecv(std::vector<ElemT> & data, int source, MPI_Comm comm) {
    MPI_Status status;
    size_t d_size;
    MPI_Recv(&d_size,1,SBD_MPI_SIZE_T,source,0,comm,&status);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( d_size != 0 ) {
      data.resize(d_size);
      MPI_Recv(data.data(),d_size,DataT,source,1,comm,&status);
    }
  }

  template <typename ElemT>
  void MpiIsend(const std::vector<ElemT> & data, int dest, MPI_Comm comm) {
    size_t d_size = data.size();
    MPI_Request req;
    MPI_Isend(&d_size,1,SBD_MPI_SIZE_T,dest,0,comm,&req);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( d_size != 0 ) {
      MPI_Isend(data.data(),d_size,DataT,dest,1,comm,&req);
    }
  }

  template <typename ElemT>
  void MpiIrecv(std::vector<ElemT> & data, int source, MPI_Comm comm) {
    size_t d_size;
    MPI_Request req;
    MPI_Irecv(&d_size,1,SBD_MPI_SIZE_T,source,0,comm,&req);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( d_size != 0 ) {
      data.resize(d_size);
      MPI_Irecv(data.data(),d_size,DataT,source,1,comm,&req);
    }
  }
  
  template <typename ElemT>
  void MpiBcast(std::vector<ElemT> & data, int root, MPI_Comm comm) {
    size_t d_size;
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    if( mpi_rank == root ) {
      d_size = data.size();
    }
    MPI_Bcast(&d_size,1,SBD_MPI_SIZE_T,root,comm);
    if( mpi_rank != root ) {
      data.resize(d_size);
    }
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    MPI_Bcast(data.data(),d_size,DataT,root,comm);
  }
  
  template <>
  void MpiSend(const std::vector<std::vector<size_t>> & config, int dest, MPI_Comm comm) {
    size_t c_num = config.size();
    MPI_Send(&c_num,1,SBD_MPI_SIZE_T,dest,0,comm);
    if( c_num != 0 ) {
      size_t c_len = config[0].size();
      MPI_Send(&c_len,1,SBD_MPI_SIZE_T,dest,1,comm);
      size_t total_size = c_num*c_len;
      std::vector<size_t> config_send(total_size);
      for(size_t n=0; n < c_num; n++) {
	for(size_t i=0; i < c_len; i++) {
	  config_send[i+c_len*n] = config[n][i];
	}
      }
      MPI_Send(config_send.data(),total_size,SBD_MPI_SIZE_T,dest,2,comm);
    }
  }
  
  template <>
  void MpiRecv(std::vector<std::vector<size_t>> & config, int source, MPI_Comm comm) {
    MPI_Status status;
    size_t c_num;
    MPI_Recv(&c_num,1,SBD_MPI_SIZE_T,source,0,comm,&status);
    if( c_num != 0 ) {
      size_t c_len;
      MPI_Recv(&c_len,1,SBD_MPI_SIZE_T,source,1,comm,&status);
      size_t total_size = c_num*c_len;
      std::vector<size_t> config_recv(total_size);
      MPI_Recv(config_recv.data(),total_size,SBD_MPI_SIZE_T,source,2,comm,&status);
      config = std::vector<std::vector<size_t>>(c_num,std::vector<size_t>(c_len));
      for(size_t n=0; n < c_num; n++) {
	for(size_t i=0; i < c_len; i++) {
	  config[n][i] = config_recv[i+c_len*n];
	}
      }
    }
  }

  template <>
  void MpiIsend(const std::vector<std::vector<size_t>> & config, int dest, MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    std::cout << " MpiIsend for std::vector<std::vector<size_t>> is called at rank " << mpi_rank;
    MPI_Request req;
    size_t c_num = config.size();
    std::cout << " c_num = " << c_num;
    MPI_Isend(&c_num,1,SBD_MPI_SIZE_T,dest,0,comm,&req);
    if( c_num != 0 ) {
      size_t c_len = config[0].size();
      std::cout << " c_len = " << c_len;
      MPI_Isend(&c_len,1,SBD_MPI_SIZE_T,dest,1,comm,&req);
      size_t total_size = c_num*c_len;
      std::cout << " total size = " << total_size;
      std::vector<size_t> config_send(total_size);
      for(size_t n=0; n < c_num; n++) {
	for(size_t i=0; i < c_len; i++) {
	  config_send[i+c_len*n] = config[n][i];
	}
      }
      MPI_Isend(config_send.data(),total_size,SBD_MPI_SIZE_T,dest,2,comm,&req);
    }
    std::cout << std::endl;
  }

  template <>
  void MpiIrecv(std::vector<std::vector<size_t>> & config, int source, MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    std::cout << " MpiIrecv for std::vector<std::vector<size_t>> is called at rank "
	      << mpi_rank;
    MPI_Request req;
    size_t c_num;
    MPI_Irecv(&c_num,1,SBD_MPI_SIZE_T,source,0,comm,&req);
    std::cout << " c_num = " << c_num;
    if( c_num != 0 ) {
      size_t c_len;
      MPI_Irecv(&c_len,1,SBD_MPI_SIZE_T,source,1,comm,&req);
      std::cout << " c_len = " << c_len << std::endl;
      size_t total_size = c_num*c_len;
      std::cout << " total size = " << total_size << std::endl;
      std::vector<size_t> config_recv(total_size);
      MPI_Irecv(config_recv.data(),total_size,SBD_MPI_SIZE_T,source,2,comm,&req);
      config.resize(c_num,std::vector<size_t>(c_len));
      for(size_t n=0; n < c_num; n++) {
	for(size_t i=0; i < c_len; i++) {
	  config[n][i] = config_recv[i+c_len*n];
	}
      }
    }
  }
  
  
  template <>
  void MpiBcast(std::vector<std::vector<size_t>> & config,
		int root,
		MPI_Comm comm) {

    size_t c_num;
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    if( mpi_rank == root ) {
      c_num = config.size();
    }
    MPI_Bcast(&c_num,1,SBD_MPI_SIZE_T,root,comm);
    if( c_num != 0 ) {
      
      size_t c_len;
      if( mpi_rank == root ) {
	c_len = config[0].size();
      }
      MPI_Bcast(&c_len,1,SBD_MPI_SIZE_T,root,comm);
      size_t total_size = c_num*c_len;
      std::vector<size_t> config_transfer(total_size);
      if( mpi_rank == root ) {
	for(size_t n=0; n < c_num; n++) {
	  for(size_t i=0; i < c_len; i++) {
	    config_transfer[i+c_len*n] = config[n][i];
	  }
	}
      }
      MPI_Bcast(config_transfer.data(),static_cast<int>(total_size),SBD_MPI_SIZE_T,root,comm);
      if( mpi_rank != root ) {
	config = std::vector<std::vector<size_t>>(c_num,std::vector<size_t>(c_len));
	for(size_t n=0; n < c_num; n++) {
	  for(size_t i=0; i < c_len; i++) {
	    config[n][i] = config_transfer[i+c_len*n];
	  }
	}
      }
    }
  }
  
  template <typename ElemT>
  void MpiIncSlide(const std::vector<ElemT> & A,
		   std::vector<ElemT> & B,
		   MPI_Comm comm) {
    
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    
    if( mpi_size % 2 == 0 ) {
      if( mpi_rank % 2 == 0 ) {
	MpiSend(A,mpi_rank+1,comm);
      } else {
	MpiRecv(B,mpi_rank-1,comm);
      }
      if( mpi_rank % 2 == 1 ) {
	if( mpi_rank == mpi_size-1 ) {
	  MpiSend(A,0,comm);
	} else {
	  MpiSend(A,mpi_rank+1,comm);
	}
      } else {
	if( mpi_rank == 0 ) {
	  MpiRecv(B,mpi_size-1,comm);
	} else {
	  MpiRecv(B,mpi_rank-1,comm);
	}
      }
    } else {
      if( mpi_rank % 2 == 0 && mpi_rank != mpi_size-1 ) {
	MpiSend(A,mpi_rank+1,comm);
      } else {
	MpiRecv(B,mpi_rank-1,comm);
      }
      if( mpi_rank % 2 == 1 ) {
	MpiSend(A,mpi_rank+1,comm);
      } else {
	MpiRecv(B,mpi_rank-1,comm);
      }
      if( mpi_rank == mpi_size-1 ) {
	MpiSend(A,0,comm);
      }
      if( mpi_rank == 0 ) {
	MpiRecv(B,mpi_size-1,comm);
      }
    }
  }
  
  template <typename ElemT>
  void MpiDecSlide(const std::vector<ElemT> & A, std::vector<ElemT> & B, MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    if( mpi_size % 2 == 0 ) {
      if( mpi_rank % 2 == 0 ) {
	MpiRecv(B,mpi_rank+1,comm);
      } else {
	MpiSend(A,mpi_rank-1,comm);
      }
      if( mpi_rank % 2 == 1 ) {
	if( mpi_rank == mpi_size-1 ) {
	  MpiRecv(B,0,comm);
	} else {
	  MpiRecv(B,mpi_rank+1,comm);
	}
      } else {
	if( mpi_rank == 0 ) {
	  MpiSend(A,mpi_size-1,comm);
	} else {
	  MpiSend(A,mpi_rank-1,comm);
	}
      }
    } else {
      if( mpi_rank % 2 == 0 && mpi_rank != mpi_size-1 ) {
	MpiRecv(B,mpi_rank+1,comm);
      } else {
	MpiSend(A,mpi_rank-1,comm);
      }
      if( mpi_rank % 2 == 1 ) {
	MpiRecv(B,mpi_rank+1,comm);
      } else {
	MpiSend(A,mpi_rank-1,comm);
      }
      if( mpi_rank == mpi_size-1 ) {
	MpiRecv(B,0,comm);
      }
      if( mpi_rank == 0 ) {
	MpiSend(A,mpi_size-1,comm);
      }
    }
  }

  template <typename ElemT>
  void MpiSlide(const std::vector<ElemT> & A,
		std::vector<ElemT> & B,
		int slide,
		MPI_Comm comm) {
    
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;

    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = A.size();
    MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( send_size != 0 ) {
      MPI_Isend(A.data(),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
      MPI_Irecv(B.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }

    if( send_size != 0 && recv_size != 0 ) {
      MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
      MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
      MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }

  }

  

  template <>
  void MpiSlide(const std::vector<std::vector<size_t>> & A,
		std::vector<std::vector<size_t>> & B,
		int slide,
		MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;
    
    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);

    // We first check the size
    std::vector<size_t> size_send(2);
    std::vector<size_t> size_recv(2);
    size_send[0] = A.size();
    if( A.size() != 0 ) {
      size_send[1] = A[0].size();
    } else {
      size_send[1] = static_cast<size_t>(0);
    }
    MPI_Isend(size_send.data(),2,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),2,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());

    size_t total_send_size = size_send[0]*size_send[1];
    size_t total_recv_size = size_recv[0]*size_recv[1];

    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

    std::vector<size_t> data_send(total_send_size);
    std::vector<size_t> data_recv(total_recv_size);
    if( size_recv[0] != 0 ) {
      B.resize(size_recv[0],std::vector<size_t>(size_recv[1]));
    } else {
      B.resize(0);
    }

    for(size_t n=0; n < size_send[0]; n++) {
      for(size_t k=0; k < size_send[1]; k++) {
	data_send[n*size_send[1]+k] = A[n][k];
      }
    }

    if( total_send_size != 0 ) {
      MPI_Isend(data_send.data(),total_send_size,SBD_MPI_SIZE_T,mpi_dest,1,comm,&req_data[0]);
    }
    if( total_recv_size != 0 ) {
      MPI_Irecv(data_recv.data(),total_recv_size,SBD_MPI_SIZE_T,mpi_source,1,comm,&req_data[1]);
    }
    
    if( total_send_size != 0 && total_recv_size != 0 ) {
      MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( total_send_size != 0 && total_recv_size == 0 ) {
      MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( total_send_size == 0 && total_recv_size != 0 ) {
      MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }

    for(size_t n=0; n < size_recv[0]; n++) {
      for(size_t k=0; k < size_recv[1]; k++) {
	B[n][k] = data_recv[n*size_recv[1]+k];
      }
    }
    
  }


  template <>
  void MpiSlide(const std::vector<size_t> & A,
		std::vector<size_t> & B,
		int slide,
		MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_dest   = (mpi_size+mpi_rank+slide) % mpi_size;
    int mpi_source = (mpi_size+mpi_rank-slide) % mpi_size;
    
    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = A.size();
    MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,mpi_dest,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());
    
    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);
    
    MPI_Datatype DataT = SBD_MPI_SIZE_T;
    if( send_size != 0 ) {
      MPI_Isend(A.data(),send_size,DataT,mpi_dest,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
      MPI_Irecv(B.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
    
    if( send_size != 0 && recv_size != 0 ) {
      MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
      MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
      MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
  }

#ifdef SBD_TRADMODE
  template <typename ElemT>
  void MpiAllreduce(std::vector<ElemT> & A, MPI_Op op, MPI_Comm comm) {
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    std::vector<ElemT> B(A);
#if MPI_VERSION >= 4
    MPI_Allreduce_c(B.data(),A.data(),A.size(),DataT,op,comm);
#else
    if (A.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI_Allreduce: count exceeds INT_MAX (MPI<4). Use MPI-4 *_c API.");
    }
    MPI_Allreduce(B.data(),A.data(),static_cast<int>(A.size()),DataT,op,comm);
#endif
  }

  template <>
  void MpiAllreduce(std::vector<size_t> & A, MPI_Op op, MPI_Comm comm) {
    MPI_Datatype DataT = SBD_MPI_SIZE_T;
    std::vector<size_t> B(A);
#if MPI_VERSION >= 4
    MPI_Allreduce_c(B.data(),A.data(),A.size(),DataT,op,comm);
#else
    if (A.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI_Allreduce: count exceeds INT_MAX (MPI<4). Use MPI-4 *_c API.");
    }
    MPI_Allreduce(B.data(),A.data(),static_cast<int>(A.size()),DataT,op,comm);
#endif
  }
#else
  template <typename ElemT>
  void MpiAllreduce(std::vector<ElemT> & A, MPI_Op op, MPI_Comm comm) {
    MPI_Datatype DataT;
    if constexpr (std::is_same_v<ElemT,size_t>) {
      DataT = SBD_MPI_SIZE_T;
    } else {
      DataT = GetMpiType<ElemT>::MpiT;
    }
    std::vector<ElemT> B(A);
#if MPI_VERSION >= 4
    MPI_Allreduce_c(B.data(),A.data(),A.size(),DataT,op,comm);
#else
    if (A.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI_Allreduce: count exceeds INT_MAX (MPI<4). Use MPI-4 *_c API.");
    }
    MPI_Allreduce(B.data(),A.data(),static_cast<int>(A.size()),DataT,op,comm);
#endif
  }
#endif

  template <typename ElemT>
  void Mpi2dSlide(const std::vector<ElemT> & A,
		  std::vector<ElemT> & B,
		  int x_size,
		  int y_size,
		  int x_slide,
		  int y_slide,
		  MPI_Comm comm) {
    // Assuming mpi_rank = x_rank * y_size + y_rank;

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    int x_rank = mpi_rank / y_size;
    int y_rank = mpi_rank % y_size;

    int x_dist = ( x_rank + x_slide + x_size ) % x_size;
    int y_dist = ( y_rank + y_slide + y_size ) % y_size;
    int mpi_dist = x_dist * y_size + y_dist;

    int x_source = ( x_rank - x_slide + x_size ) % x_size;
    int y_source = ( y_rank - y_slide + y_size ) % y_size;
    int mpi_source = x_source * y_size + y_source;

#ifdef SBD_DEBUG_MPI_UTILITY
    std::cout << " Mpi2dSlide at rank " << mpi_rank << " = (" << x_rank << "," << y_rank
	      << "): distination rank = " << mpi_dist << " = (" << x_dist << "," << y_dist
	      << "), source rank = " << mpi_source << " = (" << x_source << "," << y_source
	      << ")" << std::endl;
#endif
    std::vector<MPI_Request> req_size(2);
    std::vector<MPI_Status> sta_size(2);
    std::vector<size_t> size_send(1);
    std::vector<size_t> size_recv(1);
    size_send[0] = A.size();
    
    MPI_Isend(size_send.data(),1,SBD_MPI_SIZE_T,
	      mpi_dist,0,comm,&req_size[0]);
    MPI_Irecv(size_recv.data(),1,SBD_MPI_SIZE_T,
	      mpi_source,0,comm,&req_size[1]);
    MPI_Waitall(2,req_size.data(),sta_size.data());

    size_t send_size = size_send[0];
    size_t recv_size = size_recv[0];
    B.resize(recv_size);
    std::vector<MPI_Request> req_data(2);
    std::vector<MPI_Status> sta_data(2);

#if MPI_VERSION >= 4
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( send_size != 0 ) {
      MPI_Isend_c(A.data(),send_size,DataT,mpi_dist,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
      MPI_Irecv_c(B.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#else
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( send_size != 0 ) {
      MPI_Isend(A.data(),send_size,DataT,mpi_dist,1,comm,&req_data[0]);
    }
    if( recv_size != 0 ) {
      MPI_Irecv(B.data(),recv_size,DataT,mpi_source,1,comm,&req_data[1]);
    }
#endif

    if( send_size != 0 && recv_size != 0 ) {
      MPI_Waitall(2,req_data.data(),sta_data.data());
    } else if ( send_size != 0 && recv_size == 0 ) {
      MPI_Waitall(1,&req_data[0],&sta_data[0]);
    } else if ( send_size == 0 && recv_size != 0 ) {
      MPI_Waitall(1,&req_data[1],&sta_data[1]);
    }
    
  }
		    
  
}

#endif
