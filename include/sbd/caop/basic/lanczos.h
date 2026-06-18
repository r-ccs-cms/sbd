/**
@file sbd/chemistry/tpb/lanzcos.h
@brief lanczos for parallel task management for distributed basis
*/
#ifndef SBD_CAOP_BASIC_LANCZOS_H
#define SBD_CAOP_BASIC_LANCZOS_H

#include "sbd/framework/hp_numeric.h"
#include "sbd/framework/dm_vector.h"
#include "sbd/framework/timestamp.h"

namespace sbd {

  template <typename ElemT, typename RealT>
  void Lanczos(const std::vector<ElemT> & hii,
	       std::vector<ElemT> & W,
	       const det_vector<size_t> & bs,
	       const size_t bit_length,
	       const std::vector<int> & slide,
	       const GeneralOp<ElemT> & Ham,
	       bool sign,
	       MPI_Comm h_comm,
	       MPI_Comm b_comm,
	       MPI_Comm t_comm,
	       int max_iteration,
	       int num_block,
	       RealT eps) {
    
    char jobz = 'V';
    char uplo = 'U';
    int lda   = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    RealT * A = (RealT *) calloc(num_block*num_block,sizeof(RealT));
    RealT * U = (RealT *) calloc(num_block*num_block,sizeof(RealT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));

    std::vector<ElemT> HC(W);
    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<ElemT> edot;

    for(int it=0; it < max_iteration; it++) {

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	HC[is] = ElemT(0.0);
      }

      int n=0;
      ElemT Aii;
      RealT E_old = 1.0e+8;
      bool stop_it = false;

      for(int ib=0; ib < num_block; ib++) {
	
	n++;
	int ii = ib + lda * ib;
	int ij = ib + lda * (ib+1);
	int ji = ib+1 + lda * ib;
	mult(hii,C[ib],HC,
	     bs,bit_length,
	     slide,Ham,sign,h_comm,b_comm,t_comm);
	InnerProduct(C[ib],HC,Aii,b_comm);
	A[ii] = GetReal(Aii);
	for(int i=0; i < n; i++) {
	  for(int j=0; j < n; j++) {
	    U[i+lda*j] = A[i+lda*j];
	  }
	}
	hp_numeric::MatHeev(jobz,uplo,n,U,lda,E);

	if( std::abs(E[0]-E_old) < eps ) {
	  stop_it = true;
	  break;
	}
	E_old = E[0];
	if( ib+1 == num_block ) {
	  break;
	}

#pragma omp parallel for
	for(size_t is=0; is < C[ib].size(); is++) {
	  HC[is] -= Aii * C[ib][is];
	}

	MGS(C,ib+1,HC,edot,b_comm);
	MGS(C,ib+1,HC,edot,b_comm);

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  C[ib+1][is] = HC[is];
	}

	Normalize(C[ib+1],A[ij],b_comm);
	A[ji] = A[ij];

	if( (mpi_rank_h == 0) &&
	    (mpi_rank_b == 0) &&
	    (mpi_rank_t == 0) ) {
	  std::cout << " " << make_timestamp()
		    << " Lanczos iteration " << it
		    << ", step " << ib
		    << ": (A,B)=(" << A[ii]
		    << "," << A[ij] << "):";
	  for(int p=0; p < std::min(n,6); p++) {
	    std::cout << " " << E[p];
	  }
	  std::cout << std::endl;
	}

	if( std::abs(A[ij]) < eps ) {
	  stop_it = true;
	  break;
	}

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] = - A[ij] * C[ib][is];
	}
      }
      for(int i=0; i < n; i++) {
	for(int j=0; j < n; j++) {
	  U[i+lda*j] = A[i+lda*j];
	}
      }
      hp_numeric::MatHeev(jobz,uplo,n,U,lda,E);
      for(int ib=0; ib < n; ib++) {
	if( ib == 0 ) {
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] = U[0] * C[ib][is];
	  }
	} else {
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += U[ib] * C[ib][is];
	  }
	}
      }
      RealT NormW;
      Normalize(W,NormW,b_comm);
      if( stop_it ) {
	break;
      }
    }

    free(A);
    free(E);
    free(U);
  }

  template <typename ElemT, typename RealT>
  void Lanczos(const std::vector<ElemT> & hii,
	       const std::vector<std::vector<std::vector<size_t>>> & ih,
	       const std::vector<std::vector<std::vector<size_t>>> & jh,
	       const std::vector<std::vector<std::vector<ElemT>>> & hij,
	       std::vector<ElemT> & W,
	       const std::vector<int> & slide,
	       MPI_Comm h_comm,
	       MPI_Comm b_comm,
	       MPI_Comm t_comm,
	       int max_iteration,
	       int num_block,
	       RealT eps) {
    
    char jobz = 'V';
    char uplo = 'U';
    int lda   = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    RealT * A = (RealT *) calloc(num_block*num_block,sizeof(RealT));
    RealT * U = (RealT *) calloc(num_block*num_block,sizeof(RealT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));

    std::vector<ElemT> HC(W);
    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<ElemT> edot;

    for(int it=0; it < max_iteration; it++) {
      
      int n=0;
      ElemT Aii;
      RealT E_old = 1.0e+8;
      bool stop_it = false;
      
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	HC[is] = ElemT(0.0);
      }

      for(int ib=0; ib < num_block; ib++) {
	
	n++;
	int ii = ib + lda * ib;
	int ij = ib + lda * (ib+1);
	int ji = ib+1 + lda * ib;

	mult(hii,ih,jh,hij,C[ib],HC,
	     slide,h_comm,b_comm,t_comm);
	InnerProduct(C[ib],HC,Aii,b_comm);
	A[ii] = GetReal(Aii);
	for(int i=0; i < n; i++) {
	  for(int j=0; j < n; j++) {
	    U[i+lda*j] = A[i+lda*j];
	  }
	}
	hp_numeric::MatHeev(jobz,uplo,n,U,lda,E);

	if( std::abs(E[0]-E_old) < eps ) {
	  stop_it = true;
	  break;
	}
	E_old = E[0];
	if( ib+1 == num_block ) {
	  break;
	}

#pragma omp parallel for
	for(size_t is=0; is < C[ib].size(); is++) {
	  HC[is] -= Aii * C[ib][is];
	}

	MGS(C,ib+1,HC,edot,b_comm);
	MGS(C,ib+1,HC,edot,b_comm);

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  C[ib+1][is] = HC[is];
	}

	Normalize(C[ib+1],A[ij],b_comm);
	A[ji] = A[ij];

	if( (mpi_rank_h == 0) &&
	    (mpi_rank_b == 0) &&
	    (mpi_rank_t == 0) ) {
	  std::cout << " " << make_timestamp()
		    << " Lanczos iteration " << it
		    << ", step " << ib
		    << ": (A,B)=(" << A[ii]
		    << "," << A[ij] << "):";
	  for(int p=0; p < std::min(n,4); p++) {
	    std::cout << " " << E[p];
	  }
	  std::cout << std::endl;
	}

	if( std::abs(A[ij]) < eps ) {
	  stop_it = true;
	  break;
	}

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] = - A[ij] * C[ib][is];
	}
      }

      for(int i=0; i < n; i++) {
	for(int j=0; j < n; j++) {
	  U[i+lda*j] = A[i+lda*j];
	}
      }
      hp_numeric::MatHeev(jobz,uplo,n,U,lda,E);
      for(int ib=0; ib < n; ib++) {
	if( ib == 0 ) {
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] = U[0] * C[ib][is];
	  }
	} else {
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += U[ib] * C[ib][is];
	  }
	}
      }
      RealT NormW;
      Normalize(W,NormW,b_comm);
      if( stop_it ) {
	break;
      }
    }
    free(A);
    free(E);
    free(U);
  }

  
}

#endif
