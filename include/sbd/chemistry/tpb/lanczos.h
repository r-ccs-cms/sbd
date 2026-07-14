/**
@file sbd/chemistry/tpb/lanzcos.h
@brief lanczos for parallel task management for distributed basis
*/
#ifndef SBD_CHEMISTRY_TPB_LANCZOS_H
#define SBD_CHEMISTRY_TPB_LANCZOS_H

#include "sbd/framework/hp_numeric.h"
#include "sbd/framework/dm_vector.h"

namespace sbd {

  template <typename ElemT, typename RealT>
  void Lanczos(const std::vector<ElemT> & hii,
	       const std::vector<std::vector<size_t*>> & ih,
	       const std::vector<std::vector<size_t*>> & jh,
	       const std::vector<std::vector<ElemT*>> & hij,
	       const std::vector<std::vector<size_t>> & len,
	       const std::vector<size_t> & tasktype,
	       const std::vector<size_t> & adetshift,
	       const std::vector<size_t> & bdetshift,
	       const size_t adet_comm_size,
	       const size_t bdet_comm_size,
	       std::vector<ElemT> & W,
	       MPI_Comm h_comm,
	       MPI_Comm b_comm,
	       MPI_Comm t_comm,
	       int max_iteration,
	       int num_block,
	       size_t bit_length,
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
	mult(hii,ih,jh,hij,len,
	     tasktype,adetshift,bdetshift,adet_comm_size,bdet_comm_size,
	     C[ib],HC,bit_length,h_comm,b_comm,t_comm);

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

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  C[ib+1][is] = HC[is];
	}

	Normalize(C[ib+1],A[ij],b_comm);
	A[ji] = A[ij];

	if( (mpi_rank_h == 0) &&
	    (mpi_rank_b == 0) &&
	    (mpi_rank_t == 0) ) {
	  std::cout << " Lanczos iteration " << it
		    << ", step " << ib
		    << ": (A,B)=(" << A[ii]
		    << "," << A[ij] << "):";
	  for(int p=0; p < std::min(n,6); p++) {
	    std::cout << " " << E[p];
	  }
	  std::cout << std::endl;
	}

	if( std::abs(A[ij]) < eps ) {
	  break;
	}

	ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] = - A[ij] * volp * C[ib][is];
	}
	MpiAllreduce(HC,MPI_SUM,t_comm);
	MpiAllreduce(HC,MPI_SUM,h_comm);
#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] *= volp;
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
	       std::vector<ElemT> & W,
	       const det_vector<size_t, det_kind::half> & adets,
	       const det_vector<size_t, det_kind::half> & bdets,
	       const size_t bit_length,
	       const size_t norbs,
	       const size_t adet_comm_size,
	       const size_t bdet_comm_size,
	       const std::vector<TaskHelpers> & helper,
	       const ElemT & I0,
	       const oneInt<ElemT> & I1,
	       const twoInt<ElemT> & I2,
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
	mult(hii,C[ib],HC,
	     adets,bdets,bit_length,norbs,
	     adet_comm_size,bdet_comm_size,
	     helper,I0,I1,I2,
	     h_comm,b_comm,t_comm);

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

#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  C[ib+1][is] = HC[is];
	}

	Normalize(C[ib+1],A[ij],b_comm);
	A[ji] = A[ij];

	if( (mpi_rank_h == 0) &&
	    (mpi_rank_b == 0) &&
	    (mpi_rank_t == 0) ) {
	  std::cout << " Lanczos iteration " << it
		    << ", step " << ib
		    << ": (A,B)=(" << A[ii]
		    << "," << A[ij] << "):";
	  for(int p=0; p < std::min(n,4); p++) {
	    std::cout << " " << E[p];
	  }
	  std::cout << std::endl;
	}

	if( std::abs(A[ij]) < eps ) {
	  break;
	}

	ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] = - A[ij] * volp * C[ib][is];
	}
	MpiAllreduce(HC,MPI_SUM,t_comm);
	MpiAllreduce(HC,MPI_SUM,h_comm);
#pragma omp parallel for
	for(size_t is=0; is < HC.size(); is++) {
	  HC[is] *= volp;
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
