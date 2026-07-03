/**
@file sbd/caop/basic/msdavidson.h
@brief multi-start davidson for creation-annihilation operator models
 */
#ifndef SBD_CAOP_BASIC_MSDAVIDSON_H
#define SBD_CAOP_BASIC_MSDAVIDSON_H

#include "sbd/framework/hp_numeric.h"
#include "sbd/framework/ssutils.h"
#include "sbd/framework/dm_vector.h"
#include "sbd/framework/timestamp.h"

namespace sbd {

  template <typename ElemT>
  void InitVectorCAOP(std::vector<ElemT> & w,
		      const std::vector<std::vector<size_t>> & basis,
		      MPI_Comm h_comm,
		      MPI_Comm b_comm,
		      MPI_Comm t_comm,
		      int init,
		      size_t seed) {
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    if( init == 0 ) {
      if( mpi_rank_t == 0 ) {
	w.resize(basis.size());
	Randomize(seed,w,b_comm,h_comm);
      }
      MpiBcast(w,0,t_comm);
    }
  }

  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
		std::vector<ElemT> & W,
		const std::vector<std::vector<size_t>> & bs,
		const size_t bit_length,
		const std::vector<int> & slide,
		const GeneralOp<ElemT> & Ham,
		bool sign,
		MPI_Comm h_comm,
		MPI_Comm b_comm,
		MPI_Comm t_comm,
		int max_iteration,
		int num_block,
		int num_initvec,
		RealT eps,
		size_t seed) {

    // RealT eps_reg = 1.0e-12;
    size_t w_size = W.size();
    RealT eps_reg = 1.0e-8;

    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<std::vector<ElemT>> V(num_initvec);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    std::vector<ElemT> vdot;
    ElemT vnrm;
    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

    bool do_continue = true;

    for(int iv=0; iv < num_initvec; iv++) {
      if( iv == 0 ) {
	V[0] = std::move(W);
      } else {
	if( mpi_rank_t == 0 ) {
	  if( mpi_rank_h == 0 ) {
	    V[iv].resize(w_size);
	    Randomize(seed+iv,V[iv],b_comm);
	    MGS(V,iv,V[iv],vdot,b_comm);
	    MGS(V,iv,V[iv],vdot,b_comm);
	    Normalize(V[iv],vnrm,b_comm);
	  }
	  MpiBcast(V[iv],0,h_comm);
	}
	MpiBcast(V[iv],0,t_comm);
      }
    }

    for(int it=0; it < max_iteration; it++) {
      for(int iv=0; iv < num_initvec; iv++) {
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  C[iv][is] = V[iv][is];
	}
      }

      for(int ib=0; ib < nb; ib++) {
	Zero(HC[ib]);
	mult(hii,C[ib],HC[ib],
	     bs,bit_length,
	     slide,Ham,sign,h_comm,b_comm,t_comm);
	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	if( ib < num_initvec-1 ) continue;

	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}
	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	for(int iv=0; iv < num_initvec; iv++) {
	  ElemT x = U[0+nb*iv];
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    V[iv][is] = C[0][is] * x;
	  }
	  for(int kb=1; kb <= ib; kb++) {
	    x = U[kb+nb*iv];
#pragma omp parallel for
	    for(size_t is=0; is < w_size; is++) {
	      V[iv][is] += C[kb][is] * x;
	    }
	  }
	}

	
	ElemT x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  R[is] += E[0]*V[0][is];
	}

	for(int iv=0; iv < num_initvec; iv++) {
	  MpiAllreduce(V[iv],MPI_SUM,t_comm);
	  MpiAllreduce(V[iv],MPI_SUM,h_comm);
	}
	MpiAllreduce(R,MPI_SUM,t_comm);
	MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*mpi_size_t));
	for(int iv=0; iv < num_initvec; iv++) {
#pragma	omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    V[iv][is] *= volp;
	  }
	  RealT norm_V;
	  Normalize(V[iv],norm_V,b_comm);
	}
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
          R[is] *= volp;
	}
	RealT norm_R;
	Normalize(R,norm_R,b_comm);

	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    if( mpi_rank_b == 0 ) {
	      std::cout << " " << make_timestamp()
			<< " Davidson iteration " << it << "." << ib
			<< " (tol=" << norm_R << "):";
	      for(int p=0; p < std::min(ib+1,4); p++) {
		std::cout << " " << E[p];
	      }
	      std::cout << std::endl;
	    }
	  }
	}
	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // 2-step Gram-Schmidt orthogonalization
	  MGS(C,ib+1,C[ib+1],vdot,b_comm);
	  MGS(C,ib+1,C[ib+1],vdot,b_comm);
	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);
	}
      }
      if( !do_continue ) {
	break;
      }
    }

    W = std::move(V[0]);
    /*
#pragma omp parallel for
    for(size_t is=0; is < W.size(); is++) {
      W[is] = V[0][is];
    }
    */

    free(H);
    free(U);
    free(E);
    
  }
		
  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
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
		int num_initvec,
		RealT eps,
		size_t seed) {

    // RealT eps_reg = 1.0e-12;
    size_t w_size = W.size();
    RealT eps_reg = 1.0e-8;

    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<std::vector<ElemT>> V(num_initvec);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    std::vector<ElemT> vdot;
    ElemT vnrm;
    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

    bool do_continue = true;

    for(int iv=0; iv < num_initvec; iv++) {
      if( iv == 0 ) {
	V[0] = std::move(W);
      } else {
	if( mpi_rank_t == 0 ) {
	  if( mpi_rank_h == 0 ) {
	    V[iv].resize(w_size);
	    Randomize(seed+iv,V[iv],b_comm);
	    MGS(V,iv,V[iv],vdot,b_comm);
	    MGS(V,iv,V[iv],vdot,b_comm);
	    Normalize(V[iv],vnrm,b_comm);
	  }
	  MpiBcast(V[iv],0,h_comm);
	}
	MpiBcast(V[iv],0,t_comm);
      }
    }

    for(int it=0; it < max_iteration; it++) {
      for(int iv=0; iv < num_initvec; iv++) {
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  C[iv][is] = V[iv][is];
	}
      }

      for(int ib=0; ib < nb; ib++) {
	Zero(HC[ib]);
	mult(hii,ih,jh,hij,C[ib],HC[ib],
	     slide,h_comm,b_comm,t_comm);

	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	if( ib < num_initvec-1 ) continue;
	
	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}
	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	for(size_t iv=0; iv < num_initvec; iv++) {
	  ElemT x = U[0+nb*iv];
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    V[iv][is] = C[0][is] * x;
	  }
	  for(int kb=1; kb <= ib; kb++) {
	    x = U[kb+nb*iv];
#pragma omp parallel for
	    for(size_t is=0; is < w_size; is++) {
	      V[iv][is] += C[kb][is] * x;
	    }
	  }
	}
	
	ElemT x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
	  R[is] += E[0]*V[0][is];
	}

	for(int iv=0; iv < num_initvec; iv++) {
	  MpiAllreduce(V[iv],MPI_SUM,t_comm);
	  MpiAllreduce(V[iv],MPI_SUM,h_comm);
	}
	MpiAllreduce(R,MPI_SUM,t_comm);
	MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*mpi_size_t));
	for(int iv=0; iv < num_initvec; iv++) {
#pragma	omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    V[iv][is] *= volp;
	  }
	  RealT norm_V;
	  Normalize(V[iv],norm_V,b_comm);
	}
#pragma omp parallel for
	for(size_t is=0; is < w_size; is++) {
          R[is] *= volp;
	}
	RealT norm_R;
	Normalize(R,norm_R,b_comm);

	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    if( mpi_rank_b == 0 ) {
	      std::cout << " " << make_timestamp()
			<< " Davidson iteration " << it << "." << ib
			<< " (tol=" << norm_R << "):";
	      for(int p=0; p < std::min(ib+1,4); p++) {
		std::cout << " " << E[p];
	      }
	      std::cout << std::endl;
	    }
	  }
	}
	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < w_size; is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // Gram-Schmidt orthogonalization
	  MGS(C,ib+1,C[ib+1],vdot,b_comm);
	  MGS(C,ib+1,C[ib+1],vdot,b_comm);
	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);
	}
      }
      if( !do_continue ) {
	break;
      }
    }

    W = std::move(V[0]);
    /*
#pragma omp parallel for
    for(size_t is=0; is < W.size(); is++) {
      W[is] = V[0][is];
    }
    */

    free(H);
    free(U);
    free(E);
    
  }
		
  
}

#endif
