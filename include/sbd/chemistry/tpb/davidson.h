/**
@file sbd/chemistry/tpb/davidson.h
@brief davidson for parallel task management for distributed basis
*/
#ifndef SBD_CHEMISTRY_TPB_DAVIDSON_H
#define SBD_CHEMISTRY_TPB_DAVIDSON_H

#include "sbd/framework/jacobi.h"
#include "sbd/framework/hp_numeric.h"
#include "sbd/framework/dm_vector.h"

namespace sbd {

/**

   --- b_comm --- color rule: y = const

    o -- o -- o -- o

    o -- o -- o -- o

    o -- o -- o -- o

x = 0    1    2    3

   --- t_comm --- color rule: x = const
                      y
    o    o    o    o  2
    |    |    |    |
    o    o    o    o  1
    |    |    |    |
    o    o    o    o  0
x = 0    1    2    3

   -----------------------

   - h_comm is the communicator in direction
     perpendicular to b_comm times k_comm plane.

 */

/**
   Initializer for the wave function
   @tparam ElemT: Type of elements for the Hamiltonian and the wave functions
   @param[out] W: The wave vector to be initialized.
   @param[in] helper: helper array to construct the Hamiltonian
   @param[in] h_comm: Communicator to split the row-index when performing the Hamiltonian operations
   @param[in] b_comm: Communicator to split the wave function data
   @param[in] t_comm: Communicator to split the operation in column basis when performing the Hamiltonian operations
   @param[in] init: Select type of initial state. init==0 corresponds to the HF solution. init==1 is the random state.
 */


  template <typename ElemT>
  void BasisInitVector(std::vector<ElemT> & W,
		       const std::vector<TaskHelpers> helper,
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

    size_t AlphaSize = 0;
    size_t BetaSize  = 0;
    if( helper.size() != 0 ) {
      AlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
      BetaSize  = helper[0].braBetaEnd  - helper[0].braBetaStart;
    }
    W.resize(AlphaSize*BetaSize,ElemT(0.0));
    if( init == 0 ) { // default = start from fermi sea
      if( mpi_rank_b == 0 ) {
	W[0] = ElemT(1.0);
      }
      MpiBcast(W,0,t_comm);
    } else if ( init == 1 ) {
      if( mpi_rank_t == 0 ) {
	Randomize(seed,W,b_comm,h_comm);
      }
      MpiBcast(W,0,t_comm);
    }
  }

/**
   Initializer for the wave function
   @tparam ElemT: Type of elements for the Hamiltonian and the wave functions
   @param[out] W: The wave vector to be initialized.
   @param[in] adet: Array of bit string for alpha det
   @param[in] bdet: Array of bit string for beta det
   @param[in] adet_comm_size: Number of communicators used to split the adet
   @param[in] bdet_comm_size: Number of communicators used to split the bdet
   @param[in] h_comm: Communicator to split the row-index when performing the Hamiltonian operations
   @param[in] b_comm: Communicator to split the wave function data
   @param[in] t_comm: Communicator to split the operation in column basis when performing the Hamiltonian operations
   @param[in] init: Select type of initial state. init==0 corresponds to the HF solution. init==1 is the random state.
 */


  template <typename ElemT>
  void BasisInitVector(std::vector<ElemT> & W,
		       const std::vector<std::vector<size_t>> & adet,
		       const std::vector<std::vector<size_t>> & bdet,
		       const size_t adet_comm_size,
		       const size_t bdet_comm_size,
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

    int adet_rank = mpi_rank_b / bdet_comm_size;
    int bdet_rank = mpi_rank_b % bdet_comm_size;
    size_t adet_start = 0;
    size_t adet_end   = adet.size();
    size_t bdet_start = 0;
    size_t bdet_end   = bdet.size();
    get_mpi_range(adet_comm_size,adet_rank,adet_start,adet_end);
    get_mpi_range(bdet_comm_size,bdet_rank,bdet_start,bdet_end);
    size_t adet_size = adet_end-adet_start;
    size_t bdet_size = bdet_end-bdet_start;
    W.resize(adet_size*bdet_size,ElemT(0.0));
    if( init == 0 ) { // default = start from fermi sea
      if( mpi_rank_b == 0 ) {
	W[0] = ElemT(1.0);
      }
    } else if ( init == 1 ) {
      if( mpi_rank_t == 0 ) {
	Randomize(seed,W,b_comm,h_comm);
      }
      MpiBcast(W,0,t_comm);
    }
  }

  /**
     Davidson method for the stored Hamiltonian constructed by the TaskHelpers
     @tparam ElemT: Type of the elements of the Hamiltonian and the wave functions
     @param[in] hii: Diagonal elements for the Hamiltonian matrix
     @param[in] ih: row-index of the off-diagonal term
     @param[in] jh: column-index of the off-diagonal term
     @param[in] hij: Value of elements of off-diagonal term
     @param[in] len: Length of the each (ih[x],jh[x],hij[x])
     @param[in] tasktype: Type of task
     @param[in] adetshift: Shift of ket-side block of alpha-string index
     @param[in] bdetshift: Shift of ket-side block of beta-string index
     @param[in] adet_comm_size: Number of division in alpha-string array
     @param[in] bdet_comm_size: Number of division in beta-string array
     @param[in/out] W: Initial vector in input, ground state wave function in output.
     @param[in] h_comm: Communicator used to cyclicly split the row-index when performing the multiplication of Hamiltonian
     @param[in] b_comm: Communicator to split the wave vector
     @param[in] t_comm: Communicator to split the tasks in column-index when performing the multiplication
     @param[in] max_iteration: Number of maximum interation of the Davidson iteration
     @param[in] num_block: Maximum size of Litz vector space
     @param[in] eps: error torelance (norm of the residual vector)
   */

  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
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

    RealT eps_reg = 1.0e-12;

    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

#ifdef SBD_DEBUG_DAVIDSON
    std::cout << " diagonal term at mpi process (h,b,t) = ("
	      << mpi_rank_h << "," << mpi_rank_b << ","
	      << mpi_rank_t << "): ";
    for(size_t id=0; id < std::min(W.size(),static_cast<size_t>(6)); id++) {
      std::cout << " " << dii[id];
    }
    std::cout << std::endl;
#endif

    bool do_continue = true;

    for(int it=0; it < max_iteration; it++) {
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

      for(int ib=0; ib < nb; ib++) {

	Zero(HC[ib]);
	mult(hii,ih,jh,hij,len,
	     tasktype,adetshift,bdetshift,adet_comm_size,bdet_comm_size,
	     C[ib],HC[ib],bit_length,h_comm,b_comm,t_comm);

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): Hv(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << HC[ib][n];
	}
	std::cout << " ... " << HC[ib][W.size()-1] << std::endl;
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "):  v(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << C[ib][n];
	}
	std::cout << " ... " << C[ib][W.size()-1] << std::endl;
#endif

	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}

	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	ElemT x = U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  W[is] = C[0][is] * x;
	}
	x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += C[kb][is] * x;
	  }
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] += E[0]*W[is];
	}

	// #ifdef SBD_FUAGKUPATCH
	if (mpi_size_t > 1) MpiAllreduce(W,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(W,MPI_SUM,h_comm);
	if (mpi_size_t > 1) MpiAllreduce(R,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
        for(size_t is=0; is < W.size(); is++) {
          W[is] *= volp;
	}
#pragma omp parallel for
	for(size_t is=0; is < R.size(); is++) {
          R[is] *= volp;
	}
	// #endif

	RealT norm_W;
	Normalize(W,norm_W,b_comm);

	RealT norm_R;
	Normalize(R,norm_R,b_comm);

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " Davidson iteration " << it << "." << ib
		  << " at mpi (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): (tol=" << norm_R << "):";
	for(int p=0; p < std::min(ib+1,4); p++) {
	  std::cout << " " << E[p];
	}
	std::cout << std::endl;
#else
	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    if( mpi_rank_b == 0 ) {
	      std::cout << " Davidson iteration " << it << "." << ib
			<< " (tol=" << norm_R << "):";
	      for(int p=0; p < std::min(ib+1,4); p++) {
		std::cout << " " << E[p];
	      }
	      std::cout << std::endl;
	    }
	  }
	}
#endif

	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // Gram-Schmidt orthogonalization
	  for(int kb=0; kb < ib+1; kb++) {
	    ElemT olap;
	    InnerProduct(C[kb],C[ib+1],olap,b_comm);
	    olap *= ElemT(-1.0);
#pragma omp parallel for
	    for(size_t is=0; is < W.size(); is++) {
	      C[ib+1][is] += C[kb][is]*olap;
	    }
	  }

	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);

	}
      } // end for(int ib=0; ib < nb; ib++)

      if( !do_continue ) {
	break;
      }

      // Restart with C[0] = W;
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

    } // end for(int it=0; it < max_iteration; it++)

    free(H);
    free(U);
    free(E);

  }

  /**
     Davidson method for the direct multiplication using TaskHelpers
     @tparam ElemT: Type of the Hamiltonian and wave functions
     @tparam RealT: Real type of ElemT
     @param[in] hii: Diagonal elements for the Hamiltonian matrix
     @param[in/out] W: Initialized wave function in input. Obtained ground state at output.
     @param[in] adets: bit strings to span the Hilbert space for alpha spins
     @param[in] bdets: bit strings to span the Hilbert space for beta spins
     @param[in] bit_length: Bit length managed by each size_t in adets and bdets
     @param[in] norbs: Overall bit length of alpha-det and beta-det, corresponding to the number of orbitals
     @param[in] adet_comm_size: number of nodes used to split the alpha-dets
     @param[in] bdet_comm_size: number of nodes used to split the beta-dets
     @param[in] helper: TaskHelpers to perform the Hamiltonian operatorations
     @param[in] I0: Constant shift of energy
     @param[in] I1: One-body integrals
     @param[in] I2: Two-body integrals
     @param[in] h_comm: Communicator used to cyclicly split the row-index when performing the multiplication of Hamiltonian
     @param[in] b_comm: Communicator to split the wave vector
     @param[in] t_comm: Communicator to split the tasks in column-index when performing the multiplication
     @param[in] max_iteration: Number of maximum interation of the Davidson iteration
     @param[in] num_block: Maximum size of Litz vector space
     @param[in] eps: error torelance (norm of the residual vector)
  */

  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
		std::vector<ElemT> & W,
		const std::vector<std::vector<size_t>> & adets,
		const std::vector<std::vector<size_t>> & bdets,
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

    RealT eps_reg = 1.0e-12;

    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

#ifdef SBD_DEBUG_DAVIDSON
    std::cout << " diagonal term at mpi process (h,b,t) = ("
	      << mpi_rank_h << "," << mpi_rank_b << ","
	      << mpi_rank_t << "): ";
    for(size_t id=0; id < std::min(W.size(),static_cast<size_t>(6)); id++) {
      std::cout << " " << dii[id];
    }
    std::cout << std::endl;
#endif

    bool do_continue = true;

    for(int it=0; it < max_iteration; it++) {

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

      for(int ib=0; ib < nb; ib++) {

	Zero(HC[ib]);
	mult(hii,C[ib],HC[ib],
	     adets,bdets,bit_length,norbs,
	     adet_comm_size,bdet_comm_size,
	     helper,I0,I1,I2,
	     h_comm,b_comm,t_comm);
    ElemT t;
    InnerProduct(HC[ib], HC[ib], t, b_comm);
    std::cout << "  dot(HC) = " << t << std::endl;

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): Hv(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << HC[ib][n];
	}
	std::cout << " ... " << HC[ib][W.size()-1] << std::endl;
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "):  v(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << C[ib][n];
	}
	std::cout << " ... " << C[ib][W.size()-1] << std::endl;
#endif

	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}

	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	ElemT x = U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  W[is] = C[0][is] * x;
	}
	x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += C[kb][is] * x;
	  }
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] += E[0]*W[is];
	}

	// #ifdef SBD_FUAGKUPATCH
	if (mpi_size_t > 1) MpiAllreduce(W,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(W,MPI_SUM,h_comm);
	if (mpi_size_t > 1) MpiAllreduce(R,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
        for(size_t is=0; is < W.size(); is++) {
          W[is] *= volp;
	}
#pragma omp parallel for
	for(size_t is=0; is < R.size(); is++) {
          R[is] *= volp;
	}
	// #endif
	RealT norm_W;
	Normalize(W,norm_W,b_comm);

	RealT norm_R;
	Normalize(R,norm_R,b_comm);

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " Davidson iteration " << it << "." << ib
		  << " at mpi (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): (tol=" << norm_R << "):";
	for(int p=0; p < std::min(ib+1,4); p++) {
	  std::cout << " " << E[p];
	}
	std::cout << std::endl;
#else
	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    if( mpi_rank_b == 0  ) {
	      std::cout << " Davidson iteration " << it << "." << ib
			<< " (tol=" << norm_R << "):";
	      for(int p=0; p < std::min(ib+1,4); p++) {
		std::cout << " " << E[p];
	      }
	      std::cout << std::endl;
	    }
	  }
	}
#endif

	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // Gram-Schmidt orthogonalization
	  for(int kb=0; kb < ib+1; kb++) {
	    ElemT olap;
	    InnerProduct(C[kb],C[ib+1],olap,b_comm);
	    olap *= ElemT(-1.0);
#pragma omp parallel for
	    for(size_t is=0; is < W.size(); is++) {
	      C[ib+1][is] += C[kb][is]*olap;
	    }
	  }

	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);

	}
      } // end for(int ib=0; ib < nb; ib++)

      if( !do_continue ) {
	break;
      }

      // Restart with C[0] = W;
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

    } // end for(int it=0; it < max_iteration; it++)

    free(H);
    free(U);
    free(E);

  }

#ifndef SBD_THRUST
  /**
     Davidson method for the direct multiplication using TaskHelpers, specialized for the SQD loop calculation.
     @tparam ElemT: Type of the Hamiltonian and wave functions
     @tparam RealT: Real type of ElemT
     @param[in] hii: Diagonal elements for the Hamiltonian matrix
     @param[in/out] W: Initialized wave function in input. Obtained ground state at output.
     @param[in] adets: bit strings to span the Hilbert space for alpha spins
     @param[in] bdets: bit strings to span the Hilbert space for beta spins
     @param[in] bit_length: Bit length managed by each size_t in adets and bdets
     @param[in] norbs: Overall bit length of alpha-det and beta-det, corresponding to the number of orbitals
     @param[in] adet_comm_size: number of nodes used to split the alpha-dets
     @param[in] bdet_comm_size: number of nodes used to split the beta-dets
     @param[in] helper: TaskHelpers to perform the Hamiltonian operatorations
     @param[in] I0: Constant shift of energy
     @param[in] I1: One-body integrals
     @param[in] I2: Two-body integrals
     @param[in] h_comm: Communicator used to cyclicly split the row-index when performing the multiplication of Hamiltonian
     @param[in] b_comm: Communicator to split the wave vector
     @param[in] t_comm: Communicator to split the tasks in column-index when performing the multiplication
     @param[in] max_iteration: Number of maximum interation of the Davidson iteration
     @param[in] num_block: Maximum size of Litz vector space
     @param[in] eps: error torelance (norm of the residual vector)
     @param[in] max_time: Maximum time allowed to perform the calculation
   */

  template <typename ElemT, typename RealT>
  void Davidson(const std::vector<ElemT> & hii,
		std::vector<ElemT> & W,
		const std::vector<std::vector<size_t>> & adets,
		const std::vector<std::vector<size_t>> & bdets,
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
		RealT eps,
		RealT max_time) {

    RealT eps_reg = 1.0e-12;

    std::vector<std::vector<ElemT>> C(num_block,W);
    std::vector<std::vector<ElemT>> HC(num_block,W);
    std::vector<ElemT> R(W);
    std::vector<ElemT> dii(hii);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

    ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
    RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
    char jobz = 'V';
    char uplo = 'U';
    int nb = num_block;
    MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
    MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;

    GetTotalD(hii,dii,h_comm);

#ifdef SBD_DEBUG_DAVIDSON
    std::cout << " diagonal term at mpi process (h,b,t) = ("
	      << mpi_rank_h << "," << mpi_rank_b << ","
	      << mpi_rank_t << "): ";
    for(size_t id=0; id < std::min(W.size(),static_cast<size_t>(6)); id++) {
      std::cout << " " << dii[id];
    }
    std::cout << std::endl;
#endif

    bool do_continue = true;

    std::vector<double> onestep_times(num_block*max_iteration,0.0);
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int it=0; it < max_iteration; it++) {

#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

      for(int ib=0; ib < nb; ib++) {

	auto step_start = std::chrono::high_resolution_clock::now();

	Zero(HC[ib]);
	mult(hii,C[ib],HC[ib],
	     adets,bdets,bit_length,norbs,
	     adet_comm_size,bdet_comm_size,
	     helper,I0,I1,I2,
	     h_comm,b_comm,t_comm);

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): Hv(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << HC[ib][n];
	}
	std::cout << " ... " << HC[ib][W.size()-1] << std::endl;
	std::cout << " (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "):  v(" << ib << ") =";
	for(size_t n=0; n < std::min(static_cast<size_t>(6),W.size()); n++) {
	  std::cout << " " << C[ib][n];
	}
	std::cout << " ... " << C[ib][W.size()-1] << std::endl;
#endif

	for(int jb=0; jb <= ib; jb++) {
	  InnerProduct(C[jb],HC[ib],H[jb+nb*ib],b_comm);
	  H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	}
	for(int jb=0; jb <= ib; jb++) {
	  for(int kb=0; kb <= ib; kb++) {
	    U[jb+nb*kb] = H[jb+nb*kb];
	  }
	}

#ifdef SBD_NO_LAPACK
	hp_numeric::JacobiHeev(ib+1,U,nb,E);
#else
	hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);
#endif

	ElemT x = U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  W[is] = C[0][is] * x;
	}
	x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] = HC[0][is] * x;
	}
	for(int kb=1; kb <= ib; kb++) {
	  x = U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    W[is] += C[kb][is] * x;
	  }
	  x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    R[is] += HC[kb][is] * x;
	  }
	}
#pragma omp parallel for
	for(size_t is=0; is < W.size(); is++) {
	  R[is] += E[0]*W[is];
	}

	/**
	   Patch for stability on Fugaku
	 */
	// #ifdef SBD_FUAGKUPATCH
	if (mpi_size_t > 1) MpiAllreduce(W,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(W,MPI_SUM,h_comm);
	if (mpi_size_t > 1) MpiAllreduce(R,MPI_SUM,t_comm);
	if (mpi_size_h > 1) MpiAllreduce(R,MPI_SUM,h_comm);
        ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
        for(size_t is=0; is < W.size(); is++) {
          W[is] *= volp;
	}
#pragma omp parallel for
	for(size_t is=0; is < R.size(); is++) {
          R[is] *= volp;
	}
	// #endif

	RealT norm_W;
	Normalize(W,norm_W,b_comm);

	RealT norm_R;
	Normalize(R,norm_R,b_comm);

#ifdef SBD_DEBUG_DAVIDSON
	std::cout << " Davidson iteration " << it << "." << ib
		  << " at mpi (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << ","
		  << mpi_rank_t << "): (tol=" << norm_R << "):";
	for(int p=0; p < std::min(ib+1,4); p++) {
	  std::cout << " " << E[p];
	}
	std::cout << std::endl;
#else
	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    if( mpi_rank_b == 0 ) {
	      std::cout << " Davidson iteration " << it << "." << ib
			<< " (tol=" << norm_R << "):";
	      for(int p=0; p < std::min(ib+1,4); p++) {
		std::cout << " " << E[p];
	      }
	      std::cout << std::endl;
	    }
	  }
	}
#endif

	if( norm_R < eps ) {
	  do_continue = false;
	  break;
	}

	if( ib < nb-1 ) {
	// Determine
#pragma omp parallel for
	  for(size_t is=0; is < W.size(); is++) {
	    if( std::abs(E[0]-dii[is]) > eps_reg ) {
	      C[ib+1][is] = R[is]/(E[0] - dii[is]);
	    } else {
	      C[ib+1][is] = R[is]/(E[0] - dii[is] - eps_reg);
	    }
	  }

	  // Gram-Schmidt orthogonalization
	  for(int kb=0; kb < ib+1; kb++) {
	    ElemT olap;
	    InnerProduct(C[kb],C[ib+1],olap,b_comm);
	    olap *= ElemT(-1.0);
#pragma omp parallel for
	    for(size_t is=0; is < W.size(); is++) {
	      C[ib+1][is] += C[kb][is]*olap;
	    }
	  }

	  RealT norm_C;
	  Normalize(C[ib+1],norm_C,b_comm);

	}

	auto step_end = std::chrono::high_resolution_clock::now();
	onestep_times[it*nb+ib] = std::chrono::duration<double>(step_end-step_start).count();
	double ave_time_per_step = 0.0;
	for(int ks=0; ks <= it*nb+ib; ks++) {
	  ave_time_per_step += onestep_times[ks];
	}
	ave_time_per_step /= (it*nb+ib+1);

	auto current_time = std::chrono::high_resolution_clock::now();
	double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();
	double predicted_next_end = total_elapsed + ave_time_per_step;
	if( mpi_rank_h == 0 ) {
	  if( mpi_rank_t == 0 ) {
	    MPI_Bcast(&predicted_next_end,1,MPI_DOUBLE,0,b_comm);
	  }
	  MPI_Bcast(&predicted_next_end,1,MPI_DOUBLE,0,t_comm);
	}
	MPI_Bcast(&predicted_next_end,1,MPI_DOUBLE,0,h_comm);

	if( predicted_next_end > max_time ) {
	  do_continue = false;
          if (mpi_rank_h + mpi_rank_t + mpi_rank_b == 0) std::cerr << "halting iterations due to max_time limit" << std::endl;
	  break;
	}


      } // end for(int ib=0; ib < nb; ib++)

      if( !do_continue ) {
	break;
      }

      // Restart with C[0] = W;
#pragma omp parallel for
      for(size_t is=0; is < W.size(); is++) {
	C[0][is] = W[is];
      }

    } // end for(int it=0; it < max_iteration; it++)

    free(H);
    free(U);
    free(E);

  }
#endif	// SBD_THRUST


}

#endif
