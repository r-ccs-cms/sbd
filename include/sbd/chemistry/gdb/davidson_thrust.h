/**
@file sbd/chemistry/gdb/davidson_thrust.h
@brief davidson for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_DAVIDSON_THRUST_H
#define SBD_CHEMISTRY_GDB_DAVIDSON_THRUST_H

namespace sbd {
  namespace gdb {

    template <typename ElemT, typename DetsContainer>
    void BasisInitVector(std::vector<ElemT> & w,
			 const DetsContainer & det,
			 MPI_Comm h_comm,
			 MPI_Comm b_comm,
			 MPI_Comm t_comm,
			 int init) {
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      w.resize(det.size());
      if( init == 0 ) {
	if( mpi_rank_b == 0 ) {
	  w[0] = ElemT(1.0);
	}
	MpiBcast(w,0,t_comm);
      } else if ( init == 1 ) {
	if( mpi_rank_t == 0 ) {
	  Randomize(w,b_comm,h_comm);
	}
	MpiBcast(w,0,t_comm);
      }
    }

    template <typename ElemT, typename RealT>
    void Davidson(const std::vector<ElemT> & hii,
		  std::vector<ElemT> & w,
		  const std::vector<std::vector<size_t>> & det,
		  const size_t bit_length,
		  const size_t norb,
		  const DetIndexMap & idxmap,
		  const std::vector<ExcitationLookup> & exidx,
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

      std::vector<std::vector<ElemT>> v(num_block,w);
      std::vector<std::vector<ElemT>> Hv(num_block,w);
      std::vector<ElemT> r(w);
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

      bool do_continue = true;

      for(int it=0; it < max_iteration; it++) {

#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}

	for(int ib=0; ib < nb; ib++) {

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << ", wave function weight before applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << v[ib][is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=v[ib].size()/2-2; is < v[ib].size()/2+2; is++) {
		      std::cout << "," << v[ib][is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=v[ib].size()-2; is < v[ib].size(); is++) {
		    std::cout << "," << v[ib][is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif

	  Zero(Hv[ib]);
	  mult(hii,v[ib],Hv[ib],bit_length,norb,
	       det,idxmap,exidx,I0,I1,I2,
	       h_comm,b_comm,t_comm);

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " wave function weight after applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << Hv[ib][is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=Hv[ib].size()/2-2; is < Hv[ib].size()/2+2; is++) {
		      std::cout << "," << Hv[ib][is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=Hv[ib].size()-2; is < Hv[ib].size(); is++) {
		    std::cout << "," << Hv[ib][is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif


	  for(int jb=0; jb <= ib; jb++) {
	    InnerProduct(v[jb],Hv[ib],H[jb+nb*ib],b_comm);
	    H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	  }
	  for(int jb=0; jb <= ib; jb++) {
	    for(int kb=0; kb <= ib; kb++) {
	      U[jb+nb*kb] = H[jb+nb*kb];
	    }
	  }

#ifdef SBD_DEBUG_MULT
	  if( mpi_rank_h == 0 &&
	      mpi_rank_b == 0 &&
	      mpi_rank_t == 0 ) {
	    std::cout << " " << make_timestamp()
		      << " sbd: davidson step "
		      << it << "," << ib
		      << " effective matrix = [";
	    for(int kb=0; kb <= ib; kb++) {
	      for(int jb=0; jb <= ib; jb++) {
		std::cout << ( (jb==0) ? "[" : "," ) << U[jb+nb*kb];
	      }
	      std::cout << "]";
	    }
	    std::cout << "]" << std::endl;
	  }

#endif

	  hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);
	  ElemT x = U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] = v[0][is] * x;
	  }
	  x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] = Hv[0][is] * x;
	  }
	  for(int kb=1; kb <= ib; kb++) {
	    x = U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      w[is] += v[kb][is] * x;
	    }
	    x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      r[is] += Hv[kb][is] * x;
	    }
	  }
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] += E[0]*w[is];
	  }

	  MpiAllreduce(w,MPI_SUM,t_comm);
	  MpiAllreduce(w,MPI_SUM,h_comm);
	  MpiAllreduce(r,MPI_SUM,t_comm);
	  MpiAllreduce(r,MPI_SUM,h_comm);
	  ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] *= volp;
	  }
#pragma omp parallel for
	  for(size_t is=0; is < r.size(); is++) {
	    r[is] *= volp;
	  }

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " residual vector at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << r[is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=r.size()/2-2; is < r.size()/2+2; is++) {
		      std::cout << "," << r[is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=r.size()-2; is < r.size(); is++) {
		    std::cout << "," << r[is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif

	  RealT norm_w;
	  Normalize(w,norm_w,b_comm);

	  RealT norm_r;
	  Normalize(r,norm_r,b_comm);

	  if( mpi_rank_h == 0 ) {
	    if( mpi_rank_t == 0 ) {
	      if( mpi_rank_b == 0 ) {
		std::cout << " Davidson iteration " << it << "." << ib
			  << " (tol=" << norm_r << "):";
		for(int p=0; p < std::min(ib+1,4); p++) {
		  std::cout << " " << E[p];
		}
		std::cout << std::endl;
	      }
	    }
	  }
	  if( norm_r < eps ) {
	    do_continue = false;
	    break;
	  }
	  if( ib < nb-1 ) {
	    // Determine
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      if( std::abs(E[0]-dii[is]) > eps_reg ) {
		v[ib+1][is] = r[is]/(E[0] - dii[is]);
	      } else {
		v[ib+1][is] = r[is]/(E[0] - dii[is] - eps_reg);
	      }
	    }
	    // Gram-Schmidt orthogonalization
	    for(int kb=0; kb < ib+1; kb++) {
	      ElemT olap;
	      InnerProduct(v[kb],v[ib+1],olap,b_comm);
	      olap *= ElemT(-1.0);
#pragma omp parallel for
	      for(size_t is=0; is < w.size(); is++) {
		v[ib+1][is] += v[kb][is]*olap;
	      }
	    }
	    RealT norm_v;
	    Normalize(v[ib+1],norm_v,b_comm);
	  }
	} // end for(int ib=0; ib < nb; ib++)
	if( !do_continue ) {
	  break;
	}
	// Restart with C[0] = W;
#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}
      } // end for(int it=0; it < max_iteration; it++)
      free(H);
      free(U);
      free(E);
    }


template <typename ElemT, typename RealT>
void Davidson(const std::vector<ElemT> & hii,
		const std::vector<std::vector<size_t*>> & ih,
		const std::vector<std::vector<size_t*>> & jh,
		const std::vector<std::vector<ElemT*>> & hij,
		const std::vector<std::vector<size_t>> & len,
		const std::vector<int> & slide,
		std::vector<ElemT> & w,
		MPI_Comm h_comm,
		MPI_Comm b_comm,
		MPI_Comm t_comm,
		int max_iteration,
		int num_block,
		RealT eps)
{
	// not used for Thrust
}


} // end namespace gdb
} // end namespace sbd

#endif
