/**
@file sbd/caop/basic/sbdiag.h
@brief Function used for sample-based diagonalization
 */
#ifndef SBD_CAOP_BASIC_SBDIAG_H
#define SBD_CAOP_BASIC_SBDIAG_H

#include "sbd/framework/timestamp.h"

namespace sbd {
  namespace caop {

    struct SBD {
      int t_comm_size = 1;
      int b_comm_size = 1;
      int h_comm_size = 1;

      int method = 0;
      int max_it = 1;
      int max_nb = 10;
      int max_iv = 1;
      double eps = 1.0e-4;
      int init = 0;
      size_t seed = 0;

      size_t system_size = 64;
      size_t bit_length = 32;
      bool sign = false;
      bool do_sort_basis = false;
      bool do_redist_basis = false;
      double ratio = 0.0;
    };

    SBD generate_sbd_data(int argc, char * argv[]) {
      SBD sbd_data;
      for(int i=0; i < argc; i++) {
	if( std::string(argv[i]) == "--t_comm_size" ) {
	  sbd_data.t_comm_size = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--b_comm_size" ) {
	  sbd_data.b_comm_size = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--method" ) {
	  sbd_data.method = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--iteration" ) {
	  sbd_data.max_it = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--block" ) {
	  sbd_data.max_nb = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--numivec" ) {
	  sbd_data.max_iv = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--tolerance" ) {
	  sbd_data.eps = std::atof(argv[++i]);
	}
	if( std::string(argv[i]) == "--system_size" ) {
	  sbd_data.system_size = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--bit_length" ) {
	  sbd_data.bit_length = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--fermionsign" ) {
	  if( std::atoi(argv[++i]) != 0 ) {
	    sbd_data.sign = true;
	  }
	}
	if( std::string(argv[i]) == "--init" ) {
	  sbd_data.init = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--seed" ) {
	  sbd_data.seed = std::stoull(argv[++i]);
	}
	if( std::string(argv[i]) == "--do_sort_basis" ) {
	  if( std::atoi(argv[++i]) != 0 ) {
	    sbd_data.do_sort_basis = true;
	  }
	}
	if( std::string(argv[i]) == "--do_redist_basis" ) {
	  if( std::atoi(argv[++i]) != 0 ) {
	    sbd_data.do_redist_basis = true;
	  }
	}
      }
      return sbd_data;
    }

    void cout_options(const caop::SBD sbd_data) {
      std::cout.precision(16);
      std::cout << "# t_comm_size: " << sbd_data.t_comm_size << std::endl;
      std::cout << "# b_comm_size: " << sbd_data.b_comm_size << std::endl;
      std::cout << "# method: (" << sbd_data.method << ")";
      if( sbd_data.method == 0 ) {
	std::cout << " Davidson method without storing Hamiltonian data " << std::endl;
      } else if ( sbd_data.method == 1 ) {
	std::cout << " Davidson method and store Hamiltonian data to accelerate Hamiltonian operation" << std::endl;
      } else if ( sbd_data.method == 2 ) {
	std::cout << " Lanczos method without storing Hamiltonian data " << std::endl;
      } else if ( sbd_data.method == 3 ) {
	std::cout << " Lanczos method and store Hamiltonian data to accelerate Hamiltonian operation" << std::endl;
      }
      std::cout << "# max_it: " << sbd_data.max_it << std::endl;
      std::cout << "# block size: " << sbd_data.max_nb << std::endl;
      std::cout << "# number of  initiial vectors: " << sbd_data.max_iv << std::endl;
      std::cout << "# tolerance: " << sbd_data.eps << std::endl;
      std::cout << "# init method: " << sbd_data.init << std::endl;
      std::cout << "# system size: " << sbd_data.system_size << std::endl;
      std::cout << "# bit length: " << sbd_data.bit_length << std::endl;
      std::cout << "# fermion sign: " << sbd_data.sign << std::endl;
      std::cout << "# do basis sort: " << sbd_data.do_sort_basis << std::endl;
      std::cout << "# do redistribution of basis: " << sbd_data.do_redist_basis << std::endl;
      std::cout << "# carryover ratio: " << sbd_data.ratio << std::endl;
    }
    
    template <typename ElemT>
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const GeneralOp<ElemT> & H,
	      const std::vector<std::vector<size_t>> & basis,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<std::vector<size_t>> & co_basis) {
      
      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);
      int t_comm_size = sbd_data.t_comm_size;
      int b_comm_size = sbd_data.b_comm_size;
      int h_comm_size = mpi_size / (t_comm_size*b_comm_size);
      int method = sbd_data.method;
      int max_it = sbd_data.max_it;
      int max_nb = sbd_data.max_nb;
      int max_iv = sbd_data.max_iv;
      int init   = sbd_data.init;
      size_t seed = sbd_data.seed;
      bool seed_was_random = (seed == 0);
      if (seed == 0) {
        if (mpi_rank == 0) {
          seed = static_cast<size_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
        unsigned long long bcast_seed = (unsigned long long)seed;
        MPI_Bcast(&bcast_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
        seed = (size_t)bcast_seed;
      }
      if (mpi_rank == 0) {
        std::cout << "# seed: " << seed
                  << (seed_was_random ? " (random)" : " (user-specified)") << std::endl;
      }
      double eps = sbd_data.eps;
      size_t system_size = sbd_data.system_size;
      size_t bit_length = sbd_data.bit_length;
      bool sign = sbd_data.sign;

      /**
	 Setup helpers
       */
      std::vector<int> slide;
      MPI_Comm h_comm;
      MPI_Comm b_comm;
      MPI_Comm t_comm;
      setup_communicator(comm,h_comm_size,b_comm_size,t_comm_size,
			 h_comm,b_comm,t_comm);
      make_slide(slide,b_comm,t_comm);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      /**
	 Initialize vector
       */
      auto time_start_init = std::chrono::high_resolution_clock::now();
      std::vector<ElemT> W;
      if ( loadname.empty() ) {
	InitVectorCAOP(W,basis,h_comm,b_comm,t_comm,init,seed);
      } else {
	LoadWavefunction(loadname,basis,h_comm,b_comm,t_comm,W);
      }
      auto time_end_init = std::chrono::high_resolution_clock::now();
      auto elapsed_init_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_init-time_start_init).count();
      double elapsed_init = 0.000001 * elapsed_init_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " Elapsed time for init " << elapsed_init << " (sec) " << std::endl;
      }
      
      /**
	 Diagonalization
      */
      if( method == 0 || method == 2 ) {
	auto time_start_davidson = std::chrono::high_resolution_clock::now();
	std::vector<ElemT> hii;
	makeCAOpHamDiagTerms(basis,bit_length,slide,H,hii);
	if( method == 0 ) {
	  Davidson(hii,W,basis,bit_length,slide,H,sign,
		   h_comm,b_comm,t_comm,
		   max_it,max_nb,max_iv,eps,seed);
	} else if ( method == 2 ) {
	  Lanczos(hii,W,basis,bit_length,slide,H,sign,
		  h_comm,b_comm,t_comm,
		  max_it,max_nb,eps);
	}
	auto time_end_davidson = std::chrono::high_resolution_clock::now();
	auto elapsed_davidson_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_davidson-time_start_davidson).count();
	double elapsed_davidson = 0.000001 * elapsed_davidson_count;
	if( mpi_rank == 0 ) {
	  if( method == 0 ) {
	    std::cout << " " << make_timestamp()
		      << " Elapsed time for davidson " << elapsed_davidson << " (sec) " << std::endl;
	  } else if ( method == 2 ) {
	    std::cout << " " << make_timestamp()
		      << " Elapsed time for lanczos " << elapsed_davidson << " (sec) " << std::endl;
	  }
	}

	/**
	   Evaluation of energy
	 */
	auto time_start_mult = std::chrono::high_resolution_clock::now();
	std::vector<ElemT> C(W.size(),0.0);
	mult(hii,W,C,basis,bit_length,slide,H,sign,
	     h_comm,b_comm,t_comm);
	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
        double elapsed_mult = 0.000001 * elapsed_mult_count;
	if ( mpi_rank == 0 ) {
          std::cout << " " << make_timestamp()
		    << " Elapsed time for mult " << elapsed_mult << " (sec) " << std::endl;
        }
	ElemT E = 0.0;
	InnerProduct(W,C,E,b_comm);
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " Energy = " << GetReal(E) << std::endl;
	}
	energy = GetReal(E);
	
      } else {
	std::vector<ElemT> hii;
	std::vector<std::vector<std::vector<size_t>>> ib;
	std::vector<std::vector<std::vector<size_t>>> ik;
	std::vector<std::vector<std::vector<ElemT>>> hij;
	auto time_start_diag = std::chrono::high_resolution_clock::now();
	
	auto time_start_mkham = std::chrono::high_resolution_clock::now();
	makeCAOpHam(basis,bit_length,slide,H,sign,
		    hii,ib,ik,hij,
		    h_comm,b_comm,t_comm);
	auto time_end_mkham = std::chrono::high_resolution_clock::now();
	auto elapsed_mkham_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mkham-time_start_mkham).count();
	double elapsed_mkham = 0.000001 * elapsed_mkham_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " Elapsed time for make Hamiltonian " << elapsed_mkham << " (sec) " << std::endl;
	}
	
	auto time_start_davidson = std::chrono::high_resolution_clock::now();
	if( method == 1 ) {
	  Davidson(hii,ib,ik,hij,W,slide,
		   h_comm,b_comm,t_comm,
		   max_it,max_nb,max_iv,eps,seed);
	} else if ( method == 3 ) {
	  Lanczos(hii,ib,ik,hij,W,slide,
		  h_comm,b_comm,t_comm,
		  max_it,max_nb,eps);
	}
        auto time_end_davidson = std::chrono::high_resolution_clock::now();
        auto elapsed_davidson_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_davidson-time_start_davidson).count();
        double elapsed_davidson = 0.000001 * elapsed_davidson_count;
	if( mpi_rank == 0 ) {
	  if( method == 1 ) {
	    std::cout << " " << make_timestamp()
		      << " Elapsed time for davidson " << elapsed_davidson << " (sec) " << std::endl;
	  } else if ( method == 3 ) {
	    std::cout << " " << make_timestamp()
		      << " Elapsed time for lanczos " << elapsed_davidson << " (sec) " << std::endl;
	  }
	}

	/**
	   Evaluation of Hamiltonian expectation value
	 */
	std::vector<ElemT> C(W.size(),0.0);
	auto time_start_mult = std::chrono::high_resolution_clock::now();
	mult(hii,ib,ik,hij,W,C,slide,
	     h_comm,b_comm,t_comm);
	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
	double elapsed_mult = 0.000001 * elapsed_mult_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " Elapsed time for mult " << elapsed_mult << " (sec) " << std::endl;
	}
	ElemT E = 0.0;
	InnerProduct(W,C,E,b_comm);
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " Energy = " << GetReal(E) << std::endl;
	}
	energy = GetReal(E);
      }

      if( sbd_data.ratio != 0.0 ) {
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start carryover selection" << std::endl;
	}
	size_t n_kept = static_cast<size_t>(sbd_data.ratio * basis.size() * mpi_size_b);
	double truncated_weight = 0.0;
	CarryOverBasis(W,basis,b_comm,n_kept,co_basis,truncated_weight);
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end carryover selection" << std::endl;
	}
      }

      /**
	 Save wavefunctions
       */
      if( !savename.empty() ) {
	SaveWavefunction(savename,basis,h_comm,b_comm,t_comm,W);
      }
			MPI_Comm_free(&h_comm);
			MPI_Comm_free(&t_comm);
			MPI_Comm_free(&b_comm);
		}

    /**
       Main function to perform the diagonalization for selected basis on creation/annihilation operator model
       @param[in] comm: communicator
       @param[in] sbd_data: parameters for setup
       @param[in] hamiltonianfile: filename for fcidump data
       @param[in] basisname: bitstrings for basis
       @param[in] loadname: load filename for wavefunction data. if string is empty, use HF det as a initial state.
       @param[in] savename: save filename for wavefunction data. if string is empty, do not save.
       @param[out] energy: obtained energy after davidson method
       @param[out] co_basis: carryover basis
     */
    template <typename ElemT>
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const std::string & hamiltonianfile,
	      const std::vector<std::string> & basisfiles,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<std::vector<size_t>> & co_basis) {

      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);

      int t_comm_size = sbd_data.t_comm_size;
      int b_comm_size = sbd_data.b_comm_size;
      int h_comm_size = mpi_size / (t_comm_size*b_comm_size);
      size_t bit_length = sbd_data.bit_length;
      size_t system_size = sbd_data.system_size;
      det_vector<size_t>::init_elem_size((system_size + bit_length - 1) / bit_length);
      MPI_Comm h_comm;
      MPI_Comm b_comm;
      MPI_Comm t_comm;
      setup_communicator(comm,
			 h_comm_size,b_comm_size,t_comm_size,
			 h_comm,b_comm,t_comm);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      /**
	 Load Hamiltonian file
       */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start load Hamiltonian" << std::endl;
      }
      GeneralOp<ElemT> hamiltonian;
      bool sign;
      load_GeneralOp_from_file(hamiltonianfile,hamiltonian,sign,
			       h_comm,b_comm,t_comm);
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end load Hamiltonian" << std::endl;
      }
#ifdef SBD_DEBUG
      if( mpi_rank_b == 0 && mpi_rank_t == 0 ) {
	for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	  if( mpi_rank_h == rank_h ) {
	    std::cout << " Hamiltonian at mpi_rank ("
		      << mpi_rank_h << ","
		      << mpi_rank_b << ","
		      << mpi_rank_t << ") ---------" << std::endl;
	    std::cout << hamiltonian;
	  }
	  MPI_Barrier(h_comm);
	}
      }
#endif
      /**
	 Load basis data
       */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start load basis" << std::endl;
      }
      std::vector<std::vector<size_t>> basis;
      if( mpi_rank_h == 0 ) {
	if( mpi_rank_t == 0 ) {
	  load_basis_from_files(basisfiles,basis,
				bit_length,system_size,
				b_comm);
	  sort_bitarray(basis);
	  if( sbd_data.do_sort_basis ) {
	    redistribution(basis,bit_length,system_size,b_comm);
	    reordering(basis,bit_length,system_size,b_comm);
	  } else if ( sbd_data.do_redist_basis ) {
	    redistribution(basis,bit_length,system_size,b_comm);
	  }
	}
	MpiBcast(basis,0,t_comm);
      }
      MpiBcast(basis,0,h_comm);
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end load basis" << std::endl;
      }
      diag(comm,sbd_data,hamiltonian,basis,loadname,savename,
	   energy,co_basis);
		 MPI_Comm_free(&h_comm); // my fix
		 MPI_Comm_free(&t_comm); // my fix
		 MPI_Comm_free(&b_comm); // my fix
		}
  }
}

#endif
