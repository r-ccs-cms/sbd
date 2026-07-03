/**
@file sbd/chemistry/gdb/sbdiag.h
@brief Function used for selected-basis diagonalization with general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_DBDIAG_H
#define SBD_CHEMISTRY_GDB_DBDIAG_H

#include "sbd/framework/timestamp.h"

namespace sbd {
  namespace gdb {
    struct SBD {
      int h_comm_size = 1;
      int b_comm_size = 1;
      int t_comm_size = 1;

      int method = 0;
      int max_it = 1;
      int max_nb = 10;
      double eps = 1.0e-4;
      double max_time = 86400.0;
      int init = 0;
      int do_shuffle = 0;
      int do_rdm = 0;
      int carryover_type = 0;
      double ratio = 0.0;
      double threshold = 0.01;
      double heatbath_cutoff = 1.0e-4;
      double heatbath_truncation = 0.0;
      size_t heatbath_batch_size = 200000000;
      size_t bit_length = 20;
      size_t seed = 1729;
      bool timing_barriers = false;
      bool do_sort_det = false;
      bool do_redist_det = false;
      bool do_redist_alpha_eq = true;
    };

    SBD generate_sbd_data(int argc, char * argv[]) {
      SBD sbd_data;
      for(int i=0; i < argc; i++) {
	if ( std::string(argv[i]) == "--init" ) {
	  sbd_data.init = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--seed" ) {
	  sbd_data.seed = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--b_comm_size" ) {
	  sbd_data.b_comm_size = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--t_comm_size" ) {
	  sbd_data.t_comm_size = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--method" ) {
	  sbd_data.method = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--iteration" ) {
	  sbd_data.max_it = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--block" ) {
	  sbd_data.max_nb = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--tolerance" ) {
	  sbd_data.eps = std::atof(argv[++i]);
	}
	if ( std::string(argv[i]) == "--carryover_type" ) {
	  sbd_data.carryover_type = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--carryover_ratio" ) {
	  sbd_data.ratio = std::atof(argv[++i]);
	}
	if ( std::string(argv[i]) == "--carryover_threshold" ) {
	  sbd_data.threshold = std::atof(argv[++i]);
	}
	if ( std::string(argv[i]) == "--heatbath_cutoff" ) {
	  sbd_data.heatbath_cutoff = std::atof(argv[++i]);
	}
	if ( std::string(argv[i]) == "--heatbath_truncation" ) {
	  sbd_data.heatbath_truncation = std::atof(argv[++i]);
	}
	if ( std::string(argv[i]) == "--heatbath_batch_size" ) {
	  sbd_data.heatbath_batch_size = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--shuffle" ) {
	  sbd_data.do_shuffle = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--rdm" ) {
	  sbd_data.do_rdm = std::atoi(argv[++i]);
	}
	if ( std::string(argv[i]) == "--bit_length" ) {
	  sbd_data.bit_length = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--timing_barriers" ) {
	  sbd_data.timing_barriers = ( std::atoi(argv[++i]) != 0 );
	}
	if( std::string(argv[i]) == "--do_sort_det" ) {
	  if( std::atoi(argv[++i]) != 0 ) {
	    sbd_data.do_sort_det = true;
	  }
	}
	if( std::string(argv[i]) == "--do_redist_det" ) {
	  if( std::atoi(argv[++i]) != 0 ) {
	    sbd_data.do_redist_det = true;
	  }
	}
	if( std::string(argv[i]) == "--do_redist_alpha_eq" ) {
	  sbd_data.do_redist_alpha_eq = ( std::atoi(argv[++i]) != 0 );
	}
      }
      return sbd_data;
    }

    void cout_options(const gdb::SBD sbd_data) {
      std::cout.precision(16);
      std::cout << "# t_comm_size: " << sbd_data.t_comm_size << std::endl;
      std::cout << "# b_comm_size: " << sbd_data.b_comm_size << std::endl;
      std::cout << "# method: (" << sbd_data.method << ")";
      if( sbd_data.method == 0 ) {
	std::cout << " Davidson method without storing Hamiltonian data " << std::endl;
      } else if ( sbd_data.method == 1 ) {
	std::cout << " Davidson method and store Hamiltonian data to accelerate Hamiltonian operation" << std::endl;
      }
      std::cout << "# max_it: " << sbd_data.max_it << std::endl;
      std::cout << "# block size: " << sbd_data.max_nb << std::endl;
      std::cout << "# tolerance: " << sbd_data.eps << std::endl;
      std::cout << "# init method: " << sbd_data.init << std::endl;
      std::cout << "# seed for initialization: " << sbd_data.seed << std::endl;
      std::cout << "# bit length: " << sbd_data.bit_length << std::endl;
      std::cout << "# timing_barriers: " << sbd_data.timing_barriers << std::endl;
      std::cout << "# do basis sort: " << sbd_data.do_sort_det << std::endl;
      std::cout << "# do redistribution of basis: " << sbd_data.do_redist_det << std::endl;
      std::cout << "# do equal-bra_a redistribution: " << sbd_data.do_redist_alpha_eq << std::endl;
      if( sbd_data.do_rdm != 0.0 ) {
	std::cout << "# do rdm: " << sbd_data.do_rdm << std::endl;
      }
      if( sbd_data.carryover_type == 0 ) {
	std::cout << "# carryover type: none" << std::endl;
      } else  if( sbd_data.carryover_type == 1 ) {
	std::cout << "# carryover type: weight truncation" << std::endl;
	std::cout << "# carryover ratio: " << sbd_data.ratio << std::endl;
      } else if ( sbd_data.carryover_type == 2 || sbd_data.carryover_type == 3 ) {
	std::cout << "# carryover type: heatbath expansion" << std::endl;
	std::cout << "# heatbath truncation: " << sbd_data.heatbath_truncation << std::endl;
	std::cout << "# heatbath cutoff: " << sbd_data.heatbath_cutoff << std::endl;
      }
    }

    /**
       Main function to perform the selected basis diagonalization
       @param[in] comm: communicator
       @param[in] sbd_data: parameters for setup
       @param[in] fcidump: sbd::FCIDump data
       @param[in] det: bitstrings for all alpha-spin and beta-spin orbitals
       @param[in] loadname: load filename for wavefunction data.
       @param[in] savename: save filename for wavefunction data.
       @param[out] energy: obtained energy after davidson method
       @param[out] density: diagonal part of 1pRDM for configuration recovery
       @param[out] carryover_det: dominant bitstrings for alpha-spin and beta-spin orbitals
       @param[out] one_p_rdm: one-particle reduced density matrix if sbd_data.do_rdm != 0
       @param[out] two_p_rdm: two-particle reduced density matrix if sbd_data.do_rdm != 0
     */
    template <typename ElemT>
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const sbd::FCIDump & fcidump,
	      const sbd::det_vector<size_t> & det,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<double> & density,
	      sbd::det_vector<size_t> & rdet,
	      std::vector<std::vector<ElemT>> & one_p_rdm,
	      std::vector<std::vector<ElemT>> & two_p_rdm) {
      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);
      int b_comm_size = sbd_data.b_comm_size;
      int t_comm_size = sbd_data.t_comm_size;
      int h_comm_size = mpi_size / (t_comm_size * b_comm_size);
      int L;
      int N;
      int method = sbd_data.method;
#ifdef SBD_THRUST
	  method &= 1;
#endif
      int max_it = sbd_data.max_it;
      int max_nb = sbd_data.max_nb;
      double eps = sbd_data.eps;
      double max_time = sbd_data.max_time;
      int init = sbd_data.init;
      size_t seed = sbd_data.seed;
      int do_shuffle = sbd_data.do_shuffle;
      int do_rdm = sbd_data.do_rdm;
      double ratio = sbd_data.ratio;
      double threshold = sbd_data.threshold;
      int co_type = sbd_data.carryover_type;
      double hb_truncation = sbd_data.heatbath_truncation;
      double hb_cutoff = sbd_data.heatbath_cutoff;
      size_t hb_batch_size = sbd_data.heatbath_batch_size;
      size_t bit_length = sbd_data.bit_length;
      /**
	 Setup system parameters from fcidump
      */
      if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	  if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start integral construction" << std::endl;
      }
      auto time_start_model = std::chrono::high_resolution_clock::now();
      double I0;
      sbd::oneInt<ElemT> I1;
      sbd::twoInt<ElemT> I2;
      sbd::SetupIntegrals(fcidump,L,N,I0,I1,I2);
      if( sbd_data.timing_barriers ) MPI_Barrier(comm);
      auto time_end_model = std::chrono::high_resolution_clock::now();
      auto elapsed_model_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_model-time_start_model).count();
      double elapsed_model = 1.0e-6 * elapsed_model_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end integral construction [Elapsed time "
		  << elapsed_model << " (sec)]" << std::endl;
      }
      /**
	 Setup helpers
      */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start helper construction" << std::endl;
      }
      auto time_start_helper = std::chrono::high_resolution_clock::now();
      DetIndexMap idxmap;
      std::vector<ExcitationLookup> exidx;
      MPI_Comm h_comm;
      MPI_Comm b_comm;
      MPI_Comm t_comm;
      DetBasisCommunicator(comm,h_comm_size,b_comm_size,t_comm_size,
			   h_comm,b_comm,t_comm);
      MakeHelpers(det,bit_length,static_cast<size_t>(L),
		  idxmap,exidx,h_comm,b_comm,t_comm);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      if( sbd_data.timing_barriers ) MPI_Barrier(comm);
      auto time_end_helper = std::chrono::high_resolution_clock::now();
      auto elapsed_helper_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_helper-time_start_helper).count();
      double elapsed_helper = 1.0e-6 * elapsed_helper_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end helper construction [Elapsed time "
		  << elapsed_helper << " (sec)]" << std::endl;
      }
      /**
	 Initialize/Load wave function
      */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start initialize/load wave function" << std::endl;
      }
      auto time_start_init = std::chrono::high_resolution_clock::now();
      std::vector<ElemT> w;
      if( loadname.empty() ) {
	sbd::gdb::BasisInitVector(w,det,h_comm,b_comm,t_comm,init,seed);
      } else {
	sbd::LoadWavefunction(loadname,det,h_comm,b_comm,t_comm,w);
      }
      if( sbd_data.timing_barriers ) MPI_Barrier(comm);
      auto time_end_init = std::chrono::high_resolution_clock::now();
      auto elapsed_init_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_init-time_start_init).count();
      double elapsed_init = 1.0e-6 * elapsed_init_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end initialize/load wave function [Elapsed time "
		  << elapsed_init << " (sec)]" << std::endl;
      }
      /**
	 Diagonalization
      */
#ifdef SBD_THRUST
	// multiplyer class for TPB on Thrust
	MultGDBThrust<double> device_mult;
#endif
      if( method == 0 ) {

	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start diagonalization" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: start make diagonal term" << std::endl;
	}
	auto time_start_mkham = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
	thrust::device_vector<ElemT> hii;
	device_mult.Init(bit_length,static_cast<size_t>(L),det,
					idxmap,exidx,I0,I1,I2,
		 			h_comm,b_comm,t_comm);
	device_mult.makeQChamDiagTerms(hii);
#else
	std::vector<ElemT> hii;
	makeQChamDiagTerms(det,bit_length,L,
			   idxmap,exidx,I0,I1,I2,hii,
			   h_comm,b_comm,t_comm);
#endif
    if( sbd_data.timing_barriers ) MPI_Barrier(comm);
    auto time_end_mkham = std::chrono::high_resolution_clock::now();
	auto elapsed_mkham_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mkham-time_start_mkham).count();
	double elapsed_mkham = 1.0e-6 * elapsed_mkham_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end make diagonal term [Elapsed time "
		    << elapsed_mkham << " (sec)]" << std::endl;
	}
	/**
	   Davidson
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start davidson" << std::endl;
	}
	auto time_start_david = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
	sbd::Davidson(hii, w, device_mult,
			max_it,max_nb,eps,max_time);
#else
	Davidson(hii,w,det,bit_length,static_cast<size_t>(L),
		 idxmap,exidx,I0,I1,I2,
		 h_comm,b_comm,t_comm,
		 max_it,max_nb,eps);
#endif
    if( sbd_data.timing_barriers ) MPI_Barrier(comm);
    auto time_end_david = std::chrono::high_resolution_clock::now();
	auto elapsed_david_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_david-time_start_david).count();
	auto elapsed_diag_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_david-time_start_mkham).count();
	double elapsed_david = 1.0e-6 * elapsed_david_count;
	double elapsed_diag = 1.0e-6 * elapsed_diag_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end davidson [Elapsed time "
		    << elapsed_david << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: end diagonalization [Elapsed time "
		    << elapsed_diag << " (sec)]" << std::endl;
	}
	/**
	   Evaluation of Hamiltonian expectation value
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start Hamiltonian expectation value" << std::endl;
	}
	auto time_start_mult = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
    // copyin W
    thrust::device_vector<double> w_dev(w.size());
    thrust::copy_n(w.begin(), w.size(), w_dev.begin());

    thrust::device_vector<double> v(w.size(), 0.0);

	device_mult.run(hii, w_dev, v);

	ElemT E;
	InnerProduct(w_dev,v,E,b_comm);
#else
	std::vector<ElemT> v(w.size(),ElemT(0.0));
	mult(hii,w,v,bit_length,static_cast<size_t>(L),det,
	     idxmap,exidx,I0,I1,I2,
	     h_comm,b_comm,t_comm);
	ElemT E;
	InnerProduct(w,v,E,b_comm);
	energy = GetReal(E);
#endif
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
	double elapsed_mult = 1.0e-6 * elapsed_mult_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end Hamiltonian expectation value [Elapsed time "
		    << elapsed_mult << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: Energy = " << GetReal(E) << std::endl;
	}
      } else if ( method == 1 ) {
	/**
	   Make Hamiltonian
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start diagonalization" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: start make Hamiltonian" << std::endl;
	}
	auto time_start_mkham = std::chrono::high_resolution_clock::now();
	std::vector<ElemT> hii;
	std::vector<std::vector<size_t*>> ih;
	std::vector<std::vector<size_t*>> jh;
	std::vector<std::vector<ElemT*>> hij;
	std::vector<std::vector<size_t>> len;
	std::vector<int> slide;
	std::vector<std::vector<size_t>> storage_int;
	std::vector<std::vector<ElemT>> storage_elem;
	makeQCham(det,bit_length,static_cast<size_t>(L),
		  idxmap,exidx,I0,I1,I2,
		  hii,ih,jh,hij,len,slide,
		  storage_int,storage_elem,
		  h_comm,b_comm,t_comm);
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_mkham = std::chrono::high_resolution_clock::now();
	auto elapsed_mkham_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mkham-time_start_mkham).count();
	double elapsed_mkham = 1.0e-6 * elapsed_mkham_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end make Hamiltonian [Elapsed time "
		    << elapsed_mkham << " (sec)]" << std::endl;
	}
	/**
	   Davidson
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start davidson" << std::endl;
	}
	auto time_start_david = std::chrono::high_resolution_clock::now();
	sbd::gdb::Davidson(hii,ih,jh,hij,len,slide,w,
			   h_comm,b_comm,t_comm,max_it,max_nb,eps);
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_david = std::chrono::high_resolution_clock::now();
	auto elapsed_david_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_david-time_start_david).count();
	auto elapsed_diag_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_david-time_start_mkham).count();
	double elapsed_david = 1.0e-6 * elapsed_david_count;
	double elapsed_diag = 1.0e-6 * elapsed_diag_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end davidson [Elapsed time "
		    << elapsed_david << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: end diagonalization [Elapsed time "
		    << elapsed_diag << " (sec)]" << std::endl;
	}
	/**
	   Evaluation of Hamiltonian expectation value
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start Hamiltonian expectation value" << std::endl;
	}
	auto time_start_mult = std::chrono::high_resolution_clock::now();
	std::vector<ElemT> v(w.size(),ElemT(0.0));
	sbd::gdb::mult(hii,ih,jh,hij,len,slide,
		       w,v,h_comm,b_comm,t_comm);
	ElemT E = 0.0;
	InnerProduct(w,v,E,b_comm);
	std::cout.precision(16);
	energy = GetReal(E);
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
	double elapsed_mult = 1.0e-6 * elapsed_mult_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end Hamiltonian expectation value [Elapsed time "
		    << elapsed_mult << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: Energy = " << GetReal(E) << std::endl;
	}
      }

      /**
	 Evaluation of expectation values
      */
      if( do_rdm == 0 ) {
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start occupation density calculation"
		    << std::endl;
	}
	auto time_start_occd = std::chrono::high_resolution_clock::now();
	if( mpi_rank_t == 0 ) {
	  size_t oidx_start = 0;
	  size_t oidx_end   = static_cast<size_t>(L);
	  get_mpi_range(mpi_size_h,mpi_rank_h,oidx_start,oidx_end);
	  size_t oidx_size = oidx_end - oidx_start;
	  std::vector<int> oidx(oidx_size);
	  std::iota(oidx.begin(),oidx.end(),static_cast<int>(oidx_start));
	  std::vector<double> res_density;
	  OccupationDensity(oidx,w,det,bit_length,b_comm,res_density);
	  density.resize(2*L,0.0);
	  for(size_t io=0; io < oidx.size(); io++) {
	    density[2*oidx[io]+0] = res_density[2*io+0];
	    density[2*oidx[io]+1] = res_density[2*io+1];
	  }
	  MpiAllreduce(density,MPI_SUM,h_comm);
	}
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_occd = std::chrono::high_resolution_clock::now();
	auto elapsed_occd_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_occd-time_start_occd).count();
	double elapsed_occd = 1.0e-6 * elapsed_occd_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end occupation density calculation [Elapsed time "
		    << elapsed_occd << " (sec)]" << std::endl;
	}
      } else {
	/**
	   do_rdm != 0: calculate all one- and two-particle rdm
	*/
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start rdm calculation" << std::endl;
	}
	auto time_start_rdm = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
	device_mult.correlation(w,one_p_rdm,two_p_rdm);
#else
	Correlation(w,det,bit_length,static_cast<size_t>(L),
		    idxmap,exidx,h_comm,b_comm,t_comm,
		    one_p_rdm,two_p_rdm);
#endif
	density.resize(2*L);
	for(size_t io=0; io < L; io++) {
	  density[2*io+0] = GetReal(one_p_rdm[0][io+L*io]);
	  density[2*io+1] = GetReal(one_p_rdm[1][io+L*io]);
	}
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_rdm = std::chrono::high_resolution_clock::now();
	auto elapsed_rdm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_rdm-time_start_rdm).count();
	double elapsed_rdm = 1.0e-6 * elapsed_rdm_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end rdm calculation [Elapsed time "
		    << elapsed_rdm << " (sec)]" << std::endl;
	}
      }
      /**
	 Carryover selection
      */
      if( co_type == 1 ) {
	if( ratio != 0.0 ) {
	  if( mpi_rank == 0 ) {
	    std::cout << " " << make_timestamp()
		      << " sbd: start carryover selection" << std::endl;
	  }
	  auto time_start_co = std::chrono::high_resolution_clock::now();
	  size_t n_kept = static_cast<size_t>(ratio * det.size()*mpi_size_b);
	  double truncated_weight = 0.0;
	  CarryOverDet(w,det,b_comm,n_kept,rdet,truncated_weight);
	  auto time_end_co = std::chrono::high_resolution_clock::now();
	  auto elapsed_co_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_co-time_start_co).count();
	  double elapsed_co = 1.0e-6 * elapsed_co_count;
	  if( mpi_rank == 0 ) {
	    std::cout << " " << make_timestamp()
		      << " sbd: end carryover selection [Elapsed time "
		      << elapsed_co << " (sec)]" << std::endl;
	  }
	}
      } else if ( co_type == 2 || co_type == 3 ) {
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start weight truncation" << std::endl;
	}
	auto time_start_wt = std::chrono::high_resolution_clock::now();
	std::vector<ElemT> cw;
	sbd::det_vector<size_t> cdet;
	WeightTruncation(w,det,hb_truncation,cw,cdet);
	auto time_end_wt = std::chrono::high_resolution_clock::now();
	auto elapsed_wt_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_wt-time_start_wt).count();
	double elapsed_wt = 1.0e-6 * elapsed_wt_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end weight truncation [Elapsed time "
		    << elapsed_wt << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: start redistribution of wavefunction data" << std::endl;
	}
	auto time_start_rd = std::chrono::high_resolution_clock::now();
	redistribution_bitarray(cdet,cw,b_comm);
	auto time_end_rd = std::chrono::high_resolution_clock::now();
	auto elapsed_rd_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_rd-time_start_rd).count();
	double elapsed_rd = 1.0e-6 * elapsed_rd_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end redistribution of wavefunction data [Elapsed time "
		    << elapsed_rd << " (sec)]" << std::endl;
	  std::cout << " " << make_timestamp()
		    << " sbd: start heatbath expansion" << std::endl;
	}
	auto time_start_hb = std::chrono::high_resolution_clock::now();
	int hb_type = (co_type == 2) ? 0 : 1;
	HeatbathExpansion(cdet,cw,bit_length,static_cast<size_t>(L),I0,I1,I2,
			  hb_type,hb_cutoff,hb_batch_size,rdet,b_comm,comm);
	auto time_end_hb = std::chrono::high_resolution_clock::now();
	auto elapsed_hb_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_hb-time_start_hb).count();
	double elapsed_hb = 1.0e-6 * elapsed_hb_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end heatbath expansion [Elapsed time "
		    << elapsed_hb << " (sec)]" << std::endl;
	}
      }

      /**
	 Save wavefunction
      */
      if( !savename.empty() ) {
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: start save wavefunction" << std::endl;
	}
	auto time_start_save = std::chrono::high_resolution_clock::now();
	SaveWavefunction(savename,det,h_comm,b_comm,t_comm,w);
	if( sbd_data.timing_barriers ) MPI_Barrier(comm);
	auto time_end_save = std::chrono::high_resolution_clock::now();
	auto elapsed_save_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_save-time_start_save).count();
	double elapsed_save = 1.0e-6 * elapsed_save_count;
	if( mpi_rank == 0 ) {
	  std::cout << " " << make_timestamp()
		    << " sbd: end save wavefunction [elapsed time "
		    << elapsed_save << " (sec)]" << std::endl;
	}
      }
    } // end void diag function

    /**
       Main function to perform the selected basis diagonalization
       @param[in] comm: communicator
       @param[in] sbd_data: parameters for setup
       @param[in] fcidumpfile: filename for fcidump data
       @param[in] detfiles: determinant files
       @param[in] loadname: load filename for wavefunction data.
       @param[in] savename: save filename for wavefunction data.
       @param[out] energy: obtained energy after davidson
       @param[out] density: diagonal part of 1p-rdm
       @param[out] rdet: carryover determinants
       @param[out] one_p_rdm: 1p-rdm if sbd_data.do_rdm != 0
       @param[out] two_p_rdm: 2p-rdm if sbd_data.do_rdm != 0
    */
    template <typename ElemT>
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const std::string & fcidumpfile,
	      const std::vector<std::string> & detfiles,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<double> & density,
	      sbd::det_vector<size_t> & rdet,
	      std::vector<std::vector<ElemT>> & one_p_rdm,
	      std::vector<std::vector<ElemT>> & two_p_rdm) {
      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);

      /**
	 Load fcidump data
      */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start load fcidump data" << std::endl;
      }
      auto time_start_fcidump = std::chrono::high_resolution_clock::now();
      size_t L;
      size_t N;
      sbd::FCIDump fcidump;
      if( mpi_rank == 0 ) {
	fcidump = sbd::LoadFCIDump(fcidumpfile);
      }
      sbd::MpiBcast(fcidump,0,comm);
      for(const auto & [key,value] : fcidump.header) {
	if( key == std::string("NORB") ) {
	  L = std::atoi(value.c_str());
	}
	if( key == std::string("NELEC") ) {
	  N = std::atoi(value.c_str());
	}
      }
      auto time_end_fcidump = std::chrono::high_resolution_clock::now();
      auto elapsed_fcidump_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_fcidump-time_start_fcidump).count();
      double elapsed_fcidump = 1.0e-6 * elapsed_fcidump_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end load fcidump data [Elapsed time "
		  << elapsed_fcidump << " (sec)]" << std::endl;
      }
      /**
	 Load dets files
      */
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: start load det data" << std::endl;
      }
      auto time_start_ldet = std::chrono::high_resolution_clock::now();
      int t_comm_size = sbd_data.t_comm_size;
      int b_comm_size = sbd_data.b_comm_size;
      int h_comm_size = mpi_size / (t_comm_size*b_comm_size);
      size_t bit_length = sbd_data.bit_length;
      det_vector<size_t>::init_elem_size((2*L + bit_length - 1) / bit_length);
      det_vector<size_t, det_kind::half>::init_elem_size((L + bit_length - 1) / bit_length);
      MPI_Comm h_comm;
      MPI_Comm b_comm;
      MPI_Comm t_comm;
      DetBasisCommunicator(comm,h_comm_size,b_comm_size,t_comm_size,
			   h_comm,b_comm,t_comm);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      det_vector<size_t> det;
      if( mpi_rank_h == 0 ) {
	if( mpi_rank_t == 0 ) {
	  load_basis_from_files(detfiles,det,bit_length,2*L,b_comm);
	  if( sbd_data.do_sort_det ) {
	    redistribution(det,bit_length,2*L,b_comm);
	    reordering(det,bit_length,2*L,b_comm);
	  } else if ( sbd_data.do_redist_det ) {
	    redistribution(det,bit_length,2*L,b_comm);
	  } else if ( sbd_data.do_redist_alpha_eq ) {
	    redistribution_equal_bra_a(det,bit_length,2*L,b_comm);
	  }
	}
	MpiBcast(det,0,t_comm);
      }
      MpiBcast(det,0,h_comm);
      auto time_end_ldet = std::chrono::high_resolution_clock::now();
      auto elapsed_ldet_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_ldet-time_start_ldet).count();
      double elapsed_ldet = 1.0e-6 * elapsed_ldet_count;
      if( mpi_rank == 0 ) {
	std::cout << " " << make_timestamp()
		  << " sbd: end load det data [Elapsed time "
		  << elapsed_ldet << " (sec)]" << std::endl;
      }
      diag(comm,sbd_data,fcidump,det,loadname,savename,
	   energy,density,rdet,one_p_rdm,two_p_rdm);
    }
  } // end namespace gdb
} // end namespace sbd

#endif
