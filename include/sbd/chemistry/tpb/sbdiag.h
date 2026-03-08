/**
@file sbd/chemistry/tpb/sbdiag.h
@brief Function used for sample-based diagonalization
*/
#ifndef SBD_CHEMISTRY_TPB_SBDIAG_H
#define SBD_CHEMISTRY_TPB_SBDIAG_H

#ifdef USE_OMP_OFFLOAD
#include "../basic/omp_offload.h"
#endif

namespace sbd {

  namespace tpb {

    struct SBD {
      int task_comm_size = 1;
      int adet_comm_size = 1;
      int bdet_comm_size = 1;
      int h_comm_size = 1;

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

      size_t bit_length = 20;

      std::string dump_matrix_form_wf;

#ifdef SBD_THRUST
	  bool use_precalculated_dets = true;
	  int max_memory_gb_for_determinants = -1;
#endif
	};

    SBD generate_sbd_data(int argc, char *argv[]) {
      SBD sbd_data;
      for(int i=0; i < argc; i++) {
	if( std::string(argv[i]) == "--adet_comm_size" ) {
	  sbd_data.adet_comm_size = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--bdet_comm_size" ) {
	  sbd_data.bdet_comm_size = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--task_comm_size" ) {
	  sbd_data.task_comm_size = std::atoi(argv[++i]);
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
	if( std::string(argv[i]) == "--tolerance" ) {
	  sbd_data.eps = std::atof(argv[++i]);
	}
	if( std::string(argv[i]) == "--carryover_type" ) {
	  sbd_data.carryover_type = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--carryover_ratio" ) {
	  sbd_data.ratio = std::atof(argv[++i]);
	}
	if( std::string(argv[i]) == "--carryover_threshold" ) {
	  sbd_data.threshold = std::atof(argv[++i]);
	}
	if( std::string(argv[i]) == "--max_time" ) {
		sbd_data.max_time = std::atof(argv[++i]);
	}
	if( std::string(argv[i]) == "--shuffle" ) {
	  sbd_data.do_shuffle = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--rdm" ) {
	  sbd_data.do_rdm = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--bit_length" ) {
	  sbd_data.bit_length = std::atoi(argv[++i]);
	}
	if( std::string(argv[i]) == "--dump_matrix_form_wf" ) {
	  sbd_data.dump_matrix_form_wf = std::string(argv[++i]);
	}
#ifdef SBD_THRUST
	if( std::string(argv[i]) == "--use_precalculated_dets" ) {
	  sbd_data.use_precalculated_dets = std::atoi(argv[i+1]) == 1;
	  i++;
	}
	if( std::string(argv[i]) == "--max_memory_gb_for_determinants" ) {
	  sbd_data.max_memory_gb_for_determinants = std::atoi(argv[i+1]);
	  i++;
	}
#endif
      }
      return sbd_data;
    }

    /**
       Main function to perform the diagonalization for selected basis
       @param[in] comm: communicator
       @param[in] sbd_data: parameters for setup
       @param[in] fcidump: sbd::FCIDump data
       @param[in] adet: bitstrings for alpha-spin orbitals
       @param[in] bdet: bitstrings for beta-spin orbitals
       @param[in] loadname: load filename for wavefunction data. if string is empty, use HF det as a initial state.
       @param[in] savename: save filename for wavefunction data. if string is empty, do not save.
       @param[out] energy: obtained energy after davidson method
       @param[out] density: diagonal part of 1pRDM for configuration recovery
       @param[out] carryover_bitstrings: dominant bitstrings for alpha-spin orbitals
       @param[out] one_p_rdm: one-particle reduced density matrix if sbd_data.do_rdm != 0
       @param[out] two_p_rdm: two-particle reduced density matrix if sbd_data.do_rdm != 0
     */
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const sbd::FCIDump & fcidump,
	      const std::vector<std::vector<size_t>> & adet,
	      const std::vector<std::vector<size_t>> & bdet,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<double> & density,
	      std::vector<std::vector<size_t>> & co_adet,
	      std::vector<std::vector<size_t>> & co_bdet,
	      std::vector<std::vector<double>> & one_p_rdm,
	      std::vector<std::vector<double>> & two_p_rdm) {

      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);
      int task_comm_size = sbd_data.task_comm_size;
      int adet_comm_size = sbd_data.adet_comm_size;
      int bdet_comm_size = sbd_data.bdet_comm_size;
      int base_comm_size = adet_comm_size*bdet_comm_size;
      int h_comm_size = mpi_size / (task_comm_size * base_comm_size);

      int L;
      int N;

      int method = sbd_data.method;
#ifdef SBD_THRUST
	  method = method & 1;
#endif
	  int max_it = sbd_data.max_it;
      int max_nb = sbd_data.max_nb;
      double eps = sbd_data.eps;
      double max_time = sbd_data.max_time;
      int init = sbd_data.init;
      int do_shuffle = sbd_data.do_shuffle;
      int do_rdm = sbd_data.do_rdm;
      double ratio = sbd_data.ratio;
      double threshold = sbd_data.threshold;

      size_t bit_length = sbd_data.bit_length;

      /**
	 Setup system parameters from fcidump
       */
      double I0;
      sbd::oneInt<double> I1;
      sbd::twoInt<double> I2;
      sbd::SetupIntegrals(fcidump,L,N,I0,I1,I2);

      int norbs = L;

#ifdef USE_OMP_OFFLOAD
      // Flatten I1/I2 integrals once and keep on GPU for all tasks
      I1_size = (2 * norbs) * (2 * norbs);
      I2_size = (norbs * (norbs + 1) / 2) * ((norbs * (norbs + 1) / 2) + 1) / 2;
      I2_Direct_size = norbs * norbs;
      I2_Exchange_size = norbs * norbs;
      std::vector<double> I1_flat(I1_size);
      std::vector<double> I2_flat(I2_size);
      std::vector<double> I2_Direct_flat(I2_Direct_size);
      std::vector<double> I2_Exchange_flat(I2_Exchange_size);

      for (size_t i = 0; i < 2 * norbs; i++) {
        for (size_t j = 0; j < 2 * norbs; j++) {
          I1_flat[i * (2 * norbs) + j] = I1.Value(i, j);
        }
      }
      for (size_t ij = 0; ij < I2_size; ij++) {
        I2_flat[ij] = I2.store[ij];
      }
      for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
          I2_Direct_flat[i + norbs * j] = I2.DirectValue(i, j);
          I2_Exchange_flat[i + norbs * j] = I2.ExchangeValue(i, j);
        }
      }

      I1_ptr = I1_flat.data();
      I2_ptr = I2_flat.data();
      I2_Direct_ptr = I2_Direct_flat.data();
      I2_Exchange_ptr = I2_Exchange_flat.data();
#pragma omp target enter data map(to : I1_ptr[0 : I1_size],                    \
                                       I2_ptr[0 : I2_size],                     \
                                       I2_Direct_ptr[0 : I2_Direct_size],       \
                                       I2_Exchange_ptr[0 : I2_Exchange_size])
#endif

      /**
	 Setup helpers
       */
      std::vector<sbd::TaskHelpers> helper;
      MPI_Comm h_comm;
      MPI_Comm b_comm;
      MPI_Comm t_comm;
      sbd::TaskCommunicator(comm,
			    h_comm_size,adet_comm_size,bdet_comm_size,task_comm_size,
			    h_comm,b_comm,t_comm);

      auto time_start_help = std::chrono::high_resolution_clock::now();
      sbd::MakeHelpers(adet,bdet,bit_length,L,helper,
		       h_comm,b_comm,t_comm,
		       adet_comm_size,bdet_comm_size);
      sbd::RemakeHelpers(adet,bdet,bit_length,L,helper,
			 h_comm,b_comm,t_comm,
			 adet_comm_size,bdet_comm_size);
      auto time_end_help = std::chrono::high_resolution_clock::now();
      auto elapsed_help_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_help-time_start_help).count();
      double elapsed_help = 0.000001 * elapsed_help_count;
      if( mpi_rank == 0 ) {
	std::cout << " Elapsed time for helper construction " << elapsed_help << " (sec) " << std::endl;
      }

      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);

      /**
	 Initialize/Load wave function
      */
      auto time_start_init = std::chrono::high_resolution_clock::now();
      std::vector<double> W;
      if( loadname == std::string("") ) {
	sbd::BasisInitVector(W,adet,bdet,adet_comm_size,bdet_comm_size,h_comm,b_comm,t_comm,init);
      } else {
	sbd::LoadWavefunction(loadname,adet,bdet,
			      adet_comm_size,bdet_comm_size,
			      h_comm,b_comm,t_comm,W);
      }
      auto time_end_init = std::chrono::high_resolution_clock::now();
      auto elapsed_init_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_init-time_start_init).count();
      double elapsed_init = 1.0e-6 * elapsed_init_count;
      if( mpi_rank == 0 ) {
	std::cout << " Elapsed time for init " << elapsed_init << " (sec) " << std::endl;
      }

#ifdef SBD_THRUST
	// multiplyer class for TPB on Thrust
	MultTPBThrust<double> device_mult;
#endif

      /**
	 Diagonalization
      */
      if( method == 0 || method == 2 ) {

	/**
	   Default method 0: Calculation without storing hamiltonian elements
	*/

	auto time_start_diag = std::chrono::high_resolution_clock::now();
	auto time_start_davidson = std::chrono::high_resolution_clock::now();
	auto time_start_qcham = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
	auto time_start_mult_init = std::chrono::high_resolution_clock::now();
	device_mult.Init(adet, bdet, bit_length, static_cast<size_t>(L),
					adet_comm_size,bdet_comm_size, helper, I0, I1, I2,
					h_comm,b_comm,t_comm,
	                sbd_data.use_precalculated_dets, sbd_data.max_memory_gb_for_determinants);
	auto time_end_mult_init = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_init_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult_init-time_start_mult_init).count();
	double elapsed_mult_init = 1.0e-6 * elapsed_mult_init_count;
	if( mpi_rank == 0 ) {
		std::cout << " Elapsed time for mult.Init() " << elapsed_mult_init << " (sec) " << std::endl;
	}

	thrust::device_vector<double> hii;
	device_mult.makeQChamDiagTerms(hii);
#else
	std::vector<double> hii;
	sbd::makeQChamDiagTerms(adet,bdet,bit_length,L,
				helper,I0,I1,I2,hii,
				h_comm,b_comm,t_comm);
#endif
	auto time_end_qcham = std::chrono::high_resolution_clock::now();
	auto elapsed_qcham_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_qcham-time_start_qcham).count();
	double elapsed_qcham = 1.0e-6 * elapsed_qcham_count;
	if( mpi_rank == 0 ) {
		std::cout << " Elapsed time for makeQChamDiagTerms " << elapsed_qcham << " (sec) " << std::endl;
	}

#ifdef SBD_THRUST
	if( method == 0 ) {
		sbd::Davidson(hii, W, device_mult,
				max_it,max_nb,eps,max_time);
	} else {
		sbd::Lanczos(hii, W, device_mult,
				max_it,max_nb,eps);
	}
#else
	if( method == 0 ) {
	  sbd::Davidson(hii,W,
			adet,bdet,bit_length,static_cast<size_t>(L),
			adet_comm_size,bdet_comm_size,helper,
			I0,I1,I2,
			h_comm,b_comm,t_comm,
			max_it,max_nb,eps,max_time);
	} else if ( method == 2 ) {
	  sbd::Lanczos(hii,W,
		       adet,bdet,bit_length,static_cast<size_t>(L),
		       adet_comm_size,bdet_comm_size,helper,
		       I0,I1,I2,
		       h_comm,b_comm,t_comm,
		       max_it,max_nb,eps);
	}
#endif
	auto time_end_davidson = std::chrono::high_resolution_clock::now();
	auto elapsed_davidson_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_davidson-time_start_davidson).count();
	double elapsed_davidson = 0.000001 * elapsed_davidson_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for davidson " << elapsed_davidson << " (sec) " << std::endl;
	}

	auto time_end_diag = std::chrono::high_resolution_clock::now();
	auto elapsed_diag_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_diag-time_start_diag).count();
	double elapsed_diag = 0.000001 * elapsed_diag_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for diagonalization " << elapsed_diag << " (sec) " << std::endl;
	}

	/**
	   Evaluation of Hamiltonian expectation value
	*/

	std::vector<double> C(W.size(),0.0);

	auto time_start_mult = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
    // copyin W
    thrust::device_vector<double> W_dev(W.size());
    thrust::copy_n(W.begin(), W.size(), W_dev.begin());

    thrust::device_vector<double> C_dev(C.size(), 0.0);

	device_mult.run(hii, W_dev, C_dev);

	thrust::copy_n(C_dev.begin(), C_dev.size(), C.begin());
#else
	sbd::mult(hii,W,C,adet,bdet,bit_length,static_cast<size_t>(L),
		  adet_comm_size,bdet_comm_size,helper,
		  I0,I1,I2,h_comm,b_comm,t_comm);
#endif

	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
	double elapsed_mult = 0.000001 * elapsed_mult_count;
	if ( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for mult " << elapsed_mult << " (sec) " << std::endl;
	}

	double E = 0.0;
	sbd::InnerProduct(W,C,E,b_comm);
	double EE = 0.0;
	std::vector<double> Cd(C);
	sbd::InnerProduct(Cd,C,EE,b_comm);
	std::cout.precision(16);
	if( mpi_rank == 0 ) {
	  std::cout << " Energy = " << E << std::endl;
	}
	energy = E;
      } else if ( method == 1 || method == 3 ) {

	/**
	   Method 1: Calculation with storing hamiltonian elements
	*/
	auto time_start_diag = std::chrono::high_resolution_clock::now();
	auto time_start_mkham = std::chrono::high_resolution_clock::now();
	std::vector<double> hii;

#ifdef SBD_THRUST
	// initialize hii
	sbd::makeQChamDiagTerms(adet, bdet, bit_length, L,
				helper, I0, I1, I2, hii,
				h_comm, b_comm, t_comm);
	device_data.Init(adet, bdet, bit_length, static_cast<size_t>(L), helper, I0, I1, I2,
 		             sbd_data.use_precalculated_dets, sbd_data.max_memory_gb_for_determinants);
#else
	std::vector<std::vector<size_t*>> ih;
	std::vector<std::vector<size_t*>> jh;
	std::vector<std::vector<double*>> hij;
	std::vector<std::vector<size_t>> len;
	std::vector<size_t> tasktype;
	std::vector<size_t> adetshift;
	std::vector<size_t> bdetshift;
	std::vector<size_t> sharedSizeT;
	std::vector<double> sharedElemT;

	sbd::makeQCham(adet,bdet,bit_length,L,helper,I0,I1,I2,
		       hii,ih,jh,hij,len,tasktype,adetshift,bdetshift,
		       sharedSizeT,sharedElemT,
		       h_comm,b_comm,t_comm);
    auto time_end_mkham = std::chrono::high_resolution_clock::now();
	auto elapsed_mkham_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mkham-time_start_mkham).count();
	double elapsed_mkham = 0.000001 * elapsed_mkham_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for make Hamiltonian " << elapsed_mkham << " (sec) " << std::endl;
	}

	auto time_start_davidson = std::chrono::high_resolution_clock::now();
	sbd::BasisInitVector(W,adet,bdet,adet_comm_size,bdet_comm_size,h_comm,b_comm,t_comm,init);
	if( method == 1 ) {
	  sbd::Davidson(hii,ih,jh,hij,len,tasktype,
			adetshift,bdetshift,adet_comm_size,bdet_comm_size,
			W,
			h_comm,b_comm,t_comm,
			max_it,max_nb,bit_length,eps);
	} else if ( method == 3 ) {
	  sbd::Lanczos(hii,ih,jh,hij,len,tasktype,
		       adetshift,bdetshift,adet_comm_size,bdet_comm_size,
		       W,
		       h_comm,b_comm,t_comm,
		       max_it,max_nb,bit_length,eps);
	}
	auto time_end_davidson = std::chrono::high_resolution_clock::now();
	auto elapsed_davidson_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_davidson-time_start_davidson).count();
	double elapsed_davidson = 0.000001 * elapsed_davidson_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for davidson " << elapsed_davidson << " (sec) " << std::endl;
	}

	auto time_end_diag = std::chrono::high_resolution_clock::now();
	auto elapsed_diag_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_diag-time_start_diag).count();
	double elapsed_diag = 0.000001 * elapsed_diag_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for diagonalization " << elapsed_diag << " (sec) " << std::endl;
	}

	/**
	   Evaluation of Hamiltonian expectation value
	*/

	std::vector<double> C(W.size(),0.0);

	auto time_start_mult = std::chrono::high_resolution_clock::now();
#endif
#ifdef SBD_THRUST
	sbd::mult(hii, W, C, device_data,
		  adet_comm_size, bdet_comm_size,
		  h_comm, b_comm, t_comm);
#else
	sbd::mult(hii,ih,jh,hij,len,
		  tasktype,adetshift,bdetshift,
		  adet_comm_size,bdet_comm_size,
		  W,C,bit_length,h_comm,b_comm,t_comm);
	auto time_end_mult = std::chrono::high_resolution_clock::now();
	auto elapsed_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_mult-time_start_mult).count();
	double elapsed_mult = 0.000001 * elapsed_mult_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for mult " << elapsed_mult << " (sec) " << std::endl;
	}

	double E = 0.0;
	sbd::InnerProduct(W,C,E,b_comm);
	std::cout.precision(16);
	if( mpi_rank == 0 ) {
	  std::cout << " Energy = " << E << std::endl;
	}
	energy = E;

      }
#endif
#ifdef USE_OMP_OFFLOAD
      // Clean up GPU memory for integrals
#pragma omp target exit data map(delete : I1_ptr[0 : I1_size],                    \
                                          I2_ptr[0 : I2_size],                     \
                                          I2_Direct_ptr[0 : I2_Direct_size],      \
                                          I2_Exchange_ptr[0 : I2_Exchange_size])
#endif

      /**
	 Evaluation of expectation values
      */
      if( do_rdm == 0 ) {

	/**
	   do_rdm == 0: calculation only the diagonal part of 1p-RDM
	 */
	auto time_start_meas = std::chrono::high_resolution_clock::now();

	int p_size = mpi_size_t * mpi_size_h;
	int p_rank = mpi_rank_h * mpi_size_t + mpi_rank_t;
	size_t o_start = 0;
	size_t o_end   = L;
	sbd::get_mpi_range(p_size,p_rank,o_start,o_end);
	size_t o_size = o_end - o_start;
	std::vector<int> oIdx(o_size);
	std::iota(oIdx.begin(),oIdx.end(),o_start);
	std::vector<double> res_density;
	sbd::OccupationDensity(oIdx,W,adet,bdet,bit_length,
			       adet_comm_size,bdet_comm_size,
			       b_comm,res_density);
	std::vector<double> density_rank(2*L,0.0);
	std::vector<double> density_group(2*L,0.0);
	density.resize(2*L,0.0);
	for(size_t io=o_start; io < o_end; io++) {
	  density_rank[2*io]   = res_density[2*(io-o_start)];
	  density_rank[2*io+1] = res_density[2*(io-o_start)+1];
	}
	MPI_Allreduce(density_rank.data(),density_group.data(),2*L,MPI_DOUBLE,MPI_SUM,t_comm);
	MPI_Allreduce(density_group.data(),density.data(),2*L,MPI_DOUBLE,MPI_SUM,h_comm);
	auto time_end_meas = std::chrono::high_resolution_clock::now();
	auto elapsed_meas_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_meas-time_start_meas).count();
	double elapsed_meas = 0.000001 * elapsed_meas_count;
	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for measurement " << elapsed_meas << " (sec) " << std::endl;
	}

      } else {

	/**
	   do_rdm != 0: calculation all one- and two-particle RDM
	 */

	auto time_start_meas = std::chrono::high_resolution_clock::now();
#ifdef SBD_THRUST
	Correlation(W,
		device_mult,
		one_p_rdm,
	    two_p_rdm);
#else
	Correlation(W,adet,bdet,bit_length,L,
		    adet_comm_size,bdet_comm_size,
		    helper,
		    h_comm,b_comm,t_comm,
		    one_p_rdm,
		    two_p_rdm);
#endif
	auto time_end_meas = std::chrono::high_resolution_clock::now();
	auto elapsed_meas_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_meas-time_start_meas).count();
	double elapsed_meas = 0.000001 * elapsed_meas_count;
	density.resize(2*L);
	for(size_t io=0; io < L; io++) {
	  density[2*io+0] = one_p_rdm[0][io+L*io];
	  density[2*io+1] = one_p_rdm[1][io+L*io];
	}

	if( mpi_rank == 0 ) {
	  std::cout << " Elapsed time for measurement " << elapsed_meas << " (sec) " << std::endl;
	  /*
	  for(size_t io=0; io < L; io++) {
	    std::cout << " Occupation density for orbital " << io
		      << ": " << density[2*io]+density[2*io+1]
		      << ", " << density[2*io]
		      << " for alpha, " << density[2*io+1]
		      << " for beta " << std::endl;
	  }
	  */
	}

      }

      /**
	 Evaluation of carry-over bit-strings
       */
      if ( sbd_data.carryover_type == 1 || sbd_data.carryover_type == 2 ) {
	if ( ratio == 0.0 ) {
	  sbd::CarryOverAdet(W,adet,bdet,
			     adet_comm_size,bdet_comm_size,
			     b_comm,co_adet,threshold);
	  sbd::CarryOverBdet(W,adet,bdet,
			     adet_comm_size,bdet_comm_size,
			     b_comm,co_bdet,threshold);
	} else {
	  size_t n_kept_a = static_cast<size_t>(ratio * adet.size());
	  double truncated_weight = 0.0;
	  sbd::CarryOverAdet(W,adet,bdet,
			     adet_comm_size,bdet_comm_size,
			     b_comm,n_kept_a,co_adet,truncated_weight);
	  if( mpi_rank == 0 ) {
	    std::cout << " truncated weight in carry-over for alpha-det = " << truncated_weight << std::endl;
	  }
	  size_t n_kept_b = static_cast<size_t>(ratio * bdet.size());
	  sbd::CarryOverBdet(W,adet,bdet,
			     adet_comm_size,bdet_comm_size,
			     b_comm,n_kept_b,co_bdet,truncated_weight);
	  if( mpi_rank == 0 ) {
	    std::cout << " truncated weight in carry-over for beta-det = " << truncated_weight << std::endl;
	  }
	}
	if( sbd_data.carryover_type == 2 ) {
	  std::vector<std::vector<size_t>> res_adet;
	  std::vector<std::vector<size_t>> res_bdet;
	  sbd::SinglesExtendHalfdets(co_adet,co_bdet,bit_length,L,
				     adet_comm_size,bdet_comm_size,b_comm,
				     res_adet,res_bdet);
	  co_adet = res_adet;
	  co_bdet = res_bdet;
	}
      } else if ( sbd_data.carryover_type == 3 ) {
	double total_weight = 0.0;
	sbd::SinglesExtendHalfdets(W,adet,bdet,bit_length,L,
				   adet_comm_size,bdet_comm_size,b_comm,
				   threshold,co_adet,co_bdet,total_weight);
	if( mpi_rank == 0 ) {
	  std::cout << " percentage of wf used in augmentation: "
		    << total_weight*100.0 << std::endl;
	}
      }

      /**
	 Save wavefunctions
       */

      if( savename != std::string("") ) {
	SaveWavefunction(savename,adet,bdet,
			 adet_comm_size,bdet_comm_size,
			 h_comm,b_comm,t_comm,W);
      }

      /**
	 Save wavefunction in matrix form
       */
      if( sbd_data.dump_matrix_form_wf != std::string("") ) {
	SaveMatrixFormWF(sbd_data.dump_matrix_form_wf,
			 bit_length,L,adet,bdet,
			 adet_comm_size,bdet_comm_size,
			 h_comm,b_comm,t_comm,W);
      }

      FreeHelpers(helper);

    } // end void diag function

    /**
       Main function to perform the diagonalization for selected basis
       @param[in] comm: communicator
       @param[in] sbd_data: parameters for setup
       @param[in] fcidumpfile: filename for fcidump data
       @param[in] adetfile: bitstrings for alpha-spin orbitals
       @param[in] loadname: load filename for wavefunction data. if string is empty, use HF det as a initial state.
       @param[in] savename: save filename for wavefunction data. if string is empty, do not save.
       @param[out] energy: obtained energy after davidson method
       @param[out] density: diagonal part of 1pRDM for configuration recovery
       @param[out] carryover_bitstrings: dominant bitstrings for alpha-spin orbitals
       @param[out] one_p_rdm: one-particle reduced density matrix if sbd_data.do_rdm != 0
       @param[out] two_p_rdm: two-particle reduced density matrix if sbd_data.do_rdm != 0
     */
    void diag(const MPI_Comm & comm,
	      const SBD & sbd_data,
	      const std::string & fcidumpfile,
	      const std::string & adetfile,
	      const std::string & loadname,
	      const std::string & savename,
	      double & energy,
	      std::vector<double> & density,
	      std::vector<std::vector<size_t>> & co_adet,
	      std::vector<std::vector<size_t>> & co_bdet,
	      std::vector<std::vector<double>> & one_p_rdm,
	      std::vector<std::vector<double>> & two_p_rdm) {

      int mpi_master = 0;
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);


      size_t L;
      size_t N;

      /**
	 Load fcidump data
       */
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
      /**
	 Load dets file
       */

      int do_shuffle = sbd_data.do_shuffle;
      std::vector<std::vector<size_t>> adet;
      std::vector<std::vector<size_t>> bdet;

      if( mpi_rank == 0 ) {
	sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
	sbd::sort_bitarray(adet);
      }

      if( do_shuffle == 0 ) {
	sbd::MpiBcast(adet,0,comm);
	bdet = adet;
      } else if ( do_shuffle == 1 ) {
	if( mpi_rank == 0 ) {
	  unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	  sbd::ShuffleDet(adet,seed);
	}
	sbd::MpiBcast(adet,0,comm);
	bdet = adet;
      } else if ( do_shuffle == 2 ) {
	if( mpi_rank == 0 ) {
	  unsigned int seeda = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	  bdet = adet;
	  sbd::ShuffleDet(adet,seeda);
	  unsigned int seedb = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	  sbd::ShuffleDet(bdet,seedb);
	}
	sbd::MpiBcast(adet,0,comm);
	sbd::MpiBcast(bdet,0,comm);
      } else if ( do_shuffle == 3 ) {
	if( mpi_rank == 0 ) {
	  unsigned int HitchhikerGuide = 42;
	  sbd::ShuffleDet(adet,HitchhikerGuide);
	}
	sbd::MpiBcast(adet,0,comm);
	bdet = adet;
      } else if ( do_shuffle == 4 ) {
	if( mpi_rank == 0 ) {
	  bdet = adet;
	  unsigned int Taxi = 1729;
	  sbd::ShuffleDet(adet,Taxi);
	  unsigned int Magic = 137;
	  sbd::ShuffleDet(bdet,Magic);
	}
	sbd::MpiBcast(adet,0,comm);
	sbd::MpiBcast(bdet,0,comm);
      }

      diag(comm,sbd_data,fcidump,adet,bdet,
	   loadname,savename,
	   energy,density,co_adet,co_bdet,
	   one_p_rdm,two_p_rdm);

    } // end diag for file-name version


  } // end namespace for tpb (tensor-product basis)

} // end namespace for sbd

#endif
