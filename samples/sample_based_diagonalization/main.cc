#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <deque>

#include <unistd.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "sbd/sbd.h"
#include "mpi.h"



int main(int argc, char * argv[]) {

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);

  auto sbd_data = sbd::tpb::generate_sbd_data(argc,argv);

  std::string adetfile("alphadets.txt");
  std::string fcidumpfile("fcidump.txt");
  std::string loadname("");
  std::string savename("");
  std::string carryoverfile("");
  
  for(int i=0; i < argc; i++) {
    if( std::string(argv[i]) == "--adetfile" ) {
      adetfile = std::string(argv[i+1]);
      i++;
    }
    if( std::string(argv[i]) == "--fcidump" ) {
      fcidumpfile = std::string(argv[i+1]);
      i++;
    }
    if( std::string(argv[i]) == "--loadname" ) {
      loadname = std::string(argv[i+1]);
      i++;
    }
    if( std::string(argv[i]) == "--savename" ) {
      savename = std::string(argv[i+1]);
      i++;
    }
    if( std::string(argv[i]) == "--carryoverfile" ) {
      carryoverfile = std::string(argv[i+1]);
      i++;
    }
  }
  
  int L;
  int N;
  double energy;
  std::vector<double> density;
  std::vector<std::vector<size_t>> cobits;
  std::vector<std::vector<double>> one_p_rdm;
  std::vector<std::vector<double>> two_p_rdm;
  sbd::FCIDump fcidump;

  std::cout.precision(16);
  
#ifdef SBD_FILEIN

  /**
     sample-based diagonalization using fcidump file and adet file
   */
  sbd::tpb::diag(comm,sbd_data,fcifumpfile,adetfile,loadname,savename,
		 energy,density,cobits,one_p_rdm,two_p_rdm);

  /**
     Get L (number of orbitals) and N (number of electrons) from fcidump data for output
   */
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

#else

  /**
     load fcidump data
   */
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
     setup determinants for alpha and beta spin orbitals
   */
  std::vector<std::vector<size_t>> adet;
  std::vector<std::vector<size_t>> bdet;
  if( mpi_rank == 0 ) {
    sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
  }

  if( sbd_data.do_shuffle == 0 ) {
    sbd::MpiBcast(adet,0,comm);
    bdet = adet;
  } else {
    if( mpi_rank == 0 ) {
      unsigned int taxi = 1729;
      unsigned int magic = 137;
      sbd::ShuffleDet(adet,taxi);
      sbd::ShuffleDet(bdet,magic);
    }
    sbd::MpiBcast(adet,0,comm);
    sbd::MpiBcast(bdet,0,comm);
  }

  /**
     sample-based diagonalization using data for fcidump, adet, bdet.
   */
  sbd::tpb::diag(comm,sbd_data,fcidump,adet,bdet,loadname,savename,
		 energy,density,cobits,one_p_rdm,two_p_rdm);
  
#endif

  if( mpi_rank == 0 ) {
    std::cout << " Sample-based diagonalization: Energy = " << energy << std::endl;
    std::cout << " Sample-based diagonalization: density = ";
    for(size_t i=0; i < density.size()/2; i++) {
      std::cout << ( (i==0) ? "[" : "," )
		<< density[2*i]+density[2*i+1];
    }
    std::cout << std::endl;
    std::cout << " Sample-based diagonalization: carryover bitstrings = ";
    for(size_t i=0; i < std::min(cobits.size(),static_cast<size_t>(6)); i++) {
      std::cout << " " << sbd::makestring(cobits[i],sbd_data.bit_length,L);
    }
    if( cobits.size() > static_cast<size_t>(6) ) {
      std::cout << " ... " << sbd::makestring(cobits[cobits.size()-1],sbd_data.bit_length,L);
    }
    std::cout << std::endl;
    if( carryoverfile != std::string("") ) {
      std::ofstream ofs_co(carryoverfile);
      for(size_t i=0; i < cobits.size(); i++) {
	ofs_co << sbd::makestring(cobits[i],sbd_data.bit_length,L) << std::endl;
      }
      ofs_co.close();
    }
    if( one_p_rdm.size() != 0 ) {
      auto time_start_dump = std::chrono::high_resolution_clock::now();
      std::ofstream ofs_one("1pRDM.txt");
      ofs_one.precision(16);
      for(int io=0; io < L; io++) {
	for(int jo=0; jo < L; jo++) {
	  ofs_one << io << " " << jo << " 0 " << one_p_rdm[0][io+L*jo] << std::endl;
	  ofs_one << io << " " << jo << " 1 " << one_p_rdm[1][io+L*jo] << std::endl;
	}
      }
      auto time_end_dump = std::chrono::high_resolution_clock::now();
      auto elapsed_dump_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_dump-time_start_dump).count();
      double elapsed_dump = 0.000001 * elapsed_dump_count;
      std::cout << " Elapse time for dumping one-particle rdm = " << elapsed_dump << std::endl;
    }
    if( two_p_rdm.size() != 0 ) {
      auto time_start_dump = std::chrono::high_resolution_clock::now();
      std::ofstream ofs_two("2pRDM.txt");
      ofs_two.precision(16);
      for(int io=0; io < L; io++) {
	for(int jo=0; jo < L; jo++) {
	  for(int ia=0; ia < L; ia++) {
	    for(int ja=0; ja < L; ja++) {
	      ofs_two << io << " " << jo << " "
		      << ia << " " << ja << " 0 0 "
		      << two_p_rdm[0][io+L*jo+L*L*ja+L*L*L*ia] << std::endl;
	      ofs_two << io << " " << jo << " "
		      << ia << " " << ja << " 1 0 "
		      << two_p_rdm[1][io+L*jo+L*L*ja+L*L*L*ia] << std::endl;
	      ofs_two << io << " " << jo << " "
		      << ia << " " << ja << " 0 1 "
		      << two_p_rdm[2][io+L*jo+L*L*ja+L*L*L*ia] << std::endl;
	      ofs_two << io << " " << jo << " "
		      << ia << " " << ja << " 1 1 "
		      << two_p_rdm[3][io+L*jo+L*L*ja+L*L*L*ia] << std::endl;
	    }
	  }
	}
      }
      auto time_end_dump = std::chrono::high_resolution_clock::now();
      auto elapsed_dump_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_dump-time_start_dump).count();
      double elapsed_dump = 0.000001 * elapsed_dump_count;
      std::cout << " Elapse time for dumping two-particle rdm = " << elapsed_dump << std::endl;
    }
  }

  /**
     Finalize
  */

  MPI_Finalize();
  return 0;
}
