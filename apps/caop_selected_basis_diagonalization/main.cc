#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <deque>

#include <unistd.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "sbd/sbd.h"
#include "sbd/caop/basic/inc_all.h"

#ifdef _COMPLEX
using ElemType = std::complex<double>;
#else
using ElemType = double;
#endif

int main(int argc, char * argv[]) {

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);
  
  auto sbd_data = sbd::caop::generate_sbd_data(argc,argv);

  std::string hamfile;
  std::vector<std::string> basisfiles;
  std::string loadname;
  std::string savename;
  std::string basis_list;

  for(int i=0; i < argc; i++) {
    if( std::string(argv[i]) == "--hamfile" ) {
      hamfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--basisfiles" ) {
      std::stringstream ss(argv[++i]);
      std::string token;
      while (std::getline(ss,token,',')) {
	basisfiles.push_back(token);
      }
    }
    if( std::string(argv[i]) == "--basis_list" ) {
      basis_list = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--loadname" ) {
      loadname = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--savename" ) {
      savename = std::string(argv[++i]);
    }
  }

  if( !basis_list.empty() ) {
    if( mpi_rank == 0 ) {
      std::ifstream ifs(basis_list);
      std::string line;
      while( std::getline(ifs,line) ) {
	basisfiles.push_back(line);
      }
    }
    sbd::mpi_bcast_string_vector(basisfiles,0,comm);
  }

  if( mpi_rank == 0 ) {
    cout_options(sbd_data);
    std::cout << "# hamiltonian file: " << hamfile << std::endl;
    std::cout << "# basis files:";
    for(auto file : basisfiles) {
      std::cout << " " << file;
    }
    std::cout << std::endl;
    if( !loadname.empty() ) {
      std::cout << "# load wavefunction file name: " << loadname << std::endl;
    }
    if( !savename.empty() ) {
      std::cout << "# save wavefunction file name: " << savename << std::endl;
    }
  }

  double energy;
  sbd::det_vector<size_t> cobasis;
  sbd::caop::diag<ElemType>(comm,sbd_data,hamfile,basisfiles,
			    loadname,savename,energy,cobasis);


  MPI_Finalize();

  return 0;
}
