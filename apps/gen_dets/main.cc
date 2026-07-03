#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>

#define _USE_MATH_DEFINES
#include <cmath>

#include "sbd/sbd.h"
#include "mpi.h"

int main(int argc, char * argv[]) {
  int provided;
  int mpi_ierr = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);

  std::string adetfile;
  std::vector<std::string> detfiles;
  int L;
  int bit_length;
  for(size_t i=0; i < argc; i++) {
    if( std::string(argv[i]) == "--adetfile" ) {
      adetfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--detfiles" ) {
      std::stringstream ss(argv[++i]);
      std::string token;
      while ( std::getline(ss,token,',') ) {
	detfiles.push_back(token);
      }
    }
    if( std::string(argv[i]) == "--bit_length" ) {
      bit_length = std::atoi(argv[++i]);
    }
    if( std::string(argv[i]) == "--norb" ) {
      L = std::atoi(argv[++i]);
    }
  }

  std::vector<std::vector<size_t>> adet;
  if( mpi_rank == 0 ) {
    sbd::LoadAlphaDets(adetfile,adet,bit_length,L);
    sbd::sort_bitarray(adet);
  }
  sbd::MpiBcast(adet,0,comm);
  std::vector<std::vector<size_t>> bdet(adet);

  std::vector<std::vector<size_t>> det;

  if( mpi_rank == 0 ) {
    size_t adet_size = adet.size();
    size_t bdet_size = bdet.size();
    size_t det_size = adet_size * bdet_size;
    det.resize(det_size);
    for(size_t i=0; i < adet_size; i++) {
      for(size_t j=0; j < bdet_size; j++) {
	det[i*bdet_size+j] =
	  sbd::DetFromAlphaBeta(adet[i],bdet[j],
				static_cast<size_t>(bit_length),
				static_cast<size_t>(L));
      }
    }
    sbd::sort_bitarray(det);
    int num_size = static_cast<int>(detfiles.size());
    for(int num_rank=0; num_rank < num_size; num_rank++) {
      size_t i_begin = 0;
      size_t i_end   = det_size;
      sbd::get_mpi_range(num_size,num_rank,i_begin,i_end);
      std::ofstream ofs(detfiles[num_rank]);
      for(size_t i=i_begin; i < i_end; i++) {
	ofs << sbd::makestring(det[i],static_cast<size_t>(bit_length),
			       static_cast<size_t>(2*L)) << std::endl;
      }
      ofs.close();
    }
  }
  MPI_Finalize();
  return 0;
}
