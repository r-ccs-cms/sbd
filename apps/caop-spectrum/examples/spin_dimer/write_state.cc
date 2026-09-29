// Exact singlet fixture, saved by public SBD (no eigensolver required).
#include "sbd/sbd.h"
#include <cmath>
#include <iostream>
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(argc!=2) {if(!rank) std::cerr<<"usage: write_state PREFIX\n";MPI_Abort(MPI_COMM_WORLD,1);}
  if(!rank) {
    sbd::det_vector<std::size_t>::init_elem_size(1);
    sbd::det_vector<std::size_t> basis;
    basis.push_back(std::vector<std::size_t>{1});
    basis.push_back(std::vector<std::size_t>{2});
    const double c=1/std::sqrt(2.0);
    sbd::SaveWavefunction(std::string(argv[1]),basis,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,
        std::vector<double>{c,-c});
  }
  MPI_Finalize();return 0;
}
