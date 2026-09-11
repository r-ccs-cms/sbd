#include "sbd/sbd.h"

#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(argc != 2 || size > 4) {
    MPI_Finalize();
    return 2;
  }
  const std::string directory(argv[1]);
  const std::string detfile = directory + "/det-" + std::to_string(rank) + ".txt";
  const std::string bitstrings[] = {
      "00001111", "00010111", "00011011", "00011101"};
  {
    std::ofstream output(detfile);
    output << bitstrings[rank] << '\n';
  }
  if(rank == 0) {
    std::ofstream output(directory + "/FCIDUMP");
    output << " &FCI NORB=4,NELEC=4,MS2=0,\n"
              "  ORBSYM=1,1,1,1,\n"
              "  ISYM=1,\n"
              " &END\n";
#ifdef _UHF
    // SBD-UHF FCIDUMP indices are interleaved spin-orbital indices even
    // though NORB remains the number of spatial orbitals.
    output << " 0.2 3 1 0 0\n"
              " 0.1 1 1 1 1\n"
              " 0.08 3 1 3 1\n";
#else
    output << " 0.2 2 1 0 0\n"
              " 0.1 1 1 1 1\n";
#endif
    output << " 0.0 0 0 0 0\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<std::string> detfiles;
  for(int r = 0; r < size; ++r)
    detfiles.push_back(directory + "/det-" + std::to_string(r) + ".txt");
  sbd::det_vector<std::size_t>::init_elem_size(1);
  sbd::det_vector<std::size_t> basis;
  sbd::load_basis_from_files(detfiles, basis, 64, 8, MPI_COMM_WORLD);
  std::vector<double> coefficients(basis.size(), rank + 1.0);
  sbd::SaveWavefunction(directory + "/state-", basis,
                        MPI_COMM_SELF, MPI_COMM_WORLD, MPI_COMM_SELF,
                        coefficients);
  MPI_Finalize();
  return 0;
}
