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

#include "sbd/framework/nvtx.h"

int main(int argc, char * argv[]) {

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);

#ifdef USE_OMP_OFFLOAD
  // set GPU device
  int myDevice, numDevices;
  numDevices = omp_get_num_devices();
  if (numDevices > 0) myDevice = mpi_rank % numDevices;
  else                myDevice = 0;
  omp_set_default_device(myDevice);
#endif

#ifdef SBD_THRUST
  int numDevices, myDevice;
#ifdef __CUDACC__
  cudaGetDeviceCount(&numDevices);
  myDevice = mpi_rank % numDevices;
  cudaSetDevice(myDevice);
#else
  hipGetDeviceCount(&numDevices);
  myDevice = mpi_rank % numDevices;
  hipSetDevice(myDevice);
#endif
#endif

  auto sbd_data = sbd::tpb::generate_sbd_data(argc,argv);

  std::string adetfile("alphadets.txt");
  std::string bdetfile;
  std::string fcidumpfile("fcidump.txt");
  std::string loadname;
  std::string savename;
  std::string carryover_adetfile;
  std::string carryover_bdetfile;

  for(int i=0; i < argc; i++) {
    if( std::string(argv[i]) == "--adetfile" ) {
      adetfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--bdetfile" ) {
      bdetfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--fcidump" ) {
      fcidumpfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--loadname" ) {
      loadname = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--savename" ) {
      savename = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--carryover_adetfile" ) {
      carryover_adetfile = std::string(argv[++i]);
    }
    if( std::string(argv[i]) == "--carryover_bdetfile" ) {
      carryover_bdetfile = std::string(argv[++i]);
    }
  }

  int L;
  int N;
  double energy;
  std::vector<double> density;
  sbd::det_vector<size_t, sbd::det_kind::half> co_adet;
  sbd::det_vector<size_t, sbd::det_kind::half> co_bdet;
  std::vector<std::vector<double>> one_p_rdm;
  std::vector<std::vector<double>> two_p_rdm;
  sbd::FCIDump fcidump;

  std::cout.precision(16);

#ifdef SBD_FILEIN

  /**
     sample-based diagonalization using fcidump file and adet file
   */
  {
      SBD_NVTX_RANGE("diag", __LINE__);
      sbd::tpb::diag(comm,sbd_data,fcifumpfile,adetfile,loadname,savename,
                     energy,density,co_adet,co_bdet,one_p_rdm,two_p_rdm);
  }

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
  sbd::det_vector<size_t, sbd::det_kind::half> adet;
  sbd::det_vector<size_t, sbd::det_kind::half> bdet;
  if( bdetfile.empty() ) {
    if( mpi_rank == 0 ) {
      sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
      sbd::sort_bitarray(adet);
    }

    sbd::MpiBcast(adet,0,comm);
    bdet = adet;
    if( sbd_data.do_shuffle != 0 ) {
      if( mpi_rank == 0 ) {
	unsigned int taxi = 1729;
	unsigned int magic = 137;
	sbd::ShuffleDet(adet,taxi);
	sbd::ShuffleDet(bdet,magic);
      }
      sbd::MpiBcast(adet,0,comm);
      sbd::MpiBcast(bdet,0,comm);
    }
  } else {
    if( mpi_rank == 0 ) {
      sbd::LoadAlphaDets(adetfile,adet,sbd_data.bit_length,L);
      sbd::sort_bitarray(adet);
      sbd::LoadAlphaDets(bdetfile,bdet,sbd_data.bit_length,L);
      sbd::sort_bitarray(bdet);
    }
    sbd::MpiBcast(adet,0,comm);
    sbd::MpiBcast(bdet,0,comm);
    if( sbd_data.do_shuffle != 0 ) {
      if( mpi_rank == 0 ) {
	unsigned int taxi = 1729;
	unsigned int magic = 137;
	sbd::ShuffleDet(adet,taxi);
	sbd::ShuffleDet(bdet,magic);
      }
      sbd::MpiBcast(adet,0,comm);
      sbd::MpiBcast(bdet,0,comm);
    }
  }

  /**
     sample-based diagonalization using data for fcidump, adet, bdet.
   */
  {
      SBD_NVTX_RANGE("diag");
      sbd::tpb::diag(comm,sbd_data,fcidump,adet,bdet,loadname,savename,
                     energy,density,co_adet,co_bdet,one_p_rdm,two_p_rdm);
  }

#endif

  if( mpi_rank == 0 ) {
    std::cout << " Sample-based diagonalization: Energy = " << energy << std::endl;
    std::cout << " Sample-based diagonalization: density = ";
    for(size_t i=0; i < density.size()/2; i++) {
      std::cout << ( (i==0) ? "[" : "," )
		<< density[2*i]+density[2*i+1];
    }
    std::cout << std::endl;
    std::cout << " Sample-based diagonalization: carryover bitstrings = [";
    for(size_t i=0; i < std::min(co_adet.size(),static_cast<size_t>(4)); i++) {
      std::cout << ((i==0) ? "" : ", ") << sbd::makestring(co_adet[i],sbd_data.bit_length,L);
    }
    if( co_adet.size() > static_cast<size_t>(4) ) {
      std::cout << " ..., " << sbd::makestring(co_adet[co_adet.size()-1],sbd_data.bit_length,L);
    }
    std::cout << "], size = " << co_adet.size() << std::endl;

    if( !carryover_adetfile.empty() ) {
      std::ofstream ofs_co(carryover_adetfile);
      for(size_t i=0; i < co_adet.size(); i++) {
	ofs_co << sbd::makestring(co_adet[i],sbd_data.bit_length,L) << std::endl;
      }
      ofs_co.close();
    }
    if( !carryover_bdetfile.empty() ) {
      std::ofstream ofs_co(carryover_bdetfile);
      for(size_t i=0; i < co_bdet.size(); i++) {
	ofs_co << sbd::makestring(co_bdet[i],sbd_data.bit_length,L) << std::endl;
      }
      ofs_co.close();
    }

#ifdef SBD_PREFECT
    std::cout << "Davidson energy: " << energy << std::endl;
    std::ofstream ofs_energy("davidson_energy.txt");
    ofs_energy.precision(16);
    ofs_energy << energy << std::endl;
    ofs_energy.close();
    std::ofstream ofs_occa("occ_a.txt");
    ofs_occa.precision(16);
    std::ofstream ofs_occb("occ_b.txt");
    ofs_occb.precision(16);
    for(size_t i=0; i < density.size()/2; i++) {
      ofs_occa << density[2*i] << std::endl;
      ofs_occb << density[2*i + 1] << std::endl;
    }
    ofs_occa.close();
    ofs_occb.close();
    std::cout << "Number of carryover determinants: " << cobits.size() << std::endl;
    std::ofstream ofs_co_bin("carryover.bin", std::ios::binary);
    const size_t bytes_per_config = (L + 7) / 8;
    std::vector<uint8_t> bytes(bytes_per_config);
    for (size_t i = 0; i < cobits.size(); ++i) {
      std::fill(bytes.begin(), bytes.end(), 0);
      for (size_t j = 0; j < L; ++j) {
        size_t rev_idx = L - 1 - j;                 // sbd::makestring order
        size_t pw = rev_idx % sbd_data.bit_length;  // position in word
        size_t bw = rev_idx / sbd_data.bit_length;  // index of word
        bool bit = (cobits[i][bw] >> pw) & 1ULL;
        size_t pb = 7 - (j % 8);                    // big-endian bit order
        size_t bb = j / 8;                          // index of byte
        bytes[bb] |= static_cast<uint8_t>(bit << pb);
      }
      ofs_co_bin.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    ofs_co_bin.close();
#endif  //SBD_PREFECT

    if( one_p_rdm.size() != 0 ) {

      double zerobody = 0.0;
      double onebody = 0.0;
      double twobody = 0.0;
      double I0;
      sbd::oneInt<double> I1;
      sbd::twoInt<double> I2;
      sbd::SetupIntegrals(fcidump,L,N,I0,I1,I2);
      zerobody = I0;

      auto time_start_dump = std::chrono::high_resolution_clock::now();
      std::ofstream ofs_one("1pRDM.txt");
      ofs_one.precision(16);
      for(int io=0; io < L; io++) {
	for(int jo=0; jo < L; jo++) {
	  ofs_one << io << " " << jo << " " << one_p_rdm[0][io+L*jo]+one_p_rdm[1][io+L*jo] << std::endl;
	  onebody += I1.Value(2*io,2*jo) * (one_p_rdm[0][io+L*jo] + one_p_rdm[1][io+L*jo]);
	}
      }
      auto time_end_dump = std::chrono::high_resolution_clock::now();
      auto elapsed_dump_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_dump-time_start_dump).count();
      double elapsed_dump = 0.000001 * elapsed_dump_count;
      std::cout << " Elapse time for dumping one-particle rdm = " << elapsed_dump << std::endl;
      time_start_dump = std::chrono::high_resolution_clock::now();
      std::ofstream ofs_two("2pRDM.txt");
      ofs_two.precision(16);
      for(int io=0; io < L; io++) {
	for(int jo=0; jo < L; jo++) {
	  for(int ia=0; ia < L; ia++) {
	    for(int ja=0; ja < L; ja++) {
	      ofs_two << io << " " << jo << " "
		      << ia << " " << ja << " "
		      << two_p_rdm[0][io+L*jo+L*L*(ia+L*ja)] + two_p_rdm[1][io+L*jo+L*L*(ia+L*ja)]
		       + two_p_rdm[2][io+L*jo+L*L*(ia+L*ja)] + two_p_rdm[3][io+L*jo+L*L*(ia+L*ja)]
		      << std::endl;
	      twobody += 0.5 * I2.Value(2*io,2*ia,2*jo,2*ja) * two_p_rdm[0][io+L*jo+L*L*ia+L*L*L*ja];
	      twobody += 0.5 * I2.Value(2*io,2*ia,2*jo,2*ja) * two_p_rdm[1][io+L*jo+L*L*ia+L*L*L*ja];
	      twobody += 0.5 * I2.Value(2*io,2*ia,2*jo,2*ja) * two_p_rdm[2][io+L*jo+L*L*ia+L*L*L*ja];
	      twobody += 0.5 * I2.Value(2*io,2*ia,2*jo,2*ja) * two_p_rdm[3][io+L*jo+L*L*ia+L*L*L*ja];
	    }
	  }
	}
      }

      time_end_dump = std::chrono::high_resolution_clock::now();
      elapsed_dump_count = std::chrono::duration_cast<std::chrono::microseconds>(time_end_dump-time_start_dump).count();
      elapsed_dump = 0.000001 * elapsed_dump_count;
      std::cout << " Elapse time for dumping two-particle rdm = " << elapsed_dump << std::endl;
      std::cout << " Zero-Body energy = " << zerobody << std::endl;
      std::cout << " One-Body energy = " << onebody << std::endl;
      std::cout << " Two-Body energy = " << twobody << std::endl;
      std::cout << " One-Body + Two-Body energy = " << onebody + twobody << std::endl;
      std::cout << " Zero-Body + One-Body + Two-Body energy = " << zerobody + onebody + twobody << std::endl;

    }
  }

  /**
     Finalize
  */

  MPI_Finalize();
  return 0;
}
