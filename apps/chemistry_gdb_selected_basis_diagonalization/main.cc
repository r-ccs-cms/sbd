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
#include "mpi.h"

#ifdef _COMPLEX
using Elem = std::complex<double>;
#else
using Elem = double;
#endif

int main(int argc, char * argv[]) {

  int provided;
  int mpi_ierr = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_master = 0;
  int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
  int mpi_size; MPI_Comm_size(comm,&mpi_size);
  auto sbd_data = sbd::gdb::generate_sbd_data(argc,argv);
  if( mpi_rank == 0 ) {
    sbd::gdb::cout_options(sbd_data);
  }
  std::vector<std::string> detfiles;
  std::string fcidumpfile;
  std::string loadname;
  std::string savename;
  std::string carryovername;
  for(int i=0; i < argc; i++) {
    if( std::string(argv[i]) == "--detfiles" ) {
      std::stringstream ss(argv[++i]);
      std::string token;
      while (std::getline(ss,token,',')) {
	detfiles.push_back(token);
      }
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
    if( std::string(argv[i]) == "--carryovername" ) {
      carryovername = std::string(argv[++i]);
    }
  }

  int L;
  int N;
  double energy;
  std::vector<double> density;
  sbd::det_vector<size_t> codet;
  std::vector<std::vector<Elem>> one_p_rdm;
  std::vector<std::vector<Elem>> two_p_rdm;
  sbd::FCIDump fcidump;
  std::cout.precision(16);

#ifdef SBD_FILEIN
  /**
     call sbd::gdb::diag
  */
  sbd::gdb::diag(comm,sbd_data,fcidumpfile,detfiles,loadname,savename,
		 energy,density,codet,one_p_rdm,two_p_rdm);
  /**
     preparation for output
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
  sbd::det_vector<size_t> det;
  int t_comm_size = sbd_data.t_comm_size;
  int b_comm_size = sbd_data.b_comm_size;
  int h_comm_size = mpi_size / (t_comm_size*b_comm_size);
  size_t bit_length = sbd_data.bit_length;
  MPI_Comm h_comm;
  MPI_Comm b_comm;
  MPI_Comm t_comm;
  sbd::gdb::DetBasisCommunicator(comm,h_comm_size,b_comm_size,t_comm_size,
		       h_comm,b_comm,t_comm);
  int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
  int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
  int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
  int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
  int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
  int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
#else
  /**
     load fcidump data
  */
  if( mpi_rank == 0 ) {
    fcidump = sbd::LoadFCIDump(fcidumpfile);
  }
  sbd::MpiBcast(fcidump,0,comm);
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
     setup determinants
  */
  if( mpi_rank == 0 ) {
    std::cout << " " << sbd::make_timestamp()
	      << " start setup determinant " << std::endl;
  }
  sbd::det_vector<size_t> det;
  int t_comm_size = sbd_data.t_comm_size;
  int b_comm_size = sbd_data.b_comm_size;
  int h_comm_size = mpi_size / (t_comm_size*b_comm_size);
  size_t bit_length = sbd_data.bit_length;
  sbd::det_vector<size_t>::init_elem_size((2*L + bit_length - 1) / bit_length);
  sbd::det_vector<size_t, sbd::det_kind::half>::init_elem_size((L + bit_length - 1) / bit_length);
  MPI_Comm h_comm;
  MPI_Comm b_comm;
  MPI_Comm t_comm;
  sbd::gdb::DetBasisCommunicator(comm,h_comm_size,b_comm_size,t_comm_size,
		       h_comm,b_comm,t_comm);
  int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
  int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
  int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
  int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
  int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
  int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
  if( mpi_rank_h == 0 ) {
    if( mpi_rank_t == 0 ) {
      sbd::load_basis_from_files(detfiles,det,bit_length,2*L,b_comm);
      sbd::sort_bitarray(det);
      if( sbd_data.do_sort_det ) {
	sbd::redistribution(det,bit_length,2*L,b_comm);
	sbd::reordering(det,bit_length,2*L,b_comm);
      } else if ( sbd_data.do_redist_det ) {
	sbd::redistribution(det,bit_length,2*L,b_comm);
      } else if ( sbd_data.do_redist_alpha_eq ) {
	sbd::redistribution_equal_bra_a(det,bit_length,2*L,b_comm);
      }
    }
    sbd::MpiBcast(det,0,t_comm);
  }
  sbd::MpiBcast(det,0,h_comm);
#ifdef SBD_DEBUG
  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
	if( mpi_rank_h == rank_h &&
	    mpi_rank_b == rank_b &&
	    mpi_rank_t == rank_t ) {
	  std::cout << " " << sbd::make_timestamp()
		    << " end setup determinant at rank ("
		    << mpi_rank_h << ","
		    << mpi_rank_b << ","
		    << mpi_rank_t << "): det =";
	  for(size_t i=0; i < std::min(static_cast<size_t>(4),det.size()); i++) {
	    std::cout << ( (i==0) ? " " : "," ) << sbd::makestring(det[i],sbd_data.bit_length,2*L);
	  }
	  if( det.size() > 4 ) {
	    std::cout << " ... " << sbd::makestring(det[det.size()-1],sbd_data.bit_length,2*L);
	  }
	  std::cout << ", |det|=" << det.size() << std::endl;
	}
	MPI_Barrier(t_comm);
      }
      MPI_Barrier(b_comm);
    }
    MPI_Barrier(h_comm);
  }
#endif
  /**
     call diag
  */
  sbd::gdb::diag(comm,sbd_data,fcidump,det,loadname,savename,
		 energy,density,codet,one_p_rdm,two_p_rdm);
#endif

  /**
     end print
  */
  if( mpi_rank == 0 ) {
    std::cout << " " << sbd::make_timestamp()
	      << " sbd: Energy = " << energy << std::endl;
    std::cout << " " << sbd::make_timestamp()
	      << " sbd: density = ";
    for(size_t i=0; i < density.size()/2; i++) {
      std::cout << ( (i==0) ? "[" : "," )
		<< density[2*i]+density[2*i+1];
    }
    std::cout << std::endl;
    std::cout << " " << sbd::make_timestamp()
	      << " sbd: carryover dets = ";
    for(size_t i=0; i < std::min(codet.size(),static_cast<size_t>(6)); i++) {
      std::cout << " " << sbd::makestring(codet[i],sbd_data.bit_length,2*L);
    }
    if( codet.size() > static_cast<size_t>(6) ) {
      std::cout << " ... " << sbd::makestring(codet[codet.size()-1],sbd_data.bit_length,2*L);
    }
    std::cout << std::endl;
  }
  if( carryovername != std::string("") ) {
    if( sbd_data.carryover_type == 1 ) {
      if( mpi_rank_h == 0 ) {
	if( mpi_rank_t == 0 ) {
	  std::string filename = sbd::carryoverfilename(carryovername,mpi_rank_b);
	  std::ofstream ofs(filename);
	  for(size_t k=0; k < codet.size(); k++) {
	    ofs << sbd::makestring(codet[k],sbd_data.bit_length,2*static_cast<size_t>(L)) << "\n";
	  }
	  ofs.close();
	}
      }
    } else if ( sbd_data.carryover_type == 2 || sbd_data.carryover_type == 3 ) {
      std::string filename = sbd::carryoverfilename(carryovername,mpi_rank);
      std::ofstream ofs(filename);
      for(size_t k=0; k < codet.size(); k++) {
	ofs << sbd::makestring(codet[k],sbd_data.bit_length,2*static_cast<size_t>(L)) << "\n";
      }
      ofs.close();
    }
  }
  if( one_p_rdm.size() != 0 ) {
    if ( mpi_rank == 0 ) {
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
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: elapsed time for dumping one-particle rdm = "
		<< elapsed_dump << std::endl;
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
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: elapse time for dumping two-particle rdm = "
		<< elapsed_dump << std::endl;
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: zero-body energy = " << zerobody << std::endl;
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: one-body energy = " << onebody << std::endl;
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: two-body energy = " << twobody << std::endl;
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: one-body + two-body energy = " << onebody + twobody << std::endl;
      std::cout << " " << sbd::make_timestamp()
		<< " sbd: zero-body + one-body + two-body energy = " << zerobody + onebody + twobody << std::endl;
    }
  }
  /**
     Finalize
   */
  MPI_Finalize();
  return 0;
}
