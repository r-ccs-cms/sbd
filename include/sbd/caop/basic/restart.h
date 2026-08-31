/**
@file sbd/caop/basic/restart.h
@brief restart from intersectional bitstring
 */
#ifndef SBD_CAOP_BASIC_RESTART_H
#define SBD_CAOP_BASIC_RESTART_H

#include <iomanip>
#include <sstream>
#include <fstream>
#include <omp.h>

namespace sbd {

  std::string statefilename(const std::string & statename, int index) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename = statename + tag + ".bin";
    return filename;

  }

  template <typename ElemT, typename DetsContainer>
  void SaveWavefunction(const std::string savename,
			const DetsContainer & basis,
			MPI_Comm h_comm,
			MPI_Comm b_comm,
			MPI_Comm t_comm,
			const std::vector<ElemT> & w) {
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

    size_t basis_size = basis.size();
    size_t basis_length = basis[0].size();
    if( mpi_rank_h == 0 && mpi_rank_t == 0 ) {
      std::string filename = statefilename(savename,mpi_rank_b);
      std::ofstream ofs(filename,std::ios::binary);
      ofs.write(reinterpret_cast<char *>(&basis_size),sizeof(size_t));
      ofs.write(reinterpret_cast<char *>(&basis_length),sizeof(size_t));
      for(size_t i=0; i < basis_size; i++) {
	ofs.write(reinterpret_cast<const char *>(basis[i].data()),sizeof(size_t)*basis_length);
      }
      auto rw = w;
      ofs.write(reinterpret_cast<char *>(rw.data()),sizeof(ElemT)*basis_size);
      ofs.close();
    }
  }

  template <typename ElemT, typename DetsContainer>
  void LoadWavefunction(const std::string loadname,
			const DetsContainer & basis,
			MPI_Comm h_comm,
			MPI_Comm b_comm,
			MPI_Comm t_comm,
			std::vector<ElemT> & w) {

    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    w.resize(basis.size(),ElemT(0.0));

    if( mpi_rank_h == 0 ) {

      if( mpi_rank_t == 0 ) {
	size_t load_basis_size   = 0;
	size_t load_basis_length = 0;
	det_vector<size_t> load_basis;
	std::vector<ElemT> load_w;
	std::string filename = statefilename(loadname,mpi_rank_b);
	std::ifstream ifs(filename,std::ios::binary);
	if( ifs.is_open() ) {
	  ifs.read(reinterpret_cast<char *>(&load_basis_size),sizeof(size_t));
	  ifs.read(reinterpret_cast<char *>(&load_basis_length),sizeof(size_t));
	  load_basis.resize(load_basis_size);
	  for(size_t i=0; i < load_basis_size; i++) {
	    load_basis[i].resize(load_basis_length);
	    ifs.read(reinterpret_cast<char *>(load_basis[i].data()),sizeof(size_t)*load_basis_length);
	  }
	  load_w.resize(load_basis_size);
	  ifs.read(reinterpret_cast<char *>(load_w.data()),sizeof(ElemT)*load_basis_size);
	}

	auto cmp = [](const auto & x, const auto & y) {
	  return sbd::less_from_back(x,y);
	};
	// Parallel binary search: race-free because basis is sorted and unique, so
	// each load_basis[i] maps to at most one slot.  Use std::vector<char>
	// rather than std::vector<bool> to avoid the bit-packed specialisation's
	// concurrent-write pitfall (two bools in the same byte, distinct indices).
	std::vector<char> found_flag(load_basis_size, 0);
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < load_basis_size; ++i) {
	  auto it = std::lower_bound(basis.begin(), basis.end(), load_basis[i], cmp);
	  if (it != basis.end() && *it == load_basis[i]) {
	    w[static_cast<size_t>(it - basis.begin())] = load_w[i];
	    found_flag[i] = 1;
	  }
	}
	std::vector<size_t> index_not_found;
	index_not_found.reserve(load_basis_size);
	for (size_t i = 0; i < load_basis_size; ++i)
	  if (!found_flag[i]) index_not_found.push_back(i);
	det_vector<size_t> send_basis(index_not_found.size());
	std::vector<ElemT> send_w(index_not_found.size());
	for(size_t k=0; k < index_not_found.size(); k++) {
	  send_basis[k] = load_basis[index_not_found[k]];
	  send_w[k] = load_w[index_not_found[k]];
	}

	for(int slide=1; slide < mpi_size_b; slide++) {
	  MpiSlide(send_basis,load_basis,1,b_comm);
	  MpiSlide(send_w,load_w,1,b_comm);
	  const size_t slide_sz = load_basis.size();
	  std::vector<char> slide_found(slide_sz, 0);
	  #pragma omp parallel for schedule(static)
	  for (size_t i = 0; i < slide_sz; ++i) {
	    auto it = std::lower_bound(basis.begin(), basis.end(), load_basis[i], cmp);
	    if (it != basis.end() && *it == load_basis[i]) {
	      w[static_cast<size_t>(it - basis.begin())] = load_w[i];
	      slide_found[i] = 1;
	    }
	  }
	  index_not_found.clear();
	  index_not_found.reserve(slide_sz);
	  for (size_t i = 0; i < slide_sz; ++i)
	    if (!slide_found[i]) index_not_found.push_back(i);
	  send_basis.resize(index_not_found.size());
	  send_w.resize(index_not_found.size());
	  for(size_t k=0; k < index_not_found.size(); k++) {
	    send_basis[k] = load_basis[index_not_found[k]];
	    send_w[k] = load_w[index_not_found[k]];
	  }
	}
	ElemT normw;
	Normalize(w,normw,b_comm);
      }
      MpiBcast(w,0,t_comm);
    }
    MpiBcast(w,0,h_comm);
  }


}

#endif
