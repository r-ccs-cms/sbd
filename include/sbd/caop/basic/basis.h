/**
@file sbd/caop/basic/basis.h
@brief function to setup the basis
 */
#ifndef SBD_CAOP_BASIC_BASIS_H
#define SBD_CAOP_BASIC_BASIS_H

#include <sys/stat.h>
#include <iomanip>

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {
  
  template <typename Container>
  void redistribution(Container & config,
		      size_t bit_length,
		      size_t total_bit_length,
		      MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_redistribution(config,config_begin,config_end,index_begin,index_end,
		       total_bit_length,bit_length,comm);
    
  }
  


  // Sort idx[lo..hi) so that (a[idx[i]][elem] & Mask) is in ascending order,
  // recursing on equal-valued runs through decreasing elements.
  // Mask is a template parameter so the compiler specialises each instantiation:
  // Mask=~0 eliminates the AND entirely; Mask=0x5555... is baked into code.
  template<size_t Mask = ~size_t(0), typename Container>
  void idx_sort_from_back(std::vector<size_t>& idx,
                          const Container& a,
                          size_t lo, size_t hi, int elem) {
    if (hi - lo <= 1 || elem < 0) return;
    std::sort(idx.begin() + lo, idx.begin() + hi,
              [&a, elem](size_t x, size_t y) {
                return (a[x][elem] & Mask) < (a[y][elem] & Mask);
              });
    if (elem == 0) return;
    size_t run_lo = lo;
    for (size_t i = lo + 1; i <= hi; i++) {
      if (i == hi || (a[idx[i]][elem] & Mask) != (a[idx[run_lo]][elem] & Mask)) {
        if (i - run_lo > 1)
          idx_sort_from_back<Mask>(idx, a, run_lo, i, elem - 1);
        run_lo = i;
      }
    }
  }

  // Equal-bra_a redistribution: partition dets so each rank owns a disjoint
  // contiguous range of alpha strings, giving equal bra_a across ranks.
  // Alpha occupation is at even bit positions (0,2,4,...) of the det bitstring,
  // interleaved with beta at odd positions.  Works for any clen.
  void redistribution_equal_bra_a(det_vector<size_t> & config,
                                   size_t bit_length,
                                   size_t total_bit_length,
                                   MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);

    const int clen = static_cast<int>((total_bit_length + bit_length - 1) / bit_length);
    // Alpha bits sit at even positions (0,2,4,...) of each interleaved word.
    // Masking with ALPHA_MASK zeroes beta bits in place; the remaining even-position
    // bits compare correctly with < because bit position order is preserved.
    constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;

    // Step 1: sort local dets alpha-primary by masking beta bits out of each word.
    // Det order within each alpha group is irrelevant; Step 7 re-sorts beta-primary.
    std::vector<size_t> idx(config.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    idx_sort_from_back<ALPHA_MASK>(idx, config, 0, idx.size(), clen - 1);
    {
      det_vector<size_t> tmp(config.size());
      for (size_t i = 0; i < config.size(); i++) tmp[i] = config[idx[i]];
      config = std::move(tmp);
    }

    // Step 2: collect local unique alpha keys from sorted config (mask on the fly).
    det_vector<size_t> local_alphas;
    {
      std::vector<size_t> prev(clen, ~size_t(0));
      for (size_t j = 0; j < config.size(); j++) {
        bool diff = false;
        for (int k = 0; k < clen && !diff; k++)
          diff = ((config[j][k] & ALPHA_MASK) != prev[k]);
        if (diff) {
          for (int k = 0; k < clen; k++)
            prev[k] = config[j][k] & ALPHA_MASK;
          local_alphas.emplace_back(prev);
        }
      }
    }

    // Step 3: allgather unique alpha keys → global sorted unique list.
    // Each alpha key is clen words; exchange as flat size_t arrays.
    int local_n = static_cast<int>(local_alphas.size());
    std::vector<int> all_counts(mpi_size);
    MPI_Allgather(&local_n, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
    int total_n = 0;
    std::vector<int> ag_displs_w(mpi_size, 0), ag_counts_w(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
      ag_displs_w[r] = total_n * clen;
      ag_counts_w[r] = all_counts[r] * clen;
      total_n += all_counts[r];
    }
    std::vector<size_t> flat_local(local_n * clen), flat_all(total_n * clen);
    for (int i = 0; i < local_n; i++)
      for (int k = 0; k < clen; k++) flat_local[i * clen + k] = local_alphas[i][k];
    MPI_Allgatherv(flat_local.data(), local_n * clen, SBD_MPI_SIZE_T,
                   flat_all.data(), ag_counts_w.data(), ag_displs_w.data(), SBD_MPI_SIZE_T, comm);
    det_vector<size_t> all_alphas(static_cast<size_t>(total_n));
    for (int i = 0; i < total_n; i++)
      for (int k = 0; k < clen; k++) all_alphas[i][k] = flat_all[i * clen + k];
    sort_bitarray(all_alphas);
    size_t global_bra_a = all_alphas.size();

    // Step 4: assign alpha strings to ranks with equal bra_a.
    size_t chunk = global_bra_a / static_cast<size_t>(mpi_size);
    size_t rem   = global_bra_a % static_cast<size_t>(mpi_size);
    std::vector<size_t> alpha_start(mpi_size + 1, 0);
    for (int r = 0; r < mpi_size; r++)
      alpha_start[r+1] = alpha_start[r] + chunk + (static_cast<size_t>(r) < rem ? 1 : 0);

    // Step 5: for each local det compute destination rank.
    // all_alphas entries are already masked; apply ALPHA_MASK to config[j] inline.
    std::vector<int> dest_per_det(config.size());
    std::vector<int> sendcounts(mpi_size, 0);
    for (size_t j = 0; j < config.size(); j++) {
      size_t pos = static_cast<size_t>(
        std::lower_bound(all_alphas.begin(), all_alphas.end(), config[j],
          [clen](const auto& alpha, const auto& det) {
            constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;
            for (int k = clen - 1; k >= 0; k--) {
              if (alpha[k] < (det[k] & ALPHA_MASK)) return true;
              if (alpha[k] > (det[k] & ALPHA_MASK)) return false;
            }
            return false;
          })
        - all_alphas.begin());
      int dest = static_cast<int>(
        std::upper_bound(alpha_start.begin(), alpha_start.end(), pos)
        - alpha_start.begin()) - 1;
      if (dest < 0) dest = 0;
      if (dest >= mpi_size) dest = mpi_size - 1;
      dest_per_det[j] = dest;
      sendcounts[dest]++;
    }

    // Step 6: MPI_Alltoallv (clen words per det).
    std::vector<int> sendcounts_w(mpi_size), sdispls(mpi_size, 0);
    for (int r = 0; r < mpi_size; r++) sendcounts_w[r] = sendcounts[r] * clen;
    for (int r = 1; r < mpi_size; r++) sdispls[r] = sdispls[r-1] + sendcounts_w[r-1];
    std::vector<int> recvcounts_w(mpi_size), rdispls(mpi_size, 0);
    MPI_Alltoall(sendcounts_w.data(), 1, MPI_INT, recvcounts_w.data(), 1, MPI_INT, comm);
    for (int r = 1; r < mpi_size; r++) rdispls[r] = rdispls[r-1] + recvcounts_w[r-1];
    int total_recv_w = rdispls[mpi_size-1] + recvcounts_w[mpi_size-1];

    std::vector<size_t> sendbuf(config.size() * clen);
    {
      std::vector<int> fill(sdispls);
      for (size_t j = 0; j < config.size(); j++) {
        int d = dest_per_det[j];
        for (int k = 0; k < clen; k++) sendbuf[fill[d]++] = config[j][k];
      }
    }
    std::vector<size_t> recvbuf(total_recv_w);
    MPI_Alltoallv(sendbuf.data(), sendcounts_w.data(), sdispls.data(), SBD_MPI_SIZE_T,
                  recvbuf.data(), recvcounts_w.data(), rdispls.data(), SBD_MPI_SIZE_T, comm);

    // Step 7: unpack and restore kernel sort order (beta-primary = less_from_back).
    size_t n_recv = static_cast<size_t>(total_recv_w) / static_cast<size_t>(clen);
    config.resize(n_recv);
    for (size_t i = 0; i < n_recv; i++)
      for (int k = 0; k < clen; k++) config[i][k] = recvbuf[i * clen + k];
    sort_bitarray(config);
  }

  template <typename Container>
  void reordering(Container & config,
		  size_t bit_length,
		  size_t total_bit_length,
		  MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,
		      total_bit_length,bit_length,comm);
  }
  
  // I/O for basis
  template<typename Container>
  void load_basis_from_file(const std::string & filename,
			    Container & config,
			    size_t bit_length,
			    size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ifstream ifs(filename);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string file.");
      }
      std::string line;
      std::vector<std::string> lines;
      while( std::getline(ifs,line) ) {
	lines.push_back(line);
      }
      config.resize(lines.size());
      for(size_t i=0; i < lines.size(); i++) {
	config[i] = from_string(lines[i],bit_length,total_bit_length);
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ifstream ifs(filename, std::ios::binary);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string binary file.");
      }

      size_t inner_size = (total_bit_length+bit_length-1)/bit_length;
      ifs.seekg(0, std::ios::end);
      std::streampos file_size = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      size_t bytes_per_line = inner_size * sizeof(size_t);

      if( file_size % bytes_per_line != 0 ) {
	throw std::runtime_error("Binary file size mismatch");
      }

      size_t num_lines = file_size / bytes_per_line;

      config.resize(num_lines);
      for(size_t i=0; i < num_lines; i++) {
	config[i].resize(inner_size);
	ifs.read(reinterpret_cast<char*>(config[i].data()),bytes_per_line);
	if (!ifs) {
	  throw std::runtime_error("Failed to read binary basis data.");
	}
      }
    }
  }
  
  template<typename Container>
  void save_basis_to_file(const std::string & filename,
			  Container & config,
			  size_t bit_length,
			  size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ofstream ofs(filename);
      for(size_t i=0; i < config.size(); i++) {
	ofs << makestring(config[i],bit_length,total_bit_length) << std::endl;
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ofstream ofs(filename,std::ios::binary);
      for(auto & b : config) {
	ofs.write(reinterpret_cast<char*>(b.data()),sizeof(size_t)*b.size());
      }
    }
  }

  // basis file name for multiple nodes
  std::string basisfilename(const std::string & basisname, int index, int filetype) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename;
    if( filetype == 0 ) {
      filename = basisname + tag + ".txt";
    } else if ( filetype == 1 ) {
      filename = basisname + tag + ".bin";
    }
    return filename;
  }

  template<typename Container>
  void load_basis_from_files(const std::vector<std::string> & all_filenames,
			     Container & config,
			     size_t bit_length,
			     size_t total_bit_length,
			     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const int num_files = static_cast<int>(all_filenames.size());
    config.clear();
    
    if (num_files == 0) return;
    
    const int base = num_files / mpi_size;
    const int rem  = num_files % mpi_size;

    int my_first = 0;
    int my_count = 0;
    if (mpi_rank < rem) {
      my_count = base + 1;
      my_first = mpi_rank * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1) + (mpi_rank - rem) * base;
    }
    const int my_last = my_first + my_count;
    
    for (int i = my_first; i < my_last; ++i) {
      const std::string & fname = all_filenames[i];
      
      Container local;
      load_basis_from_file(fname, local, bit_length, total_bit_length);
      
      config.insert(config.end(),
		    std::make_move_iterator(local.begin()),
		    std::make_move_iterator(local.end()));
    }
    sort_bitarray(config);
  }
  
  // load single file
  void load_basis_from_single_binary(const std::string & filename,
				     std::vector<std::vector<size_t>> & config,
				     size_t bit_length,
				     size_t total_bit_length,
				     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const size_t inner_size    = (total_bit_length + bit_length - 1) / bit_length;
    const size_t bytes_per_line = inner_size * sizeof(size_t);
    
    std::uint64_t num_lines_u64 = 0;
    
    if (mpi_rank == 0) {
      std::ifstream ifs(filename, std::ios::binary);
      if (!ifs.is_open()) {
	throw std::runtime_error("Failed to open basis binary file: " + filename);
      }
      
      ifs.seekg(0, std::ios::end);
      std::streampos file_size_pos = ifs.tellg();
      ifs.seekg(0, std::ios::beg);
      
    if (file_size_pos < 0) {
      throw std::runtime_error("tellg() failed for file: " + filename);
    }
    
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_size_pos);
    
    if (file_size % bytes_per_line != 0) {
      throw std::runtime_error("Binary file size mismatch in " + filename);
    }
    
    num_lines_u64 = file_size / bytes_per_line;
    }
    
    MPI_Bcast(&num_lines_u64, 1, MPI_UINT64_T, 0, comm);

    if (num_lines_u64 == 0) {
      config.clear();
      return;
    }
    
    const std::size_t num_lines = static_cast<std::size_t>(num_lines_u64);

    const std::size_t base = num_lines / mpi_size;
    const std::size_t rem  = num_lines % mpi_size;
    
    std::size_t my_first = 0;
    std::size_t my_count = 0;
    if (static_cast<std::size_t>(mpi_rank) < rem) {
      my_count = base + 1;
      my_first = static_cast<std::size_t>(mpi_rank) * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1)
	+ (static_cast<std::size_t>(mpi_rank) - rem) * base;
    }
    const std::size_t my_last = my_first + my_count;
    
    config.clear();
    config.resize(my_count);
    for (auto & row : config) {
      row.resize(inner_size);
    }
    
    if (my_count == 0) {
      return;
    }
    
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open basis binary file (per rank): " + filename);
    }

    const std::uint64_t my_offset_bytes =
      static_cast<std::uint64_t>(my_first) * bytes_per_line;
    
    ifs.seekg(static_cast<std::streamoff>(my_offset_bytes), std::ios::beg);
    if (!ifs) {
      throw std::runtime_error("seekg failed for basis binary file: " + filename);
    }
    
    for (std::size_t i = 0; i < my_count; ++i) {
      ifs.read(reinterpret_cast<char*>(config[i].data()), bytes_per_line);
      if (!ifs) {
	throw std::runtime_error("Failed to read basis data from: " + filename);
      }
    }
    sort_bitarray(config);
  }

  inline void mpi_bcast_string_vector(std::vector<std::string> & vec,
				      int root,
				      MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int count = static_cast<int>(vec.size());
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    
    if (rank != root) {
      vec.resize(count);
    }
    
    std::vector<int> lengths(count);
    if (rank == root) {
      for (int i = 0; i < count; i++) {
	lengths[i] = static_cast<int>(vec[i].size());
      }
    }
    MPI_Bcast(lengths.data(), count, MPI_INT, root, comm);
    
    for (int i = 0; i < count; i++) {
      if (rank != root) {
	vec[i].resize(lengths[i]);
      }
      if (lengths[i] > 0) {
#ifdef SBD_TRADMODE
	char * ptr = &vec[i][0];
	MPI_Bcast(ptr, lengths[i], MPI_CHAR, root, comm);
#else
	MPI_Bcast(vec[i].data(), lengths[i], MPI_CHAR, root, comm);
#endif
      }
    }
  }
  
  
}

#endif
