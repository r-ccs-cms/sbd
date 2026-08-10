/**
@file sbd/chemistry/basic/makedeterminants.h
@brief Setup the set of determinants from
*/
#ifndef SBD_CHEMISTRY_BASIC_MAKEDETERMINANTS_H
#define SBD_CHEMISTRY_BASIC_MAKEDETERMINANTS_H

namespace sbd {

  void SetupDeterminants(const std::vector<std::vector<size_t>> & AlphaDet,
			 const size_t bit_length,
			 const size_t L,
			 std::vector<std::vector<size_t>> & Det,
			 MPI_Comm comm) {

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    size_t NcA = AlphaDet.size();
    size_t NcB = AlphaDet.size();
    size_t Nc = NcA*NcB;

    size_t na_begin = 0;
    size_t na_end = NcA;
    get_mpi_range(mpi_size,mpi_rank,na_begin,na_end);
    Det.resize((na_end-na_begin)*NcB);
    for(size_t na=na_begin; na < na_end; na++) {
      for(size_t nb=0; nb < NcB; nb++) {
	Det[NcB*(na-na_begin)+nb] = DetFromAlphaBeta(AlphaDet[na],
						     AlphaDet[nb],
						     bit_length,L);
      }
    }
  }

  void SetupDeterminants(const std::vector<std::vector<size_t>> & AlphaDet,
			 const std::vector<std::vector<size_t>> & BetaDet,
			 const size_t bit_length,
			 const size_t L,
			 std::vector<std::vector<size_t>> & Det,
			 MPI_Comm comm) {

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    size_t NcA = AlphaDet.size();
    size_t NcB = BetaDet.size();
    size_t na_begin = 0;
    size_t na_end = NcA;
    get_mpi_range(mpi_size,mpi_rank,na_begin,na_end);

    Det.resize((na_end-na_begin)*NcB);
    for(size_t na=na_begin; na < na_end; na++) {
      for(size_t nb=0; nb < NcB; nb++) {
	Det[NcB*(na-na_begin)+nb] = DetFromAlphaBeta(AlphaDet[na],BetaDet[nb],bit_length,L);
      }
    }
  }


  std::vector<std::vector<size_t>>
  DecodeAlphaDetsBits(const std::string& filename, size_t norb) {
    // open binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file");
    }

    // byte number of each bit array
    size_t byte_length = (norb + 7) / 8;

    // output vector for bit array
    std::vector<std::vector<size_t>> all_bit_sequences;

    // decode reading the file
    std::vector<uint8_t> buffer(byte_length);  // buffa of each bit array
    while (file.read(reinterpret_cast<char*>(buffer.data()), byte_length)) {
      // decode the one bit array
      std::vector<size_t> bit_sequence;
      size_t bit_index = 0;


      for (size_t byte : buffer) {
	for (int i = 7; i >= 0; --i) { // decode from higher bit
	  if ( 8*byte_length-norb <= bit_index ) {
	    size_t bit = (byte >> i) & 1;
	    bit_sequence.push_back(bit);
	  }
	  ++bit_index;
	}
      }

      // reverse bit sequence
      std::vector<size_t> reverse_bit_sequence(bit_sequence.size());
      for(int i=0; i < bit_sequence.size(); i++) {
	reverse_bit_sequence[i] = bit_sequence[bit_sequence.size()-i-1];
      }

      // store the decoded bit array
      // all_bit_sequences.push_back(bit_sequence);
      all_bit_sequences.push_back(reverse_bit_sequence);
    }

    return all_bit_sequences;
  }

  det_vector<size_t, det_kind::half>
  DecodeAlphaDets(const std::string& filename,
		  size_t norb,
		  size_t bit_length) {
    auto all_bit_sequences = DecodeAlphaDetsBits(filename, norb);
    change_bitlength(1, all_bit_sequences, bit_length);
    det_vector<size_t, det_kind::half> result;
    result.assign(all_bit_sequences.begin(), all_bit_sequences.end());
    return result;
  }

  void LoadFromAlphaDets(const std::string & filename,
			 std::vector<std::vector<size_t>> & config,
			 size_t norb,
			 size_t bit_length,
			 MPI_Comm comm) {

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_root = 0;

    std::vector<std::vector<size_t>> alphadets;
    if( mpi_rank == mpi_root) {
      alphadets = DecodeAlphaDetsBits(filename,norb);
    }
    MpiBcast(alphadets,mpi_root,comm);

    size_t Na = alphadets.size();
    size_t Nb = Na;
    size_t ib_start = 0;
    size_t ib_end = Nb;

    get_mpi_range(mpi_size,mpi_rank,ib_start,ib_end);

    size_t config_size = (2*norb-1)/bit_length+1;
    config.resize(Na*(ib_end-ib_start));

    for(size_t ib=ib_start; ib < ib_end; ib++) {
      for(size_t ia=0; ia < Na; ia++) {
	std::vector<size_t> bst(2*norb);
	for(int p=0; p < norb; p++) {
	  bst[p] = alphadets[ia][p];
	}
	for(int p=0; p < norb; p++) {
	  bst[norb+p] = alphadets[ib][p];
	}
	change_bitlength(1,bst,bit_length);
	config[Na*(ib-ib_start)+ia] = bst;
      }
    }
  }

  /**
     Function to load the half determinant data
     @param[in] filename
     @param[out] adet: loaded alpha-det file
     @param[in] bit_length: length of bit string managed by a size_t
     @param[in] L: total length of bit string
   */

  void LoadAlphaDets(const std::string & adetfile,
		     det_vector<size_t, det_kind::half> & adet,
		     size_t bit_length,
		     size_t total_bit_length) {

    if( sbd::get_extension(adetfile) == std::string("txt") ) {
      std::ifstream inadet(adetfile);
      if( !inadet.is_open() ) {
        throw std::runtime_error("Failed to open alpha det file.");
      }
      std::vector<std::string> lines;
      std::string line;
      while( std::getline(inadet,line) ) {
        lines.push_back(std::move(line));
      }
      adet.resize(lines.size(),
                  std::vector<size_t>((total_bit_length + bit_length - 1) / bit_length));
      for(size_t i=0; i < lines.size(); i++) {
        adet[i] = sbd::from_string(lines[i],bit_length,total_bit_length);
      }
    } else if ( sbd::get_extension(adetfile) == std::string("bin") ) {
      adet = sbd::DecodeAlphaDets(adetfile,total_bit_length,bit_length);
    }
    sbd::sort_bitarray(adet);
  }

}

#endif
