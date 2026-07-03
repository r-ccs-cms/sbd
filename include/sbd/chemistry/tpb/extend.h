/**
@file sbd/chemistry/tpb/extend.h
@brief function to extend the determinant basis
*/
#ifndef SBD_CHEMISTRY_TPB_EXTEND_H
#define SBD_CHEMISTRY_TPB_EXTEND_H

namespace sbd {

  /**
     Function to construct an extended half-determinant subspace by generating all single-particle excitations from half-determinants appearing in dominant determinants.
   */
  template <typename ElemT, typename RealT>
  void SinglesExtendHalfdets(const std::vector<ElemT> & w,
			     const std::vector<std::vector<size_t>> & adet,
			     const std::vector<std::vector<size_t>> & bdet,
			     size_t bit_length,
			     size_t norb,
			     size_t adet_comm_size,
			     size_t bdet_comm_size,
			     MPI_Comm comm,
			     RealT cutoff,
			     std::vector<std::vector<size_t>> & res_adet,
			     std::vector<std::vector<size_t>> & res_bdet,
			     RealT & total_weight) {

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    size_t adet_begin = 0;
    size_t adet_end   = adet.size();
    size_t bdet_begin = 0;
    size_t bdet_end   = bdet.size();
    int a_comm_size = static_cast<int>(adet_comm_size);
    int b_comm_size = static_cast<int>(bdet_comm_size);
    assert( mpi_size == a_comm_size * b_comm_size );
    int a_comm_rank = mpi_rank / b_comm_size;
    int b_comm_rank = mpi_rank % b_comm_size;

    get_mpi_range(a_comm_size,a_comm_rank,adet_begin,adet_end);
    get_mpi_range(b_comm_size,b_comm_rank,bdet_begin,bdet_end);

    size_t adet_size = adet_end - adet_begin;
    size_t bdet_size = bdet_end - bdet_begin;
    std::vector<size_t> adet_count(adet.size(),0);
    std::vector<size_t> bdet_count(bdet.size(),0);
    RealT total_weight_local = 0.0;
    for(size_t ia=adet_begin; ia < adet_end; ia++) {
      for(size_t ib=bdet_begin; ib < bdet_end; ib++) {
	size_t idx = (ia - adet_begin) * bdet_size + ib - bdet_begin;
	RealT weight = SquaredNorm(w[idx]);
	if( weight > cutoff ) {
	  total_weight_local += weight;
	  adet_count[ia]++;
	  bdet_count[ib]++;
	}
      }
    }

    MpiAllreduce(adet_count,MPI_SUM,comm);
    MpiAllreduce(bdet_count,MPI_SUM,comm);
    MPI_Allreduce(&total_weight_local,&total_weight,1,
		  GetMpiType<RealT>::MpiT,MPI_SUM,comm);

    size_t num_one_a = static_cast<size_t>(bitcount(adet[0],bit_length,norb));
    size_t num_one_b = static_cast<size_t>(bitcount(bdet[0],bit_length,norb));
    
    size_t max_single_from_a = num_one_a * (norb-num_one_a);
    size_t max_single_from_b = num_one_b * (norb-num_one_b);

    size_t reduced_adet_size = 0;
    size_t reduced_bdet_size = 0;
    for(size_t ia=adet_begin; ia < adet_end; ia++) {
      if( adet_count[ia] > 0 ) {
	reduced_adet_size++;
      }
    }
    for(size_t ib=bdet_begin; ib < bdet_end; ib++) {
      if( bdet_count[ib] > 0 ) {
	reduced_bdet_size++;
      }
    }

    size_t max_single_adet = max_single_from_a * reduced_adet_size + reduced_adet_size;
    size_t max_single_bdet = max_single_from_a * reduced_bdet_size + reduced_bdet_size;

    std::vector<std::vector<size_t>> new_adet_local(max_single_adet);
    std::vector<std::vector<size_t>> new_bdet_local(max_single_bdet);

    size_t ia_count = 0;
    for(size_t ia=adet_begin; ia < adet_end; ia++) {
      if( adet_count[ia] > 0 ) {
	new_adet_local[ia_count++] = adet[ia];
      }
    }

    std::vector<std::vector<size_t>> hdet_ex(max_single_from_a);
    std::vector<int> open_adet(norb-num_one_a);
    std::vector<int> closed_adet(num_one_a);
    for(size_t ia=0; ia < reduced_adet_size; ia++) {
      int nc = getOpenClosed(new_adet_local[ia],bit_length,norb,open_adet,closed_adet);
      size_t numc = static_cast<size_t>(nc);
      single_from_hdet(new_adet_local[ia],bit_length,norb,numc,open_adet,closed_adet,hdet_ex);
      for(size_t k=0; k < hdet_ex.size(); k++) {
	new_adet_local[ia_count++] = hdet_ex[k];
      }
    }

    size_t ib_count = 0;
    for(size_t ib=bdet_begin; ib < bdet_end; ib++) {
      if( bdet_count[ib] > 0 ) {
	new_bdet_local[ib_count++] = bdet[ib];
      }
    }
    hdet_ex.resize(max_single_from_b);
    std::vector<int> open_bdet(norb-num_one_b);
    std::vector<int> closed_bdet(num_one_b);
    for(size_t ib=0; ib < reduced_bdet_size; ib++) {
      int nc = getOpenClosed(new_bdet_local[ib],bit_length,norb,open_bdet,closed_bdet);
      size_t numc = static_cast<size_t>(nc);
      single_from_hdet(new_bdet_local[ib],bit_length,norb,numc,open_bdet,closed_bdet,hdet_ex);
      for(size_t k=0; k < hdet_ex.size(); k++) {
	new_bdet_local[ib_count++] = hdet_ex[k];
      }
    }

    MPI_Comm adet_comm;
    MPI_Comm bdet_comm;
    
    MPI_Comm_split(comm,b_comm_rank,a_comm_rank,&adet_comm);
    MPI_Comm_split(comm,a_comm_rank,b_comm_rank,&bdet_comm);

    std::vector<std::vector<std::vector<size_t>>> temp_adet(adet_comm_size);
    std::vector<std::vector<std::vector<size_t>>> temp_bdet(bdet_comm_size);

    for(int rank=0; rank < a_comm_size; rank++) {
      if( rank == a_comm_rank ) {
	temp_adet[rank] = new_adet_local;
      }
      MpiBcast(temp_adet[rank],rank,adet_comm);
    }

    for(int rank=0; rank < b_comm_size; rank++) {
      if( rank == b_comm_rank ) {
	temp_bdet[rank] = new_bdet_local;
      }
      MpiBcast(temp_bdet[rank],rank,bdet_comm);
    }
    
    size_t res_adet_size = 0;
    size_t res_bdet_size = 0;
    for(size_t rank=0; rank < adet_comm_size; rank++) {
      res_adet_size += temp_adet[rank].size();
    }
    for(size_t rank=0; rank < bdet_comm_size; rank++) {
      res_bdet_size += temp_bdet[rank].size();
    }
    res_adet.resize(res_adet_size);
    res_bdet.resize(res_bdet_size);
    size_t res_adet_count=0;
    size_t res_bdet_count=0;
    for(size_t rank=0; rank < adet_comm_size; rank++) {
      for(size_t k=0; k < temp_adet[rank].size(); k++) {
	res_adet[res_adet_count++] = temp_adet[rank][k];
      }
    }
    for(size_t rank=0; rank < bdet_comm_size; rank++) {
      for(size_t k=0; k < temp_bdet[rank].size(); k++) {
	res_bdet[res_bdet_count++] = temp_bdet[rank][k];
      }
    }
    sort_bitarray(res_adet);
    sort_bitarray(res_bdet);

    MPI_Comm_free(&adet_comm);
    MPI_Comm_free(&bdet_comm);
  }


  /**
     Function to construct the complete half-determinant subspace
     generated by single-particle excitations of all half-determinants,
     without amplitude-based selection.
     This function assumes carryover bitstrings as input.
   */
  void SinglesExtendHalfdets(const std::vector<std::vector<size_t>> & adet,
			     const std::vector<std::vector<size_t>> & bdet,
			     size_t bit_length,
			     size_t norb,
			     const size_t adet_comm_size,
			     const size_t bdet_comm_size,
			     MPI_Comm comm,
			     std::vector<std::vector<size_t>> & res_adet,
			     std::vector<std::vector<size_t>> & res_bdet) {
    
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    size_t adet_begin = 0;
    size_t adet_end   = adet.size();
    size_t bdet_begin = 0;
    size_t bdet_end   = bdet.size();
    int a_comm_size = static_cast<int>(adet_comm_size);
    int b_comm_size = static_cast<int>(bdet_comm_size);
    assert( mpi_size == a_comm_size * b_comm_size );
    int a_comm_rank = mpi_rank / b_comm_size;
    int b_comm_rank = mpi_rank % b_comm_size;

    get_mpi_range(a_comm_size,a_comm_rank,adet_begin,adet_end);
    get_mpi_range(b_comm_size,b_comm_rank,bdet_begin,bdet_end);

    size_t adet_size = adet_end - adet_begin;
    size_t bdet_size = bdet_end - bdet_begin;

    size_t num_one_a = static_cast<size_t>(bitcount(adet[0],bit_length,norb));
    size_t num_one_b = static_cast<size_t>(bitcount(bdet[0],bit_length,norb));
    
    size_t max_single_from_a = num_one_a * (norb-num_one_a);
    size_t max_single_from_b = num_one_b * (norb-num_one_b);

    size_t max_single_adet = max_single_from_a * adet_size + adet_size;
    size_t max_single_bdet = max_single_from_a * bdet_size + bdet_size;

    std::vector<std::vector<size_t>> new_adet_local(max_single_adet);
    std::vector<std::vector<size_t>> new_bdet_local(max_single_bdet);

    size_t ia_count = 0;
    for(size_t ia=adet_begin; ia < adet_end; ia++) {
      new_adet_local[ia_count++] = adet[ia];
    }
    std::vector<std::vector<size_t>> hdet_ex(max_single_from_a);
    std::vector<int> open_adet(norb-num_one_a);
    std::vector<int> closed_adet(num_one_a);
    for(size_t ia=0; ia < adet_size; ia++) {
      int nc = getOpenClosed(new_adet_local[ia],bit_length,norb,open_adet,closed_adet);
      size_t numc = static_cast<size_t>(nc);
      single_from_hdet(new_adet_local[ia],bit_length,norb,numc,open_adet,closed_adet,hdet_ex);
      for(size_t k=0; k < hdet_ex.size(); k++) {
	new_adet_local[ia_count++] = hdet_ex[k];
      }
    }

    size_t ib_count = 0;
    for(size_t ib=bdet_begin; ib < bdet_end; ib++) {
      new_bdet_local[ib_count++] = bdet[ib];
    }
    hdet_ex.resize(max_single_from_b);
    std::vector<int> open_bdet(norb-num_one_a);
    std::vector<int> closed_bdet(num_one_a);
    for(size_t ib=0; ib < bdet_size; ib++) {
      int nc = getOpenClosed(new_bdet_local[ib],bit_length,norb,open_bdet,closed_bdet);
      size_t numc = static_cast<size_t>(nc);
      single_from_hdet(new_bdet_local[ib],bit_length,norb,numc,open_bdet,closed_bdet,hdet_ex);
      for(size_t k=0; k < hdet_ex.size(); k++) {
	new_bdet_local[ib_count++] = hdet_ex[k];
      }
    }

    MPI_Comm adet_comm;
    MPI_Comm bdet_comm;
    MPI_Comm_split(comm,b_comm_rank,a_comm_rank,&adet_comm);
    MPI_Comm_split(comm,a_comm_rank,b_comm_rank,&bdet_comm);

    std::vector<std::vector<std::vector<size_t>>> temp_adet(adet_comm_size);
    std::vector<std::vector<std::vector<size_t>>> temp_bdet(bdet_comm_size);

    for(int rank=0; rank < a_comm_size; rank++) {
      if( a_comm_rank == rank ) {
	temp_adet[rank].resize(new_adet_local.size());
	for(size_t k=0; k < new_adet_local.size(); k++) {
	  temp_adet[rank][k].resize(new_adet_local[k].size());
	  for(size_t l=0; l < new_adet_local[k].size(); l++) {
	    temp_adet[rank][k][l] = new_adet_local[k][l];
	  }
	}
      }
      MpiBcast(temp_adet[rank],rank,adet_comm);
    }

    for(int rank=0; rank < b_comm_size; rank++) {
      if( b_comm_rank == rank ) {
	temp_bdet[rank].resize(new_bdet_local.size());
	for(size_t k=0; k < new_bdet_local.size(); k++) {
	  temp_bdet[rank][k].resize(new_bdet_local[k].size());
	  for(size_t l=0; l < new_bdet_local[k].size(); l++) {
	    temp_bdet[rank][k][l] = new_bdet_local[k][l];
	  }
	}
      }
      MpiBcast(temp_bdet[rank],rank,bdet_comm);
    }
    
    size_t res_adet_size = 0;
    size_t res_bdet_size = 0;
    for(size_t rank=0; rank < adet_comm_size; rank++) {
      res_adet_size += temp_adet[rank].size();
    }
    for(size_t rank=0; rank < bdet_comm_size; rank++) {
      res_bdet_size += temp_bdet[rank].size();
    }
    res_adet.resize(res_adet_size);
    res_bdet.resize(res_bdet_size);
    size_t res_adet_count=0;
    size_t res_bdet_count=0;
    for(size_t rank=0; rank < adet_comm_size; rank++) {
      for(size_t k=0; k < temp_adet[rank].size(); k++) {
	res_adet[res_adet_count++] = temp_adet[rank][k];
      }
    }
    for(size_t rank=0; rank < bdet_comm_size; rank++) {
      for(size_t k=0; k < temp_bdet[rank].size(); k++) {
	res_bdet[res_bdet_count++] = temp_bdet[rank][k];
      }
    }
    sort_bitarray(res_adet);
    sort_bitarray(res_bdet);

    MPI_Comm_free(&adet_comm);
    MPI_Comm_free(&bdet_comm);
  }
  
}

#endif
