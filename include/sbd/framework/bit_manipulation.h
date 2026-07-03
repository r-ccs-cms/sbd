/// This is a part of sbd
/**
@file bit_manipulation.h
@brief mathematical tools for bit manipulations
*/

#ifndef SBD_FRAMEWORK_BIT_MANIPULATION_H
#define SBD_FRAMEWORK_BIT_MANIPULATION_H

#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <stdexcept>
#include <bitset>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

#include "mpi.h"
#include "sbd/framework/det_vector.h"

#define SBD_BIT_LENGTH 20

  bool operator < (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();

    assert( a_size == b_size );

    bool res = false;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] < b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] > b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator <= (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size < b_size ) {
      return true;
    } else if ( a_size > b_size ) {
      return false;
    }

    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] < b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] > b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator > (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size > b_size ) {
      return true;
    } else if ( a_size < b_size ) {
      return false;
    }

    bool res = false;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] > b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] < b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator >= (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size > b_size ) {
      return true;
    } else if ( a_size < b_size ) {
      return false;
    }

    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] > b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] < b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator == (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size != b_size ) {
      return false;
    }
    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] != b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

namespace sbd {

  /**
   * @brief Compare two vectors in right-to-left lexicographical order.
   *
   * This function performs a lexicographical comparison starting from the
   * last element of each vector (i.e., from back to front).
   * The vector lengths are assumed to be identical.
   *
   * The ordering is equivalent to:
   *   Compare a[n-1] with b[n-1], then a[n-2] with b[n-2], and so on,
   *   and decide at the first position where they differ.
   *
   * @param a First vector.
   * @param b Second vector.
   * @return true if a is considered smaller than b under this ordering,
   *         false otherwise.
   */
  template <typename VecA, typename VecB>
  inline bool less_from_back(const VecA & a, const VecB & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();

    assert( a_size == b_size );

    bool res = false;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] < b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] > b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  /**
   * @brief Compare two vectors in reverse right-to-left lexicographical order.
   *
   * This function defines the opposite ordering of less_from_back(a, b).
   * It is equivalent to calling less_from_back(b, a).
   *
   * @param a First vector.
   * @param b Second vector.
   * @return true if a is considered greater than b under this ordering,
   *         false otherwise.
   */
  template <typename VecA, typename VecB>
  inline bool greater_from_back(const VecA & a, const VecB & b) {
    return less_from_back(b,a);
  }
  
  /**
     Function for making a string from the bitarray data
     @param[in] config: target bitarray
     @param[in] bit_length: length of the bitstring managed by each size_t
     @param[in] L: number of total bits in the bitstring
   */
  std::string makestring(const std::vector<size_t> & config,
			 size_t bit_length,
			 size_t L) {
    std::string s;
    for(size_t i=L; i > 0; i--) {
      size_t p = (i-1) % bit_length;
      size_t b = (i-1) / bit_length;
      if( config[b] & (static_cast<size_t>(1) << p) ) {
	s += std::string("1").c_str();
      }
      else {
	s += std::string("0").c_str();
      }
    }
    return s;
  }

  /**
     Function for construct the bitarray data from a string
     @param[in] s: target string
     @param[in] bit_length: length of the bitstring managed by each size_t
     @param[in] L: number of total bits in the bitstring
   */
  std::vector<size_t> from_string(const std::string & s,
				  size_t bit_length,
				  size_t L) {
    size_t num_words = (L + bit_length - 1) / bit_length;
    std::vector<size_t> result(num_words, 0);
    for (size_t i = 0; i < L; ++i) {
      char bit = s[L - 1 - i];
      if (bit == '1') {
	size_t p = i % bit_length;
	size_t b = i / bit_length;
	result[b] |= (static_cast<size_t>(1) << p);
      }
    }
    return result;
  }

  /**
     Function for finding a mpi process which manages the target bit string
     @param[in] config: target configuration
     @param[in] config_begin: first element for each mpi
  */
  void mpi_process_search(const std::vector<size_t> & target_config,
			  const std::vector<std::vector<size_t>> & config_begin,
			  const std::vector<std::vector<size_t>> & config_end,
			  int & target_mpi_rank, bool & mpi_exist) {
    size_t mpi_size = config_begin.size();
    mpi_exist = false;
    for(int rank=0; rank < mpi_size; rank++) {
      if( config_begin[rank] <= target_config && target_config < config_end[rank] ) {
	target_mpi_rank = rank;
	mpi_exist = true;
	break;
      }
    }
  }

  /**
     Function for finding the state index of target bit string
     @param[in] target_config: target configuration
     @param[in] config: all configuration managed by target mpi process
     @param[in] index_begin: index of first element managed by target mpi process
     @param[in] index_end: +1 index of last element managed by target mpi process
  */

  void bisection_search(const std::vector<size_t> & target_config,
			const std::vector<std::vector<size_t>> & config,
			const size_t & index_begin,
			const size_t & index_end,
			size_t & index, bool & exist) {
    bool do_bisection = true;
    size_t index_a = index_begin;
    size_t index_b = index_end-1;
    exist = false;
    while( do_bisection ) {
      size_t index_c = (index_a + index_b)/2;
      if( config[index_c] < target_config ) {
	index_a = index_c+1;
      } else if ( config[index_c] > target_config ) {
	index_b = index_c-1;
      } else {
	index = index_c;
	do_bisection = false;
	exist = true;
	return;
      }
      if( (index_b - index_a) < 2 && do_bisection ) {
	do_bisection = false;
	if( config[index_b] == target_config ) {
	  index = index_b;
	  exist = true;
	} else if ( config[index_a] == target_config ) {
	  index = index_a;
	  exist = true;
	} else if ( target_config < config[index_a] ) {
	  index = index_a;
	  exist = false;
	} else if( target_config < config[index_b] ) {
	  index = index_b;
	  exist = false;
	} else {
	  index = index_b+1;
	  exist = false;
	}
      }
    }
  }

  /**
     Function for finding the state index of target bit string
     @param[in] target_config: target configuration
     @param[in] config: all configuration managed by target mpi process
     @param[in] index_begin: index of first element managed by target mpi process
     @param[in] index_end: +1 index of last element managed by target mpi process
  */
  void bisection_search_mpi(const std::vector<size_t> & target_config,
			    const std::vector<std::vector<size_t>> & config,
			    const size_t & index_begin,
			    const size_t & index_end,
			    int target_mpi_rank, size_t & index, bool & exist,
			    MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    if( mpi_rank == target_mpi_rank ) {
      bool do_bisection = true;
      std::vector<size_t> config_a = config[index_begin];
      std::vector<size_t> config_b = config[index_end-1];
      size_t index_a = index_begin;
      size_t index_b = index_end-1;
      while( do_bisection ) {
	size_t index_c = (index_a + index_b)/2;
	std::vector<size_t> config_c = config[index_c-index_begin];
	if( config_c < target_config ) {
	  config_a = config_c;
	  index_a = index_c;
	} else if ( config_c > target_config ) {
	  config_b = config_c;
	  index_b = index_c;
	} else {
	  index = index_c;
	  do_bisection = false;
	  exist = true;
	}
	if( index_b-index_a < 2 ) {
	  do_bisection = false;
	  if( config_b == target_config ) {
	    index = index_b;
	    exist = true;
	  } else if ( config_a == target_config ) {
	    index = index_a;
	    exist = true;
	  } else if ( target_config < config_a ) {
	    index = index_a;
	    exist = false;
	  } else if( target_config < config_b ) {
	    index = index_b;
	    exist = false;
	  } else {
	    index = index_b+1;
	    exist = false;
	  }
	}
      }
    } // end if for mpi process
  } // end void function bisection_search

  /**
     Function for adding 1 to the bitstring
     @param[in/out] a: bitstring
     @param[in] bit_length: length for the bitstring managed by each size_t
   */

  template<typename RowType>
  void bitadvance(RowType& a, int bit_length) {
    size_t x;
    size_t d = (((size_t) 1) << bit_length) - 1;
    size_t v = (size_t) 1;
    for(size_t n=0; n < a.size(); n++) {
      x = a[n]+v;
      a[n] = (x & d);
      v = x >> bit_length;
    }
  }

  /**
     Function for sorting the array of bit strings.
     @param[in/out] a: set of bit strings to be sorted
   */
  void sort_bitarray(std::vector<std::vector<size_t>> & a) {
    /*
    std::sort(a.begin(),a.end(),
	      [](const std::vector<size_t> & x,
		 const std::vector<size_t> & y)
	      { return x < y; });
    */
    std::sort(a.begin(),a.end(),
	      [](const std::vector<size_t> & x,
		 const std::vector<size_t> & y)
	      { return less_from_back(x,y); });
    a.erase(std::unique(a.begin(),a.end()),a.end());
  }

  // Generic sort_from_back: works on any RowType with operator[] returning size_t.
  // Instantiate with size_t* for pointer-into-flat-array rows (swaps 8 bytes, not 24).
  template<typename RowType>
  void sort_from_back_t(std::vector<RowType>& rows, size_t lo, size_t hi, int elem) {
    if (hi - lo <= 1 || elem < 0) return;
    std::sort(rows.begin() + lo, rows.begin() + hi,
              [elem](const RowType& x, const RowType& y) {
                return x[elem] < y[elem];
              });
    if (elem == 0) return;
    size_t run_lo = lo;
    for (size_t i = lo + 1; i <= hi; i++) {
      if (i == hi || rows[i][elem] != rows[run_lo][elem]) {
        if (i - run_lo > 1)
          sort_from_back_t(rows, run_lo, i, elem - 1);
        run_lo = i;
      }
    }
  }

  // det_vector overload: pointer-sort avoids moving non-constructible row objects.
  void sort_bitarray(det_vector<size_t>& a) {
    size_t n = a.size();
    if (n <= 1) return;
    size_t row_len = a.elem_size();
    int elem = static_cast<int>(row_len) - 1;
    std::vector<size_t*> ptrs(n);
    for (size_t i = 0; i < n; i++) ptrs[i] = a[i].data();
    sort_from_back_t(ptrs, 0, n, elem);
    auto end_it = std::unique(ptrs.begin(), ptrs.end(),
        [row_len](const size_t* x, const size_t* y) {
            return std::memcmp(x, y, row_len * sizeof(size_t)) == 0;
        });
    size_t unique_n = static_cast<size_t>(end_it - ptrs.begin());
    det_vector<size_t> result(unique_n);
    for (size_t i = 0; i < unique_n; i++)
        std::memcpy(result[i].data(), ptrs[i], row_len * sizeof(size_t));
    a = std::move(result);
  }



  /**
     Function to redistribute the bit string on each mpi processes.
     @param[in/out] config: set of bit strings to be redistributed
     @param[in/out] config_begin: first bit string for each mpi process.
     @param[in/out] config_end: last bit string for each mpi process.
     @param[in/out] index_begin: first index of bit string for each mpi process.
     @param[in/out] index_end: last index of bit string for each mpi process.
     @param[in] total_bit_length: total bit length represented by `Container`
     @param[in] bit_length: length of bit string managed by each `size_t`
     @param[in] comm: mpi communicator
   */
  template<typename Container>
  void mpi_redistribution(Container & config,
			  Container & config_begin,
			  Container & config_end,
			  std::vector<size_t> & index_begin,
			  std::vector<size_t> & index_end,
			  size_t total_bit_length,
			  size_t bit_length,
			  MPI_Comm comm) {

    int mpi_master = 0;
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    size_t config_length = (total_bit_length+bit_length-1) / bit_length;
    std::vector<size_t> send_config_size(mpi_size,0);
    std::vector<size_t> config_size(mpi_size,0);
    send_config_size[mpi_rank] = config.size();
    MPI_Allreduce(send_config_size.data(),config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
    size_t total_size = 0;
    for(int rank=0; rank < mpi_size; rank++) {
      total_size += config_size[rank];
    }
    std::fill(index_begin.begin(),index_begin.end(),static_cast<size_t>(0));
    std::fill(index_end.begin(),index_end.end(),static_cast<size_t>(0));
    for(int rank=1; rank < mpi_size; rank++) {
      index_begin[rank] = index_begin[rank-1] + config_size[rank-1];
    }
    for(int rank=0; rank < mpi_size; rank++) {
      index_end[rank] = index_begin[rank] + config_size[rank];
    }
    std::vector<size_t> i_begin(mpi_size,0);
    std::vector<size_t> i_end(mpi_size,total_size);
    for(size_t rank=0; rank < mpi_size; rank++) {
      get_mpi_range(mpi_size,rank,i_begin[rank],i_end[rank]);
    }

    size_t new_config_size = i_end[mpi_rank]-i_begin[mpi_rank];
    Container new_config;
    for(int recv_rank=0; recv_rank < mpi_size; recv_rank++) {
      // find i_begin and i_end mpi process
      int mpi_rank_begin = 0;
      int mpi_rank_end   = mpi_size-1;
      for(int rank=0; rank < mpi_size; rank++) {
	if ( ( index_begin[rank] <= i_begin[recv_rank] )
	     && ( i_begin[recv_rank] < index_end[rank] ) ) {
	  mpi_rank_begin = rank;
	  break;
	}
      }
      for(int rank=mpi_size-1; rank > -1; rank--) {
	if ( ( index_begin[rank] < i_end[recv_rank] ) && ( i_end[recv_rank] <= index_end[rank] ) ) {
	  mpi_rank_end = rank;
	  break;
	}
      }
      for(int send_rank=mpi_rank_begin; send_rank <= mpi_rank_end; send_rank++) {
	if( mpi_rank == send_rank ) {
	  size_t ii_min = std::max(i_begin[recv_rank],index_begin[send_rank]);
	  size_t ii_max = std::min(i_end[recv_rank],  index_end[send_rank]);
	  Container config_transfer;
	  if( (ii_max - ii_min) > 0 ) {
	    config_transfer.resize(ii_max-ii_min);
	    for(size_t i=ii_min; i < ii_max; i++) {
	      config_transfer[i-ii_min] = config[i-index_begin[send_rank]];
	    }
	  } else {
	    config_transfer.resize(0);
	  }
	  if( send_rank != recv_rank ) {
	    MpiSend(config_transfer,recv_rank,comm);
	  } else {
	    new_config.insert(new_config.end(),config_transfer.begin(),config_transfer.end());
	  }
	}
	if( ( mpi_rank == recv_rank ) && ( send_rank != recv_rank ) ) {
	  Container config_transfer;
	  MpiRecv(config_transfer,send_rank,comm);
	  new_config.insert(new_config.end(),config_transfer.begin(),config_transfer.end());
	}
      } // end for(int send_rank=mpi_rank_begin; send_rank <= mpi_rank_end; send_rank++)
      MPI_Barrier(comm);
    } // end for(int recv_rank=0; recv_rank < mpi_size; recv_rank++)

    sort_bitarray(new_config);
    config = std::move(new_config);

    std::fill(send_config_size.begin(),send_config_size.end(),static_cast<size_t>(0));
    std::fill(config_size.begin(),config_size.end(),static_cast<size_t>(0));
    send_config_size[mpi_rank] = config.size();
    MPI_Allreduce(send_config_size.data(),config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
    index_begin[0] = 0;
    for(int rank=1; rank < mpi_size; rank++) {
      index_begin[rank] = index_begin[rank-1] + config_size[rank-1];
    }
    for(int rank=0; rank < mpi_size; rank++) {
      index_end[rank] = index_begin[rank] + config_size[rank];
    }

    for(int rank=0; rank < mpi_size; rank++) {
      if( rank == mpi_rank ) {
	config_begin[rank] = config[0];
      } else {
	config_begin[rank].resize(config_length);
      }

      MPI_Bcast(config_begin[rank].data(),config_length,SBD_MPI_SIZE_T,rank,comm);
    }
    for(int rank=0; rank < mpi_size-1; rank++) {
      config_end[rank] = config_begin[rank+1];
    }
    if( mpi_rank == mpi_size-1 ) {
      config_end[mpi_rank] = config[config.size()-1];
      bitadvance(config_end[mpi_rank],bit_length);
    } else {
      config_end[mpi_size-1].resize(config_length);
    }
    MPI_Bcast(config_end[mpi_size-1].data(),config_length,SBD_MPI_SIZE_T,mpi_size-1,comm);

  }

  /**
     Function to sort the set of bit string distributed on the all mpi processes.
     @param[in/out] config: set of bit strings to be redistributed
     @param[in/out] config_begin: first bit string for each mpi process.
     @param[in/out] config_end: last bit string for each mpi process.
     @param[in/out] index_begin: first index of bit string for each mpi process.
     @param[in/out] index_end: last index of bit string for each mpi process.
     @param[in] total_bit_length: total bit length represented by the Container element type
     @param[in] bit_length: length of bit string managed by each `size_t`
     @param[in] comm: mpi communicator
   */
  template<typename Container>
  void mpi_sort_bitarray(Container & config,
			 Container & config_begin,
			 Container & config_end,
			 std::vector<size_t> & index_begin,
			 std::vector<size_t> & index_end,
			 size_t total_bit_length,
			 size_t bit_length,
			 MPI_Comm comm,
			 int id=0) {
    int mpi_master = 0;
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    size_t bit_size = (total_bit_length + bit_length - 1 ) / bit_length;
    if( mpi_size == 1 ) {
      sort_bitarray(config);
      config_begin[mpi_rank] = config[0];
      config_end[mpi_rank] = config[config.size()-1];
      index_begin[mpi_rank] = 0;
      index_end[mpi_rank] = config.size()-1;
    } else {
      // mpi_size = 2 -> mpi_size_half = 1, mpi_rank / mpi_size_half = 0 or 1
      // mpi_size = 3 -> mpi_size_half = 2, mpi_rank / mpi_size_half = 0 or 1
      int mpi_ierr;
      int mpi_size_half = (mpi_size / 2) + ( mpi_size % 2 );
      int mpi_color = mpi_rank / mpi_size_half;
      int mpi_key   = ( mpi_rank < mpi_size_half ) ? ( mpi_rank ) : ( mpi_rank - mpi_size_half );
      // int mpi_key mpi_rank % mpi_size_half;
      MPI_Comm new_comm;
      int new_id = id + mpi_color * mpi_size_half;
      MPI_Comm_split(comm,mpi_color,mpi_key,&new_comm);
      mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,new_comm,new_id);
      MPI_Comm_free(&new_comm);

#ifdef SBD_DEBUG_BIT
      // for(int rank=0; rank < mpi_size; rank++) {
      // if( mpi_rank == rank ) {
      if( mpi_rank == 0 ) {
	std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		  << ": start parallel merge sort for size " << mpi_size
		  << " (" << mpi_size_half << "," << mpi_size-mpi_size_half << ")" << std::endl;
      }
      // MPI_Barrier(comm);
      // }
      sleep(1);
#endif

      // determine the range to be managed by each node
      int mpi_size_a = mpi_size_half;
      int mpi_size_b = mpi_size - mpi_size_half;
      int mpi_master_a = 0;
      int mpi_master_b = mpi_size_half;
      std::vector<std::vector<size_t>> config_begin_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_middle_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_end_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_begin_b(mpi_size_b,std::vector<size_t>(bit_size));
      std::vector<std::vector<size_t>> config_end_b(mpi_size_b,std::vector<size_t>(bit_size));
      std::vector<size_t> config_end_a_end(bit_size,0);
      std::vector<size_t> config_end_b_end(bit_size,0);
      if( mpi_color == 0 ) {
	config_begin_a[mpi_key].assign(config[0].begin(), config[0].end());
	config_middle_a[mpi_key].assign(config[config.size()/2].begin(), config[config.size()/2].end());
	if( mpi_key == mpi_size_a - 1 ) {
	  config_end_a_end.assign(config[config.size()-1].begin(), config[config.size()-1].end());
	  bitadvance(config_end_a_end,bit_length);
	}
      }
      if( mpi_color == 1 ) {
	config_begin_b[mpi_key].assign(config[0].begin(), config[0].end());
	if( mpi_key == mpi_size_b - 1 ) {
	  config_end_b_end.assign(config[config.size()-1].begin(), config[config.size()-1].end());
	  bitadvance(config_end_b_end,bit_length);
	}
      }
      for(int rank=0; rank < mpi_size_a; rank++) {
	MPI_Bcast(config_begin_a[rank].data(),bit_size,SBD_MPI_SIZE_T,rank,comm);
	MPI_Bcast(config_middle_a[rank].data(),bit_size,SBD_MPI_SIZE_T,rank,comm);
      }
      for(int rank=0; rank < mpi_size_b; rank++) {
	MPI_Bcast(config_begin_b[rank].data(),bit_size,SBD_MPI_SIZE_T,mpi_master_b+rank,comm);
      }
      MPI_Bcast(config_end_a_end.data(),bit_size,SBD_MPI_SIZE_T,mpi_master_b-1,comm);
      MPI_Bcast(config_end_b_end.data(),bit_size,SBD_MPI_SIZE_T,mpi_size-1,comm);

      for(int rank=0; rank < mpi_size_a-1; rank++) {
	config_end_a[rank] = config_begin_a[rank+1];
      }
      config_end_a[mpi_size_a-1] = config_end_a_end;
      for(int rank=0; rank < mpi_size_b-1; rank++) {
	config_end_b[rank] = config_begin_b[rank+1];
      }
      config_end_b[mpi_size_b-1] = config_end_b_end;

      if( config_end_a_end > config_begin_b[0] ) {

#ifdef SBD_DEBUG_BIT
	MPI_Barrier(comm);
	// for(int rank=0; rank < mpi_size; rank++) {
	// if( mpi_rank == rank ) {
	if( mpi_rank == 0 ) {
	  std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		      << ": end config_begin and config_end set up, and enter case necessary to merge " << std::endl;
	}
	// MPI_Barrier(comm);
	// }
	sleep(1);
#endif

	for(int rank=0; rank < mpi_size_half; rank++) {
	  if( 2*rank == mpi_size-1 ) {
	    config_begin[2*rank] = config_begin_a[rank];
	    config_end[2*rank] = config_end_a[rank];
	  } else if( 2*rank < mpi_size ) {
	    config_begin[2*rank] = config_begin_a[rank];
	    config_end[2*rank]   = config_middle_a[rank];
	  }
	  if( 2*rank+1 < mpi_size ) {
	    config_begin[2*rank+1] = config_middle_a[rank];
	    config_end[2*rank+1] = config_end_a[rank];
	  }
	}
	if( config_begin_b[0] < config_begin[0] ) {
	  config_begin[0] = config_begin_b[0];
	}
	if( config_end[mpi_size-1] < config_end_b_end ) {
	  config_end[mpi_size-1] = config_end_b_end;
	}

#ifdef SBD_DEBUG_BIT
	MPI_Barrier(comm);
	// for(int rank=0; rank < mpi_size; rank++) {
	//  if( mpi_rank == rank ) {
	if( mpi_rank == 0 ) {
	  std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		    << ": complete range definition:";
	  for(int rank_t=0; rank_t < mpi_size; rank_t++) {
	    std::cout << " [" << makestring(config_begin[rank_t],bit_length,total_bit_length)
		      << ","  << makestring(config_end[rank_t],bit_length,total_bit_length) << ")";
	  }
	  std::cout << std::endl;
	}
	// MPI_Barrier(comm);
	// }
	sleep(1);
#endif

	Container new_config_b;
	for(int r_rank=0; r_rank < mpi_size; r_rank++) {
	  for(int s_rank=0; s_rank < mpi_size_b; s_rank++) {

	    if( ( config_begin[r_rank] < config_end_b[s_rank] )
	     && ( config_begin_b[s_rank] < config_end[r_rank] ) ) {

	      if( (s_rank + mpi_master_b) == mpi_rank ) {

		size_t i_begin = 0;
		size_t i_end   = config.size();
		if( config.size() > 0 ) {
		  if( config[0] < config_begin[r_rank] ) {
		    auto itb = std::lower_bound(config.begin(),config.end(),
						config_begin[r_rank],
						[](const auto & lhs,
						   const auto & rhs) {
						  return lhs < rhs;
						});
		    i_begin = static_cast<size_t>(std::distance(config.begin(),itb));
		  }
		  if( config_end[r_rank] <= config[config.size()-1] ) {
		    auto ite = std::lower_bound(config.begin(),config.end(),
						config_end[r_rank],
						[](const auto & lhs,
						   const auto & rhs) {
						  return lhs < rhs;
						});
		    i_end = static_cast<size_t>(std::distance(config.begin(),ite));
		  }
		}

#ifdef SBD_DEBUG_BIT
		std::cout << " rank = " << mpi_rank << " id " << id << ": send data [" << i_begin << "," << i_end
			  << ") from " << s_rank + mpi_master_b << " to " << r_rank
			  << " because [" << makestring(config_begin[r_rank],bit_length,total_bit_length)
			  << " < " << makestring(config_end_b[s_rank],bit_length,total_bit_length)
			  << "] = " << ( config_begin[r_rank] < config_end_b[s_rank] )
			  << " and [" << makestring(config_begin_b[s_rank],bit_length,total_bit_length)
			  << " < " << makestring(config_end[r_rank],bit_length,total_bit_length)
			  << "] = " << ( config_begin_b[s_rank] < config_end[r_rank] ) << std::endl;
#endif

		Container config_transfer;
		size_t transfer_size = i_end-i_begin;
		if( i_end-i_begin > 0 ) {
		  config_transfer.resize(transfer_size);
		}
		for(size_t i = i_begin; i < i_end; i++) {
		  config_transfer[i-i_begin] = config[i];
		}
		if( s_rank+mpi_master_b != r_rank ) {
		  MpiSend(config_transfer,r_rank,comm);
		} else {
		  new_config_b.insert(new_config_b.end(),config_transfer.begin(),config_transfer.end());
		}

	      }
	      if( ( r_rank == mpi_rank ) && ( s_rank+mpi_master_b != r_rank ) ) {

		Container config_transfer;
		MpiRecv(config_transfer,s_rank+mpi_master_b,comm);
		new_config_b.insert(new_config_b.end(),config_transfer.begin(),config_transfer.end());

	      }
	    }
	  }
	} // end send configs from b-block

	sort_bitarray(new_config_b);

	Container new_config_a;
	for(int s_rank=0; s_rank < mpi_size_a; s_rank++) {
	  int r_rank = 2 * s_rank;
	  if( r_rank < mpi_size ) {
	    if( mpi_rank == s_rank ) {
	      size_t i_begin = 0;
	      size_t i_end = config.size();
	      if( r_rank == mpi_size-1 ) {
		i_begin = 0;
		i_end = config.size();
	      } else {
		i_begin = 0;
		i_end = config.size()/2;
	      }
	      Container config_transfer;
	      config_transfer.resize(0);
	      size_t transfer_size = i_end-i_begin;
	      if( transfer_size > 0 ) {
		config_transfer.resize(transfer_size);
	      }
	      for(size_t i=i_begin; i < i_end; i++) {
		config_transfer[i-i_begin] = config[i];
	      }
	      if( s_rank != r_rank ) {
		MpiSend(config_transfer,r_rank,comm);
	      } else {
		new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	      }
	    }
	    if( mpi_rank == r_rank && s_rank != r_rank ) {
	      Container config_transfer;
	      MpiRecv(config_transfer,s_rank,comm);
	      new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	    }
	  }
	  r_rank = 2*s_rank+1;
	  if( r_rank < mpi_size ) {
	    if( mpi_rank == s_rank ) {
	      size_t i_begin = config.size()/2;
	      size_t i_end = config.size();
	      Container config_transfer;
	      config_transfer.resize(0);
	      size_t transfer_size = i_end-i_begin;
	      if( transfer_size > 0 ) {
		config_transfer.resize(transfer_size);
	      }
	      for(size_t i=i_begin; i < i_end; i++) {
		config_transfer[i-i_begin] = config[i];
	      }
	      if( s_rank != r_rank ) {
		MpiSend(config_transfer,r_rank,comm);
	      } else {
		new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	      }
	    }
	    if( ( mpi_rank == r_rank ) && ( s_rank != r_rank ) ) {
	      Container config_transfer;
	      MpiRecv(config_transfer,s_rank,comm);
	      new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	    }
	  }
	} // end send configs from a-block

	sort_bitarray(new_config_a);

	// Sorted config from sorted a-configs and sorted b-configs (O(N) method)
	size_t config_size = new_config_a.size() + new_config_b.size();
	config.resize(0);
	config.reserve(config_size);

	std::set_union(new_config_a.begin(),new_config_a.end(),
		       new_config_b.begin(),new_config_b.end(),
		       std::back_inserter(config),
		       [](const auto & lhs,
			  const auto & rhs) {
			 return lhs < rhs;
		       });

	std::vector<size_t> new_config_size(mpi_size,0);
	std::vector<size_t> send_new_config_size(mpi_size,0);
	send_new_config_size[mpi_rank] = config.size();
	MPI_Allreduce(send_new_config_size.data(),new_config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
	index_begin[0] = 0;
	for(int rank=1; rank < mpi_size; rank++) {
	  index_begin[rank] = index_begin[rank-1] + new_config_size[rank-1];
	}
	for(int rank=0; rank < mpi_size; rank++) {
	  index_end[rank] = index_begin[rank]+new_config_size[rank];
	}


#ifdef SBD_DEBUG_BIT
	if( mpi_rank == 0 ) {
	  std::cout << " [config_begin, config_end) at rank " << mpi_rank << " id " << id << " before redistribution";
	  for(int p_rank=0; p_rank < mpi_size; p_rank++) {
	    std::cout << " [" << makestring(config_begin[p_rank],bit_length,total_bit_length)
		      << "_" << index_begin[p_rank]
		      << "," << makestring(config_end[p_rank],bit_length,total_bit_length)
		      << "_" << index_end[p_rank] << ")";
	  }
	  std::cout << std::endl;
	}
	MPI_Barrier(comm);
	usleep(500000);
	for(int rank=0; rank < mpi_size; rank++) {
	  if( mpi_rank == rank ) {
	    std::cout << " config at rank " << mpi_rank << " id " << id << " before redistribution = [";
	    for(size_t k=0; k < config.size(); k++) {
	      std::cout << ( (k==0) ? "" : "," ) << makestring(config[k],bit_length,total_bit_length);
	    }
	    std::cout << "]" << std::endl;
	  }
	  MPI_Barrier(comm);
	  usleep(100000);
	}
#endif

	mpi_redistribution(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,comm);

#ifdef SBD_DEBUG_BIT
	if( mpi_rank == 0 ) {
	  std::cout << " [config_begin, config_end) at rank " << mpi_rank << " id " << id << " after redistribution";
	  for(int p_rank=0; p_rank < mpi_size; p_rank++) {
	    std::cout << " [" << makestring(config_begin[p_rank],bit_length,total_bit_length)
		      << "_" << index_begin[p_rank]
		      << "," << makestring(config_end[p_rank],bit_length,total_bit_length)
		      << "_" << index_end[p_rank] << ")";
	  }
	  std::cout << std::endl;
	}
	MPI_Barrier(comm);
	usleep(500000);
	for(int rank=0; rank < mpi_size; rank++) {
	  if( mpi_rank == rank ) {
	    std::cout << " config at rank " << mpi_rank << " id " << id << " before redistribution = [";
	    for(size_t k=0; k < config.size(); k++) {
	      std::cout << ( (k==0) ? "" : "," ) << makestring(config[k],bit_length,total_bit_length);
	    }
	    std::cout << "]" << std::endl;
	  }
	  MPI_Barrier(comm);
	  usleep(100000);
	}
#endif


      } else {

	for(int rank=0; rank < mpi_size_a; rank++) {
	  config_begin[rank] = config_begin_a[rank];
	  config_end[rank] = config_end_a[rank];
	}
	for(int rank=0; rank < mpi_size_b; rank++) {
	  config_begin[rank+mpi_master_b] = config_begin_b[rank];
	  config_end[rank+mpi_master_b] = config_end_b[rank];
	}

	std::vector<size_t> new_config_size(mpi_size,0);
	std::vector<size_t> send_new_config_size(mpi_size,0);
	send_new_config_size[mpi_rank] = config.size();
	MPI_Allreduce(send_new_config_size.data(),new_config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
	index_begin[0] = 0;
	for(int rank=1; rank < mpi_size; rank++) {
	  index_begin[rank] = index_begin[rank-1] + new_config_size[rank-1];
	}
	for(int rank=0; rank < mpi_size; rank++) {
	  index_end[rank] = index_begin[rank]+new_config_size[rank];
	}

	mpi_redistribution(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,comm);

      } // if( config_end_a_end > config_begin_b_begin ) to skip case where it is already sorted.
    }
  }


  /**
     Function to change the length of bit string managed by each `size_t`
     @param[in] bit_length_a: the input length of bit string managed by each `size_t`
     @param[in/out] b: target bit string
     @param[in] bit_length_b: target length of bit string managed by each `size_t`
   */
  void change_bitlength(size_t bit_length_a, std::vector<size_t> & b, size_t bit_length_b) {
    std::vector<size_t> a = b;
    size_t a_size = a.size();
    size_t total_bit_length_a = a_size * bit_length_a;
    size_t b_size = total_bit_length_a / bit_length_b;
    if( total_bit_length_a % bit_length_b != 0 ) {
      b_size++;
    }
    b.resize(b_size);

    size_t a_max_order = bit_length_a;
    size_t min_order = 0;
    size_t b_max_order = bit_length_b;
    size_t maxbit_b = (((size_t) 1) << bit_length_b) - 1;

    size_t v_a = a[0];
    size_t i_a = 0;
    for(size_t i=0; i < b_size; i++) {
      while ( a_max_order < b_max_order ) {
	i_a++;
	if( i_a < a_size ) {
	  v_a += (a[i_a] << (a_max_order - min_order));
	}
	a_max_order += bit_length_a;
      }
      b[i] = v_a & maxbit_b;
      v_a = (bit_length_b >= 64) ? 0 : (v_a >> bit_length_b);
      min_order += bit_length_b;
      b_max_order += bit_length_b;
    }
  }

  /**
     Function same with previous change_bitlengh but for set of bitstrings
     @param[in] bit_length_a: the input length of bit string managed by each `size_t`
     @param[in/out] b: target set of bit strings
     @param[in] bit_length_b: target length of bit string managed by each `size_t`
   */
  void change_bitlength(size_t bit_length_a, std::vector<std::vector<size_t>> & b, size_t bit_length_b) {
    std::vector<size_t> a = b[0];
    size_t a_size = a.size();
    size_t total_bit_length_a = a_size * bit_length_a;
    size_t b_size = total_bit_length_a / bit_length_b;
    if( total_bit_length_a % bit_length_b != 0 ) {
      b_size++;
    }
    size_t maxbit_b;
    if (bit_length_b == 64)
      maxbit_b = 0xffffffffffffffff;
    else
      maxbit_b = (((size_t) 1) << bit_length_b) - 1;

    for(size_t k=0; k < b.size(); k++) {
      a = b[k];
      b[k].resize(b_size);

      size_t a_max_order = bit_length_a;
      size_t min_order = 0;
      size_t b_max_order = bit_length_b;
      size_t v_a = a[0];
      size_t i_a = 0;
      for(size_t i=0; i < b_size; i++) {
	while ( a_max_order < b_max_order ) {
	  i_a++;
	  if( i_a < a_size ) {
	    v_a += (a[i_a] << (a_max_order - min_order));
	  }
	  a_max_order += bit_length_a;
	}
	b[k][i] = (v_a & maxbit_b);
	v_a = (bit_length_b >= 64) ? 0 : (v_a >> bit_length_b);
	min_order += bit_length_b;
	b_max_order += bit_length_b;
      }
    }
  }

  /**
     Function to count the sign bit string
     @param[in] w: input bitstring
     @param[in] bit_length: the length of bit string managed by each `size_t` for `w`.
     @param[in] x: position of bit. Actual position corresponds to x + bit_length * r
     @param[in] r: position of bit.
   */
  inline int bit_string_sign_factor(const std::vector<size_t> & w,
				    int bit_length,
				    size_t x,
				    size_t r) {
    int sign = 1;
    size_t size_t_one = 1;
    for(size_t k=0; k < r; k++) {
      for(size_t l=0; l < bit_length; l++) {
	if( (w[k] & (size_t_one << l)) != 0 ) {
	  sign *= -1;
	}
      }
    }
    for(size_t l=0; l < x; l++) {
      if( ( w[r] & (size_t_one << l) ) != 0 ) {
	sign *= -1;
      }
    }
    return sign;
  }


  /**
     Function to change the integer to a string filled by zero from left
     @param[in] i: input integer
     @param[out] s: output string
   */
  void convert_int_to_string(int i, std::string & s) {
    std::string snum = std::to_string(i);
    if( i < 10 )
      {
	std::string zeros("0000000");
	s = zeros + snum;
      }
    else if( i < 100 )
      {
	std::string zeros("000000");
	s = zeros + snum;
      }
    else if( i < 1000 )
      {
	std::string zeros("00000");
	s = zeros + snum;
      }
    else if( i < 10000 )
      {
	std::string zeros("0000");
	s = zeros + snum;
      }
    else if( i < 100000 )
      {
	std::string zeros("000");
	s = zeros + snum;
      }
    else if( i < 1000000 )
      {
	std::string zeros("00");
	s = zeros + snum;
      }
    else if( i < 10000000 )
      {
	std::string zeros("0");
	s = zeros + snum;
      }
    else
      {
	s = snum;
      }
  }

  /**
     Function to get extension of the filename.
     @param[in] path: filename
   */
  inline std::string get_extension(const std::string &path) {
    std::string ext;
    size_t pos1 = path.rfind('.');
    if(pos1 != std::string::npos){
      ext = path.substr(pos1+1, path.size()-pos1);
      std::string::iterator itr = ext.begin();
      while(itr != ext.end()){
	*itr = tolower(*itr);
	itr++;
      }
      itr = ext.end()-1;
      while(itr != ext.begin()){
	if(*itr == 0 || *itr == 32){
	  ext.erase(itr--);
	}
	else{
	  itr--;
	}
      }
    }
    return ext;
  }

  /**
     Function to remove the extension of the filename.
     @param[in] filename: filename
   */
  std::string remove_extension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
  }

  /**
     Function to define the name of binary file with zero-filled integer
     @param[in] i: integer
     @param[in] filename: filename
   */
  std::string get_binary_file_name(int i, const std::string & filename) {
    std::string name = remove_extension(filename);
    std::string label;
    convert_int_to_string(i,label);
    std::string res = name + label + ".dat";
    return res;
  }

  /**
     Function to save the set of bitstring
     @param[in] os: output stream
     @param[in] config: the set of bitstrings
   */
  void SaveConfig(std::ostream & os,
		  std::vector<std::vector<size_t>> & config) {
    size_t size_v = config.size();
    size_t size_b = 0;
    if( config.size() != 0 ) {
      size_b = config[0].size();
    }
    os.write(reinterpret_cast<char *>(&size_v),sizeof(size_t));
    os.write(reinterpret_cast<char *>(&size_b),sizeof(size_t));
    for(size_t n=0; n < size_v; n++) {
      os.write(reinterpret_cast<char *>(config[n].data()),
	       sizeof(size_t)*size_b);
    }
  }

  /**
     Function to load the set of bitstring
     @param[in] is: input stream
     @param[in] config: the set of bitstrings
   */
  void LoadConfig(std::istream & is,
		  std::vector<std::vector<size_t>> & config) {
    size_t size_v;
    size_t size_b;
    is.read(reinterpret_cast<char *>(&size_v),sizeof(size_t));
    is.read(reinterpret_cast<char *>(&size_b),sizeof(size_t));
    config = std::vector<std::vector<size_t>>(size_v,std::vector<size_t>(size_b));
    for(size_t n=0; n < size_v; n++) {
      is.read(reinterpret_cast<char *>(config[n].data()),
	      sizeof(size_t)*size_b);
    }
  }

  /**
     structure for Hash function for sort: XOR base (while it has intersection risk, it is covered by equal)
   */
  struct BitVecHash {
    size_t operator()(const std::vector<size_t> & v) const {
      size_t hash = v.size();
      for (size_t x : v) {
	hash ^= x + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  /**
     Equal function: whether bit is equal or not
   */
  struct BitVecEqual {
    bool operator()(const std::vector<size_t> & a,
		    const std::vector<size_t>& b) const {
      return a == b; // std::vector<size_t>
    }
  };

  /**
     merge sequences D and W to Dn and Wn if D has same entries
     @param[in] D: input set of bit strings
     @param[in] W: weight for each bit string
     @param[out] Dn: unique set of bit strings
     @param[out] Wn: corresponding weights for each bit string
   */
  template<typename T>
  void merge_bit_sequences(
	 const std::vector<std::vector<size_t>> & D,
	 const std::vector<T> & W,
	 std::vector<std::vector<size_t>>& Dn,
	 std::vector<T>& Wn) {

    Dn = D;
    sort_bitarray(Dn);
    Wn.resize(Dn.size(),T(0.0));
    for(size_t i=0; i < D.size(); i++) {
      auto itn = std::lower_bound(Dn.begin(),Dn.end(),
				  D[i],
				  [](const std::vector<size_t> & x,
				     const std::vector<size_t> & y)
				  { return x < y; });
      size_t n = std::distance(Dn.begin(),itn);
      Wn[n] += W[i];
    }


  }

  inline bool equal_bitarray_from_back(const std::vector<size_t> &a,
                                     const std::vector<size_t> &b) {
    return !less_from_back(a, b) && !less_from_back(b, a);
  }

  template<typename DetsContainer>
  void sort_unique_local_bitarray(DetsContainer& dets) {
    sort_bitarray(dets);
  }

  inline int destination_by_splitters(
    const std::vector<size_t> &det,
    const std::vector<std::vector<size_t>> &splitters) {
    auto it = std::upper_bound(splitters.begin(), splitters.end(), det,
			       [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
				 return less_from_back(a, b); });
    return static_cast<int>(it - splitters.begin());
  }

  template<typename DetsContainer>
  void sort_global_bitarray(DetsContainer &dets,
                          MPI_Comm comm) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    sort_unique_local_bitarray(dets);
    size_t local_num_words = 0;
    if (!dets.empty()) {
      local_num_words = dets[0].size();
#ifndef NDEBUG
      for (const auto &det : dets) {
	assert(det.size() == local_num_words);
      }
#endif
    }
    
    size_t num_words = 0;
    MPI_Allreduce(&local_num_words, &num_words, 1,
		  SBD_MPI_SIZE_T, MPI_MAX, comm);
    
    if (num_words == 0) {
      dets.clear();
      return;
    }
    
    if (mpi_size == 1) {
      return;
    }
    
    const size_t local_n = dets.size();
    
    /*
     * Sample selection.
     * Oversampling factor = 4.
     */
    
    const size_t oversample = 4;
    const size_t max_samples =
      std::min(local_n, oversample * static_cast<size_t>(mpi_size - 1));
    std::vector<size_t> local_samples_flat;
    local_samples_flat.reserve(max_samples * num_words);
    if (max_samples > 0) {
      for (size_t s = 0; s < max_samples; ++s) {
	size_t idx = ((s + 1) * local_n) / (max_samples + 1);
	if (idx >= local_n) idx = local_n - 1;
	local_samples_flat.insert(local_samples_flat.end(),
				  dets[idx].begin(), dets[idx].end());
      }
    }
    
    int local_sample_count = static_cast<int>(max_samples);
    std::vector<int> sample_counts;
    if (mpi_rank == 0) {
      sample_counts.resize(mpi_size);
    }
    
    MPI_Gather(&local_sample_count, 1, MPI_INT,
	       sample_counts.data(), 1, MPI_INT,
	       0, comm);
    
    std::vector<int> sample_word_counts;
    std::vector<int> sample_word_displs;
    std::vector<size_t> all_samples_flat;
    
    if (mpi_rank == 0) {
      sample_word_counts.resize(mpi_size);
      sample_word_displs.resize(mpi_size);
      int total_sample_words = 0;
      for (int r = 0; r < mpi_size; ++r) {
	sample_word_counts[r] =
          static_cast<int>(static_cast<size_t>(sample_counts[r]) * num_words);
	sample_word_displs[r] = total_sample_words;
	total_sample_words += sample_word_counts[r];
      }
      all_samples_flat.resize(total_sample_words);
    }
    
    MPI_Gatherv(local_samples_flat.data(),
		static_cast<int>(local_samples_flat.size()),
		SBD_MPI_SIZE_T,
		all_samples_flat.data(),
		sample_word_counts.data(),
		sample_word_displs.data(),
		SBD_MPI_SIZE_T,
		0, comm);
    
    std::vector<std::vector<size_t>> splitters;
    
    if (mpi_rank == 0) {
      
      const size_t total_samples =
        all_samples_flat.size() / num_words;
      
      std::vector<std::vector<size_t>> all_samples(total_samples,
						   std::vector<size_t>(num_words));
      
      for (size_t i = 0; i < total_samples; ++i) {
	std::copy(all_samples_flat.begin() + i * num_words,
		  all_samples_flat.begin() + (i + 1) * num_words,
		  all_samples[i].begin());
      }
      
      sort_unique_local_bitarray(all_samples);
      const size_t ns = all_samples.size();
      const size_t nsplit =
        std::min(static_cast<size_t>(mpi_size - 1), ns);
      
      splitters.resize(nsplit, std::vector<size_t>(num_words));
      for (size_t s = 0; s < nsplit; ++s) {
	size_t idx = ((s + 1) * ns) / mpi_size;
	if (idx >= ns) idx = ns - 1;
	splitters[s] = all_samples[idx];
      }
      sort_unique_local_bitarray(splitters);
    }
    
    int nsplitters = 0;
    if (mpi_rank == 0) {
      nsplitters = static_cast<int>(splitters.size());
    }
    
    MPI_Bcast(&nsplitters, 1, MPI_INT, 0, comm);
    std::vector<size_t> splitters_flat(
		  static_cast<size_t>(nsplitters) * num_words);
    if (mpi_rank == 0) {
      for (int i = 0; i < nsplitters; ++i) {
	std::copy(splitters[i].begin(), splitters[i].end(),
		  splitters_flat.begin() + static_cast<size_t>(i) * num_words);
      }
    }
    
    MPI_Bcast(splitters_flat.data(),
	      static_cast<int>(splitters_flat.size()),
	      SBD_MPI_SIZE_T,
	      0, comm);
    
    if (mpi_rank != 0) {
      splitters.resize(nsplitters, std::vector<size_t>(num_words));
      for (int i = 0; i < nsplitters; ++i) {
	std::copy(splitters_flat.begin() + static_cast<size_t>(i) * num_words,
		  splitters_flat.begin() + static_cast<size_t>(i + 1) * num_words,
		  splitters[i].begin());
      }
    }
    
    /*
      
     * Bucket assignment.
     
     */
    
    std::vector<int> send_counts_det(mpi_size, 0);
    
    for (const auto &det : dets) {
      int dest = destination_by_splitters(det, splitters);
      if (dest >= mpi_size) dest = mpi_size - 1;
      ++send_counts_det[dest];
    }
    
    std::vector<int> send_counts_words(mpi_size, 0);
    std::vector<int> recv_counts_words(mpi_size, 0);
    std::vector<int> send_displs_words(mpi_size, 0);
    std::vector<int> recv_displs_words(mpi_size, 0);
    
    for (int r = 0; r < mpi_size; ++r) {
      send_counts_words[r] =
        static_cast<int>(static_cast<size_t>(send_counts_det[r]) * num_words);
    }
    
    MPI_Alltoall(send_counts_words.data(), 1, MPI_INT,
		 recv_counts_words.data(), 1, MPI_INT,
		 comm);
    
    for (int r = 1; r < mpi_size; ++r) {
      send_displs_words[r] = send_displs_words[r - 1] + send_counts_words[r - 1];
      recv_displs_words[r] = recv_displs_words[r - 1] + recv_counts_words[r - 1];
    }
    
    const int total_send_words =
      std::accumulate(send_counts_words.begin(), send_counts_words.end(), 0);
    
    const int total_recv_words =
      std::accumulate(recv_counts_words.begin(), recv_counts_words.end(), 0);
    
    std::vector<size_t> sendbuf(total_send_words);
    std::vector<size_t> recvbuf(total_recv_words);
    std::vector<int> current_displs = send_displs_words;
    
    for (const auto &det : dets) {
      int dest = destination_by_splitters(det, splitters);
      if (dest >= mpi_size) dest = mpi_size - 1;
      int pos = current_displs[dest];
      std::copy(det.begin(), det.end(), sendbuf.begin() + pos);
      current_displs[dest] += static_cast<int>(num_words);
    }
    
    MPI_Alltoallv(sendbuf.data(),
		  send_counts_words.data(),
		  send_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  recvbuf.data(),
		  recv_counts_words.data(),
		  recv_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  comm);
    
    const size_t recv_n = recvbuf.size() / num_words;

    dets.clear();
    dets.resize(recv_n, std::vector<size_t>(num_words));

    for (size_t i = 0; i < recv_n; ++i) {
      std::copy(recvbuf.begin() + i * num_words,
		recvbuf.begin() + (i + 1) * num_words,
		dets[i].begin());
    }
    sort_unique_local_bitarray(dets);
  }

  inline size_t balanced_begin(const size_t global_n,
			       const int mpi_size,
			       const int rank) {
    const size_t q = global_n / static_cast<size_t>(mpi_size);
    const size_t r = global_n % static_cast<size_t>(mpi_size);
    return q * static_cast<size_t>(rank)
           + std::min(static_cast<size_t>(rank), r);
  }

  inline size_t balanced_end(const size_t global_n,
                           const int mpi_size,
                           const int rank) {
    return balanced_begin(global_n, mpi_size, rank + 1);
  }

  template<typename DetsContainer>
  void redistribution_bitarray(DetsContainer &dets,
			       MPI_Comm comm) {

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const size_t local_n = dets.size();
    size_t local_num_words = 0;
    if (!dets.empty()) {
      local_num_words = dets[0].size();
#ifndef NDEBUG
      for (const auto &det : dets) {
	assert(det.size() == local_num_words);
      }
#endif
    }
    
    size_t num_words = 0;
    
    MPI_Allreduce(&local_num_words, &num_words, 1,
		  SBD_MPI_SIZE_T, MPI_MAX, comm);
    
    if (num_words == 0) {
      dets.clear();
      return;
    }
    
    std::vector<size_t> all_local_n(mpi_size, 0);
    MPI_Allgather(&local_n, 1, SBD_MPI_SIZE_T,
		  all_local_n.data(), 1, SBD_MPI_SIZE_T,
		  comm);
    
    size_t global_n = 0;
    for (size_t n : all_local_n) {
      global_n += n;
    }
    
    if (global_n == 0) {
      dets.clear();
      return;
    }

    size_t local_begin = 0;
    for (int r = 0; r < mpi_rank; ++r) {
      local_begin += all_local_n[r];
    }

    const size_t local_end = local_begin + local_n;
    std::vector<int> send_counts_words(mpi_size, 0);
    std::vector<int> recv_counts_words(mpi_size, 0);
    std::vector<int> send_displs_words(mpi_size, 0);
    std::vector<int> recv_displs_words(mpi_size, 0);
    for (int dest = 0; dest < mpi_size; ++dest) {
      const size_t target_begin = balanced_begin(global_n, mpi_size, dest);
      const size_t target_end   = balanced_end(global_n, mpi_size, dest);
      const size_t overlap_begin = std::max(local_begin, target_begin);
      const size_t overlap_end   = std::min(local_end, target_end);
      if (overlap_begin < overlap_end) {
	const size_t nsend_det = overlap_end - overlap_begin;
	send_counts_words[dest] =
          static_cast<int>(nsend_det * num_words);
      }
    }

    MPI_Alltoall(send_counts_words.data(), 1, MPI_INT,
		 recv_counts_words.data(), 1, MPI_INT,
		 comm);

    for (int r = 1; r < mpi_size; ++r) {
      send_displs_words[r] =
        send_displs_words[r - 1] + send_counts_words[r - 1];
      recv_displs_words[r] =
        recv_displs_words[r - 1] + recv_counts_words[r - 1];
    }

    const int total_send_words =
      std::accumulate(send_counts_words.begin(),
		      send_counts_words.end(), 0);

    const int total_recv_words =
      std::accumulate(recv_counts_words.begin(),
		      recv_counts_words.end(), 0);

    std::vector<size_t> sendbuf(total_send_words);
    std::vector<size_t> recvbuf(total_recv_words);

    for (int dest = 0; dest < mpi_size; ++dest) {
      const size_t target_begin = balanced_begin(global_n, mpi_size, dest);
      const size_t target_end   = balanced_end(global_n, mpi_size, dest);
      const size_t overlap_begin = std::max(local_begin, target_begin);
      const size_t overlap_end   = std::min(local_end, target_end);
      if (overlap_begin >= overlap_end) continue;
      const size_t local_offset = overlap_begin - local_begin;
      const size_t nsend_det = overlap_end - overlap_begin;
      size_t *ptr = sendbuf.data() + send_displs_words[dest];
      for (size_t i = 0; i < nsend_det; ++i) {
	const auto &det = dets[local_offset + i];
	std::copy(det.begin(), det.end(), ptr + i * num_words);
      }
    }

    MPI_Alltoallv(sendbuf.data(),
		  send_counts_words.data(),
		  send_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  recvbuf.data(),
		  recv_counts_words.data(),
		  recv_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  comm);

    const size_t new_local_n =
      static_cast<size_t>(total_recv_words) / num_words;

    DetsContainer new_dets(new_local_n, std::vector<size_t>(num_words));

    for (size_t i = 0; i < new_local_n; ++i) {
      std::copy(recvbuf.begin() + i * num_words,
		recvbuf.begin() + (i + 1) * num_words,
		new_dets[i].begin());
    }
    dets.swap(new_dets);
  }

  template <typename ElemT, typename DetsContainer>
  void redistribution_bitarray(DetsContainer &dets,
			       std::vector<ElemT> &w,
			       MPI_Comm comm) {
    assert(dets.size() == w.size());

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const size_t local_n = dets.size();
    
    size_t local_num_words = 0;
    if (!dets.empty()) {
      local_num_words = dets[0].size();
#ifndef NDEBUG
      for (const auto &det : dets) {
	assert(det.size() == local_num_words);
      }
#endif
    }

    size_t num_words = 0;
    MPI_Allreduce(&local_num_words, &num_words, 1,
		  SBD_MPI_SIZE_T, MPI_MAX, comm);

    if (num_words == 0) {
      dets.clear();
      w.clear();
      return;
    }

    std::vector<size_t> all_local_n(mpi_size, 0);
    MPI_Allgather(&local_n, 1, SBD_MPI_SIZE_T,
		  all_local_n.data(), 1, SBD_MPI_SIZE_T,
		  comm);
    
    size_t global_n = 0;
    for (size_t n : all_local_n) {
      global_n += n;
    }
    
    if (global_n == 0) {
      dets.clear();
      w.clear();
      return;
    }
    
    size_t local_begin = 0;
    for (int r = 0; r < mpi_rank; ++r) {
      local_begin += all_local_n[r];
    }
    const size_t local_end = local_begin + local_n;
    
    std::vector<int> send_counts_det(mpi_size, 0);
    std::vector<int> recv_counts_det(mpi_size, 0);
    std::vector<int> send_displs_det(mpi_size, 0);
    std::vector<int> recv_displs_det(mpi_size, 0);
    
    for (int dest = 0; dest < mpi_size; ++dest) {
      const size_t target_begin = balanced_begin(global_n, mpi_size, dest);
      const size_t target_end   = balanced_end(global_n, mpi_size, dest);
      
      const size_t overlap_begin = std::max(local_begin, target_begin);
      const size_t overlap_end   = std::min(local_end, target_end);
      
      if (overlap_begin < overlap_end) {
	send_counts_det[dest] =
          static_cast<int>(overlap_end - overlap_begin);
      }
    }
    
    MPI_Alltoall(send_counts_det.data(), 1, MPI_INT,
		 recv_counts_det.data(), 1, MPI_INT,
		 comm);
    
    for (int r = 1; r < mpi_size; ++r) {
      send_displs_det[r] =
        send_displs_det[r - 1] + send_counts_det[r - 1];
      recv_displs_det[r] =
        recv_displs_det[r - 1] + recv_counts_det[r - 1];
    }
    
    const int total_send_det =
      std::accumulate(send_counts_det.begin(),
                      send_counts_det.end(), 0);
    
    const int total_recv_det =
      std::accumulate(recv_counts_det.begin(),
                      recv_counts_det.end(), 0);
    
    std::vector<int> send_counts_words(mpi_size, 0);
    std::vector<int> recv_counts_words(mpi_size, 0);
    std::vector<int> send_displs_words(mpi_size, 0);
    std::vector<int> recv_displs_words(mpi_size, 0);
    
    for (int r = 0; r < mpi_size; ++r) {
      send_counts_words[r] =
        static_cast<int>(static_cast<size_t>(send_counts_det[r]) * num_words);
      recv_counts_words[r] =
        static_cast<int>(static_cast<size_t>(recv_counts_det[r]) * num_words);
    }
    
    for (int r = 1; r < mpi_size; ++r) {
      send_displs_words[r] =
        send_displs_words[r - 1] + send_counts_words[r - 1];
      recv_displs_words[r] =
        recv_displs_words[r - 1] + recv_counts_words[r - 1];
    }
    
    const int total_send_words =
      std::accumulate(send_counts_words.begin(),
                      send_counts_words.end(), 0);
    
    const int total_recv_words =
      std::accumulate(recv_counts_words.begin(),
                      recv_counts_words.end(), 0);
    
    std::vector<size_t> sendbuf_det(total_send_words);
    std::vector<size_t> recvbuf_det(total_recv_words);
    
    std::vector<ElemT> sendbuf_w(total_send_det);
    std::vector<ElemT> recvbuf_w(total_recv_det);
    
    for (int dest = 0; dest < mpi_size; ++dest) {
      const size_t target_begin = balanced_begin(global_n, mpi_size, dest);
      const size_t target_end   = balanced_end(global_n, mpi_size, dest);
      
      const size_t overlap_begin = std::max(local_begin, target_begin);
      const size_t overlap_end   = std::min(local_end, target_end);
      
      if (overlap_begin >= overlap_end) continue;
      
      const size_t local_offset = overlap_begin - local_begin;
      const size_t nsend_det = overlap_end - overlap_begin;
      
      size_t *ptr_det =
        sendbuf_det.data() + send_displs_words[dest];
      
      ElemT *ptr_w =
        sendbuf_w.data() + send_displs_det[dest];
      
      for (size_t i = 0; i < nsend_det; ++i) {
	const auto &det = dets[local_offset + i];
	
	std::copy(det.begin(), det.end(),
		  ptr_det + i * num_words);
	
	ptr_w[i] = w[local_offset + i];
      }
    }

    MPI_Alltoallv(sendbuf_det.data(),
		  send_counts_words.data(),
		  send_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  recvbuf_det.data(),
		  recv_counts_words.data(),
		  recv_displs_words.data(),
		  SBD_MPI_SIZE_T,
		  comm);
    
    MPI_Alltoallv(sendbuf_w.data(),
		  send_counts_det.data(),
		  send_displs_det.data(),
		  GetMpiType<ElemT>::MpiT,
		  recvbuf_w.data(),
		  recv_counts_det.data(),
		  recv_displs_det.data(),
		  GetMpiType<ElemT>::MpiT,
		  comm);
    
    const size_t new_local_n = static_cast<size_t>(total_recv_det);
    
    DetsContainer new_dets(new_local_n, std::vector<size_t>(num_words));

    for (size_t i = 0; i < new_local_n; ++i) {
      std::copy(recvbuf_det.begin() + i * num_words,
		recvbuf_det.begin() + (i + 1) * num_words,
		new_dets[i].begin());
    }
    dets.swap(new_dets);
    w.swap(recvbuf_w);
  }


}

#endif
