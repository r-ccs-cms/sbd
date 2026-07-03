/**
@file sbd/chemistry/basic/determinants.h
@brief Functions to handle the bit-string basis
*/
#ifndef SBD_CHEMISTRY_BASIC_DETERMINANTS_H
#define SBD_CHEMISTRY_BASIC_DETERMINANTS_H

namespace sbd {

#ifdef SBD_TRADMODE

  std::vector<size_t> DetFromAlphaBeta(const std::vector<size_t> & A,
                                       const std::vector<size_t> & B,
                                       const size_t bit_length,
                                       const size_t L) {
    int dsize = (2*L + bit_length - 1) / bit_length;
    std::vector<size_t> D(dsize);
    int fsize = L / bit_length;
    int half = bit_length / 2;
    int extra = L - fsize*bit_length;
    int case1 = (extra > 0) && (extra <= half);
    int case2 = (extra > 0) && (extra >  half);
    for (int j = 0; j < fsize; j++) {
       D[2*j] = 0; D[2*j + 1] = 0;
       for (int i = 0; i < half; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
          if (A[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i));
          if (B[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i + 1));
       }
    }
    if (case1) {
       int j = fsize;
       D[2*j] = 0;
       for (int i = 0; i < extra; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
       }
    }
    if (case2) {
       int j = fsize;
       D[2*j] = 0; D[2*j + 1] = 0;
       for (int i = 0; i < half; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
       }
       for (int i = 0; i < extra - half; i++) {
          if (A[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i));
          if (B[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i + 1));
       }
    }
    return D;
  }

  void DetFromAlphaBeta(const std::vector<size_t> & A,
                        const std::vector<size_t> & B,
                        const size_t bit_length,
                        const size_t L,
                        std::vector<size_t> & D) {
    int fsize = L / bit_length;
    int half = bit_length / 2;
    int extra = L - fsize*bit_length;
    int case1 = (extra > 0) && (extra <= half);
    int case2 = (extra > 0) && (extra >  half);
    for (int j = 0; j < fsize; j++) {
       D[2*j] = 0; D[2*j + 1] = 0;
       for (int i = 0; i < half; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
          if (A[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i));
          if (B[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i + 1));
       }
    }
    if (case1) {
       int j = fsize;
       D[2*j] = 0;
       for (int i = 0; i < extra; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
       }
    }
    if (case2) {
       int j = fsize;
       D[2*j] = 0; D[2*j + 1] = 0;
       for (int i = 0; i < half; i++) {
          if (A[j] & (1L << i)) D[2*j] |= (1L << (2*i));
          if (B[j] & (1L << i)) D[2*j] |= (1L << (2*i + 1));
       }
       for (int i = 0; i < extra - half; i++) {
          if (A[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i));
          if (B[j] & (1L << (i + half))) D[2*j + 1] |= (1L << (2*i + 1));
       }
    }
  }

#else
  std::vector<size_t> DetFromAlphaBeta(const std::vector<size_t>& A,
				       const std::vector<size_t>& B,
				       const size_t bit_length,
				       const size_t L) {
    size_t D_size = (2*L+bit_length-1)/bit_length;
    std::vector<size_t> D(D_size,0);
    for(size_t i=0; i < L; ++i) {
      size_t block = i / bit_length;
      size_t bit_pos = i % bit_length;
      size_t new_block_A = (2*i) / bit_length;
      size_t new_bit_pos_A = (2*i) % bit_length;
      size_t new_block_B = (2*i+1) / bit_length;
      size_t new_bit_pos_B = (2*i+1) % bit_length;

      if ( A[block] & (size_t(1) << bit_pos) ) {
	D[new_block_A] |= size_t(1) << new_bit_pos_A;
      }
      if( B[block] & (size_t(1) << bit_pos) ) {
	D[new_block_B] |= size_t(1) << new_bit_pos_B;
      }
    }
    return D;
  }

  void DetFromAlphaBeta(const std::vector<size_t> & A,
			const std::vector<size_t> & B,
			const size_t bit_length,
			const size_t L,
			std::vector<size_t> & D) {
    std::fill(D.begin(),D.end(),static_cast<size_t>(0));
    for(size_t i=0; i < L; ++i) {
      size_t block = i / bit_length;
      size_t bit_pos = i % bit_length;
      size_t new_block_A = (2*i) / bit_length;
      size_t new_bit_pos_A = (2*i) % bit_length;
      size_t new_block_B = (2*i+1) / bit_length;
      size_t new_bit_pos_B = (2*i+1) % bit_length;

      if ( A[block] & (size_t(1) << bit_pos) ) {
	D[new_block_A] |= size_t(1) << new_bit_pos_A;
      }
      if( B[block] & (size_t(1) << bit_pos) ) {
	D[new_block_B] |= size_t(1) << new_bit_pos_B;
      }
    }
  }
#endif




  // Set the specified bit (x) in the vector of size_t (bit representation)
  void setocc(std::vector<size_t> & dets, const size_t bit_length, int x, bool y) {
    if (x < 0) {
      throw std::invalid_argument("Bit index cannot be negative");
    }

    size_t block = x / bit_length; // Determine the block
    size_t bit = x % bit_length;   // Determine the bit within the block

    if (block >= dets.size()) {
      throw std::out_of_range("Bit index is out of range for the given vector");
    }

    if (y) {
      dets[block] |= (size_t(1) << bit); // Set the bit to 1
    } else {
      dets[block] &= ~(size_t(1) << bit); // Set the bit to 0
    }
  }

  //
  template<typename DetT>
  inline bool getocc(const DetT & det,
		     const size_t bit_length, int x) {
    size_t index = x / bit_length;
    size_t bit_pos = x % bit_length;
    return (det[index] >> bit_pos) & 1;
  }

  inline int bitcount(const std::vector<size_t> & det,
		      const size_t bit_length,
		      const size_t L) {
    int count = 0;
    size_t full_words = L / bit_length;
    size_t remaining_bits = L % bit_length;
    for (size_t i = 0; i < full_words; ++i) {
      count += __builtin_popcountll(det[i]);  // 64-bit popcount
    }
    if (remaining_bits > 0) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) - 1;
      count += __builtin_popcountll(det[full_words] & mask);
    }
    return count;
  }

  int getClosed(const std::vector<size_t> & det,
		const size_t bit_length,
		const size_t L,
		std::vector<int> & closed) {
    int cindex = 0;
    int csize = bitcount(det,bit_length,L);
    closed.resize(csize);
    for(int i=0; i < L; i++) {
      if( getocc(det, bit_length, i) ) {
	closed.at(cindex)=i;
	cindex++;
      }
    }
    return cindex;
  }

  int getOpenClosed(const std::vector<size_t> & det,
		    const size_t bit_length,
		    const size_t L,
		    std::vector<int> & open,
		    std::vector<int> & closed) {
    int cindex = 0;
    int oindex = 0;
    int csize = bitcount(det,bit_length,L);
    // closed.resize(csize);
    // open.resize(L-csize);
    for(int i=0; i < L; i++) {
      if( getocc(det, bit_length, i) ) {
	closed.at(cindex) = i;
	cindex++;
      } else {
	open.at(oindex) = i;
	oindex++;
      }
    }
    return cindex;
  }

  void parity(const std::vector<size_t> & dets,
	      const size_t bit_length,
	      const int & start, const int & end, double& sgn) {
    if (start > end) {
      throw std::invalid_argument("Start index cannot be greater than end index");
    }

    if (bit_length <= 0 || bit_length > static_cast<size_t>(8 * sizeof(size_t))) {
      throw std::invalid_argument("Invalid bit length");
    }

    size_t blockStart = start / bit_length;
    size_t bitStart = start % bit_length;

    size_t blockEnd = end / bit_length;
    size_t bitEnd = end % bit_length;

    int nonZeroBits = 0; // counter for nonzero bits

    // 1. Count bits in the start block
    if (blockStart == blockEnd) {
        // the case where start and end is same block
        size_t mask = ((size_t(1) << bitEnd) - 1) ^ ((size_t(1) << bitStart) - 1);
        nonZeroBits += __builtin_popcountll(dets[blockStart] & mask);
    } else {
        // 2. Handle the partial bits in the start block
        if (bitStart != 0) {
            size_t mask = ~((size_t(1) << bitStart) - 1); // count after bitStart
            nonZeroBits += __builtin_popcountll(dets[blockStart] & mask);
            blockStart++;
        }

        // 3. Handle full blocks in between
        for (size_t i = blockStart; i < blockEnd; i++) {
            nonZeroBits += __builtin_popcountll(dets[i]);
        }

        // 4. Handle the partial bits in the end block
        if (bitEnd != 0) {
            size_t mask = (size_t(1) << bitEnd) - 1; // count before bitEnd
            nonZeroBits += __builtin_popcountll(dets[blockEnd] & mask);
        }
    }

    // parity estimation
    sgn *= (-2. * (nonZeroBits % 2) + 1);

    // flip sign if start == 1
    if ((dets[start / bit_length] >> (start % bit_length)) & 1) {
        sgn *= -1.;
    }
  }

  // Calculate parity for four particles
  void parity(const std::vector<size_t>& dets,
	      const size_t bit_length,
	      const int i, const int j, const int a, const int b,
	      double& sgn) {
    parity(dets, bit_length, std::min(i,a), std::max(i,a), sgn);
    if( !getocc(dets,bit_length,i) ) {
      throw std::invalid_argument("parity: bit 0 at occupied site");
    }
    if( getocc(dets,bit_length,a) ) {
      throw std::invalid_argument("parity: bit 1 at empty site");
    }
    setocc(const_cast<std::vector<size_t>&>(dets), bit_length, i, false);
    setocc(const_cast<std::vector<size_t>&>(dets), bit_length, a, true);
    parity(dets, bit_length, std::min(j,b), std::max(j,b), sgn);
    setocc(const_cast<std::vector<size_t>&>(dets), bit_length, a, false);
    setocc(const_cast<std::vector<size_t>&>(dets), bit_length, j, true);
  }

  std::vector<size_t> getAlpha(const std::vector<size_t> & det,
			       size_t bit_length,
			       size_t norb) {
    std::vector<size_t> alpha((norb + bit_length - 1) / bit_length, 0);  //

    for (size_t i = 0; i < norb; ++i) {
      bool occ = getocc(det, bit_length, 2 * i);  //
      if (occ) {
	setocc(alpha, bit_length, i, true);  //
      }
    }
    return alpha;
  }

  std::vector<size_t> getBeta(const std::vector<size_t>& det,
			      size_t bit_length,
			      size_t norb) {
    std::vector<size_t> beta((norb + bit_length - 1) / bit_length, 0);

    for (size_t i = 0; i < norb; ++i) {
      bool occ = getocc(det, bit_length, 2 * i + 1);
      if (occ) {
	setocc(beta, bit_length, i, true);
      }
    }
    return beta;
  }

  void getAdet(const std::vector<size_t> & det,
	       size_t bit_length,
	       size_t norb,
	       std::vector<size_t> & adet) {
    size_t adet_size = (norb + bit_length - 1 ) / bit_length;
    if( adet_size != adet.size() ) {
      adet.resize(adet_size);
    }
    std::fill(adet.begin(),adet.end(),static_cast<size_t>(0));
    for (size_t i = 0; i < norb; ++i) {
      bool occ = getocc(det, bit_length, 2 * i);  //
      if (occ) {
	setocc(adet, bit_length, i, true);  //
      }
    }
  }

  void getBdet(const std::vector<size_t>& det,
	       size_t bit_length,
	       size_t norb,
	       std::vector<size_t> & bdet) {
    size_t bdet_size = (norb + bit_length - 1 ) / bit_length;
    if( bdet_size != bdet.size() ) {
      bdet.resize(bdet_size);
    }
    std::fill(bdet.begin(),bdet.end(),static_cast<size_t>(0));
    for (size_t i = 0; i < norb; ++i) {
      bool occ = getocc(det, bit_length, 2 * i + 1);
      if (occ) {
	setocc(bdet, bit_length, i, true);
      }
    }
  }


  template<typename RowA, typename RowB>
  int difference(const RowA & a,
		 const RowB & b,
		 size_t bit_length,
		 size_t L) {
    int count = 0;
    size_t full_words = L / bit_length;
    size_t remaining_bits = L % bit_length;
    // error if sizes of a and b are different
    if (a.size() != b.size()) {
      throw std::invalid_argument("Vectors a and b must have the same size.");
    }
    // full-check part
    for (size_t i = 0; i < full_words; ++i) {
      count += __builtin_popcountll(a[i] ^ b[i]);  // XOR
    }
    // remaining bits
    if (remaining_bits > 0) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) - 1;  //
      count += __builtin_popcountll((a[full_words] ^ b[full_words]) & mask);
    }
    return count;
  }

  template<typename RowA, typename RowB>
  void OrbitalDifference(const RowA & a,
			 const RowB & b,
			 size_t bit_length,
			 size_t L,
			 std::vector<int> & x,
			 std::vector<int> & y) {
    x.clear();
    y.clear();

    size_t full_words = L / bit_length;
    size_t remaining_bits = L % bit_length;

    if (a.size() != b.size()) {
      throw std::invalid_argument("Vectors a and b must have the same size.");
    }

    for (size_t i = 0; i < full_words; ++i) {
      size_t diff_a = a[i] & ~b[i];
      size_t diff_b = b[i] & ~a[i];

      for (size_t bit_pos = 0; bit_pos < bit_length; ++bit_pos) {
	if (diff_a & (static_cast<size_t>(1) << bit_pos)) {
	  x.push_back(i * bit_length + bit_pos);
	}
	if (diff_b & (static_cast<size_t>(1) << bit_pos)) {
	  y.push_back(i * bit_length + bit_pos);
	}
      }
    }

    if (remaining_bits > 0) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) - 1;
      size_t diff_a = (a[full_words] & ~b[full_words]) & mask;
      size_t diff_b = (b[full_words] & ~a[full_words]) & mask;

      for (size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
	if (diff_a & (static_cast<size_t>(1) << bit_pos)) {
	  x.push_back(full_words * bit_length + bit_pos);
	}
	if (diff_b & (static_cast<size_t>(1) << bit_pos)) {
	  y.push_back(full_words * bit_length + bit_pos);
	}
      }
    }
  }

  template <typename ElemT>
  ElemT ZeroExcite(const std::vector<size_t> & det,
		   const size_t bit_length,
		   const size_t L,
		   const ElemT & I0,
		   const oneInt<ElemT> & I1,
		   const twoInt<ElemT> & I2) {
        ElemT energy(0.0);

        for (int i = 0; i < 2 * L; i++) {
            if (getocc(det, bit_length, i)) {
                energy += I1.Value(i, i);
                for (int j = i + 1; j < 2 * L; j++) {
                    if (getocc(det, bit_length, j)) {
                        energy += I2.DirectValue(i, j);
                        if ((i % 2) == (j % 2)) {
                            energy -= I2.ExchangeValue(i, j);
                        }
                    }
                }
            }
        }
        return energy + I0;
  }

  template <typename ElemT>
  ElemT OneExcite(const std::vector<size_t> & det,
		  const size_t bit_length,
		  int & i,
		  int & a,
		  const oneInt<ElemT> & I1,
		  const twoInt<ElemT> & I2) {
        double sgn = 1.0;
        parity(det, bit_length, std::min(i, a), std::max(i, a), sgn);
        ElemT energy = I1.Value(a, i);
        for (int x = 0; x < det.size(); x++) {
            size_t bits = det[x];
            for (int pos = 0; pos < bit_length; pos++) {
                if ((bits & 1ULL) == 1ULL) {
                    int j = x * bit_length + pos;
                    energy += (I2.Value(a, i, j, j) - I2.Value(a, j, j, i));
                }
                bits >>= 1;
            }
        }
        energy *= ElemT(sgn);
        return energy;
  }

  template <typename ElemT>
  ElemT TwoExcite(const std::vector<size_t> & det,
		  const size_t bit_length,
		  int & i,
		  int & j,
		  int & a,
		  int & b,
		  const oneInt<ElemT> & I1,
		  const twoInt<ElemT> & I2) {
    double sgn = 1.0;
    int I = std::min(i,j);
    int J = std::max(i,j);
    int A = std::min(a,b);
    int B = std::max(a,b);
    parity(det,bit_length,std::min(I,A),std::max(I,A),sgn);
    parity(det,bit_length,std::min(J,B),std::max(J,B),sgn);
    if( A > J || B < I ) sgn *= -1.0;
    return ElemT(sgn) * (I2.Value(A,I,B,J)-I2.Value(A,J,B,I));
  }

  template <typename ElemT>
  ElemT Hij(const std::vector<size_t> & DetA,
	    const std::vector<size_t> & DetB,
	    const size_t & bit_length,
	    const size_t & L,
	    const ElemT & I0,
	    const oneInt<ElemT> & I1,
	    const twoInt<ElemT> & I2,
	    size_t & orbDiff) {
    std::vector<int> c;
    std::vector<int> d;
    size_t nc=0;
    size_t nd=0;

    size_t full_words = (2*L) / bit_length;
    size_t remaining_bits = (2*L) % bit_length;

    for(size_t i=0; i < full_words; ++i) {
      size_t diff_c = DetA[i] & ~DetB[i];
      size_t diff_d = DetB[i] & ~DetA[i];
      for(size_t bit_pos=0; bit_pos < bit_length; ++bit_pos) {
	if(diff_c & (static_cast<size_t>(1) << bit_pos)) {
	  c.push_back(i*bit_length+bit_pos);
	  nc++;
	}
	if(diff_d & (static_cast<size_t>(1) << bit_pos)) {
	  d.push_back(i*bit_length+bit_pos);
	  nd++;
	}
      }
    }

    if( remaining_bits > 0 ) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) -1;
      size_t diff_c = (DetA[full_words] & ~DetB[full_words]) & mask;
      size_t diff_d = (DetB[full_words] & ~DetA[full_words]) & mask;
      for(size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
	if( diff_c & (static_cast<size_t>(1) << bit_pos) ) {
	  c.push_back(bit_length*full_words+bit_pos);
	  nc++;
	}
	if( diff_d & (static_cast<size_t>(1) << bit_pos) ) {
	  d.push_back(bit_length*full_words+bit_pos);
	  nd++;
	}
      }
    }

    if( nc == 0 ) {
      orbDiff = static_cast<size_t>(0);
      return ZeroExcite(DetB,bit_length,L,I0,I1,I2);
    } else if ( nc == 1 ) {
      orbDiff = static_cast<size_t>(c[0] * L + d[0]);
      return OneExcite(DetB,bit_length,d[0],c[0],I1,I2);
    } else if ( nc == 2 ) {
      orbDiff = static_cast<size_t>(c[1]*L*L*L+d[1]*L*L+c[0]*L+d[0]);
      return TwoExcite(DetB,bit_length,d[0],d[1],c[0],c[1],I1,I2);
    }
    return ElemT(0.0);
  }

  template <typename ElemT>
  ElemT Hij(const std::vector<size_t> & DetA,
	    const std::vector<size_t> & DetB,
	    const size_t & bit_length,
	    const size_t & L,
	    std::vector<int> & c,
	    std::vector<int> & d,
	    const ElemT & I0,
	    const oneInt<ElemT> & I1,
	    const twoInt<ElemT> & I2,
	    size_t & orbDiff) {
    size_t nc=0;
    size_t nd=0;

    size_t full_words = (2*L) / bit_length;
    size_t remaining_bits = (2*L) % bit_length;

    for(size_t i=0; i < full_words; ++i) {
      size_t diff_c = DetA[i] & ~DetB[i];
      size_t diff_d = DetB[i] & ~DetA[i];
      for(size_t bit_pos=0; bit_pos < bit_length; ++bit_pos) {
	if(diff_c & (static_cast<size_t>(1) << bit_pos)) {
	  c[nc] = i*bit_length+bit_pos;
	  nc++;
	}
	if(diff_d & (static_cast<size_t>(1) << bit_pos)) {
	  d[nd] = i*bit_length+bit_pos;
	  nd++;
	}
      }
    }

    if( remaining_bits > 0 ) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) -1;
      size_t diff_c = (DetA[full_words] & ~DetB[full_words]) & mask;
      size_t diff_d = (DetB[full_words] & ~DetA[full_words]) & mask;
      for(size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
	if( diff_c & (static_cast<size_t>(1) << bit_pos) ) {
	  c[nc] = bit_length*full_words+bit_pos;
	  nc++;
	}
	if( diff_d & (static_cast<size_t>(1) << bit_pos) ) {
	  d[nd] = bit_length*full_words+bit_pos;
	  nd++;
	}
      }
    }

    if( nc == 0 ) {
      orbDiff = static_cast<size_t>(0);
      return ZeroExcite(DetB,bit_length,L,I0,I1,I2);
    } else if ( nc == 1 ) {
      orbDiff = static_cast<size_t>(c[0] * L + d[0]);
      return OneExcite(DetB,bit_length,d[0],c[0],I1,I2);
    } else if ( nc == 2 ) {
      orbDiff = static_cast<size_t>(c[1]*L*L*L+d[1]*L*L+c[0]*L+d[0]);
      return TwoExcite(DetB,bit_length,d[0],d[1],c[0],c[1],I1,I2);
    }
    return ElemT(0.0);
  }

  void ShuffleDet(std::vector<std::vector<size_t>> & det,
		  unsigned int seed) {
    if( det.size() <= 1 ) return;
    std::mt19937 g(seed);
    std::shuffle(det.begin()+1,det.end(),g);
  }



} // end namespace sbd

#endif // end SBD_CHEMISTRY_DETERMINANTS_H
