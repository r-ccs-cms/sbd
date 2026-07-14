/**
@file sbd/chemistry/tpb/qcham.h
@brief Function to make Hamiltonian for parallel taskers for distributed basis
*/
#ifndef SBD_CHEMISTRY_TPB_QCHAM_H
#define SBD_CHEMISTRY_TPB_QCHAM_H

namespace sbd {

  template <typename ElemT>
  void makeQCham(const det_vector<size_t, det_kind::half> & adets,
		 const det_vector<size_t, det_kind::half> & bdets,
		 const size_t bit_length,
		 const size_t norbs,
		 const std::vector<TaskHelpers> & helper,
		 ElemT & I0,
		 oneInt<ElemT> & I1,
		 twoInt<ElemT> & I2,
		 std::vector<ElemT> & hii,
		 std::vector<std::vector<size_t*>> & ih,
		 std::vector<std::vector<size_t*>> & jh,
		 std::vector<std::vector<ElemT*>> & hij,
		 std::vector<std::vector<size_t>> & len,
		 std::vector<size_t> & tasktype,
		 std::vector<size_t> & adetshift,
		 std::vector<size_t> & bdetshift,
		 std::vector<size_t> & sharedInt,
		 std::vector<ElemT> & sharedElemT,
		 MPI_Comm h_comm,
		 MPI_Comm b_comm,
		 MPI_Comm t_comm) {

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);

    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;
    size_t braAlphaStart = 0;
    size_t braAlphaEnd = 0;
    size_t braBetaStart = 0;
    size_t braBetaEnd = 0;
    if( helper.size() != 0 ) {
      braAlphaSize = helper[0].braAlphaEnd-helper[0].braAlphaStart;
      braBetaSize = helper[0].braBetaEnd-helper[0].braBetaStart;
      braAlphaStart = helper[0].braAlphaStart;
      braAlphaEnd   = helper[0].braAlphaEnd;
      braBetaStart  = helper[0].braBetaStart;
      braBetaEnd    = helper[0].braBetaEnd;
    }
    size_t braSize = braAlphaSize*braBetaSize;
    size_t num_threads = omp_get_max_threads();
    hii.resize(braSize,ElemT(0.0));

#pragma omp parallel
    {
      auto det = DetFromAlphaBeta(adets[0],bdets[0],bit_length,norbs);
      
#pragma omp for
      for(size_t ia=braAlphaStart; ia < braAlphaEnd; ia++) {
	for(size_t ib=braBetaStart; ib < braBetaEnd; ib++) {
	  size_t k = (ia-braAlphaStart)*braBetaSize+ib-braBetaStart;
	  if( (k % mpi_size_h) != mpi_rank_h ) continue;
	  DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,det);
	  hii[k] = ZeroExcite(det,bit_length,norbs,I0,I1,I2);
	}
      }
    }
    
    tasktype.resize(helper.size());
    adetshift.resize(helper.size());
    bdetshift.resize(helper.size());
    len.resize(helper.size(),std::vector<size_t>(num_threads));
    
    // size evaluation
#pragma omp parallel
    {
      size_t thread_id = omp_get_thread_num();
      size_t ia_start = thread_id + braAlphaStart;
      size_t ia_end   = braAlphaEnd;
      for(size_t task = 0; task < helper.size(); task++) {
	len[task][thread_id] = 0;
	for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	  for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
	    size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
	                    +ib-helper[task].braBetaStart;
	    if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	    if ( helper[task].taskType == 0 ) {
	      // two-particle excitation composed of single alpha and single beta
	      len[task][thread_id] += helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]
		                    * helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart];
	    }
	    else if ( helper[task].taskType == 2 ) {
	      // single alpha excitation
	      len[task][thread_id] += helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart];
	      // double alpha excitation
	      len[task][thread_id] += helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart];
	    }
	    else if( helper[task].taskType == 1 ) {
	      // single beta excitation
	      len[task][thread_id] += helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart];
	      // double beta excitation
	      len[task][thread_id] += helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart];
	    }
	  } // end ib loop
	} // end ia loop
	tasktype[task] = helper[task].taskType;
	adetshift[task] = helper[task].adetShift;
	bdetshift[task] = helper[task].bdetShift;
      } // end tasktype loop
    } // end omp parallel for

    size_t sharedElemT_size = 0;
    for(size_t w=0; w < helper.size(); w++) {
      for(size_t t=0; t < num_threads; t++) {
	sharedElemT_size += len[w][t];
      }
    }
    size_t sharedInt_size = 2 * sharedElemT_size;
    
    sharedInt.resize(sharedInt_size);
    sharedElemT.resize(sharedElemT_size);

    size_t * begin_int = sharedInt.data();
    size_t counter_int = 0;

    ih.resize(helper.size(),std::vector<size_t*>(num_threads));
    jh.resize(helper.size(),std::vector<size_t*>(num_threads));
    hij.resize(helper.size(),std::vector<ElemT*>(num_threads));

    for(size_t task=0; task < helper.size(); task++) {
      for(size_t thread=0; thread < num_threads; thread++) {
	ih[task][thread] = begin_int + counter_int;
	counter_int += len[task][thread];
	jh[task][thread] = begin_int + counter_int;
	counter_int += len[task][thread];
      }
    }
    
    ElemT * begin_ElemT = sharedElemT.data();
    size_t counter_ElemT = 0;
    for(size_t task=0; task < helper.size(); task++) {
      for(size_t thread=0; thread < num_threads; thread++) {
	hij[task][thread] = begin_ElemT + counter_ElemT;
	counter_ElemT += len[task][thread];
      }
    }
    using RealT = typename GetRealType<ElemT>::RealT;
    size_t total_memory_size_count = counter_int * sizeof(size_t)
      + counter_ElemT * sizeof(ElemT);
    RealT total_memory_size = 1.0 * total_memory_size_count / 1073741824.0;
    if( mpi_rank_h == 0 ) {
      if( mpi_rank_t == 0 ) {
	if( mpi_rank_b == 0 ) {
	  std::cout << " Memory size for Hamiltonian = "
		    << total_memory_size 
		    << " GiB " << std::endl;
	}
      }
    }
    
    // size_t chunk_size = (helper.braBetaEnd-helper.braBetaStart) / num_threads;
#pragma omp parallel
    {
      size_t thread_id = omp_get_thread_num();
      size_t ia_start = thread_id + braAlphaStart;
      size_t ia_end   = braAlphaEnd;

      auto DetI = DetFromAlphaBeta(adets[0],bdets[0],bit_length,norbs);
      std::vector<int> c(2,0);
      std::vector<int> d(2,0);

      // alpha-beta excitation

      for(size_t task = 0; task < helper.size(); task++) {

	size_t ketAlphaSize = helper[task].ketAlphaEnd-helper[task].ketAlphaStart;
	size_t ketBetaSize  = helper[task].ketBetaEnd-helper[task].ketBetaStart;
	size_t address = 0;
	
	for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	  for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {

	    size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
	                    +ib-helper[task].braBetaStart;
	    if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	  
	    DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);

	    // two-particle excitation composed of single alpha and single beta
	    if( helper[task].taskType == 0 ) {
	      
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		for(size_t k=0; k < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; k++) {
		  size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][k];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                  +jb-helper[task].ketBetaStart;
		  ElemT eij = TwoExcite(DetI,bit_length,
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+0],
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+1],
					I1,I2);
		  ih[task][thread_id][address] = braIdx;
		  jh[task][thread_id][address] = ketIdx;
		  hij[task][thread_id][address] = eij;
		  address++;
		}
	      }
	    }
	    
	    if( helper[task].taskType == 2 ) {
	    
	      // single alpha excitation
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                +ib-helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				      I1,I2);
		ih[task][thread_id][address] = braIdx;
		jh[task][thread_id][address] = ketIdx;
		hij[task][thread_id][address] = eij;
		address++;
	      }
	      // double alpha excitation
	      for(size_t j=0; j < helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].DoublesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		               + ib-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+0],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+1],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+2],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+3],
				      I1,I2);
		ih[task][thread_id][address] = braIdx;
		jh[task][thread_id][address] = ketIdx;
		hij[task][thread_id][address] = eij;
		address++;
	      }
	    }
	    
	    if( helper[task].taskType == 1 ) {
	      // single beta excitation
	      for(size_t j=0; j < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		               + jb - helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+0],
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+1],
				      I1,I2);
		ih[task][thread_id][address] = braIdx;
		jh[task][thread_id][address] = ketIdx;
		hij[task][thread_id][address] = eij;
		address++;
	      }
	      // double beta excitation
	      for(size_t j=0; j < helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].DoublesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		               + jb-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+0],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+1],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+2],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+3],
				      I1,I2);
		ih[task][thread_id][address] = braIdx;
		jh[task][thread_id][address] = ketIdx;
		hij[task][thread_id][address] = eij;
		address++;
	      }
	    }
	  } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	} // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
      } // end for(size_t task=0; task < helper.size(); task++)
    } // end pragma paralell
  } // end function

  template <typename ElemT>
  void makeQChamDiagTerms(const det_vector<size_t, det_kind::half> & adets,
			  const det_vector<size_t, det_kind::half> & bdets,
			  const size_t bit_length,
			  const size_t norbs,
			  const std::vector<TaskHelpers> & helper,
			  ElemT & I0,
			  oneInt<ElemT> & I1,
			  twoInt<ElemT> & I2,
			  std::vector<ElemT> & hii,
			  MPI_Comm h_comm,
			  MPI_Comm b_comm,
			  MPI_Comm t_comm) {

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);

    size_t braAlphaStart = 0;
    size_t braAlphaEnd   = 0;
    size_t braBetaStart  = 0;
    size_t braBetaEnd    = 0;
    if( helper.size() != 0 ) {
      braAlphaStart = helper[0].braAlphaStart;
      braAlphaEnd   = helper[0].braAlphaEnd;
      braBetaStart  = helper[0].braBetaStart;
      braBetaEnd    = helper[0].braBetaEnd;
    }
    size_t braAlphaSize = braAlphaEnd-braAlphaStart;
    size_t braBetaSize = braBetaEnd-braBetaStart;
    size_t braSize = braAlphaSize*braBetaSize;
    
    size_t num_threads = omp_get_max_threads();
    
    hii.resize(braSize,ElemT(0.0));

#pragma omp parallel
    {
      auto det = DetFromAlphaBeta(adets[0],bdets[0],bit_length,norbs);
      
#pragma omp for
      for(size_t ia=braAlphaStart; ia < braAlphaEnd; ia++) {
	for(size_t ib=braBetaStart; ib < braBetaEnd; ib++) {
	  size_t k = (ia-braAlphaStart)*braBetaSize+ib-braBetaStart;
	  if( (k % mpi_size_h) != mpi_rank_h ) continue;
	  DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,det);
	  hii[k] = ZeroExcite(det,bit_length,norbs,I0,I1,I2);
	}
      }
    }
  }
  
  
} // end namespace sbd


#endif // SBD_CHEMISTRY_TPB_QCHAM_H

