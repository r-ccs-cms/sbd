/**
@file sbd/chemistry/tpb/qcham.h
@brief Function to make Hamiltonian for parallel taskers for distributed basis
*/
#ifndef SBD_CHEMISTRY_GDB_QCHAM_H
#define SBD_CHEMISTRY_GDB_QCHAM_H

namespace sbd {
  namespace gdb {

    template <typename ElemT, typename DetsContainer>
    void makeQCham(const DetsContainer & det,
		   size_t bit_length,
		   size_t norb,
		   const DetIndexMap & idxmap,
		   const std::vector<ExcitationLookup> & exidx,
		   ElemT & I0,
		   oneInt<ElemT> & I1,
		   twoInt<ElemT> & I2,
		   std::vector<ElemT> & hii,
		   std::vector<std::vector<size_t*>> & ih,
		   std::vector<std::vector<size_t*>> & jh,
		   std::vector<std::vector<ElemT*>> & hij,
		   std::vector<std::vector<size_t>> & len,
		   std::vector<int> & slide,
		   std::vector<std::vector<size_t>> & storage_int,
		   std::vector<std::vector<ElemT>> & storage_elem,
		   MPI_Comm h_comm,
		   MPI_Comm b_comm,
		   MPI_Comm t_comm) {

      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      DetIndexMap tidxmap;
      DetsContainer tdet;

      if( exidx[0].slide != 0 ) {
	sbd::gdb::MpiSlide(idxmap,tidxmap,-exidx[0].slide,b_comm);
	sbd::MpiSlide(det,tdet,-exidx[0].slide,b_comm);
      } else {
	sbd::gdb::DetIndexMapCopy(idxmap,tidxmap);
	tdet = det;
      }

      size_t num_threads = 1;
      hii.resize(det.size());

#pragma omp parallel
      {
	num_threads = omp_get_num_threads();
#pragma omp for
	for(size_t k=0; k < det.size(); k++) {
	  if( (k%mpi_size_h) != mpi_rank_h ) continue;
	  hii[k] = ZeroExcite(det[k],bit_length,norb,I0,I1,I2);
	}
      }

      ih.resize(exidx.size());
      jh.resize(exidx.size());
      hij.resize(exidx.size());
      len.resize(exidx.size());
      storage_int.resize(exidx.size());
      storage_elem.resize(exidx.size());
      slide.resize(exidx.size());
      for(size_t task=0; task < exidx.size(); task++) {
	slide[task] = exidx[task].slide;
      }

      for(size_t task=0; task < exidx.size(); task++) {
	
	ih[task].resize(num_threads);
	jh[task].resize(num_threads);
	hij[task].resize(num_threads);
	len[task].resize(num_threads);
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t ia_begin = thread_id;
	  size_t ia_end   = idxmap.AdetToDetLen.size();
	  len[task][thread_id] = 0;
	  
	  for(size_t ia=ia_begin; ia < ia_end; ia+=num_threads) {
	    for(size_t ib=0; ib < idxmap.AdetToDetLen[ia]; ib++) {
	      size_t iast = ia;
	      size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	      size_t idet = idxmap.AdetToDetSM[ia][ib];
	      if( idet % mpi_size_h != mpi_rank_h ) continue;
	      
	      if( exidx[task].SelfFromBdetLen[ibst] != 0 ) {
		size_t jbst = exidx[task].SelfFromBdetSM[ibst][0];

		// single-slpha excitations
		for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		  auto itidxa = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
						 &tidxmap.BdetToAdetSM[jbst][0]
						 +tidxmap.BdetToDetLen[jbst],
						 jast);
		  if( itidxa != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst]) ) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itidxa);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    len[task][thread_id]++;
		  }
		}

		// double-beta excitations
		for(size_t ja=0; ja < exidx[task].DoublesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].DoublesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst])) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    len[task][thread_id]++;
		  }
		}

		
	      }// if there is same beta string

	      // alpha-beta two-particle excitations
	      for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		size_t start_idx = 0;
		size_t end_idx = tidxmap.AdetToDetLen[jast];
		size_t SinglesFromBLen = exidx[task].SinglesFromBdetLen[ibst];
		for(size_t k=0; k < SinglesFromBLen; k++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][k];
		  if( start_idx >= end_idx ) break;
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0]+start_idx,
					      &tidxmap.AdetToBdetSM[jast][0]+end_idx,
					      jbst);
		  size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		  start_idx = idxb;
		  if( idxb < end_idx ) {
		    if( tidxmap.AdetToBdetSM[jast][idxb] == jbst ) {
		      len[task][thread_id]++;
		    }
		  }
		}
	      }

	      if( exidx[task].SelfFromAdetLen[iast] != 0 ) {

		size_t jast = exidx[task].SelfFromAdetSM[iast][0];
		
		// single beta excitations
		for(size_t jb=0; jb < exidx[task].SinglesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxa = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxa] != jbst ) continue;
		    len[task][thread_id]++;
		  }
		}

		// double beta excitations
		for(size_t jb=0; jb < exidx[task].DoublesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].DoublesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    len[task][thread_id]++;
		  }
		}
		
	      } // if there is same alpha string
	    } // end loop for beta-string index in bra-side
	  } // end loop for alpha-string index in bra-side
	} // end omp threading loop

	// the size for each thread in task is determined
	size_t total_size = 0;
	for(size_t tid=0; tid < num_threads; tid++) {
	  total_size += len[task][tid];
	}

	storage_int[task].resize(2*total_size);
	storage_elem[task].resize(total_size);

	size_t * begin_int = storage_int[task].data();
	ElemT * begin_elem = storage_elem[task].data();
	size_t count_int = 0;
	size_t count_elem = 0;

	for(size_t tid=0; tid < num_threads; tid++) {
	  ih[task][tid] = begin_int + count_int;
	  count_int += len[task][tid];
	  jh[task][tid] = begin_int + count_int;
	  count_int += len[task][tid];
	}
	for(size_t tid=0; tid < num_threads; tid++) {
	  hij[task][tid] = begin_elem + count_elem;
	  count_elem += len[task][tid];
	}

	// perform actual matrix memorization
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t ia_begin = thread_id;
	  size_t ia_end   = idxmap.AdetToDetLen.size();
	  size_t address = 0;
	  
	  for(size_t ia=ia_begin; ia < ia_end; ia+=num_threads) {
	    for(size_t ib=0; ib < idxmap.AdetToDetLen[ia]; ib++) {
	      size_t iast = ia;
	      size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	      size_t idet = idxmap.AdetToDetSM[ia][ib];
	      if( idet % mpi_size_h != mpi_rank_h ) continue;
	      
	      if( exidx[task].SelfFromBdetLen[ibst] != 0 ) {
		size_t jbst = exidx[task].SelfFromBdetSM[ibst][0];

		// single-slpha excitations
		for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		  auto itidxa = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
						 &tidxmap.BdetToAdetSM[jbst][0]
						 +tidxmap.BdetToDetLen[jbst],
						 jast);
		  if( itidxa != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst]) ) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itidxa);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
					  I1,I2);
		    ih[task][thread_id][address] = idet;
		    jh[task][thread_id][address] = jdet;
		    hij[task][thread_id][address] = eij;
		    address++;
		  }
		}

		// double-beta excitations
		for(size_t ja=0; ja < exidx[task].DoublesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].DoublesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst])) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    ElemT eij = TwoExcite(det[idet],bit_length,
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+0],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+1],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+2],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+3],
					  I1,I2);
		    ih[task][thread_id][address] = idet;
		    jh[task][thread_id][address] = jdet;
		    hij[task][thread_id][address] = eij;
		    address++;
		  }
		}
		
	      }// if there is same beta string

	      // alpha-beta two-particle excitations
	      for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		size_t start_idx = 0;
		size_t end_idx = tidxmap.AdetToDetLen[jast];
		size_t SinglesFromBLen = exidx[task].SinglesFromBdetLen[ibst];
		for(size_t k=0; k < SinglesFromBLen; k++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][k];
		  if( start_idx >= end_idx ) break;
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0]+start_idx,
					      &tidxmap.AdetToBdetSM[jast][0]+end_idx,
					      jbst);
		  size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		  start_idx = idxb;
		  if( idxb < end_idx ) {
		    if( tidxmap.AdetToBdetSM[jast][idxb] == jbst ) {
		      size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		      ElemT eij = TwoExcite(det[idet],bit_length,
					    exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
					    exidx[task].SinglesBdetCrAnSM[ibst][2*k+0],
					    exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
					    exidx[task].SinglesBdetCrAnSM[ibst][2*k+1],
					    I1,I2);
		      ih[task][thread_id][address] = idet;
		      jh[task][thread_id][address] = jdet;
		      hij[task][thread_id][address] = eij;
		      address++;
		    }
		  }
		}
	      }

	      if( exidx[task].SelfFromAdetLen[iast] != 0 ) {

		size_t jast = exidx[task].SelfFromAdetSM[iast][0];
		
		// single beta excitations
		for(size_t jb=0; jb < exidx[task].SinglesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxa = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxa] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxa];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+0],
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+1],
					  I1,I2);
		    ih[task][thread_id][address] = idet;
		    jh[task][thread_id][address] = jdet;
		    hij[task][thread_id][address] = eij;
		    address++;
		  }
		}

		// double beta excitations
		for(size_t jb=0; jb < exidx[task].DoublesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].DoublesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		    ElemT eij = TwoExcite(det[idet],bit_length,
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+0],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+1],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+2],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+3],
					  I1,I2);
		    ih[task][thread_id][address] = idet;
		    jh[task][thread_id][address] = jdet;
		    hij[task][thread_id][address] = eij;
		    address++;
		  }
		}
		
	      } // if there is same alpha string
	    } // end loop for beta-string index in bra-side
	  } // end loop for alpha-string index in bra-side
	} // end omp threading loop

	if( task != exidx.size()-1 ) {
	  int shift = exidx[task].slide-exidx[task+1].slide;
	  DetsContainer rdet;
	  DetIndexMap ridxmap;
	  std::swap(rdet,tdet);
	  DetIndexMapCopy(tidxmap,ridxmap);
	  sbd::MpiSlide(rdet,tdet,shift,b_comm);
	  sbd::gdb::MpiSlide(ridxmap,tidxmap,shift,b_comm);
	}
	
      } // end task loop
    }


    template <typename ElemT, typename DetsContainer>
    void makeQChamDiagTerms(const DetsContainer & det,
			    size_t bit_length,
			    size_t norb,
			    const DetIndexMap & idxmap,
			    const std::vector<ExcitationLookup> & exidx,
			    ElemT & I0,
			    oneInt<ElemT> & I1,
			    twoInt<ElemT> & I2,
			    std::vector<ElemT> & hii,
			    MPI_Comm h_comm,
			    MPI_Comm b_comm,
			    MPI_Comm t_comm) {

      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      hii.resize(det.size(),ElemT(0.0));
#pragma omp parallel for
      for(size_t k=0; k < det.size(); k++) {
	if( (k%mpi_size_h) != mpi_rank_h ) continue;
	hii[k] = ZeroExcite(det[k],bit_length,norb,I0,I1,I2);
      }
    }
    
  } // end namespace gdb
} // end namespace sbd

#endif
