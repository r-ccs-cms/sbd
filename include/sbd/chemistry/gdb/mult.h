/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_H
#define SBD_CHEMISTRY_GDB_MULT_H

namespace sbd {

  namespace gdb {

    template <typename ElemT, typename DetsContainer>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      size_t bit_length,
	      size_t norb,
	      const DetsContainer & det,
	      const DetIndexMap & idxmap,
	      const std::vector<ExcitationLookup> & exidx,
	      const ElemT & I0,
	      const oneInt<ElemT> & I1,
	      const twoInt<ElemT> & I2,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      std::vector<ElemT> twk;
      std::vector<ElemT> rwk;
      DetIndexMap tidxmap;
      // std::vector<std::vector<size_t>> tdet;

      if( exidx[0].slide != 0 ) {
	sbd::gdb::MpiSlide(idxmap,tidxmap,-exidx[0].slide,b_comm);
	sbd::MpiSlide(wk,twk,-exidx[0].slide,b_comm);
	// sbd::MpiSlide(det,tdet,-exidx[0].slide,b_comm);
      } else {
	DetIndexMapCopy(idxmap,tidxmap);
	twk = wk;
	// tdet = det;
      }

      size_t num_threads = omp_get_max_threads();
      if( mpi_rank_t == 0 ) {
#pragma omp parallel for
	for(size_t i=0; i < twk.size(); i++) {
	  wb[i] += hii[i] * twk[i];
	}
      }

      for(size_t task=0; task < exidx.size(); task++) {
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t ia_begin = thread_id;
	  size_t ia_end = idxmap.AdetToDetLen.size();

	  // alpha-beta excitaiton
	  for(size_t ia=ia_begin; ia < ia_end; ia+=num_threads) {
	    for(size_t ib=0; ib < idxmap.AdetToDetLen[ia]; ib++) {
	      size_t iast = ia;
	      size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	      size_t idet = idxmap.AdetToDetSM[ia][ib];
	      if( idet % mpi_size_h != mpi_rank_h ) continue;

	      // single alpha excitations
	      if( exidx[task].SelfFromBdetLen[ibst] != 0 ) {
		size_t jbst = exidx[task].SelfFromBdetSM[ibst][0];
		for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst])) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
		  }
		}

		// double alpha excitations
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
		    wb[idet] += eij * twk[jdet];
		  }
		}
	      } // if there is same beta string

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
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+end_idx) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( jbst != tidxmap.AdetToBdetSM[jast][idxb] ) continue;
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
			wb[idet] += eij * twk[jdet];
		      }
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
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+0],
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+1],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
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
		    wb[idet] += eij * twk[jdet];
		  }
		}

	      } // if there are same alpha

	    } // corresponding beta string loop for bra-side basis
	  } // alpha-based loop for bra-side basis
	} // end omp threading

	if( task != exidx.size()-1 ) {
	  int slide = exidx[task].slide-exidx[task+1].slide;
	  rwk.resize(twk.size());
	  std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	  DetIndexMap ridxmap;
	  DetIndexMapCopy(tidxmap,ridxmap);
	  sbd::MpiSlide(rwk,twk,slide,b_comm);
	  sbd::gdb::MpiSlide(ridxmap,tidxmap,slide,b_comm);
	}

      } // end task for loop

      MpiAllreduce(wb,MPI_SUM,t_comm);
      MpiAllreduce(wb,MPI_SUM,h_comm);
    } // end function for mult

    template <typename ElemT>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<std::vector<size_t*>> & ih,
	      const std::vector<std::vector<size_t*>> & jh,
	      const std::vector<std::vector<ElemT*>> & hij,
	      const std::vector<std::vector<size_t>> & len,
	      const std::vector<int> & slide,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

      std::vector<ElemT> twk;
      std::vector<ElemT> rwk;
      if( slide.size() != 0 ) {
	if( slide[0] != 0 ) {
	  sbd::MpiSlide(wk,twk,-slide[0],b_comm);
	} else {
	  twk = wk;
	}
      }

      if( mpi_rank_t == 0 ) {
#pragma omp parallel for
	for(size_t i=0; i < twk.size(); i++) {
	  wb[i] += hii[i] * twk[i];
	}
      }

      for(size_t task=0; task < slide.size(); task++) {
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t num_threads = omp_get_num_threads();
	  for(size_t k=0; k < len[task][thread_id]; k++) {
	    wb[ih[task][thread_id][k]] += hij[task][thread_id][k]
	      * twk[jh[task][thread_id][k]];
	  }
	}
	if( task != slide.size() - 1 ) {
	  int bslide = slide[task]-slide[task+1];
	  rwk.resize(twk.size());
	  std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	  sbd::MpiSlide(rwk,twk,bslide,b_comm);
	}
      }
      sbd::MpiAllreduce(wb,MPI_SUM,t_comm);
      sbd::MpiAllreduce(wb,MPI_SUM,h_comm);
    }

  } // end namespace gdb

} // end namespace sbd

#endif
