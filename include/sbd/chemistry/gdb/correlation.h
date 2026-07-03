/**
@file sbd/chemistry/tpb/correlation.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_GDB_CORRELATION_H
#define SBD_CHEMISTRY_GDB_CORRELATION_H

namespace sbd {
  namespace gdb {

    /**
     */
    template <typename ElemT, typename DetsContainer>
    void Correlation(const std::vector<ElemT> & w,
		     const DetsContainer & det,
		     size_t bit_length,
		     size_t norb,
		     const DetIndexMap idxmap,
		     const std::vector<ExcitationLookup> & exidx,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm,
		     std::vector<std::vector<ElemT>> & onebody,
		     std::vector<std::vector<ElemT>> & twobody) {
      onebody.resize(2);
      twobody.resize(4);
      onebody[0].resize(norb*norb,ElemT(0.0));
      onebody[1].resize(norb*norb,ElemT(0.0));
      twobody[0].resize(norb*norb*norb*norb,ElemT(0.0));
      twobody[1].resize(norb*norb*norb*norb,ElemT(0.0));
      twobody[2].resize(norb*norb*norb*norb,ElemT(0.0));
      twobody[3].resize(norb*norb*norb*norb,ElemT(0.0));

      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);


      std::vector<ElemT> tw;
      std::vector<ElemT> rw;
      tw.reserve(det.size());
      rw.reserve(det.size());

      DetIndexMap tidxmap;
      // std::vector<std::vector<size_t>> tdet;

      if( exidx[0].slide != 0 ) {
	sbd::gdb::MpiSlide(idxmap,tidxmap,-exidx[0].slide,b_comm);
	sbd::MpiSlide(w,tw,-exidx[0].slide,b_comm);
	// sbd::MpiSlide(det,tdet,-exidx[0].slide,b_comm);
      } else {
	DetIndexMapCopy(idxmap,tidxmap);
	tw = w;
	// tdet = det;
      }

      size_t num_threads = 1;
      num_threads = omp_get_max_threads();
      
      std::vector<std::vector<std::vector<ElemT>>> onebody_t(num_threads,onebody);
      std::vector<std::vector<std::vector<ElemT>>> twobody_t(num_threads,twobody);

      if( mpi_rank_t == 0 ) {
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t i_start = thread_id;
	  size_t i_end   = tw.size();
	  for(size_t i=i_start; i < i_end; i+=num_threads) {
	    if( ( i % mpi_size_h ) == mpi_rank_h ) {
	      ZeroDiffCorrelation(det[i],w[i],bit_length,norb,
				  onebody_t[thread_id],
				  twobody_t[thread_id]);
	    }
	  }
	}
      }

      for(size_t task=0; task < exidx.size(); task++) {
#pragma omp parallel
	{
	  size_t thread_id = omp_get_thread_num();
	  size_t ia_begin = thread_id;
	  size_t ia_end   = idxmap.AdetToDetLen.size();
	  std::vector<int> c(2,0);
	  std::vector<int> d(2,0);

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
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst]) ) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    OneDiffCorrelation(det[idet],w[idet],tw[jdet],bit_length,norb,
				       exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
				       exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
				       onebody_t[thread_id],twobody_t[thread_id]);
		    /*
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		    */
		  }
		}

		// double alpha excitations
		for(size_t ja=0; ja < exidx[task].DoublesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].DoublesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst]) ) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    TwoDiffCorrelation(det[idet],w[idet],tw[jdet],bit_length,norb,
				       exidx[task].DoublesAdetCrAnSM[ia][4*ja+0],
				       exidx[task].DoublesAdetCrAnSM[ia][4*ja+1],
				       exidx[task].DoublesAdetCrAnSM[ia][4*ja+2],
				       exidx[task].DoublesAdetCrAnSM[ia][4*ja+3],
				       onebody_t[thread_id],twobody_t[thread_id]);
		    /*
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		    */
		  }
		}
		/*
		for(size_t ja=0; ja < tidxmap.BdetToDetLen[jbst]; ja++) {
		  size_t jdet = tidxmap.BdetToDetSM[jbst][ja];
		  if( difference(det[idet],tdet[jdet],bit_length,2*norb) == 4 ) {
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		  }
		}
		*/
		
	      } // if there is same beta string

	      // alpha-beta two-particle excitations
	      for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		size_t start_idx = 0;
		size_t end_idx = tidxmap.AdetToDetLen[jast];
		size_t SinglesFromBLen = exidx[task].SinglesFromBdetLen[ibst];
		// size_t maxAtoB = tidxmap.AdetToBdetSM[jast][end_idx-1];
		for(size_t k=0; k < SinglesFromBLen; k++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][k];
		  if( start_idx >= end_idx ) break;
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0]+start_idx,
					      &tidxmap.AdetToBdetSM[jast][0]+end_idx,
					      jbst);
		  size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		  start_idx = idxb;
		  if( idxb < end_idx ) {
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		    TwoDiffCorrelation(det[idet],w[idet],tw[jdet],bit_length,norb,
				       exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
				       exidx[task].SinglesBdetCrAnSM[ibst][2*k+0],
				       exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
				       exidx[task].SinglesBdetCrAnSM[ibst][2*k+1],
				       onebody_t[thread_id],twobody_t[thread_id]);
		      /*
		      CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					      bit_length,norb,c,d,
					      onebody_t[thread_id],twobody_t[thread_id]);
		      */
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
		    OneDiffCorrelation(det[idet],w[idet],tw[jdet],bit_length,norb,
				       exidx[task].SinglesBdetCrAnSM[ibst][2*jb+0],
				       exidx[task].SinglesBdetCrAnSM[ibst][2*jb+1],
				       onebody_t[thread_id],twobody_t[thread_id]);
		    /*
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		    */
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
		    size_t idxa = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxa] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxa];
		    TwoDiffCorrelation(det[idet],w[idet],tw[jdet],bit_length,norb,
				       exidx[task].DoublesBdetCrAnSM[ibst][4*jb+0],
				       exidx[task].DoublesBdetCrAnSM[ibst][4*jb+1],
				       exidx[task].DoublesBdetCrAnSM[ibst][4*jb+2],
				       exidx[task].DoublesBdetCrAnSM[ibst][4*jb+3],
				       onebody_t[thread_id],twobody_t[thread_id]);
		    /*
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		    */
		  }
		}
		/*
		for(size_t jb = 0; jb < tidxmap.AdetToDetLen[jast]; jb++) {
		  size_t jdet = tidxmap.AdetToDetSM[jast][jb];
		  if( difference(det[idet],tdet[jdet],bit_length,2*norb) == 4 ) {
		    CorrelationTermAddition(det[idet],tdet[jdet],w[idet],tw[jdet],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		  }
		}
		*/
		
	      } // if there are same alpha
	    } // corresponding beta string loop for bra-side basis
	  } // alpha-based loop for bra-side basis
	} // end omp parallel

	if( task != exidx.size()-1 ) {
	  int slide = exidx[task].slide-exidx[task+1].slide;
	  rw.resize(tw.size());
	  std::memcpy(rw.data(),tw.data(),tw.size()*sizeof(ElemT));
	  DetIndexMap ridxmap;
	  DetIndexMapCopy(tidxmap,ridxmap);
	  sbd::MpiSlide(rw,tw,slide,b_comm);
	  sbd::gdb::MpiSlide(ridxmap,tidxmap,slide,b_comm);
	  // std::vector<std::vector<size_t>> rdet;
	  // std::swap(rdet,tdet);
	  // sbd::MpiSlide(rdet,tdet,slide,b_comm);
	}
      } // end for(size_t task=0; task < exidx.size(); task++)

      for(size_t tid=0; tid < num_threads; tid++) {
#pragma omp parallel for
	for(size_t i=0; i < norb*norb; i++) {
	  for(size_t s=0; s < onebody.size(); s++) {
	    onebody[s][i] += onebody_t[tid][s][i];
	  }
	}
      }

      for(size_t tid=0; tid < num_threads; tid++) {
#pragma omp parallel for
	for(size_t i=0; i < norb*norb*norb*norb; i++) {
	  for(size_t s=0; s < twobody.size(); s++) {
	    twobody[s][i] += twobody_t[tid][s][i];
	  }
	}
      }

      for(int s=0; s < 2; s++) {
	MpiAllreduce(onebody[s],MPI_SUM,b_comm);
	MpiAllreduce(onebody[s],MPI_SUM,t_comm);
	MpiAllreduce(onebody[s],MPI_SUM,h_comm);
      }
      for(int s=0; s < 4; s++) {
	MpiAllreduce(twobody[s],MPI_SUM,b_comm);
	MpiAllreduce(twobody[s],MPI_SUM,t_comm);
	MpiAllreduce(twobody[s],MPI_SUM,h_comm);
      }

    } // end function Correlation
    
  } // end namespace gdb
} // end namespace sbd

#endif
