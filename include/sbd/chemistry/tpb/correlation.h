/**
@file sbd/chemistry/tpb/correlation.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_TPB_CORRELATION_H
#define SBD_CHEMISTRY_TPB_CORRELATION_H

#ifdef USE_OMP_OFFLOAD
#include "../basic/omp_offload.h"
#endif

namespace sbd {


  /**
     Function to evaluate the two-particle correlation functions
   */

#ifdef USE_OMP_OFFLOAD
  template <typename ElemT>
  void Correlation(const std::vector<ElemT> & W,
		   const det_vector<size_t, det_kind::half> & adet,
		   const det_vector<size_t, det_kind::half> & bdet,
		   const size_t bit_length,
		   const size_t norb,
		   const size_t adet_comm_size,
		   const size_t bdet_comm_size,
		   const std::vector<TaskHelpers> & helper,
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

    // make simple arrays for the oneBody and twoBody data
    int norb2 = norb * norb;
    int norb4 = norb2 * norb2;
    double * oneBody = (double *) malloc(2*norb2*sizeof(double));
    double * twoBody = (double *) malloc(4*norb4*sizeof(double));
    for (int i = 0; i < 2*norb2; i++)  oneBody[i] = 0.0;
    for (int i = 0; i < 4*norb4; i++)  twoBody[i] = 0.0;

#pragma omp target enter data map(to: oneBody[0:2*norb2], twoBody[0:4*norb4])

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);
    
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize  = 0;
    if( helper.size() != 0 ) {
      braAlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
      braBetaSize  = helper[0].braBetaEnd  - helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adet.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdet.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    std::vector<ElemT> T;
    std::vector<ElemT> R;
    T.reserve(max_det_size);
    R.reserve(max_det_size);
    if( helper.size() != 0 ) {
      Mpi2dSlide(W,T,adet_comm_size,bdet_comm_size,
		 -helper[0].adetShift,-helper[0].bdetShift,b_comm);
    }

    size_t Tmax = T.size();
    MPI_Allreduce(MPI_IN_PLACE, &Tmax, 1, MPI_UNSIGNED_LONG, MPI_MAX, b_comm);

    size_t braAlphaStart = helper[0].braAlphaStart;
    size_t braBetaStart  = helper[0].braBetaStart;
    size_t braAlphaEnd   = helper[0].braAlphaEnd;
    size_t braBetaEnd    = helper[0].braBetaEnd;

    size_t detSize  = (norb + bit_length - 1) / bit_length;
    size_t AdetSize = (braAlphaEnd - braAlphaStart)*detSize;
    size_t BdetSize = (braBetaEnd - braBetaStart)*detSize;
    //-----------------------------------------------------------------------------------------
    // use simple arrays Adet, Bdet in place of std::vector<std::vector<size_t>> adet, bdet
    //-----------------------------------------------------------------------------------------
    size_t * Adet = (size_t *) malloc(AdetSize*sizeof(size_t));
    size_t * Bdet = (size_t *) malloc(BdetSize*sizeof(size_t));
    for (size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
      const size_t * ptr = adet[ia].data();
      for (size_t d = 0; d < detSize; d++) Adet[d + (ia - braAlphaStart)*detSize] = ptr[d];
    }
    for (size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
      const size_t * ptr = bdet[ib].data();
      for (size_t d = 0; d < detSize; d++) Bdet[d + (ib - braBetaStart)*detSize] = ptr[d];
    }

    // use data pointers instead of std::vector objects
    const ElemT * W_ptr = W.data();
    size_t Wsize = W.size();

    size_t Tsize = T.size();
    T.resize(Tmax);
    ElemT * T_ptr = T.data();

    R.resize(Tmax);
    ElemT * R_ptr = R.data();
    size_t Rsize = Tsize;

#pragma omp target enter data map(to: W_ptr[0:Wsize], T_ptr[0:Tmax]) map(alloc:R_ptr[0:Tmax])

    if( mpi_rank_t == 0 ) {
#pragma omp target teams distribute parallel for collapse(2) map(to: Adet[0:AdetSize], Bdet[0:BdetSize])
      for(size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
        for(size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
          size_t DetT[SBD_MAX_DETSIZE];
          size_t i = (ia - braAlphaStart) * braBetaSize +  ib - braBetaStart;
          if( ( i % mpi_size_h ) == mpi_rank_h ) {
            DetFromAlphaBeta(&Adet[(ia - braAlphaStart)*detSize], &Bdet[(ib - braBetaStart)*detSize], bit_length, norb, DetT);
            ZeroDiffCorrelation(DetT, W_ptr[i], bit_length, norb, oneBody, twoBody);
          }
        }
      }
    }

    free(Adet);
    free(Bdet);

    for(size_t task=0; task < helper.size(); task++) {

      size_t ketAlphaSize = helper[task].ketAlphaEnd - helper[task].ketAlphaStart;
      size_t ketBetaSize  = helper[task].ketBetaEnd - helper[task].ketBetaStart;
      size_t ketAlphaStart = helper[task].ketAlphaStart;
      size_t ketBetaStart  = helper[task].ketBetaStart;
      size_t braAlphaStart = helper[task].braAlphaStart;
      size_t braAlphaEnd   = helper[task].braAlphaEnd;
      size_t braBetaStart  = helper[task].braBetaStart;
      size_t braBetaEnd    = helper[task].braBetaEnd;

      size_t AdetSize = (braAlphaEnd - braAlphaStart)*detSize;
      size_t BdetSize = (braBetaEnd - braBetaStart)*detSize;
      //-----------------------------------------------------------------------------------------
      // use simple arrays Adet, Bdet in place of std::vector<std::vector<size_t>> adet, bdet
      //-----------------------------------------------------------------------------------------
      size_t * Adet = (size_t *) malloc(AdetSize*sizeof(size_t));
      size_t * Bdet = (size_t *) malloc(BdetSize*sizeof(size_t));
      for (size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
        const size_t * ptr = adet[ia].data();
        for (size_t d = 0; d < detSize; d++) Adet[d + (ia - braAlphaStart)*detSize] = ptr[d];
      }
      for (size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
        const size_t * ptr = bdet[ib].data();
        for (size_t d = 0; d < detSize; d++) Bdet[d + (ib - braBetaStart)*detSize] = ptr[d];
      }

      // Get helper array pointers for Alpha
      size_t nAlpha = helper[task].braAlphaEnd - helper[task].braAlphaStart;
      size_t *SinglesFromAlphaLen = helper[task].SinglesFromAlphaLen;
      size_t *DoublesFromAlphaLen = helper[task].DoublesFromAlphaLen;
      const size_t *SinglesFromAlphaOffset = helper[task].SinglesFromAlphaOffset.data();
      const size_t *DoublesFromAlphaOffset = helper[task].DoublesFromAlphaOffset.data();

      // Use pre-flattened arrays from helper construction 
      size_t singles_alpha_total = helper[task].SinglesFromAlpha_flat.size();
      size_t doubles_alpha_total = helper[task].DoublesFromAlpha_flat.size();
      const size_t *SinglesFromAlpha_ptr = helper[task].SinglesFromAlpha_flat.data();
      const size_t *DoublesFromAlpha_ptr = helper[task].DoublesFromAlpha_flat.data();

      // Also corresponding CrAn arrays for Singles/Alpha and Doubles/Alpha
      const int *SinglesAlphaCrAn_ptr = helper[task].SinglesAlphaCrAn_flat.data();
      const int *DoublesAlphaCrAn_ptr = helper[task].DoublesAlphaCrAn_flat.data();
      size_t singles_alpha_cran_total = helper[task].SinglesAlphaCrAn_flat.size();
      size_t doubles_alpha_cran_total = helper[task].DoublesAlphaCrAn_flat.size();

      // Get helper array pointers for Beta
      size_t nBeta = helper[task].braBetaEnd - helper[task].braBetaStart;
      size_t *SinglesFromBetaLen = helper[task].SinglesFromBetaLen;
      size_t *DoublesFromBetaLen = helper[task].DoublesFromBetaLen;
      const size_t *SinglesFromBetaOffset = helper[task].SinglesFromBetaOffset.data();
      const size_t *DoublesFromBetaOffset = helper[task].DoublesFromBetaOffset.data();

      // Use pre-flattened arrays from helper construction
      size_t singles_beta_total = helper[task].SinglesFromBeta_flat.size();
      size_t doubles_beta_total = helper[task].DoublesFromBeta_flat.size();
      const size_t *SinglesFromBeta_ptr = helper[task].SinglesFromBeta_flat.data();
      const size_t *DoublesFromBeta_ptr = helper[task].DoublesFromBeta_flat.data();

      // Also corresponding CrAn arrays for Singles/Beta and Doubles/Beta
      const int *SinglesBetaCrAn_ptr = helper[task].SinglesBetaCrAn_flat.data();
      const int *DoublesBetaCrAn_ptr = helper[task].DoublesBetaCrAn_flat.data();
      size_t singles_beta_cran_total = helper[task].SinglesBetaCrAn_flat.size();
      size_t doubles_beta_cran_total = helper[task].DoublesBetaCrAn_flat.size();

#pragma omp target enter data map(to: Adet[0:AdetSize], Bdet[0:BdetSize])
      
	if( helper[task].taskType == 2 ) { // beta range are same
	    
        #pragma omp target teams distribute parallel for collapse(2)            \
        map(to : SinglesFromAlphaLen[0 : nAlpha],                               \
                 DoublesFromAlphaLen[0 : nAlpha],                               \
                 SinglesFromAlphaOffset[0 : nAlpha + 1],                        \
                 DoublesFromAlphaOffset[0 : nAlpha + 1],                        \
                 SinglesFromAlpha_ptr[0 : singles_alpha_total],                 \
                 DoublesFromAlpha_ptr[0 : doubles_alpha_total],                 \
                 SinglesAlphaCrAn_ptr[0 : singles_alpha_cran_total],            \
                 DoublesAlphaCrAn_ptr[0 : doubles_alpha_cran_total]) thread_limit(1024)
            for(size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
              for(size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
                size_t DetI[SBD_MAX_DETSIZE];

                size_t braIdx = (ia - braAlphaStart)*braBetaSize + ib - braBetaStart;
                if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;

                DetFromAlphaBeta(&Adet[(ia - braAlphaStart)*detSize], &Bdet[(ib - braBetaStart)*detSize], bit_length, norb, DetI);
                size_t ia_local = ia - braAlphaStart;
		
		// single alpha excitation
                for(size_t j = SinglesFromAlphaOffset[ia_local]; j < SinglesFromAlphaOffset[ia_local + 1]; j++) {
                  size_t ja = SinglesFromAlpha_ptr[j];
                  size_t ketIdx = (ja - ketAlphaStart) * ketBetaSize + (ib - ketBetaStart);
                  int cr = SinglesAlphaCrAn_ptr[2*j + 0];
                  int an = SinglesAlphaCrAn_ptr[2*j + 1];
		  OneDiffCorrelation(DetI, W_ptr[braIdx], T_ptr[ketIdx], bit_length, norb, cr, an, oneBody, twoBody);
		}

		// double alpha excitation
                for (size_t j = DoublesFromAlphaOffset[ia_local]; j < DoublesFromAlphaOffset[ia_local + 1]; j++) {
                  size_t ja = DoublesFromAlpha_ptr[j];
                  size_t ketIdx = (ja - ketAlphaStart) * ketBetaSize + (ib - ketBetaStart);
                  int cr = DoublesAlphaCrAn_ptr[4*j + 0];
                  int an = DoublesAlphaCrAn_ptr[4*j + 1];
                  int bn = DoublesAlphaCrAn_ptr[4*j + 2];
                  int cn = DoublesAlphaCrAn_ptr[4*j + 3];
		  TwoDiffCorrelation(DetI, W_ptr[braIdx], T_ptr[ketIdx], bit_length, norb, cr, an, bn, cn, oneBody, twoBody);
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	  } else if ( helper[task].taskType == 1 ) { // alpha range are same
	    
#pragma omp target teams distribute parallel for collapse(2)                \
        map(to : SinglesFromBetaLen[0 : nBeta],                             \
             DoublesFromBetaLen[0 : nBeta],                                 \
             SinglesFromBetaOffset[0 : nBeta + 1],                          \
             DoublesFromBetaOffset[0 : nBeta + 1],                          \
             SinglesFromBeta_ptr[0 : singles_beta_total],                   \
             DoublesFromBeta_ptr[0 : doubles_beta_total],                   \
             SinglesBetaCrAn_ptr[0 : singles_beta_cran_total],              \
             DoublesBetaCrAn_ptr[0 : doubles_beta_cran_total]) thread_limit(1024)
            for(size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
              for(size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
                size_t DetI[SBD_MAX_DETSIZE];

                size_t braIdx = (ia - braAlphaStart)*braBetaSize + ib - braBetaStart;
                if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;

                DetFromAlphaBeta(&Adet[(ia - braAlphaStart)*detSize], &Bdet[(ib - braBetaStart)*detSize], bit_length, norb, DetI);
                size_t ib_local = ib - braBetaStart;
		
		// single beta excitation
                for (size_t j = SinglesFromBetaOffset[ib_local]; j < SinglesFromBetaOffset[ib_local + 1]; j++) {
                  size_t jb = SinglesFromBeta_ptr[j];
                  size_t ketIdx = (ia - ketAlphaStart) * ketBetaSize + (jb - ketBetaStart);
                  int cr = SinglesBetaCrAn_ptr[2*j + 0];
                  int an = SinglesBetaCrAn_ptr[2*j + 1];
		  OneDiffCorrelation(DetI, W_ptr[braIdx], T_ptr[ketIdx], bit_length, norb, cr, an, oneBody, twoBody);
		}

		// double beta excitation
                for (size_t j = DoublesFromBetaOffset[ib_local]; j < DoublesFromBetaOffset[ib_local + 1]; j++) {
                  size_t jb = DoublesFromBeta_ptr[j];
                  size_t ketIdx = (ia - ketAlphaStart) * ketBetaSize + (jb - ketBetaStart);
                  int cr = DoublesBetaCrAn_ptr[4*j + 0];
                  int an = DoublesBetaCrAn_ptr[4*j + 1];
                  int bn = DoublesBetaCrAn_ptr[4*j + 2];
                  int cn = DoublesBetaCrAn_ptr[4*j + 3];
		  TwoDiffCorrelation(DetI, W_ptr[braIdx], T_ptr[ketIdx], bit_length, norb, cr, an, bn, cn, oneBody, twoBody);
		}
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	    
	  } else {
	    
        #pragma omp target teams distribute parallel for collapse(2)               \
                map(to : SinglesFromAlphaLen[0 : nAlpha],                          \
                         SinglesFromBetaLen[0 : nBeta],                            \
                         SinglesFromAlphaOffset[0 : nAlpha + 1],                   \
                         SinglesFromBetaOffset[0 : nBeta + 1],                     \
                         SinglesFromAlpha_ptr[0 : singles_alpha_total],            \
                         SinglesFromBeta_ptr[0 : singles_beta_total],              \
                         SinglesAlphaCrAn_ptr[0 : singles_alpha_cran_total],       \
                         SinglesBetaCrAn_ptr[0 : singles_beta_cran_total]) thread_limit(1024)
           for(size_t ia = braAlphaStart; ia < braAlphaEnd; ia++) {
              for(size_t ib = braBetaStart; ib < braBetaEnd; ib++) {
                size_t DetI[SBD_MAX_DETSIZE];

                size_t braIdx = (ia - braAlphaStart)*braBetaSize + ib - braBetaStart;
                if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;

                DetFromAlphaBeta(&Adet[(ia - braAlphaStart)*detSize], &Bdet[(ib - braBetaStart)*detSize], bit_length, norb, DetI);
                size_t ia_local = ia - braAlphaStart;
                size_t ib_local = ib - braBetaStart;
		
		// two-particle excitation composed of single alpha and single beta
                for(size_t j = SinglesFromAlphaOffset[ia_local]; j < SinglesFromAlphaOffset[ia_local + 1]; j++) {
                  size_t ja = SinglesFromAlpha_ptr[j];
                  for(size_t k = SinglesFromBetaOffset[ib_local]; k < SinglesFromBetaOffset[ib_local + 1]; k++) {
                    size_t jb = SinglesFromBeta_ptr[k];
                    size_t ketIdx = (ja - ketAlphaStart) * ketBetaSize + (jb - ketBetaStart);
                    int cr = SinglesAlphaCrAn_ptr[2*j + 0];
                    int an = SinglesBetaCrAn_ptr[2*k + 0];
                    int bn = SinglesAlphaCrAn_ptr[2*j + 1];
                    int cn = SinglesBetaCrAn_ptr[2*k + 1];
		    TwoDiffCorrelation(DetI, W_ptr[braIdx], T_ptr[ketIdx], bit_length, norb, cr, an, bn, cn, oneBody, twoBody);
		  }
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	  } // if ( helper[task].taskType == 0)

#pragma omp target exit data map(delete: Adet[0:AdetSize], Bdet[0:BdetSize])
        // free data structures for alpha and beta bit-strings
        free(Adet);
        free(Bdet);

      if( helper[task].taskType == 0 && task != helper.size()-1 ) {
	int adetslide = helper[task].adetShift-helper[task+1].adetShift;
	int bdetslide = helper[task].bdetShift-helper[task+1].bdetShift;
    #pragma omp target teams distribute parallel for
        for (size_t i = 0; i < Tsize; i++) R_ptr[i] = T_ptr[i];
        Rsize = Tsize;
        Mpi2dSlide(R_ptr, Rsize, T_ptr, Tsize, adet_comm_size, bdet_comm_size, adetslide, bdetslide, b_comm);
      }
	
    } // end for(size_t task=0; task < helper.size(); task++)

#pragma omp target exit data map(delete:W_ptr[0:Wsize])

// transfer oneBody and twoBody arrays back to the CPU
#pragma omp target exit data map(from: oneBody[0:2*norb2], twoBody[0:4*norb4]) map(delete: W_ptr[0:Wsize], T_ptr[0:Tmax], R_ptr[0:Tmax])

    for(int s=0; s < 2; s++) {
      // store back to std::vector
      for (int i = 0; i < norb2; i++) onebody[s][i] = oneBody[s*norb2 + i];
      MpiAllreduce(onebody[s],MPI_SUM,b_comm);
      MpiAllreduce(onebody[s],MPI_SUM,t_comm);
      MpiAllreduce(onebody[s],MPI_SUM,h_comm);
    }
    for(int s=0; s < 4; s++) {
      // store back to std::vector
      for (int i = 0; i < norb4; i++) twobody[s][i] = twoBody[s*norb4 + i];
      MpiAllreduce(twobody[s],MPI_SUM,b_comm);
      MpiAllreduce(twobody[s],MPI_SUM,t_comm);
      MpiAllreduce(twobody[s],MPI_SUM,h_comm);
    }

  }

#else

  template <typename ElemT>
  void Correlation(const std::vector<ElemT> & W,
		   const det_vector<size_t, det_kind::half> & adet,
		   const det_vector<size_t, det_kind::half> & bdet,
		   const size_t bit_length,
		   const size_t norb,
		   const size_t adet_comm_size,
		   const size_t bdet_comm_size,
		   const std::vector<TaskHelpers> & helper,
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

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);
    
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize  = 0;
    if( helper.size() != 0 ) {
      braAlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
      braBetaSize  = helper[0].braBetaEnd  - helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adet.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdet.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    std::vector<ElemT> T;
    std::vector<ElemT> R;
    T.reserve(max_det_size);
    R.reserve(max_det_size);
    if( helper.size() != 0 ) {
      Mpi2dSlide(W,T,adet_comm_size,bdet_comm_size,
		 -helper[0].adetShift,-helper[0].bdetShift,b_comm);
    }

    size_t array_size = (2*norb + bit_length - 1 ) / bit_length;

    size_t num_threads = 1;
    num_threads = omp_get_max_threads();

    
    if( mpi_rank_t == 0 ) {
#pragma omp parallel
      {
        // round-robin assignment of work to threads
        size_t thread_id = omp_get_thread_num();
        size_t ia_start = thread_id + helper[0].braAlphaStart;
        size_t ia_end   = helper[0].braAlphaEnd;
        std::vector<size_t> DetT(array_size);
        for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
          for(size_t ib = helper[0].braBetaStart; ib < helper[0].braBetaEnd; ib++) {
            size_t i = (ia - helper[0].braAlphaStart) * braBetaSize
                     +  ib - helper[0].braBetaStart;
            if( ( i % mpi_size_h ) == mpi_rank_h ) {
              DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetT);
              ZeroDiffCorrelation(DetT,W[i],bit_length,norb,onebody,twobody);
            }
          }
        }
      }
    }

    for(size_t task=0; task < helper.size(); task++) {

      size_t ketAlphaSize = helper[task].ketAlphaEnd-helper[task].ketAlphaStart;
      size_t ketBetaSize  = helper[task].ketBetaEnd-helper[task].ketBetaStart;
      
      if( helper.size() != 0 ) {
#pragma omp parallel
        {
	  // round-robin assignment of work to threads
          size_t thread_id = omp_get_thread_num();
          size_t ia_start = thread_id + helper[task].braAlphaStart;
          size_t ia_end   = helper[task].braAlphaEnd;

	  size_t array_size = (2*norb + bit_length - 1 ) / bit_length;
	  std::vector<size_t> DetI(array_size);
	  auto DetJ = DetI;
	  std::vector<int> c(2,0);
	  std::vector<int> d(2,0);
	  
	  if( helper[task].taskType == 2 ) { // beta range are same
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		                +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// single alpha excitation
		for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                  +ib-helper[task].ketBetaStart;
		  OneDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				     helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				     onebody,twobody);
		  /*
		  DetFromAlphaBeta(adet[ja],bdet[ib],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody,twobody);
		  */
		}
		// double alpha excitation
		for(size_t j=0; j < helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].DoublesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                 + ib-helper[task].ketBetaStart;
		  TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+0],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+1],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+2],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+3],
				     onebody,twobody);
		  /*
		  DetFromAlphaBeta(adet[ja],bdet[ib],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody,twobody);
		  */
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	  } else if ( helper[task].taskType == 1 ) { // alpha range are same
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		  +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// single beta excitation
		for(size_t j=0; j < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		  size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][j];
		  size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		                 + jb-helper[task].ketBetaStart;
		  OneDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+0],
				     helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+1],
				     onebody,twobody);
		  /*
		  DetFromAlphaBeta(adet[ia],bdet[jb],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody,twobody);
		  */
		}
		// double beta excitation
		for(size_t j=0; j < helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		  size_t jb = helper[task].DoublesFromBetaSM[ib-helper[task].braBetaStart][j];
		  size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		                 + jb-helper[task].ketBetaStart;
		  TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+0],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+1],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+2],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+3],
				     onebody,twobody);
		  /*
		  DetFromAlphaBeta(adet[ia],bdet[jb],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					bit_length,norb,c,d,
					  onebody,twobody);
		  */
		}
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	    
	  } else {
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		                +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// two-particle excitation composed of single alpha and single beta
		for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  for(size_t k=0; k < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; k++) {
		    size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][k];
		    size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                    +jb-helper[task].ketBetaStart;
		    TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				       helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				       helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+0],
				       helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				       helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+1],
				       onebody,twobody);
		    /*
		    DetFromAlphaBeta(adet[ja],bdet[jb],bit_length,norb,DetJ);
		    CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					    bit_length,norb,c,d,
					    onebody,twobody);
		    */
		  }
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	  } // if ( helper[task].taskType ==  )

        } // end #pragma omp parallel

      } // end if( helper.size() != 0 )
      
      if( helper[task].taskType == 0 && task != helper.size()-1 ) {
	int adetslide = helper[task].adetShift-helper[task+1].adetShift;
	int bdetslide = helper[task].bdetShift-helper[task+1].bdetShift;
	R.resize(T.size());
	std::memcpy(R.data(),T.data(),T.size()*sizeof(ElemT));
	Mpi2dSlide(R,T,adet_comm_size,bdet_comm_size,adetslide,bdetslide,b_comm);
      }
	
    } // end for(size_t task=0; task < helper.size(); task++)

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

  }

#endif
  
} // end namespace sbd

#endif // end if for #ifndef SBD_CHEMISTRY_PTMB_CORRELATION_H
