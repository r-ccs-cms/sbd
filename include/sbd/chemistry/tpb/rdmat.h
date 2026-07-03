/**
@file sbd/chemistry/tpb/rdmat.h
@brief function to evaluate the diagonal term of the reduced density matrix (in statistical physics/quantum information)
 */
#ifndef SBD_CHEMISTRY_TPB_RDMAT_H
#define SBD_CHEMISTRY_TPB_RDMAT_H

namespace sbd {

  // Evaluate the diagonal part of the reduced density matrix
  template <typename ElemT, typename RealT>
  void DiagonalAdetReducedDensityMatrix(const std::vector<ElemT> & W,
					const size_t adet_size,
					const size_t bdet_size,
					const size_t adet_comm_size,
					const size_t bdet_comm_size,
					MPI_Comm b_comm,
					std::vector<RealT> & D) {
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);

    int adet_rank = mpi_rank_b / bdet_comm_size;
    int bdet_rank = mpi_rank_b % bdet_comm_size;

    size_t adet_start = 0;
    size_t adet_end   = adet_size;
    size_t bdet_start = 0;
    size_t bdet_end   = bdet_size;
    get_mpi_range(static_cast<int>(adet_comm_size),adet_rank,adet_start,adet_end);
    get_mpi_range(static_cast<int>(bdet_comm_size),bdet_rank,bdet_start,bdet_end);

    MPI_Comm adet_comm;
    MPI_Comm bdet_comm;
    MPI_Comm_split(b_comm,adet_rank,mpi_rank_b,&adet_comm);
    MPI_Comm_split(b_comm,bdet_rank,mpi_rank_b,&bdet_comm);

    size_t adet_range = adet_end - adet_start;
    size_t bdet_range = bdet_end - bdet_start;
    size_t det_size = adet_range * bdet_range;

    if( det_size != W.size() ) {
      std::cout << " ReducedDensityMatrix: sizes are inconsistent " << std::endl;
    }

    D.resize(adet_size);

#pragma omp parallel for
    for(size_t ia=adet_start; ia < adet_end; ia++) {
      D[ia] = 0.0;
      for(size_t ib=bdet_start; ib < bdet_end; ib++) {
	D[ia] += SquaredNorm(W[(ia-adet_start)*bdet_range+ib-bdet_start]);
      }
    }
    
    MpiAllreduce(D,MPI_SUM,adet_comm); // each elements are already obtained
    MpiAllreduce(D,MPI_SUM,bdet_comm); // fill the different elements

    MPI_Comm_free(&adet_comm);
    MPI_Comm_free(&bdet_comm);
  }

  template <typename ElemT, typename RealT>
  void CarryOverAdet(const std::vector<ElemT> & W,
		     const std::vector<std::vector<size_t>> & adet,
		     const std::vector<std::vector<size_t>> & bdet,
		     const size_t adet_comm_size,
		     const size_t bdet_comm_size,
		     MPI_Comm b_comm,
		     std::vector<std::vector<size_t>> & rdet,
		     RealT threshold) {
    
    size_t adet_size = adet.size();
    size_t bdet_size = bdet.size();
    std::vector<RealT> D(adet_size);
    DiagonalAdetReducedDensityMatrix(W,adet_size,bdet_size,adet_comm_size,bdet_comm_size,b_comm,D);

    std::vector<size_t> sortIdx(adet_size);
    std::iota(sortIdx.begin(),sortIdx.end(),0);
    std::sort(sortIdx.begin(),sortIdx.end(),
	      [&](int i, int j) {
		return D[i] > D[j];
	      });
    
    std::vector<RealT> S(adet_size);

    RealT sum = 0.0;
    for(size_t i=0; i < adet_size; i++) {
      sum += D[sortIdx[i]];
      S[i] = sum;
    }

    RealT target = 1.0 - threshold;
    auto itImax = std::upper_bound(S.begin(),S.end(),target);
    size_t Imax = static_cast<size_t>(std::distance(S.begin(),itImax)) + 1;
    rdet.resize(Imax,std::vector<size_t>(adet[0].size()));
    for(size_t k=0; k < Imax; k++) {
      rdet[k] = adet[sortIdx[k]];
    }
  }

  template <typename ElemT, typename RealT>
  void CarryOverAdet(const std::vector<ElemT> & W,
		     const std::vector<std::vector<size_t>> & adet,
		     const std::vector<std::vector<size_t>> & bdet,
		     const size_t adet_comm_size,
		     const size_t bdet_comm_size,
		     MPI_Comm b_comm,
		     size_t kept,
		     std::vector<std::vector<size_t>> & rdet,
		     RealT & discarted_weight) {
    
    size_t adet_size = adet.size();
    size_t bdet_size = bdet.size();
    std::vector<RealT> D(adet_size);
    DiagonalAdetReducedDensityMatrix(W,adet_size,bdet_size,adet_comm_size,bdet_comm_size,b_comm,D);

    std::vector<size_t> sortIdx(adet_size);
    std::iota(sortIdx.begin(),sortIdx.end(),0);
    std::sort(sortIdx.begin(),sortIdx.end(),
	      [&](int i, int j) {
		return D[i] > D[j];
	      });
    
    rdet.resize(kept,std::vector<size_t>(adet[0].size()));
    for(size_t k=0; k < kept; k++) {
      rdet[k] = adet[sortIdx[k]];
    }
    
    RealT sum = 0.0;
    for(size_t i=0; i < kept; i++) {
      sum += D[sortIdx[i]];
    }
    discarted_weight = 1.0-sum;
  }

  // Evaluate the diagonal part of the reduced density matrix
  template <typename ElemT, typename RealT>
  void DiagonalBdetReducedDensityMatrix(const std::vector<ElemT> & W,
					const size_t adet_size,
					const size_t bdet_size,
					const size_t adet_comm_size,
					const size_t bdet_comm_size,
					MPI_Comm b_comm,
					std::vector<RealT> & D) {
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);

    int adet_rank = mpi_rank_b / bdet_comm_size;
    int bdet_rank = mpi_rank_b % bdet_comm_size;

    size_t adet_start = 0;
    size_t adet_end   = adet_size;
    size_t bdet_start = 0;
    size_t bdet_end   = bdet_size;
    get_mpi_range(static_cast<int>(adet_comm_size),adet_rank,adet_start,adet_end);
    get_mpi_range(static_cast<int>(bdet_comm_size),bdet_rank,bdet_start,bdet_end);

    MPI_Comm adet_comm;
    MPI_Comm bdet_comm;
    MPI_Comm_split(b_comm,adet_rank,mpi_rank_b,&adet_comm);
    MPI_Comm_split(b_comm,bdet_rank,mpi_rank_b,&bdet_comm);

    size_t adet_range = adet_end - adet_start;
    size_t bdet_range = bdet_end - bdet_start;
    size_t det_size = adet_range * bdet_range;

    if( det_size != W.size() ) {
      std::cout << " ReducedDensityMatrix: sizes are inconsistent " << std::endl;
    }

    D.resize(bdet_size);

#pragma omp parallel for
    for(size_t ib=bdet_start; ib < bdet_end; ib++) {
      D[ib] = 0.0;
      for(size_t ia=adet_start; ia < adet_end; ia++) {
	D[ib] += SquaredNorm(W[(ia-adet_start)*bdet_range+ib-bdet_start]);
      }
    }
    
    MpiAllreduce(D,MPI_SUM,adet_comm); // each elements are already obtained
    MpiAllreduce(D,MPI_SUM,bdet_comm); // fill the different elements

    MPI_Comm_free(&adet_comm);
    MPI_Comm_free(&bdet_comm);
  }

  template <typename ElemT, typename RealT>
  void CarryOverBdet(const std::vector<ElemT> & W,
		     const std::vector<std::vector<size_t>> & adet,
		     const std::vector<std::vector<size_t>> & bdet,
		     const size_t adet_comm_size,
		     const size_t bdet_comm_size,
		     MPI_Comm b_comm,
		     std::vector<std::vector<size_t>> & rdet,
		     RealT threshold) {
    
    size_t adet_size = adet.size();
    size_t bdet_size = bdet.size();
    std::vector<RealT> D(bdet_size);
    DiagonalBdetReducedDensityMatrix(W,adet_size,bdet_size,adet_comm_size,bdet_comm_size,b_comm,D);

    std::vector<size_t> sortIdx(bdet_size);
    std::iota(sortIdx.begin(),sortIdx.end(),0);
    std::sort(sortIdx.begin(),sortIdx.end(),
	      [&](int i, int j) {
		return D[i] > D[j];
	      });
    
    std::vector<RealT> S(bdet_size);

    RealT sum = 0.0;
    for(size_t i=0; i < bdet_size; i++) {
      sum += D[sortIdx[i]];
      S[i] = sum;
    }

    RealT target = 1.0 - threshold;
    auto itImax = std::upper_bound(S.begin(),S.end(),target);
    size_t Imax = static_cast<size_t>(std::distance(S.begin(),itImax)) + 1;
    rdet.resize(Imax,std::vector<size_t>(bdet[0].size()));
    for(size_t k=0; k < Imax; k++) {
      rdet[k] = bdet[sortIdx[k]];
    }
  }

  template <typename ElemT, typename RealT>
  void CarryOverBdet(const std::vector<ElemT> & W,
		     const std::vector<std::vector<size_t>> & adet,
		     const std::vector<std::vector<size_t>> & bdet,
		     const size_t adet_comm_size,
		     const size_t bdet_comm_size,
		     MPI_Comm b_comm,
		     size_t kept,
		     std::vector<std::vector<size_t>> & rdet,
		     RealT & discarted_weight) {
    
    size_t adet_size = adet.size();
    size_t bdet_size = bdet.size();
    std::vector<RealT> D(bdet_size);
    DiagonalBdetReducedDensityMatrix(W,adet_size,bdet_size,adet_comm_size,bdet_comm_size,b_comm,D);

    std::vector<size_t> sortIdx(bdet_size);
    std::iota(sortIdx.begin(),sortIdx.end(),0);
    std::sort(sortIdx.begin(),sortIdx.end(),
	      [&](int i, int j) {
		return D[i] > D[j];
	      });
    
    rdet.resize(kept,std::vector<size_t>(bdet[0].size()));
    for(size_t k=0; k < kept; k++) {
      rdet[k] = bdet[sortIdx[k]];
    }
    
    RealT sum = 0.0;
    for(size_t i=0; i < kept; i++) {
      sum += D[sortIdx[i]];
    }
    discarted_weight = 1.0-sum;
  }


  

  
}

#endif
