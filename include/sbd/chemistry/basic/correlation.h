/**
@file sbd/chemistry/basic/correlation.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_BASIC_CORRELATION_H
#define SBD_CHEMISTRY_BASIC_CORRELATION_H

#ifdef USE_OMP_OFFLOAD
#include "../basic/omp_offload.h"
#endif

namespace sbd {
  /**
     Function for adding diagonal contribution
   */
  template <typename ElemT>
  void ZeroDiffCorrelation(const std::vector<size_t> & DetI,
			   ElemT WeightI,
			   size_t bit_length,
			   size_t norb,
			   std::vector<std::vector<ElemT>> & onebody,
			   std::vector<std::vector<ElemT>> & twobody) {
    std::vector<int> closed;
    int num_closed = getClosed(DetI,bit_length,2*norb,closed);

    for(int i=0; i < num_closed; i++) {
      int oi = closed.at(i)/2;
      int si = closed.at(i)%2;
      onebody[si][oi+norb*oi] += SquaredNorm(WeightI);
      for(int j=i+1; j < num_closed; j++) {
	int oj = closed.at(j)/2;
	int sj = closed.at(j)%2;
	twobody[si+2*sj][oi+norb*oj+norb*norb*oi+norb*norb*norb*oj]
	  += SquaredNorm(WeightI);
	twobody[sj+2*si][oj+norb*oi+norb*norb*oj+norb*norb*norb*oi]
	  += SquaredNorm(WeightI);
	if( si == sj ) {
	  twobody[si+2*sj][oi+norb*oj+norb*norb*oj+norb*norb*norb*oi]
	    += -SquaredNorm(WeightI);
	  twobody[sj+2*si][oj+norb*oi+norb*norb*oi+norb*norb*norb*oj]
	    += -SquaredNorm(WeightI);
	}
      }
    }
  }

#ifdef USE_OMP_OFFLOAD
#pragma omp declare target
  template <typename ElemT>
  void ZeroDiffCorrelation(size_t * DetI,
                           ElemT WeightI,
                           size_t bit_length,
                           size_t norb,
                           ElemT * oneBody,
                           ElemT * twoBody) {
    int closed[SBD_MAX_SPINORBITALS];
    int num_closed = 0;
    for (int i = 0; i < 2*norb; i++) {
       int block = i / bit_length;
       int pos   = i % bit_length;
       if ((DetI[block] >> pos) & 1) {
          closed[num_closed] = i;
          num_closed++;
       }
    }

    size_t norb2 = norb * norb;
    size_t norb4 = norb2 * norb2;

    for(int i=0; i < num_closed; i++) {
      int oi = closed[i]/2;
      int si = closed[i]%2;
#pragma omp atomic update
      oneBody[si*norb2 + oi + norb*oi] += SquaredNorm(WeightI);
      for(int j=i+1; j < num_closed; j++) {
        int oj = closed[j]/2;
        int sj = closed[j]%2;
#pragma omp atomic update
        twoBody[(si+2*sj)*norb4 + oi+norb*oj+norb*norb*oi+norb*norb*norb*oj] += SquaredNorm(WeightI);
#pragma omp atomic update
        twoBody[(sj+2*si)*norb4 + oj+norb*oi+norb*norb*oj+norb*norb*norb*oi] += SquaredNorm(WeightI);
        if( si == sj ) {
#pragma omp atomic update
          twoBody[(si+2*sj)*norb4 + oi+norb*oj+norb*norb*oj+norb*norb*norb*oi] += -SquaredNorm(WeightI);
#pragma omp atomic update
          twoBody[(sj+2*si)*norb4 + oj+norb*oi+norb*norb*oi+norb*norb*norb*oj] += -SquaredNorm(WeightI);
        }
      }
    }
  }
#pragma omp end declare target
#endif

  /**
     Function for adding one-occupation different contribution
   */
  template <typename ElemT>
  void OneDiffCorrelation(const std::vector<size_t> & DetI,
			  const ElemT WeightI,
			  const ElemT WeightJ,
			  const size_t bit_length,
			  const size_t norb,
			  int i,
			  int a,
			  std::vector<std::vector<ElemT>> & onebody,
			  std::vector<std::vector<ElemT>> & twobody) {
    double sgn = 1.0;
    parity(DetI,bit_length,std::min(i,a),std::max(i,a),sgn);
    int oi = i / 2;
    int si = i % 2;
    int oa = a / 2;
    int sa = a % 2;
    onebody[si][oi+norb*oa] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
    size_t one = 1;
    for(int x=0; x < DetI.size(); x++) {
      size_t bits = DetI[x];
      while(bits != 0) {
	int pos = __builtin_ffsl(bits);
	int soj = x * bit_length + pos - 1;
	int oj = soj / 2;
	int sj = soj % 2;

	twobody[si+2*sj][oa+oj*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
	twobody[sj+2*si][oj+oa*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
	
	if( si == sj ) {
	  twobody[si+2*sj][oa+oj*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
	  twobody[sj+2*si][oj+oa*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
	}
	
	bits &= ~(one << (pos-1));
      }
    }
  }

#ifdef USE_OMP_OFFLOAD
#pragma omp declare target
  template <typename ElemT>
  void OneDiffCorrelation(const size_t * DetI,
                          const ElemT WeightI,
                          const ElemT WeightJ,
                          const size_t bit_length,
                          const size_t norb,
                          int i,
                          int a,
                          ElemT * oneBody,
                          ElemT * twoBody) {
    int detSize = (2*norb + bit_length - 1) / bit_length;
    double sgn = 1.0;
    parity_device(DetI,bit_length,std::min(i,a),std::max(i,a),sgn);
    size_t norb2 = norb * norb;
    size_t norb4 = norb2 * norb2;
    int oi = i / 2;
    int si = i % 2;
    int oa = a / 2;
    int sa = a % 2;
#pragma omp atomic update
    oneBody[si*norb2 + oi+norb*oa] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
    size_t one = 1;
    for(int x=0; x < detSize; x++) {
      size_t bits = DetI[x];
      while(bits != 0) {
        int pos = __builtin_ffsl(bits);
        int soj = x * bit_length + pos - 1;
        int oj = soj / 2;
        int sj = soj % 2;

#pragma omp atomic update
        twoBody[(si+2*sj)*norb4 + oa+oj*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
#pragma omp atomic update
        twoBody[(sj+2*si)*norb4 + oj+oa*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);

        if( si == sj ) {
#pragma omp atomic update
          twoBody[(si+2*sj)*norb4 + oa+oj*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
#pragma omp atomic update
          twoBody[(sj+2*si)*norb4 + oj+oa*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
        }

        bits &= ~(one << (pos-1));
      }
    }
  }
#pragma omp end declare target
#endif

  /**
     Function for adding two-occupation different contribution
   */
  template <typename ElemT>
  void TwoDiffCorrelation(const std::vector<size_t> & DetI,
			  const ElemT WeightI,
			  const ElemT WeightJ,
			  const size_t bit_length,
			  const size_t norb,
			  int i,
			  int j,
			  int a,
			  int b,
			  std::vector<std::vector<ElemT>> & onebody,
			  std::vector<std::vector<ElemT>> & twobody) {
    double sgn = 1.0;
    int I = std::min(i,j);
    int J = std::max(i,j);
    int A = std::min(a,b);
    int B = std::max(a,b);
    parity(DetI,bit_length,std::min(I,A),std::max(I,A),sgn);
    parity(DetI,bit_length,std::min(J,B),std::max(J,B),sgn);
    if( A > J || B < I ) sgn *= -1.0;
    int oi = I / 2;
    int si = I % 2;
    int oa = A / 2;
    int sa = A % 2;
    int oj = J / 2;
    int sj = J % 2;
    int ob = B / 2;
    int sb = B % 2;

    if( si == sa ) {
      twobody[si+2*sj][oa+norb*ob+norb*norb*(oi+norb*oj)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
      twobody[sj+2*si][ob+norb*oa+norb*norb*(oj+norb*oi)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
    }

    if( si == sb ) {
      twobody[si+2*sj][oa+norb*ob+norb*norb*(oj+norb*oi)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
      twobody[sj+2*si][ob+norb*oa+norb*norb*(oi+norb*oj)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
    }

  }

#ifdef USE_OMP_OFFLOAD
#pragma omp declare target
  template <typename ElemT>
  void TwoDiffCorrelation(const size_t * DetI,
                          const ElemT WeightI,
                          const ElemT WeightJ,
                          const size_t bit_length,
                          const size_t norb,
                          int i,
                          int j,
                          int a,
                          int b,
                          ElemT * oneBody,
                          ElemT * twoBody) {
    size_t norb2 = norb * norb;
    size_t norb4 = norb2 * norb2;
    double sgn = 1.0;
    int I = std::min(i,j);
    int J = std::max(i,j);
    int A = std::min(a,b);
    int B = std::max(a,b);
    parity_device(DetI,bit_length,std::min(I,A),std::max(I,A),sgn);
    parity_device(DetI,bit_length,std::min(J,B),std::max(J,B),sgn);
    if( A > J || B < I ) sgn *= -1.0;
    int oi = I / 2;
    int si = I % 2;
    int oa = A / 2;
    int sa = A % 2;
    int oj = J / 2;
    int sj = J % 2;
    int ob = B / 2;
    int sb = B % 2;

    if( si == sa ) {
#pragma omp atomic update
      twoBody[(si+2*sj)*norb4 + oa+norb*ob+norb*norb*(oi+norb*oj)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
#pragma omp atomic update
      twoBody[(sj+2*si)*norb4 + ob+norb*oa+norb*norb*(oj+norb*oi)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
    }

    if( si == sb ) {
#pragma omp atomic update
      twoBody[(si+2*sj)*norb4 + oa+norb*ob+norb*norb*(oj+norb*oi)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
#pragma omp atomic update
      twoBody[(sj+2*si)*norb4 + ob+norb*oa+norb*norb*(oi+norb*oj)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
    }

  }
#pragma omp end declare target
#endif

  /**
     Function for adding the terms to the resulting correlation
   */
  template <typename ElemT>
  void CorrelationTermAddition(const std::vector<size_t> & DetI,
			       const std::vector<size_t> & DetJ,
			       const ElemT WeightI,
			       const ElemT WeightJ,
			       const size_t bit_length,
			       const size_t norb,
			       std::vector<int> & c,
			       std::vector<int> & d,
			       std::vector<std::vector<ElemT>> & onebody,
			       std::vector<std::vector<ElemT>> & twobody) {
    size_t nc = 0;
    size_t nd = 0;

    size_t full_words = (2*norb) / bit_length;
    size_t remaining_bits = (2*norb) % bit_length;

    for(size_t i=0; i < full_words; ++i) {
      size_t diff_c = DetI[i] & ~DetJ[i];
      size_t diff_d = DetJ[i] & ~DetI[i];
      for(size_t bit_pos=0; bit_pos < bit_length; ++bit_pos) {
	if( diff_c & (static_cast<size_t>(1) << bit_pos)) {
	  c[nc] = i*bit_length+bit_pos;
	  nc++;
	}
	if( diff_d & (static_cast<size_t>(1) << bit_pos)) {
	  d[nd] = i*bit_length+bit_pos;
	  nd++;
	}
      }
    }
    if ( remaining_bits > 0 ) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) -1;
      size_t diff_c = (DetI[full_words] & ~DetJ[full_words]) & mask;
      size_t diff_d = (DetJ[full_words] & ~DetI[full_words]) & mask;
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
      ZeroDiffCorrelation(DetI,WeightI,bit_length,norb,onebody,twobody);
    } else if ( nc == 1 ) {
      OneDiffCorrelation(DetI,WeightI,WeightJ,bit_length,norb,c[0],d[0],onebody,twobody);
    } else if ( nc == 2 ) {
      TwoDiffCorrelation(DetI,WeightI,WeightJ,bit_length,norb,c[0],c[1],d[0],d[1],onebody,twobody);
    }
  }
}
#endif
