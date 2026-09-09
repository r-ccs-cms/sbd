/**
@file sbd/chemistry/basic/integrals.h
@brief Functions to handle integrals
*/
#ifndef SBD_CHEMISTRY_BASIC_INTEGRALS_H
#define SBD_CHEMISTRY_BASIC_INTEGRALS_H

#include <algorithm>
#include <vector>

namespace sbd {

  template <typename ElemT>
  class oneInt {
  public:
    std::vector<ElemT> store;
    int norbs;
    inline ElemT & operator()(int i, int j) {
      return store.at(i*norbs+j);
    }
    
    inline ElemT Value(int i, int j) const {
      return store.at(i*norbs+j);
    }
  };
  
#ifdef _UHF
  template <typename ElemT>
  class twoInt {
  public:
    std::vector<ElemT> store;
    ElemT maxEntry;
    ElemT zero;
    int norbs;
    std::vector<ElemT> DirectMat;
    std::vector<ElemT> ExchangeMat;
    twoInt() : zero(0.0), maxEntry(100.0) {}
    inline ElemT & operator()(int i, int j, int k, int l) {
      zero = ElemT(0.0);
      if(!((i%2 == j%2)&&(k%2==l%2))) return zero;
      int I = i/2; int J = j/2; int K = k/2; int L=l/2;
      int S = i%2 + 2 * (k%2);
      return store[I+J*norbs+K*norbs*norbs+L*norbs*norbs*norbs+S*norbs*norbs*norbs*norbs];
    }
    inline ElemT & Direct(int i, int j) {
      return DirectMat[i+2*norbs*j];
    }
    inline ElemT & Exchange(int i, int j) {
      return ExchangeMat[i+2*norbs*j];
    }

    inline ElemT Value(int i, int j, int k, int l) const {
      if(!((i%2 == j%2)&&(k%2==l%2))) return zero;
      int I = i/2; int J = j/2; int K = k/2; int L=l/2;
      int S = i%2 + 2 * (k%2);
      return store[I+J*norbs+K*norbs*norbs+L*norbs*norbs*norbs+S*norbs*norbs*norbs*norbs];
    }
    inline ElemT DirectValue(int i, int j) const {
      return DirectMat[i+2*norbs*j];
    }
    inline ElemT ExchangeValue(int i, int j) const {
      return ExchangeMat[i+2*norbs*j];
    }
    
  };
#else
  template <typename ElemT>
  class twoInt {
  public:
    std::vector<ElemT> store;
    ElemT maxEntry;
    ElemT zero;
    int norbs;
    std::vector<ElemT> DirectMat;
    std::vector<ElemT> ExchangeMat;
    twoInt() : zero(0.0), maxEntry(100.0) {}
    inline ElemT & operator()(int i, int j, int k, int l) {
      zero = ElemT(0.0);
      if(!((i%2 == j%2)&&(k%2==l%2))) return zero;
      int I = i/2; int J = j/2; int K = k/2; int L=l/2;
      int ij = std::max(I,J)*(std::max(I,J)+1)/2 + std::min(I,J);
      int kl = std::max(K,L)*(std::max(K,L)+1)/2 + std::min(K,L);
      int a = std::max(ij,kl);
      int b = std::min(ij,kl);
      return store[a*(a+1)/2+b];
    }
    inline ElemT & Direct(int i, int j) {
      return DirectMat[i+2*norbs*j];
    }
    inline ElemT & Exchange(int i, int j) {
      return ExchangeMat[i+2*norbs*j];
    }

    inline ElemT Value(int i, int j, int k, int l) const {
      if(!((i%2 == j%2)&&(k%2==l%2))) return zero;
      int I = i/2; int J = j/2; int K = k/2; int L=l/2;
      int ij = std::max(I,J)*(std::max(I,J)+1)/2 + std::min(I,J);
      int kl = std::max(K,L)*(std::max(K,L)+1)/2 + std::min(K,L);
      int a = std::max(ij,kl);
      int b = std::min(ij,kl);
      return store[a*(a+1)/2+b];
    }
    inline ElemT DirectValue(int i, int j) const {
      return DirectMat[i+2*norbs*j];
    }
    inline ElemT ExchangeValue(int i, int j) const {
      return ExchangeMat[i+2*norbs*j];
    }
    
  };
#endif
  
} // end namespace sbd

#endif // end SBD_CHEMISTRY_INTEGRALS_H
