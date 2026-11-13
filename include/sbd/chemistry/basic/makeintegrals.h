/**
@file sbd/chemistry/basic/makeintegrals.h
@brief Setup oneInt and twoInt from fcidump format
*/
#ifndef SBD_CHEMISTRY_BASIC_MAKEINTEGRALS_H
#define SBD_CHEMISTRY_BASIC_MAKEINTEGRALS_H

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sbd {

  /**
     Rejects an FCIDUMP written in the other layout than this build reads, instead of reading it as
     the wrong one. A restricted file indexes spatial orbitals, 1 .. NORB. SBD's unrestricted layout
     (-D_UHF) indexes interleaved spin orbitals, 1 .. 2 NORB (alpha odd, beta even), and conserves
     spin: i, j of one spin and k, l of one spin. A restricted file nearly always has a line that
     mixes the parities of i and j (or of k and l), such as h_21, and such a line has no meaning in
     the unrestricted layout.
  */
  inline void CheckFCIDumpLayout(const FCIDump & fcidump, int L, bool unrestricted) {
    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      const int m = std::max(std::max(i, j), std::max(k, l));
      const bool mixed = (i > 0 && j > 0 && i % 2 != j % 2) || (k > 0 && l > 0 && k % 2 != l % 2);
      const char * why = nullptr;
      if(!unrestricted && m > L)
        why = "orbital index above NORB. It looks unrestricted (SBD's UHF layout): build with -DSBD_UHF=ON (CMake) or -D_UHF.";
      else if(unrestricted && m > 2 * L)
        why = "spin-orbital index above 2 NORB.";
      else if(unrestricted && mixed)
        why = "line that mixes alpha and beta spin orbitals, which SBD's UHF layout cannot hold. "
              "It looks restricted: use a build without SBD_UHF (-D_UHF).";
      if(why)
        throw std::runtime_error("FCIDUMP integral line " + std::to_string(i) + " " + std::to_string(j) + " "
                                 + std::to_string(k) + " " + std::to_string(l) + " (NORB = " + std::to_string(L)
                                 + "): " + why);
    }
  }

#ifdef _UHF
  template <typename ElemT>
  void SetupIntegrals(const FCIDump & fcidump,
		      int & L,
		      int & N,
		      ElemT & I0,
		      oneInt<ElemT> & I1,
		      twoInt<ElemT> & I2) {

    for(const auto & [key, value] : fcidump.header) {
      if( key == std::string("NORB") ) {
	L = std::atoi(value.c_str());
      }
      if( key == std::string("NELEC") ) {
	N = std::atoi(value.c_str());
      }
    }
    CheckFCIDumpLayout(fcidump, L, true);

    I1.norbs = 2 * L;
    I1.store.resize(4*L*L,ElemT(0.0));

    int B = L*L*L*L*4;
    I2.norbs = L;
    I2.store.resize(B,ElemT(0.0));

    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      if( (i==0) && (k==0) && (j==0) && (l==0) ) {
	I0 = ElemT(value);
      } else if( (k==l) && (k==0) ) {
	I1(i-1,j-1) = ElemT(value);
	I1(j-1,i-1) = Conjugate(ElemT(value));
      } else {
	I2(i-1,j-1,k-1,l-1) = ElemT(value);
	I2(j-1,i-1,k-1,l-1) = ElemT(value);
	I2(i-1,j-1,l-1,k-1) = ElemT(value);
	I2(j-1,i-1,l-1,k-1) = ElemT(value);
	I2(k-1,l-1,i-1,j-1) = ElemT(value);
	I2(k-1,l-1,j-1,i-1) = ElemT(value);
	I2(l-1,k-1,i-1,j-1) = ElemT(value);
	I2(l-1,k-1,j-1,i-1) = ElemT(value);
      }
    }

    I2.DirectMat.resize(4*L*L);
    I2.ExchangeMat.resize(4*L*L);
    for(int i=0; i < L; i++) {
      for(int j=0; j < L; j++) {
	I2.Direct(2*i,2*j) = I2.Value(2*i,2*i,2*j,2*j);
	I2.Direct(2*i,2*j+1) = I2.Value(2*i,2*i,2*j+1,2*j+1);
	I2.Direct(2*i+1,2*j) = I2.Value(2*i+1,2*i+1,2*j,2*j);
	I2.Direct(2*i+1,2*j+1) = I2.Value(2*i+1,2*i+1,2*j+1,2*j+1);
	I2.Exchange(2*i,2*j) = I2.Value(2*i,2*j,2*j,2*i);
	I2.Exchange(2*i,2*j+1) = I2.Value(2*i,2*j+1,2*j+1,2*i);
	I2.Exchange(2*i+1,2*j) = I2.Value(2*i+1,2*j,2*j,2*i+1);
	I2.Exchange(2*i+1,2*j+1) = I2.Value(2*i+1,2*j+1,2*j+1,2*i+1);
      }
    }
  }
#elif _UHFCPLX
  template <typename ElemT>
  void SetupIntegrals(const FCIDump & fcidump,
		      int & L,
		      int & N,
		      ElemT & I0,
		      oneInt<ElemT> & I1,
		      twoInt<ElemT> & I2) {

    for(const auto & [key, value] : fcidump.header) {
      if( key == std::string("NORB") ) {
	L = std::atoi(value.c_str());
      }
      if( key == std::string("NELEC") ) {
	N = std::atoi(value.c_str());
      }
    }
    CheckFCIDumpLayout(fcidump, L, true);

    I1.norbs = 2 * L;
    I1.store.resize(4*L*L,ElemT(0.0));

    int B = L*L*L*L*4;
    I2.norbs = L;
    I2.store.resize(B,ElemT(0.0));

    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      if( (i==0) && (k==0) && (j==0) && (l==0) ) {
	I0 = ElemT(value);
      } else if( (k==l) && (k==0) ) {
	I1(i-1,j-1) = ElemT(value);
	I1(j-1,i-1) = Conjugate(ElemT(value));
      } else {
	I2(i-1,j-1,k-1,l-1) = ElemT(value);
	I2(k-1,l-1,i-1,j-1) = ElemT(value);
	I2(j-1,i-1,l-1,k-1) = Conjugate(ElemT(value));
	I2(l-1,k-1,j-1,i-1) = Conjugate(ElemT(value));
      }
    }

    I2.DirectMat.resize(4*L*L);
    I2.ExchangeMat.resize(4*L*L);
    for(int i=0; i < L; i++) {
      for(int j=0; j < L; j++) {
	I2.Direct(2*i,2*j) = I2.Value(2*i,2*i,2*j,2*j);
	I2.Direct(2*i,2*j+1) = I2.Value(2*i,2*i,2*j+1,2*j+1);
	I2.Direct(2*i+1,2*j) = I2.Value(2*i+1,2*i+1,2*j,2*j);
	I2.Direct(2*i+1,2*j+1) = I2.Value(2*i+1,2*i+1,2*j+1,2*j+1);
	I2.Exchange(2*i,2*j) = I2.Value(2*i,2*j,2*j,2*i);
	I2.Exchange(2*i,2*j+1) = I2.Value(2*i,2*j+1,2*j+1,2*i);
	I2.Exchange(2*i+1,2*j) = I2.Value(2*i+1,2*j,2*j,2*i+1);
	I2.Exchange(2*i+1,2*j+1) = I2.Value(2*i+1,2*j+1,2*j+1,2*i+1);
      }
    }
  }
#else
  template <typename ElemT>
  void SetupIntegrals(const FCIDump & fcidump,
		      int & L,
		      int & N,
		      ElemT & I0,
		      oneInt<ElemT> & I1,
		      twoInt<ElemT> & I2) {

    for(const auto & [key, value] : fcidump.header) {
      if( key == std::string("NORB") ) {
	L = std::atoi(value.c_str());
      }
      if( key == std::string("NELEC") ) {
	N = std::atoi(value.c_str());
      }
    }
    CheckFCIDumpLayout(fcidump, L, false);

    I1.norbs = 2 * L;
    I1.store.resize(4*L*L,ElemT(0.0));

    int B = L*(L+1)/2;
    I2.norbs = L;
    I2.store.resize(B*(B+1)/2,ElemT(0.0));

    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      if( (i==0) && (k==0) && (j==0) && (l==0) ) {
	I0 = ElemT(value);
      } else if( (k==l) && (k==0) ) {
	I1(2*(i-1),2*(j-1)) = ElemT(value);
	I1(2*(i-1)+1,2*(j-1)+1) = ElemT(value);
	I1(2*(j-1),2*(i-1)) = ElemT(value);
	I1(2*(j-1)+1,2*(i-1)+1) = ElemT(value);
      } else {
	I2(2*(i-1),2*(j-1),2*(k-1),2*(l-1)) = ElemT(value);
      }
    }

    I2.DirectMat.resize(4*L*L);
    I2.ExchangeMat.resize(4*L*L);
    for(int i=0; i < L; i++) {
      for(int j=0; j < L; j++) {
	I2.Direct(2*i,2*j) = I2.Value(2*i,2*i,2*j,2*j);
	I2.Direct(2*i,2*j+1) = I2.Value(2*i,2*i,2*j+1,2*j+1);
	I2.Direct(2*i+1,2*j) = I2.Value(2*i+1,2*i+1,2*j,2*j);
	I2.Direct(2*i+1,2*j+1) = I2.Value(2*i+1,2*i+1,2*j+1,2*j+1);
	I2.Exchange(2*i,2*j) = I2.Value(2*i,2*j,2*j,2*i);
	I2.Exchange(2*i,2*j+1) = I2.Value(2*i,2*j+1,2*j+1,2*i);
	I2.Exchange(2*i+1,2*j) = I2.Value(2*i+1,2*j,2*j,2*i+1);
	I2.Exchange(2*i+1,2*j+1) = I2.Value(2*i+1,2*j+1,2*j+1,2*i+1);
      }
    }
  }
#endif
  
}

#endif
