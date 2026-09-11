/**
@file sbd/caop/basic/arithmetic.h
@brief arithmetic for GeneralOp class
*/
#ifndef SBD_CAOP_BASIC_ARITHMETIC_H
#define SBD_CAOP_BASIC_ARITHMETIC_H

namespace sbd {

  ProductOp operator * (const CAOp & a, const CAOp & b) {
    ProductOp res(a);
    res *= b;
    return res;
  }

  template <typename ElemT>
  GeneralOp<ElemT> operator + (const ProductOp & P, const GeneralOp<ElemT> & G) {
    GeneralOp<ElemT> res(G);
    res += P;
    return res;
  }
  
  template <typename ElemT>
  GeneralOp<ElemT> operator - (const ProductOp & P, const GeneralOp<ElemT> & G) {
    GeneralOp<ElemT> res(P);
    res -= G;
    return res;
  }

  template <typename ElemT>
  GeneralOp<ElemT> operator * (ElemT c, const CAOp & a) {
    ProductOp P(a);
    return GeneralOp<ElemT>(c,P);
  }

  template <typename ElemT>
  GeneralOp<ElemT> operator * (ElemT c, const GeneralOp<ElemT> & G) {
    GeneralOp<ElemT> res(G);
    res *= c;
    return res;
  }

  // sign=true: fermions; sign=false: hard-core bosons (not ordinary bosons).
  // Both obey {a_i, a_i^dagger}=1 and a_i^2=(a_i^dagger)^2=0.
  // At distinct sites fermions anticommute, while hard-core bosons commute.
  template <typename ElemT>
  void NormalOrdering(const ProductOp & P,
		      GeneralOp<ElemT> & G,
		      bool sign) {
    GeneralOp<ElemT> temp(P);
    std::vector<int> osign(1);
    std::vector<int> indcr(1);
    
    int n_cr = 0;
    int n_an = 0;
    
    for(int i=0; i < P.fops_.size(); i++) {
      if( P.fops_[i].d_ ) n_cr++;
      else                n_an++;
    }
    
    indcr[0] = 0;
    osign[0] = 1;
    
    int m_an;
    for(int i=0; i < n_cr; i++) {
      int msize = temp.o_.size();
      for(int m=0; m < msize; m++) {
	m_an = 0;
	ProductOp tpop = temp.o_[m];
	for(int k=indcr[m]; k < tpop.size(); k++) {
	  if( tpop.fops_[k].d_ ) {
	    m_an = k-indcr[m];
	    break;
	  }
	}
	    
	CAOp ft1(tpop.fops_[indcr[m]+m_an]);
	ft1.dagger();
	    
	for(int k=0; k < m_an; k++) {
	  if( tpop.fops_[indcr[m]+m_an-k-1] == ft1 ) {
	    ProductOp term = tpop.extract(0,indcr[m]+m_an-k-1);
	    term *= tpop.extract(indcr[m]+m_an-k+1,tpop.size());
	    temp.o_.push_back(term);
	    indcr.push_back(indcr[m]);
	    osign.push_back(osign[m]);
	  }
	  CAOp ft2(tpop.fops_[indcr[m]+m_an-k-1]);
	  if( sign || ft2 == ft1 ) {
	    osign[m] *= -1;
	  } else {
	    osign[m] *= 1;
	  }
	  tpop.fops_[indcr[m]+m_an-k-1] = tpop.fops_[indcr[m]+m_an-k];
	  tpop.fops_[indcr[m]+m_an-k] = ft2;
	}
	temp.o_[m] = tpop;
	indcr[m] += 1;
      }
    }
    
    // reordering the all Cdag operators into a regular ordering
    // i.e.,
    //    sign  cdag(x1)  cdag(x2)  ... cdag(xn)
    // -> sign' cdag(x1') cdag(x2') ... cdag(xn')
    // where  x1' <= x2' <= ... <= xn'

    for(int m=0; m < temp.o_.size(); m++) {
      ProductOp tpop = temp.o_[m];
      for(int k=0; k < indcr[m]-1; k++) {
	CAOp ft1(tpop.fops_[k]);
	for(int l=k+1; l < indcr[m]; l++) {
	  CAOp ft2(tpop.fops_[l]);
	  if( ft2 < ft1 ) {
	    if( sign ) {
	      osign[m] *= -1;
	    } else {
	      osign[m] *= 1;
	    }
	    tpop.fops_[k] = ft2;
	    tpop.fops_[l] = ft1;
	    ft1 = ft2;
	  }
	}
	temp.o_[m] = tpop;
      }
    }

    // std::cout << " End Re-ordering of cdag " << std::endl;

    // reordering the all c operators into a regular order
    //    sign  c(x1)  c(x2)  ...  c(xn)
    // -> sign' c(x1') c(x2') ...  c(xn')
    // where x1' >= x2' >= ... >= xn'

    for(int m=0; m < temp.o_.size(); m++) {
      ProductOp tpop = temp.o_[m];
      for(int k=indcr[m]; k < tpop.size(); k++) {
	CAOp ft1(tpop.fops_[k]);
	for(int l=k+1; l < tpop.size(); l++) {
	  CAOp ft2(tpop.fops_[l]);
	  if( ft1 < ft2 ) {
	    if( sign ) {
	      osign[m] *= -1;
	    } else {
	      osign[m] *= 1;
	    }
	    tpop.fops_[k] = ft2;
	    tpop.fops_[l] = ft1;
	    ft1 = ft2;
	  }
	}
	temp.o_[m] = tpop;
      }
    }

    // std::cout << " End Re-ordering of c " << std::endl;
    // std::cout << temp << std::endl;

    // eliminating (cdag)^2 and (c)^2 operators
      
    int n_op = 0;
    GeneralOp<ElemT> bres;
    for(int m=0; m < temp.o_.size(); m++) {
      bool b = true;
      ProductOp tpop = temp.o_[m];
	
      for(int k=0; k < indcr[m]; k++) {
	CAOp ft1(tpop.fops_[k]);
	for(int l=k+1; l < indcr[m]; l++) {
	  CAOp ft2(tpop.fops_[l]);
	  if( ft1 == ft2 ) {
	    b = false;
	    break;
	  }
	}
	if( b == false) {
	  break;
	}
      }
	
      for(int k=indcr[m]; k < tpop.size(); k++) {
	CAOp ft1(tpop.fops_[k]);
	for(int l=k+1; l < tpop.size(); l++) {
	  CAOp ft2(tpop.fops_[l]);
	  if( ft1 == ft2 ) {
	    b = false;
	    break;
	  }
	}
	if( b == false ) {
	  break;
	}
      }
      
      if( b == true ) {
	bres.o_.push_back(tpop);
	bres.c_.push_back(1.0*osign[m]);
	n_op++;
      }
    }


    // std::cout << " End elimination of same operators " << std::endl;
    // std::cout << bres << std::endl;

    // check whether there are same operators
    GeneralOp<ElemT> res;
    std::vector<int> isim(bres.o_.size(),1);
    
    ElemT value;
    int n_dp = 0;
    n_op = 0;
      
    for(int m=0; m < bres.o_.size(); m++) {
      if( isim[m] == 1 ) {
	// ProductOp op1 = bres.OpTerm(m);
	ProductOp op1 = bres.o_[m];
	isim[m] = 0;
	value = bres.c_[m];
	for(int l=m+1; l < bres.o_.size(); l++) {
	  if( isim[l] == 1 ) {
	    ProductOp op2 = bres.o_[l];
	    if( op1 == op2 ) {
	      isim[l] = 0;
	      value += bres.c_[l];
	    }
	  }
	}

	if( std::abs(value) > 1.0e-8 ) {
	  if( op1.check_diagonal() ) {
	    res.d_.push_back(op1);
	    res.e_.push_back(value);
	    n_dp++;
	  } else {
	    res.o_.push_back(op1);
	    res.c_.push_back(value);
	    n_op++;
	  }
	}
      }
    }

    G = res;
    
  }
    
  template <typename ElemT>
  void NormalOrdering(GeneralOp<ElemT> & res,
		      bool sign) {
    GeneralOp<ElemT> G(res);
    res = GeneralOp<ElemT>();
    int n_d = 0;
    int n_o = 0;
    for(int m=0; m < G.d_.size(); m++) {
      ProductOp pop(G.d_[m]);
      GeneralOp<ElemT> gop;
      NormalOrdering(pop,gop,sign);

      int l_d = 0;
      for(int k=0; k < gop.d_.size(); k++) {
	res.d_.push_back(gop.d_[k]);
	res.e_.push_back(G.e_[m]*gop.e_[k]);
	l_d++;
      }
      n_d += l_d;
    }

    for(int m=0; m < G.o_.size(); m++) {
      ProductOp pop(G.o_[m]);
      GeneralOp<ElemT> gop;
      NormalOrdering(pop,gop,sign);

      for(int k=0; k < gop.d_.size(); k++) {
	res.d_.push_back(gop.d_[k]);
	ElemT c = G.c_[m] * gop.e_[k];
	res.e_.push_back(c);
      }

      for(int k=0; k < gop.o_.size(); k++) {
	res.o_.push_back(gop.o_[k]);
	ElemT c = G.c_[m] * gop.c_[k];
	res.c_.push_back(c);
      }
    }
    G =res;
  }

  template <typename ElemT>
  void Simplify(GeneralOp<ElemT> & G) {
    GeneralOp<ElemT> res;
    ElemT value;
    std::vector<int> identical_d(G.d_.size(),1);
    std::vector<int> identical_o(G.o_.size(),1);
    for(int m=0; m < G.d_.size(); m++) {
      if( identical_d[m] == 1 ) {
	ProductOp op1(G.d_[m]);
	identical_d[m] = 0;
	value = G.e_[m];
	for(int l=m+1; l < G.d_.size(); l++){
	  if( identical_d[l] == 1 ) {
	    ProductOp op2(G.d_[l]);
	    if( op1 == op2 ) {
	      identical_d[l] = 0;
	      value += G.e_[l];
	    }
	  }
	}
	if( std::abs(value) > 1.0e-8 ) {
	  res.d_.push_back(op1);
	  res.e_.push_back(value);
	}
      }
    }
    
    for(int m=0; m < G.o_.size(); m++) {
      if( identical_o[m] == 1 ) {
	ProductOp op1(G.o_[m]);
	identical_o[m] = 0;
	value = G.c_[m];
	for(int l=m+1; l < G.o_.size(); l++) {
	  if( identical_o[l] == 1 ) {
	    ProductOp op2(G.o_[l]);
	    if( op1 == op2 ) {
	      identical_o[l] = 0;
	      value += G.c_[l];
	    }
	  }
	}
	if( std::abs(value) > 1.0e-8 ) {
	  res.o_.push_back(op1);
	  res.c_.push_back(value);
	}
      }
    }
    G = res;
  }

  void MpiSend(const CAOp & F,
	       int dest,
	       MPI_Comm comm) {
    MPI_Send(&F.d_,1,MPI_C_BOOL,dest,0,comm);
    MPI_Send(&F.q_,1,MPI_INT,dest,0,comm);
  }
  
  void MpiRecv(CAOp & F,
	       int source,
	       MPI_Comm comm) {
    MPI_Status status;
    MPI_Recv(&F.d_,1,MPI_C_BOOL,source,0,comm,&status);
    MPI_Recv(&F.q_,1,MPI_INT,source,0,comm,&status);
  }
  
  void MpiBcast(CAOp & F,
		int root,
		MPI_Comm comm) {
    MPI_Bcast(&F.d_,1,MPI_C_BOOL,root,comm);
    MPI_Bcast(&F.q_,1,MPI_INT,root,comm);
  }

  void MpiSend(const ProductOp & P,
	       int dest,
	       MPI_Comm comm) {
    MPI_Send(&P.n_dag_,1,MPI_INT,dest,0,comm);
    size_t p_size = P.size();
    MPI_Send(&p_size,1,SBD_MPI_SIZE_T,dest,0,comm);
    for(size_t i=0; i < p_size; i++) {
      MpiSend(P.fops_[i],dest,comm);
    }
  }
  
  void MpiRecv(ProductOp & P,
	       int source,
	       MPI_Comm comm) {
    MPI_Status status;
    MPI_Recv(&P.n_dag_,1,MPI_INT,source,0,comm,&status);
    size_t p_size;
    MPI_Recv(&p_size,1,SBD_MPI_SIZE_T,source,0,comm,&status);
    P.fops_.resize(p_size);
    for(size_t i=0; i < p_size; i++) {
      MpiRecv(P.fops_[i],source,comm);
    }
  }

  void MpiBcast(ProductOp & P,
		int root,
		MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    size_t p_size;
    if( mpi_rank == 0 ) {
      p_size = P.fops_.size();
    }
    MPI_Bcast(&P.n_dag_,1,MPI_INT,root,comm);
    MPI_Bcast(&p_size,1,SBD_MPI_SIZE_T,root,comm);
    if( mpi_rank != 0 ) {
      P.fops_.resize(p_size);
    }
    for(size_t i=0; i < p_size; i++) {
      MpiBcast(P.fops_[i],root,comm);
    }
  }
  
  // friend function
  template <typename ElemT>
  void MpiSend(const GeneralOp<ElemT> & G,
	       int dest,
	       MPI_Comm comm) {
    std::vector<size_t> sizes(4);
    sizes[0] = G.e_.size();
    sizes[1] = G.d_.size();
    sizes[2] = G.c_.size();
    sizes[3] = G.o_.size();
    MPI_Send(sizes.data(),4,SBD_MPI_SIZE_T,dest,0,comm);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( sizes[0] != 0 ) {
      MPI_Send(G.e_.data(),sizes[0],DataT,dest,0,comm);
    }
    if( sizes[1] != 0 ) {
      for(size_t i=0; i < sizes[1]; i++) {
	MpiSend(G.d_[i],dest,comm);
      }
    }
    if( sizes[2] != 0 ) {
      MPI_Send(G.c_.data(),sizes[2],DataT,dest,0,comm);
    }
    if( sizes[3] != 0 ) {
      for(size_t i=0; i < sizes[3]; i++) {
	MpiSend(G.o_[i],dest,comm);
      }
    }
  }

  template <typename ElemT>
  void MpiRecv(GeneralOp<ElemT> & G,
	       int source,
	       MPI_Comm comm) {
    MPI_Status status;
    std::vector<size_t> sizes(4);
    MPI_Recv(sizes.data(),4,SBD_MPI_SIZE_T,source,0,comm,&status);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( sizes[0] != 0 ) {
      G.e_.resize(sizes[0]);
      MPI_Recv(G.e_.data(),sizes[0],DataT,source,0,comm,&status);
    }
    if( sizes[1] != 0 ) {
      G.d_.resize(sizes[1]);
      for(size_t i=0; i < sizes[1]; i++) {
	MpiRecv(G.d_[i],source,comm);
      }
    }
    if( sizes[2] != 0 ) {
      G.c_.resize(sizes[2]);
      MPI_Recv(G.c_.data(),sizes[2],DataT,source,0,comm,&status);
    }
    if( sizes[3] != 0 ) {
      G.o_.resize(sizes[3]);
      for(size_t i=0; i < sizes[3]; i++) {
	MpiRecv(G.o_[i],source,comm);
      }
    }
  }

  template <typename ElemT>
  void MpiBcast(GeneralOp<ElemT> & G,
		int root,
		MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    std::vector<size_t> sizes(4);
    if( mpi_rank == 0 ) {
      sizes[0] = G.e_.size();
      sizes[1] = G.d_.size();
      sizes[2] = G.c_.size();
      sizes[3] = G.o_.size();
    }
    MPI_Bcast(sizes.data(),4,SBD_MPI_SIZE_T,root,comm);
    if( mpi_rank != 0 ) {
      G.e_.resize(sizes[0]);
      G.d_.resize(sizes[1]);
      G.c_.resize(sizes[2]);
      G.o_.resize(sizes[3]);
    }
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    if( sizes[0] != 0 ) {
      MPI_Bcast(G.e_.data(),sizes[0],DataT,root,comm);
    }
    if( sizes[1] != 0 ) {
      for(size_t i=0; i < sizes[1]; i++) {
	MpiBcast(G.d_[i],root,comm);
      }
    }
    if( sizes[2] != 0 ) {
      MPI_Bcast(G.c_.data(),sizes[2],DataT,root,comm);
    }
    if( sizes[3] != 0 ) {
      for(size_t i=0; i < sizes[3]; i++) {
	MpiBcast(G.o_[i],root,comm);
      }
    }
  }
  
  std::ostream & operator << (std::ostream & s,
			      const CAOp & fop) {
    if( fop.d_ ) {
      s << "Adag(";
    } else {
      s << "A(";
    }
    s << fop.q_ << ")";
    return s;
  }
  
  std::ostream & operator << (std::ostream & s,
			      const ProductOp & pop) {
    for(size_t i=0; i < pop.size(); i++) {
      s << pop.fops_[i];
    }
    return s;
  }
  
  template <typename ElemT>
  std::ostream & operator << (std::ostream & s,
			      const GeneralOp<ElemT> & gop) {
    if( gop.d_.size() != 0 ) {
      s << "   " << gop.e_[0] << gop.d_[0];
      s << " (# of creation " << gop.d_[0].n_dag() << ")";
      for(size_t m=1; m < gop.d_.size(); m++) {
	s << std::endl;
	s << " + " << gop.e_[m] << gop.d_[m];
	s << " (# of creation " << gop.d_[m].n_dag() << ")";
      }
      for(size_t m=0; m < gop.o_.size(); m++) {
	s << std::endl;
	s << " + " << gop.c_[m] << gop.o_[m];
	s << " (# of creation " << gop.o_[m].n_dag() << ")";
      }
      s << std::endl;
      s << " # of diag. ops, # of off-diag. ops = "
	<< gop.d_.size() << "(" << gop.e_.size() << "), "
	<< gop.o_.size() << "(" << gop.c_.size() << ")" << std::endl;
    } else if( gop.o_.size() != 0 ) {
      s << "   " << gop.c_[0] << gop.o_[0];
      s << " (# of creation " << gop.o_[0].n_dag() << ")";
      
      for(size_t m=1; m < gop.size(); m++) {
	s << std::endl;
	s << " + " << gop.c_[m] << gop.o_[m];
	s << " (# of creation " << gop.o_[m].n_dag() << ")";
      }
      s << std::endl;
      s << " # of diag. ops, # of off-diag. ops = "
	<< gop.d_.size() << "(" << gop.e_.size() << "), "
	<< gop.o_.size() << "(" << gop.c_.size() << ")" << std::endl;
    }
    return s;
  }


  CAOp Sp(int q) {
    return CAOp(false,q);
  }

  CAOp Sm(int q) {
    return CAOp(true,q);
  }

  template <typename ElemT>
  GeneralOp<ElemT> Sz(int q) {
    GeneralOp<ElemT> res;
    res += Sm(q) * Sp(q);
    res *= ElemT(-1.0);
    res += ElemT(0.5);
    return res;
  }

  CAOp An(int q) {
    return CAOp(false,q);
  }

  CAOp Cr(int q) {
    return CAOp(true,q);
  }
  
} // end namespace sbd

#endif // end #ifndef SBD_HCBOSON_OUT_OF_PLACE_FUNC_ARITHMETIC_H
