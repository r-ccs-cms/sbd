/**
   @file sbd/caop/basic/loadmodel.h
   @brief construct GeneralOp from file
 */
#ifndef SBD_CAOP_BASIC_LOADMODEL_H
#define SBD_CAOP_BASIC_LOADMODEL_H

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "sbd/framework/fcidump.h"

namespace sbd {

  enum class OpTokenKind {
    C,
    CDag,
    B,
    BDag,
    SPlus,
    SMinus,
    Sx,
    Sy,
    Sz,
    Unknown
  };
  
  // string to type
  inline OpTokenKind parse_op_token(const std::string& token)
  {
    if (token == "c")      return OpTokenKind::C;
    if (token == "cdag")   return OpTokenKind::CDag;
    if (token == "b")      return OpTokenKind::B;
    if (token == "bdag")   return OpTokenKind::BDag;
    if (token == "s+")     return OpTokenKind::SPlus;
    if (token == "s-")     return OpTokenKind::SMinus;
    if (token == "sx")     return OpTokenKind::Sx;
    if (token == "sy")     return OpTokenKind::Sy;
    if (token == "sz")     return OpTokenKind::Sz;
    return OpTokenKind::Unknown;
  }

  inline bool is_effectively_empty_or_comment(const std::string& line) {
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos) {
      return true; // empty 
    }
    return line[first] == '#'; // comment
  }

#ifdef SBD_TRADMODE
  template <class T> struct is_std_complex : std::false_type {};
  template <class U> struct is_std_complex<std::complex<U>> : std::true_type {};
  
  template <typename ElemT, typename TermT, typename ProductOp, typename GeneralOp>
  inline void apply_sy_impl(TermT& term, int index, std::true_type /*is_complex*/) {
    ProductOp pSp(Sp(index));
    ProductOp pSm(Sm(index));
    GeneralOp gSp(ElemT(0.0, -0.5), pSp);
    GeneralOp gSm(ElemT(0.0,  0.5), pSm);
    term *= gSp + gSm;
  }
  
  template <typename ElemT, typename TermT, typename ProductOp, typename GeneralOp>
  inline void apply_sy_impl(TermT&, int, std::false_type /*is_complex*/) {
    throw std::runtime_error("Sy operator requires complex coefficient type");
  }
#endif
  
  template <typename ElemT>
  void load_GeneralOp_from_file(const std::string & filename,
				GeneralOp<ElemT> & op,
				bool & sign,
				MPI_Comm h_comm,
				MPI_Comm b_comm,
				MPI_Comm t_comm) {


    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    size_t max_buffer_size = 1000000;
    int sign_value = 0;

    if( mpi_rank_t == 0 ) {
      if( mpi_rank_b == 0 ) {
	
	int type = 0;
	
	if( mpi_rank_h == 0 ) {
	  
	  std::vector<GeneralOp<ElemT>> send_op(mpi_size_h);
	  std::ifstream ifs(filename);
	  if (!ifs) {
	    throw std::runtime_error("load_from_file: cannot open file: " + filename);
	  }
	  std::string line;
	  bool statistics_set = false;
	  int statistics = 0; // -1: fermion, +1: hardcore boson/spin
	  
	  size_t term_count = 0;
	  size_t rank_count = 0;
	  while (std::getline(ifs, line)) {
	    if (is_effectively_empty_or_comment(line)) {
	      continue;
	    }
	    
	    // first non-comment line is interpreted as statistics line
	    if (!statistics_set) {
	      {
		std::istringstream iss(line);
		if (!(iss >> statistics)) {
		  throw std::runtime_error("load_from_file: failed to read statistics");
		}
	      }
	      statistics_set = true;
	      
	      sign_value = statistics > 0 ? 0 : 1;
	      
	      continue;
	    }
	    
	    // one-line = one-term from here
	    std::istringstream iss(line);
	    
	    ElemT coef{};
	    if (!(iss >> coef)) {
	      throw std::runtime_error("load_from_file: failed to read coefficient");
	      continue;
	    }
	    
	    //
	    ProductOp IdOp;
	    GeneralOp<ElemT> term(IdOp);
	    std::string op_sym;
	    while (iss >> op_sym) {
	      // in-line comment (# ...)
	      if (!op_sym.empty() && op_sym[0] == '#') {
		break;
	      }
	      
	      std::size_t index = 0;
	      if (!(iss >> index)) {
		throw std::runtime_error("load_from_file: missing index after " + op_sym);
	      }
	      
	      OpTokenKind kind = parse_op_token(op_sym);
	      switch (kind) {
	      case OpTokenKind::C:
		term *= An(index);
		break;
		
	      case OpTokenKind::CDag:
		term *= Cr(index);
		break;
		
	      case OpTokenKind::B:
		term *= An(index);
		break;
		
	      case OpTokenKind::BDag:
		term *= Cr(index);
		break;
		
	      case OpTokenKind::SPlus:
		term *= Sp(index);
		break;
		
	      case OpTokenKind::SMinus:
		term *= Sm(index);
		break;
		
	      case OpTokenKind::Sx:
		{
		  ProductOp pSp(Sp(index));
		  ProductOp pSm(Sm(index));
		  GeneralOp<ElemT> gSp(ElemT(0.5),pSp);
		  GeneralOp<ElemT> gSm(ElemT(0.5),pSm);
		  term *= gSp + gSm;
		  break;
		}
		
	      case OpTokenKind::Sy:
#ifdef SBD_TRADMODE
		apply_sy_impl<ElemT,
			      decltype(term),
			      ProductOp,
			      GeneralOp<ElemT>>(term, index, is_std_complex<ElemT>{});
		break;
#else
		if constexpr (std::is_same_v<ElemT,std::complex<float>> ||
			      std::is_same_v<ElemT,std::complex<double>>) {
		  ProductOp pSp(Sp(index));
		  ProductOp pSm(Sm(index));
		  GeneralOp<ElemT> gSp(ElemT(0.0,-0.5),pSp);
		  GeneralOp<ElemT> gSm(ElemT(0.0,0.5),pSm);
		  term *= gSp + gSm;
		} else {
		  throw std::runtime_error(
					   "Sy operator requires complex coefficient type");
		}
		break;
#endif
		
	      case OpTokenKind::Sz:
		term *= Sz<ElemT>(index);
		break;
		
	      case OpTokenKind::Unknown:
	      default:
		throw std::runtime_error("load_from_file: unknown operator symbol: " + op_sym);
	      }
	    }
	    size_t target_rank = rank_count % static_cast<size_t>(mpi_size_h);
	    send_op[target_rank] += coef * term;
	    term_count++;
	    rank_count++;
	    
	    if( term_count == max_buffer_size ) {
	      op += send_op[0];
	      send_op[0] = GeneralOp<ElemT>();
	      for(int rank=1; rank < mpi_size_h; rank++) {
		MPI_Send(&type,1,MPI_INT,rank,0,h_comm);
		MpiSend(send_op[rank],rank,h_comm);
		send_op[rank] = GeneralOp<ElemT>();
	      }
	      term_count = 0;
	    }
	  }
	  type = 1;
	  op += send_op[0];
	  for(int rank=1; rank < mpi_size_h; rank++) {
	    MPI_Send(&type,1,MPI_INT,rank,0,h_comm);
	    MpiSend(send_op[rank],rank,h_comm);
	  }
	  if (!statistics_set) {
	    throw std::runtime_error("load_from_file: statistics line not found");
	  }
	  
	} else {
	  while(type == 0) {
	    GeneralOp<ElemT> recv_op;
	    MPI_Status status;
	    MPI_Recv(&type,1,MPI_INT,0,0,h_comm,&status);
	    MpiRecv(recv_op,0,h_comm);
	    op += recv_op;
	  }
	}
        MPI_Bcast(&sign_value,1,MPI_INT,0,h_comm);
        sign = sign_value != 0;
        NormalOrdering(op,sign);
        Simplify(op);
      }
      MpiBcast(op,0,b_comm);
      MPI_Bcast(&sign_value,1,MPI_INT,0,b_comm);
    }
    MpiBcast(op,0,t_comm);
    MPI_Bcast(&sign_value,1,MPI_INT,0,t_comm);
    sign = sign_value != 0;
  }

  template <typename ElemT>
  void GeneralOp_From_FCIDump(const std::string & filename,
			      MPI_Comm h_comm,
			      MPI_Comm b_comm,
			      int & L,
			      int & N,
			      GeneralOp<ElemT> & H) {

    using RealT = typename GetRealType<ElemT>::RealT;
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);

    FCIDump fcidump;
    if( (mpi_rank_b == 0 ) && (mpi_rank_h == 0) ) {
      fcidump = LoadFCIDump(filename);
    }
    if( mpi_rank_b == 0 ) {
      MpiBcast(fcidump,0,h_comm);
    }
    MpiBcast(fcidump,0,b_comm);

    for(const auto & [key, value] : fcidump.header) {
      if( key == std::string("NORB") ) {
	L = std::atoi(value.c_str());
      }
      if( key == std::string("NELEC") ) {
	N = std::atoi(value.c_str());
      }
    }
    
    size_t m=0;
    H = GeneralOp<ElemT>();
    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      if( (m % mpi_size_h) == mpi_rank_h ) {
	if( (i == 0) && (k == 0) && (j == 0) && (l == 0) ) {
	  H += ElemT(value);
	} else if( (k == l) && (k == 0) ) {
	  GeneralOp<ElemT> T;
	  std::vector<std::vector<int>> index(2,std::vector<int>(2));
	  index[0][0] = i; index[0][1] = j;
	  index[1][0] = j; index[1][1] = i;
	  std::sort(index.begin(),index.end());
	  auto it = std::unique(index.begin(),index.end());
	  index.erase(it,index.end());
	  for(const auto & p : index) {
	    T += ElemT(value) * Cr(p[0]-1) * An(p[1]-1);
	    T += ElemT(value) * Cr(p[0]-1+L) * An(p[1]-1+L);
	  }
	  NormalOrdering(T,true);
	  Simplify(T);
	  H += T;
	} else {
	  // For the definition of indecies, see following link:
	  // https://theochem.github.io/horton/2.0.2/user_hamiltonian_io.html
	  GeneralOp<ElemT> V;
	  std::vector<std::vector<int>> index(8,std::vector<int>(4));
	  index[0][0] = i; index[0][1] = j; index[0][2] = k; index[0][3] = l;
	  index[1][0] = j; index[1][1] = i; index[1][2] = k; index[1][3] = l;
	  index[2][0] = i; index[2][1] = j; index[2][2] = l; index[2][3] = k;
	  index[3][0] = j; index[3][1] = i; index[3][2] = l; index[3][3] = k;
	  index[4][0] = k; index[4][1] = l; index[4][2] = i; index[4][3] = j;
	  index[5][0] = k; index[5][1] = l; index[5][2] = j; index[5][3] = i;
	  index[6][0] = l; index[6][1] = k; index[6][2] = i; index[6][3] = j;
	  index[7][0] = l; index[7][1] = k; index[7][2] = j; index[7][3] = i;
	  std::sort(index.begin(),index.end());
	  auto it = std::unique(index.begin(),index.end());
	  index.erase(it,index.end());
#ifdef SBD_DEBUG
	  std::cout << " index [" << i << "," << j << "," << k << "," << l << "]: size = " << index.size() << std::endl;
#endif
	  
	  for(const auto & p : index) {
	    V += ElemT(0.5*value) * Cr(p[0]-1) * Cr(p[2]-1) * An(p[3]-1) * An(p[1]-1);
	    V += ElemT(0.5*value) * Cr(p[0]-1) * Cr(p[2]-1+L) * An(p[3]-1+L) * An(p[1]-1);
	    V += ElemT(0.5*value) * Cr(p[0]-1+L) * Cr(p[2]-1) * An(p[3]-1) * An(p[1]-1+L);
	    V += ElemT(0.5*value) * Cr(p[0]-1+L) * Cr(p[2]-1+L) * An(p[3]-1+L) * An(p[1]-1+L);
	  }
	  NormalOrdering(V,true);
	  Simplify(V);
	  H += V;
	}
      }
      m++;
    }
    
  }
  
}

#endif
