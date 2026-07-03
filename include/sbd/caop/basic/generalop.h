/**
@file sbd/caop/basic/generalop.h
@brief Class to manage the operator
*/
#ifndef SBD_CAOP_BASIC_GENERALOP_H
#define SBD_CAOP_BASIC_GENERALOP_H

namespace sbd {

  class CAOp;

  class ProductOp;

  template <typename ElemT>
  class GeneralOp;

  class CAOp {
  public:
    CAOp()
      : d_(false), q_(0) {}
    CAOp(bool d, int q)
      : d_(d), q_(q) {}
    CAOp(const CAOp & other)
      : d_(other.d_), q_(other.q_) {}
    ~CAOp() {}

    CAOp & operator = (const CAOp & other) {
      if( this != &other ) {
	copy(other);
      }
      return *this;
    }

    bool operator == (const CAOp & other) const {
      return ( ( d_ == other.d_ ) && ( q_ == other.q_ ) );
    }

    bool operator != (const CAOp & other) const {
      return ( ( d_ != other.d_ ) || ( q_ != other.q_ ) );
    }

    bool operator < (const CAOp & other) const {
      if( q_ < other.q_ ) {
	return true;
      }
      return false;
    }

    bool operator > (const CAOp & other) const {
      if( q_ > other.q_ ) {
	return true;
      }
      return false;
    }

    void dagger() { d_ = !d_; }
    bool d() const { return d_; }
    int q() const { return q_; }

    // out of place functions

    template <typename ElemT_>
    friend void NormalOrdering(const ProductOp & P,
			       GeneralOp<ElemT_> & G,
			       bool sign);
    
    friend void MpiSend(const CAOp & F,
			int destination,
			MPI_Comm comm);

    friend void MpiRecv(CAOp & F,
			int source,
			MPI_Comm comm);

    friend void MpiBcast(CAOp & F,
			 int root,
			 MPI_Comm comm);
    
    template <typename ElemT_>
    friend void mult(const std::vector<ElemT_> & hd,
		     const std::vector<ElemT_> & wk,
		     std::vector<ElemT_> & wb,
		     const std::vector<std::vector<size_t>> & bs,
		     const size_t bit_length,
		     const std::vector<int> & slide,
		     const GeneralOp<ElemT_> & H,
		     bool sign,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm);

    template <typename ElemT_>
    friend void makeCAOpHamDiagTerms(const std::vector<std::vector<size_t>> & bs,
				     const size_t bit_length,
				     const std::vector<int> & slide,
				     const GeneralOp<ElemT_> & H,
				     std::vector<ElemT_> & hii);

    template <typename ElemT_>
    friend void makeCAOpHam(const std::vector<std::vector<size_t>> & bs,
			    const size_t bit_length,
			    const std::vector<int> & slide,
			    const GeneralOp<ElemT_> & H,
			    const bool sign,
			    std::vector<ElemT_> & hii,
			    std::vector<std::vector<std::vector<size_t>>> & ih,
			    std::vector<std::vector<std::vector<size_t>>> & jh,
			    std::vector<std::vector<std::vector<ElemT_>>> & hij,
			    MPI_Comm h_comm,
			    MPI_Comm b_comm,
			    MPI_Comm t_comm);
    
    template <typename ElemT_>
    friend void MeasHamSquare(const GeneralOp<ElemT_> & H,
			      const std::vector<ElemT_> & W,
			      const std::vector<std::vector<size_t>> & basis,
			      size_t bit_length,
			      MPI_Comm & comm,
			      ElemT_ & res,
			      bool sign);
    
    friend std::ostream & operator << (std::ostream & s,
				       const CAOp & o);

    // friend class
    friend class ProductOp;
    
    template <typename ElemT_>
    friend class GeneralOp;

  private:
    bool d_;
    int q_;
    void copy(const CAOp & other) {
      d_ = other.d_;
      q_ = other.q_;
    }
  }; // end CAOp

  

  class ProductOp {
  public:
    ProductOp()
      : n_dag_(0), fops_(0) {}

    ProductOp(const ProductOp & other)
      : n_dag_(other.n_dag_), fops_(other.fops_) {}
    
    ProductOp(const CAOp & other)
      : fops_(1,other) {
      if( other.d_ ) {
	n_dag_ = 1;
      }  else {
	n_dag_ = 0;
      }
    }
    
    ProductOp(int i)
      : n_dag_(0), fops_(i) {}
    
    ~ProductOp() {}

    ProductOp & operator = (const CAOp & other) {
      *this = ProductOp(other);
      return *this;
    }

    ProductOp & operator = (const ProductOp & other) {
      if( this != &other ) {
	copy(other);
      }
      return *this;
    }

    bool operator == (const ProductOp & other) const {
      bool res = true;
      if( fops_.size() != other.fops_.size() ) {
	return false;
      } else {
	for(size_t i=0; i < fops_.size(); i++) {
	  if( fops_[i] != other.fops_[i] ) {
	    res = false;
	    break;
	  }
	}
      }
      return res;
    }

    ProductOp operator * (const ProductOp & other) const {
      ProductOp res(*this);
      for(size_t i=0; i < other.fops_.size(); i++)
	{
	  res.fops_.push_back(other.fops_[i]);
	  if(other.fops_[i].d_)
	    {
	      res.n_dag_ += 1;
	    }
	}
      return res;
    }

    ProductOp operator * (const CAOp & other) const
      {
	ProductOp res(*this);
	res.fops_.push_back(other);
	if( other.d_ )
	  {
	    res.n_dag_ += 1;
	  }
	return res;
      }
    
    ProductOp & operator *= (const ProductOp & other)
      {
	for(size_t i=0; i < other.fops_.size(); i++)
	  {
	    fops_.push_back(other.fops_[i]);
	    if( other.fops_[i].d_ )
	      {
		n_dag_ += 1;
	      }
	  }
	return *this;
      }

    ProductOp & operator *= (const CAOp & other)
      {
	fops_.push_back(other);
	if( other.d_ )
	  {
	    n_dag_ += 1;
	  }
	return *this;
      }
    

    ProductOp extract(int i1, int i2) const
    {
      ProductOp res;
      res.n_dag_ = 0;
      for(int i=i1; i < i2; i++)
	{
	  res.fops_.push_back(this->fops_[i]);
	  if( this->fops_[i].d_ )
	    {
	      res.n_dag_ += 1;
	    }
	}
      return res;
    }

    bool check_diagonal() const {
      int nc=0;
      int na=0;
      for(int i=0; i < fops_.size(); i++) {
	if ( fops_[i].d_ ) {
	  nc++;
	} else {
	  na++;
	}
      }

      if( nc != na ) {
	return false;
      }
    
      for(int i=0; i < fops_.size(); i++) {
	if( fops_[i].d_ ) {
	  CAOp f1 = fops_[i];
	  f1.dagger();
	  bool not_find = true;
	  for(int j=0; j < fops_.size(); j++) {
	    if( i != j ) {
	      if( f1 == fops_[j] ) {
		not_find = false;
	      }
	    }
	  }
	  if( not_find ) {
	    return false;
	  }
	}
      }
      return true;
    }

    size_t size() const { return fops_.size(); }
    int n_dag() const { return n_dag_; }
    const CAOp& FOp(size_t i) const { return fops_[i]; }
    
    void dagger()
    {
      std::vector<CAOp> fops(this->fops_);
      for(size_t i=0; i < fops.size(); i++) {
	fops_[i] = fops[fops_.size()-1-i];
	fops_[i].dagger();
      }
    }

    int max_index() const {
      int int_res=0;
      for(size_t k=0; k < fops_.size(); k++) {
	if( int_res < fops_[k].q_ ) {
	  int_res = fops_[k].q_;
	}
      }
      return int_res;
    }

    // out of place functions
    friend ProductOp operator * (const CAOp & a, const CAOp & b);
    
    template <typename ElemT_>
    friend GeneralOp<ElemT_> operator * (const ProductOp & G, ElemT_ c);
    
    template <typename ElemT_>
    friend GeneralOp<ElemT_> operator + (const ProductOp & P, const GeneralOp<ElemT_> & G);

    template <typename ElemT_>
    friend GeneralOp<ElemT_> operator - (const ProductOp & P, const GeneralOp<ElemT_> & G);
    
    template <typename ElemT_>
    friend void NormalOrdering(const ProductOp & P,
			       GeneralOp<ElemT_> & G,
			       bool sign);

    template <typename ElemT_>
    friend void mult(const std::vector<ElemT_> & hd,
		     const std::vector<ElemT_> & wk,
		     std::vector<ElemT_> & wb,
		     const std::vector<std::vector<size_t>> & bs,
		     const size_t bit_length,
		     const std::vector<int> & slide,
		     const GeneralOp<ElemT_> & H,
		     bool sign,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm);

    template <typename ElemT_>
    friend void makeCAOpHamDiagTerms(const std::vector<std::vector<size_t>> & bs,
				     const size_t bit_length,
				     const std::vector<int> & slide,
				     const GeneralOp<ElemT_> & H,
				     std::vector<ElemT_> & hii);

    template <typename ElemT_>
    friend void makeCAOpHam(const std::vector<std::vector<size_t>> & bs,
			    const size_t bit_length,
			    const std::vector<int> & slide,
			    const GeneralOp<ElemT_> & H,
			    const bool sign,
			    std::vector<ElemT_> & hii,
			    std::vector<std::vector<std::vector<size_t>>> & ih,
			    std::vector<std::vector<std::vector<size_t>>> & jh,
			    std::vector<std::vector<std::vector<ElemT_>>> & hij,
			    MPI_Comm h_comm,
			    MPI_Comm b_comm,
			    MPI_Comm t_comm);

    template <typename ElemT_>
    friend void MeasHamSquare(const GeneralOp<ElemT_> & H,
			      const std::vector<ElemT_> & W,
			      const std::vector<std::vector<size_t>> & basis,
			      size_t bit_length,
			      MPI_Comm & comm,
			      ElemT_ & res,
			      bool sign);
    
    friend void MpiSend(const ProductOp & F,
			int destination,
			MPI_Comm comm);

    friend void MpiRecv(ProductOp & F,
			int source,
			MPI_Comm comm);

    friend void MpiBcast(ProductOp & F,
			 int root,
			 MPI_Comm comm);
    
    friend std::ostream & operator << (std::ostream & s, const ProductOp & op);
    
    friend class CAOp;

    template <typename ElemT_>
    friend class GeneralOp;
    
  private:
    int n_dag_;
    std::vector<CAOp> fops_;
    void copy(const ProductOp & other) {
      n_dag_ = other.n_dag_;
      fops_ = other.fops_;
    }
    
  }; // end ProductOp

  template <typename ElemT>
  class GeneralOp {

  public:
    GeneralOp()
      : e_(0), c_(0), d_(0), o_(0) {}

    GeneralOp(const ProductOp & other)
      : e_(0), c_(1,ElemT(1.0)), d_(0), o_(1,other) {}

    GeneralOp(ElemT c, const ProductOp & other)
      : e_(0), c_(1,c), d_(0), o_(1,other) {}

    GeneralOp(const GeneralOp & other)
      : e_(other.e_), c_(other.c_), d_(other.d_), o_(other.o_) {}

    ~GeneralOp() {}

    GeneralOp & operator = (const GeneralOp & other) {
      if( this != &other ) {
	copy(other);
      }
      return *this;
    }

    GeneralOp & operator = (const ProductOp & other) {
      if( other.check_diagonal() ) {
	e_.resize(1,ElemT(1.0));
	d_.resize(1,other);
	c_.resize(0);
	o_.resize(0);
      } else {
	e_.resize(0);
	d_.resize(0);
	c_.resize(1,ElemT(1.0));
	o_.resize(1,other);
      }
      return *this;
    }

    GeneralOp & operator = (const CAOp & other) {
      e_.resize(0);
      d_.resize(0);
      c_.resize(1,ElemT(1.0));
      ProductOp f(other);
      o_.resize(1,f);
      return *this;
    }

    GeneralOp operator * (const GeneralOp & other) {
      
      GeneralOp res;

      size_t nd = this->d_.size();
      size_t no = this->o_.size();
      size_t md = other.d_.size();
      size_t mo = other.o_.size();
      
      res.e_.resize(nd*md);
      res.d_.resize(nd*md);
      res.c_.resize(nd*mo+no*md+no*mo);
      res.o_.resize(nd*mo+no*md+no*mo);
      
      for(size_t i=0; i < nd; i++) {
	for(size_t j=0; j < md; j++) {
	  res.e_[j+md*i] = this->e_[i] * other.e_[j];
	  res.d_[j+md*i] = this->d_[i] * other.d_[j];
	}
      }

      int nt=0;

      for(size_t i=0; i < nd; i++) {
	for(size_t j=0; j < mo; j++) {
	  res.c_[nt+j+mo*i] = this->e_[i] * other.c_[j];
	  res.o_[nt+j+mo*i] = this->d_[i] * other.o_[j];
	}
      }
      
      nt += nd*mo;
      
      for(size_t i=0; i < no; i++) {
	for(size_t j=0; j < md; j++) {
	  res.c_[nt+j+md*i] = this->c_[i] * other.e_[j];
	  res.o_[nt+j+md*i] = this->o_[i] * other.d_[j];
	}
      }
      
      nt += no*md;
	
      for(size_t i=0; i < no; i++) {
	for(size_t j=0; j < mo; j++) {
	  res.c_[nt+j+mo*i] = this->c_[i] * other.c_[j];
	  res.o_[nt+j+mo*i] = this->o_[i] * other.o_[j];
	}
      }
      
      return res;
    }

    GeneralOp operator * (const ProductOp & other) const {
      GeneralOp res;
      
      if( other.check_diagonal() ) {
	for(size_t i=0; i < d_.size(); i++) {
	  ProductOp term = this->d_[i];
	  term *= other;
	  res.d_.push_back(term);
	  res.e_.push_back(this->e_[i]);
	}
	
	for(size_t i=0; i < o_.size(); i++) {
	  ProductOp term = this->o_[i];
	  term *= other;
	  res.o_.push_back(term);
	  res.c_.push_back(this->c_[i]);
	}
      } else {
	for(size_t i=0; i < d_.size(); i++) {
	  ProductOp term = this->d_[i];
	  term *= other;
	  res.o_.push_back(term);
	  res.c_.push_back(this->e_[i]);
	}
	    
	for(size_t i=0; i < o_.size(); i++) {
	  ProductOp term = this->o_[i];
	  term *= other;
	  res.o_.push_back(term);
	  res.c_.push_back(this->c_[i]);
	}
      }
      return res;
    }
    
    GeneralOp operator * (const CAOp & other) const {
      GeneralOp res;
      
      for(size_t i=0; i < d_.size(); i++) {
	ProductOp term = this->d_[i];
	term *= other;
	res.o_.push_back(term);
	res.c_.push_back(this->e_[i]);
      }
	
      for(size_t i=0; i < o_.size(); i++) {
	ProductOp term = this->o_[i];
	term *= other;
	res.o_.push_back(term);
	res.c_.push_back(this->c_[i]);
      }

      return res;
    }

    GeneralOp & operator *= (const GeneralOp & other) {
      GeneralOp temp(*this);
      *this = temp * other;
      return *this;
    }
    
    GeneralOp & operator *= (const ProductOp & other) {
      GeneralOp temp(*this);
      *this = temp * other;
      return *this;
    }
    
    GeneralOp & operator *= (const CAOp & other) {
      GeneralOp temp(*this);
      *this = temp * other;
      return *this;
    }

    GeneralOp & operator *= (ElemT c) {
      for(size_t i=0; i < this->d_.size(); i++) {
	this->e_[i] *= c;
      }
      for(size_t i=0; i < this->o_.size(); i++) {
	this->c_[i] *= c;
      }
      return *this;
    }
    
    GeneralOp & operator += (const GeneralOp & other) {
      for(size_t i=0; i < other.d_.size(); i++) {
	this->d_.push_back(other.d_[i]);
	this->e_.push_back(other.e_[i]);
      }
      for(size_t i=0; i < other.o_.size(); i++) {
	this->o_.push_back(other.o_[i]);
	this->c_.push_back(other.c_[i]);
      }
      return *this;
    }

    GeneralOp & operator += (const ProductOp & other) {
      if( other.check_diagonal() ) {
	this->d_.push_back(other);
	this->e_.push_back(ElemT(1.0));
      } else {
	this->o_.push_back(other);
	this->c_.push_back(ElemT(1.0));
      }
      return *this;
    }

    GeneralOp & operator += (const CAOp & other) {
      ProductOp temp(other);
      if( temp.check_diagonal() ) {
	this->d_.push_back(temp);
	this->e_.push_back(ElemT(1.0));
      } else {
	this->o_.push_back(temp);
	this->c_.push_back(ElemT(1.0));
      }
      return *this;
    }

    GeneralOp & operator += (ElemT c) {
      ProductOp one = ProductOp();
      this->d_.push_back(one);
      this->e_.push_back(c);
      return *this;
    }

    GeneralOp & operator -= (const GeneralOp & other) {
      for(size_t i=0; i < other.d_.size(); i++) {
	this->d_.push_back(other.d_[i]);
	this->e_.push_back(-other.e_[i]);
      }
      for(size_t i=0; i < other.o_.size(); i++) {
	this->o_.push_back(other.o_[i]);
	this->c_.push_back(-other.c_[i]);
      }
      return *this;
    }

    GeneralOp & operator -= (const ProductOp & other) {
      if( other.check_diagonal() ) {
	this->d_.push_back(other);
	this->c_.push_back(ElemT(-1.0));
      } else {
	this->o_.push_back(other);
	this->c_.push_back(ElemT(-1.0));
      }
      return *this;
    }

    GeneralOp & operator -= (const CAOp & other) {
      ProductOp temp(other);
      if( temp.check_diagonal() ) {
	this->d_.push_back(temp);
	this->e_.push_back(ElemT(-1.0));
      } else {
	this->o_.push_back(temp);
	this->c_.push_back(ElemT(-1.0));
      }
      return *this;
    }

    GeneralOp & operator -= (ElemT c) {
      ProductOp one = ProductOp();
      this->d_.push_back(one);
      this->e_.push_back(-c);
      return *this;
    }
    
    GeneralOp operator + (const GeneralOp & other) const {
      GeneralOp res(*this);
      res += other;
      return res;
    }
      
    GeneralOp operator + (const ProductOp & other) const {
      GeneralOp res(*this);
      res += other;
      return res;
    }

    GeneralOp operator + (const CAOp & other) const {
      GeneralOp res(*this);
      res += ProductOp(other);
      return res;
    }

    GeneralOp operator + (ElemT c) {
      GeneralOp res(*this);
      res += c;
      return res;
    }

    // take Hermitian conjugate
    void dagger() {
      for(size_t i=0; i < this->d_.size(); i++) {
	ElemT c = Conjugate(this->e_[i]);
	this->e_[i] = c;
      }
      for(size_t i=0; i < this->o_.size(); i++) {
	this->o_[i].dagger();
	ElemT c = Conjugate(this->c_[i]);
	this->c_[i] = c;
      }
    }

    size_t size() const { return d_.size()+o_.size(); }
    int num_terms() const { return d_.size()+o_.size(); }

    ProductOp & NcTerm(int i) { return d_[i]; }
    ProductOp NcTerm(int i) const { return d_[i]; }
    ElemT & NcCoef(int i) { return e_[i]; }
    ElemT NcCoef(int i) const { return e_[i]; }
    size_t NumNcTerms() const { return d_.size(); }

    ProductOp & OpTerm(int i) { return o_[i]; }
    ProductOp OpTerm(int i) const { return o_[i]; }
    ElemT & OpCoef(int i) { return c_[i]; }
    ElemT OpCoef(int i) const { return c_[i]; }
    size_t NumOpTerms() const { return o_.size(); }

    int max_index() const {
      int int_res = 0;
      for(int k=0; k < d_.size(); k++) {
	int int_temp = d_[k].max_index();
	if( int_res < int_temp ) {
	  int_res = int_temp;
	}
      }
      for(int k=0; k < o_.size(); k++) {
	int int_temp = o_[k].max_index();
	if( int_res < int_temp ) {
	  int_res = int_temp;
	}
      }
      return int_res;
    }
    
    // out-of-place function
    template <typename ElemT_>
    friend void NormalOrdering(const ProductOp & P,
			       GeneralOp<ElemT_> & G,
			       bool sign);
    
    template <typename ElemT_>
    friend void NormalOrdering(GeneralOp<ElemT_> & G,
			       bool sign);

    template <typename ElemT_>
    friend void Simplify(GeneralOp<ElemT_> & G);

    template <typename ElemT_>
    friend void mult(const std::vector<ElemT_> & hd,
		     const std::vector<ElemT_> & wk,
		     std::vector<ElemT_> & wb,
		     const std::vector<std::vector<size_t>> & bs,
		     const size_t bit_length,
		     const std::vector<int> & slide,
		     const GeneralOp<ElemT_> & H,
		     bool sign,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm);

    template <typename ElemT_>
    friend void makeCAOpHamDiagTerms(const std::vector<std::vector<size_t>> & bs,
				     const size_t bit_length,
				     const std::vector<int> & slide,
				     const GeneralOp<ElemT_> & H,
				     std::vector<ElemT_> & hii);

    template <typename ElemT_>
    friend void makeCAOpHam(const std::vector<std::vector<size_t>> & bs,
			    const size_t bit_length,
			    const std::vector<int> & slide,
			    const GeneralOp<ElemT_> & H,
			    const bool sign,
			    std::vector<ElemT_> & hii,
			    std::vector<std::vector<std::vector<size_t>>> & ih,
			    std::vector<std::vector<std::vector<size_t>>> & jh,
			    std::vector<std::vector<std::vector<ElemT_>>> & hij,
			    MPI_Comm h_comm,
			    MPI_Comm b_comm,
			    MPI_Comm t_comm);
    
    template <typename ElemT_>
    friend void MeasHamSquare(const GeneralOp<ElemT_> & H,
			      const std::vector<ElemT_> & W,
			      const std::vector<std::vector<size_t>> & basis,
			      size_t bit_length,
			      MPI_Comm & comm,
			      ElemT_ & res,
			      bool sign);
    
    
    template <typename ElemT_>
    friend void MpiSend(const GeneralOp<ElemT_> & G, int destination, MPI_Comm comm);

    template <typename ElemT_>
    friend void MpiRecv(GeneralOp<ElemT_> & G, int source, MPI_Comm comm);

    template <typename ElemT_>
    friend void MpiBcast(GeneralOp<ElemT_> & G, int root, MPI_Comm comm);

    template <typename ElemT_>
    friend std::ostream & operator << (std::ostream & s,
				       const GeneralOp<ElemT_> & gop);

  private:
    std::vector<ElemT> e_; // coefficient for diagonal part
    std::vector<ElemT> c_; // coefficient for off-diagonal part
    std::vector<ProductOp> d_; // diagonal part
    std::vector<ProductOp> o_; // off-diagonal part
    void copy(const GeneralOp & other) {
      e_ = other.e_;
      c_ = other.c_;
      d_ = other.d_;
      o_ = other.o_;
    }

    void init() {
      e_.resize(0);
      c_.resize(0);
      d_.resize(0);
      o_.resize(0);
    }
    
  }; // end GeneralOp

  
  
}

#endif // SBD_HCBOSON_GENERALOP_H

