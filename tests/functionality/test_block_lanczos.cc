#include "sbd/framework/sparse_solver/block_lanczos.h"
#include <iostream>
#include <utility>
namespace ss=sbd::sparse_solver;
using C=ss::Complex;
void require(bool v,const char* msg) {if(!v) throw std::runtime_error(msg);}
void close(C a,C b,const char* msg,double tol=2e-9) {if(!std::isfinite(std::abs(a)) || std::abs(a-b)>tol*(1+std::abs(b))) throw std::runtime_error(msg);}
void analytic_test(int rank,int size,std::size_t p) {
  const std::size_t n=7;
  auto range=std::make_pair(n/size*rank+std::min(std::size_t(rank),n%size), n/size*(rank+1)+std::min(std::size_t(rank+1),n%size));auto rows=range.second-range.first;
  std::vector<C> u(n*n),h(n*n),f(n*p);
  double energies[n]={-3,-1.4,-.2,.7,1.1,2.3,4};
  const double pi=std::acos(-1.0);
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) u[(i)*n+(j)]=std::polar(1/std::sqrt(double(n)),2*pi*i*j/n);
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) for(std::size_t k=0;k<n;++k) h[(i)*n+(j)]+=u[(i)*n+(k)]*energies[k]*std::conj(u[(j)*n+(k)]);
  for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<p;++j) f[(i)*p+(j)]=C(std::sin((i+1)*(j+1)),std::cos((i+2)*(j+1)));
  // Duplicate and zero physical observables with nontrivial complex C0.
  if(p>2) for(std::size_t i=0;i<n;++i) {f[(i)*p+(p-2)]=C(.3,.7)*f[(i)*p+(0)];f[(i)*p+(p-1)]=0;}
  std::vector<C> q(rows*p);
  for(std::size_t i=0;i<rows;++i) for(std::size_t j=0;j<p;++j) q[i*p+j]=f[(i+range.first)*p+(j)];
  std::vector<std::vector<C>> vectors(p,std::vector<C>(rows));
  for(std::size_t v=0;v<p;++v) for(std::size_t i=0;i<rows;++i) vectors[v][i]=q[i*p+v];
  auto qr=ss::orthogonalize(vectors,MPI_COMM_WORLD);
  auto apply_h=[&](const auto& x,auto& y) {
    for(std::size_t v=0;v<x.size();++v) {
      std::vector<C> global(n);
      for(std::size_t i=0;i<rows;++i) global[i+range.first]=x[v][i];
      MPI_Allreduce(MPI_IN_PLACE,global.data(),int(global.size()),sbd::GetMpiType<C>::MpiT,MPI_SUM,MPI_COMM_WORLD);
      for(std::size_t i=0;i<rows;++i) for(std::size_t k=0;k<n;++k)
        y[v][i]+=h[(i+range.first)*n+k]*global[k];
    }
  };
  const auto c=ss::block_lanczos<C>(apply_h,MPI_COMM_WORLD,std::move(vectors),rows);
  require(c.stop_reason=="rank_threshold","full chain must terminate");
  auto transformed=ss::multiply(ss::adjoint(u,n,n),f,n,n,p);
  for(double w:{-2.0,0.2,2.5}) for(double eta:{.07,.4}) {
    const C z(w,eta);auto g=ss::green_function(c,qr.factor,p,z);
    for(std::size_t a=0;a<p;++a) for(std::size_t b=0;b<p;++b) {
      C exact=0;for(std::size_t k=0;k<n;++k) exact+=std::conj(transformed[(k)*p+(a)])*transformed[(k)*p+(b)]/(z-energies[k]);
      close(g[(a)*p+(b)],exact,"analytic eigenbasis resolvent");
    }
    auto spectral=ss::spectral_matrix(g,p);
    for(std::size_t i=0;i<p;++i) {
      require(spectral[(i)*p+(i)].real()>-1e-10,"spectral positivity");
      for(std::size_t j=0;j<p;++j) close(spectral[(i)*p+(j)],std::conj(spectral[(j)*p+(i)]),"Hermitian spectral matrix");
    }
  }
  const auto zero=ss::multiply(ss::adjoint(qr.factor,qr.rank,p),qr.factor,p,qr.rank,p);
  const auto one=ss::multiply(ss::adjoint(qr.factor,qr.rank,p),ss::multiply(c.A[0],qr.factor,qr.rank,qr.rank,p),p,qr.rank,p);
  const auto exactzero=ss::multiply(ss::adjoint(f,n,p),f,p,n,p);
  const auto exactone=ss::multiply(ss::adjoint(f,n,p),ss::multiply(h,f,n,n,p),p,n,p);
  for(std::size_t i=0;i<zero.size();++i) {close(zero[i],exactzero[i],"zeroth moment");close(one[i],exactone[i],"first moment");}
  // Finite-chain comparison to direct assembled T solve tests rectangular B.
  auto short_c=c;
  if(short_c.A.size()>1) {
    short_c.A.pop_back();short_c.terminal_B=short_c.B.back();
    short_c.B.pop_back();short_c.ranks.pop_back();
  }
  std::size_t dim=0;for(std::size_t j=0;j<short_c.A.size();++j) dim+=short_c.ranks[j];
  std::vector<C> d(dim*dim),rhs(dim*p);C z(.21,.14);
  std::size_t offset=0;
  for(std::size_t j=0;j<short_c.A.size();++j) {
    const auto r=short_c.ranks[j];auto& a=short_c.A[j];
    for(std::size_t i=0;i<r;++i) for(std::size_t k=0;k<r;++k)
      d[(offset+i)*dim+offset+k]=(i==k?z:C{})-a[i*r+k];
    if(j+1<short_c.A.size()) {
      auto& b=short_c.B[j];
      for(std::size_t i=0;i<short_c.ranks[j+1];++i) for(std::size_t k=0;k<r;++k) {
        d[(offset+r+i)*dim+offset+k]=-b[i*r+k];
        d[(offset+k)*dim+offset+r+i]=-std::conj(b[i*r+k]);
      }
    }
    offset+=r;
  }
  for(std::size_t i=0;i<qr.rank;++i) for(std::size_t j=0;j<p;++j) rhs[i*p+j]=qr.factor[i*p+j];
  auto exact=ss::multiply(ss::adjoint(rhs,dim,p),ss::solve(d,rhs,dim,p),p,dim,p);
  auto cf=ss::green_function(short_c,qr.factor,p,z);
  for(std::size_t i=0;i<cf.size();++i) close(cf[i],exact[i],"truncated continued fraction");
  // Explicit metadata must reject inconsistent vector dimensions/ranks.
  auto bad=c;bad.ranks[0]++;
  bool rejected=false;
  try { ss::green_function(bad,qr.factor,p,z); } catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"invalid rank metadata accepted");

}
int main(int argc,char** argv) {
 int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
 int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 omp_set_dynamic(0);
 try {
  require(provided>=MPI_THREAD_FUNNELED,"MPI thread support");
  analytic_test(rank,size,1);analytic_test(rank,size,5);
  const std::size_t rows=rank==0?1:0;
  std::vector<std::vector<double>> q(2,std::vector<double>(rows));
  if(!rank) q[0][0]=1;
  auto qr=ss::orthogonalize(q,MPI_COMM_WORLD);
  require(qr.rank==1,"zero-column deflation");
  auto diagonal=[](const auto& x,auto& y) {
   for(std::size_t v=0;v<x.size();++v) for(std::size_t i=0;i<x[v].size();++i) y[v][i]+=2*x[v][i];
  };
  auto c=ss::block_lanczos<double>(diagonal,MPI_COMM_WORLD,std::move(q),rows);
  require(c.A.size()==1&&c.ranks.back()==0,"invariant subspace deflation");
  auto empty=ss::block_lanczos<double>(diagonal,MPI_COMM_WORLD,{},0);
  auto response=ss::green_function(empty,std::vector<double>{},3,{0,.1});
  require(response.size()==9,"zero response shape");
  for(auto x:response) close(x,0,"zero response");
  if(!rank) std::cout<<"PASS: block Lanczos analytic resolvents, moments, rank deflation, finite chain and empty ranks\n";
  MPI_Finalize();return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
