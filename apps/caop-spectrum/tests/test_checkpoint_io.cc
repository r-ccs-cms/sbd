#include "sbd/caop/spectrum/applied_operator.h"
#include <iostream>
namespace cs=sbd::caop::spectrum;
using C=std::complex<double>;
void require(bool v,const char* msg) {if(!v) throw std::runtime_error(msg);}
void close(C a,C b,const char* msg,double tol=2e-9) {if(!std::isfinite(std::abs(a)) || std::abs(a-b)>tol*(1+std::abs(b))) throw std::runtime_error(msg);}
int main(int argc,char** argv) {
 int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
 omp_set_dynamic(0);int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 try {
  require(provided>=MPI_THREAD_FUNNELED,"MPI thread support");
    // Multiword signs and support union; bit 64 sees occupied lower word.
    cs::Det target;double phase=0;
    require(cs::single_particle_excite(cs::Det{1,0},64,true,true,64,target,phase)&&phase==-1&&target[1]==1,"multiword fermion sign");
    require(!cs::single_particle_excite(cs::Det{1},0,true,true,64,target,phase),"Pauli zero");
    cs::Basis::init_elem_size(1);
    cs::Wavefunction<> state;
    for(int i=0;i<2;++i) if(i%size==rank) {
      state.basis.push_back(cs::Det{std::size_t(1)<<i});
      state.coefficients.push_back(i?.8:.6);
    }
    cs::Basis remap_target;
    if(rank==size-1) for(auto d:{4ul,2ul,2ul}) remap_target.push_back(cs::Det{d});
    auto remap=cs::RemapWavefunction(state,remap_target,MPI_COMM_WORLD);
    close(remap.retained_norm2,.64,"remap retained");close(remap.discarded_norm2,.36,"remap loss");
    std::size_t local_n=remap.state.basis.size(),total_n=0;
    MPI_Allreduce(&local_n,&total_n,1,SBD_MPI_SIZE_T,MPI_SUM,MPI_COMM_WORLD);
    require(total_n==2,"deduplicated distributed remap target");
    for(std::size_t i=0;i<local_n;++i)
      close(remap.state.coefficients[i],remap.state.basis[i][0]==2?.8:0,"remap coefficients");
    auto removed=cs::RemapWavefunction(state,{},MPI_COMM_WORLD);
    require(removed.state.basis.empty(),"empty remap target");
    close(removed.discarded_norm2,1,"all reference weight discarded");
    cs::Basis added;if(!rank) added.push_back(cs::Det{1});
    auto inserted=cs::RemapWavefunction({},added,MPI_COMM_WORLD);
    for(double x:inserted.state.coefficients) close(x,0,"empty source zero fill");
    if(argc>1&&!rank) {
      sbd::det_vector<std::size_t>::init_elem_size(1);
      // Public writer fixture: normalized one-electron state over three modes.
      sbd::det_vector<std::size_t> all;all.push_back(cs::Det{1});all.push_back(cs::Det{2});
      sbd::SaveWavefunction(std::string(argv[1])+"/state-",all,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,std::vector<double>{.6,.8});
      sbd::SaveWavefunction(std::string(argv[1])+"/complex-state-",all,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,
          std::vector<std::complex<double>>{{.6,0},{0,.8}});
      // Check the reader's contract independently of CLI normalization checks.
      const auto scaled_prefix=std::string(argv[1])+"/scaled-";
      sbd::SaveWavefunction(scaled_prefix,all,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,
          std::vector<double>{.3,.4});
      auto loaded=cs::ReadWavefunctionCheckpoint<double>(scaled_prefix,1,4,64,MPI_COMM_SELF);
      close(cs::global_norm2(loaded,MPI_COMM_SELF),.25,"reader must not normalize");
      require(loaded.basis.size()==2,"reader must restore the saved basis");
      for(std::size_t i=0;i<loaded.basis.size();++i)
        close(loaded.coefficients[i],loaded.basis[i][0]==1?.3:.4,"reader amplitude mapping");
      const auto duplicate_prefix=std::string(argv[1])+"/duplicate-";
      cs::Basis duplicate;duplicate.push_back(cs::Det{1});duplicate.push_back(cs::Det{1});
      sbd::SaveWavefunction(duplicate_prefix,duplicate,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,
          std::vector<double>{.3,.4});
      bool rejected=false;
      try { cs::ReadWavefunctionCheckpoint<double>(duplicate_prefix,1,4,64,MPI_COMM_SELF); }
      catch(const std::invalid_argument&) { rejected=true; }
      require(rejected,"reader must reject duplicate saved determinants");
      for(int k=0;k<2;++k) {
        std::string path=sbd::statefilename(std::string(argv[1])+"/two-",k);
        std::ofstream out(path,std::ios::binary);std::size_t n=1,w=1,d=std::size_t(1)<<k;double a=k?.8:.6;
        out.write(reinterpret_cast<char*>(&n),sizeof n);out.write(reinterpret_cast<char*>(&w),sizeof w);out.write(reinterpret_cast<char*>(&d),sizeof d);out.write(reinterpret_cast<char*>(&a),sizeof a);
        std::ofstream complex_out(sbd::statefilename(std::string(argv[1])+"/complex-two-",k),std::ios::binary);
        const std::complex<double> amplitude=k?std::complex<double>(0,.8):std::complex<double>(.6,0);
        complex_out.write(reinterpret_cast<char*>(&n),sizeof n);complex_out.write(reinterpret_cast<char*>(&w),sizeof w);
        complex_out.write(reinterpret_cast<char*>(&d),sizeof d);complex_out.write(reinterpret_cast<const char*>(&amplitude),sizeof amplitude);
      }
    }
    if(!rank) std::cout<<"PASS: checkpoint fixtures, signs and remap norm accounting\n";
    MPI_Finalize();return 0;
  } catch(const std::exception& e) {std::cerr<<"test rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
