#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include "sbd/caop/spectrum/checkpoint_io.h"
#include "sbd/caop/spectrum/block_lanczos_coefficient_io.h"
#include "sbd/caop/spectrum/spectrum.h"
#include "sbd/caop/basic/arithmetic.h"
#include "sbd/caop/basic/loadmodel.h"
#include "sbd/framework/timestamp.h"

#include "cli_options.h"

namespace cs=sbd::caop::spectrum;
namespace ss=sbd::sparse_solver;
namespace cli=sbd_caop_spectrum_cli;
// Only world rank zero writes. Flush workflow boundaries even to a log file.
void log_record(int rank,const std::string& record) {
  if(!rank) std::cout<<' '<<sbd::make_timestamp()<<" sbd::spectrum: "<<record<<std::endl;
}
template<class Function>
void logged_phase(int rank,const char* name,Function&& action) {
  log_record(rank,std::string("start ")+name);
  const double start=MPI_Wtime();action();
  std::ostringstream line;line<<std::setprecision(9)<<"end "<<name
      <<" elapsed="<<MPI_Wtime()-start<<" sec";log_record(rank,line.str());
}
template<class ElemT> void run(const cli::Options& o,int rank) {
    cs::SpectrumResult<ElemT> result;
    if(!o.read.empty()) logged_phase(rank,"coefficient_read",[&] {result=cs::load_coefficients<ElemT>(o.read);});
    else {
      sbd::det_vector<std::size_t>::init_elem_size((o.sites+o.bits-1)/o.bits);
      int world_size=1;MPI_Comm_size(MPI_COMM_WORLD,&world_size);
      if(o.h_size>std::size_t(world_size)||world_size%o.h_size ||
         o.t_size>std::size_t(world_size)/o.h_size || (world_size/o.h_size)%o.t_size)
        throw std::invalid_argument("h*t must divide MPI size");
      const int b_size=world_size/o.h_size/o.t_size;
      MPI_Comm h_comm,b_comm,t_comm;
      sbd::setup_communicator(MPI_COMM_WORLD,int(o.h_size),b_size,int(o.t_size),
          h_comm,b_comm,t_comm);
      int h_rank=0,t_rank=0;
      MPI_Comm_rank(h_comm,&h_rank);MPI_Comm_rank(t_comm,&t_rank);
      cs::Wavefunction<ElemT> state;cs::Basis extra;
      if(h_rank==0 && t_rank==0) {
        logged_phase(rank,"checkpoint_read",[&] {
        if(o.wavefunction_type=="real") {
          auto real_state=cs::ReadWavefunctionCheckpoint<double>(o.load,o.shards,o.sites,o.bits,b_comm);
          state.basis=std::move(real_state.basis);
          state.coefficients.assign(real_state.coefficients.begin(),real_state.coefficients.end());
        } else state=cs::ReadWavefunctionCheckpoint<ElemT>(o.load,o.shards,o.sites,o.bits,b_comm);
        });
        if(!o.remap.empty()) {
          logged_phase(rank,"reference_remap",[&] {
          auto r=cs::RemapWavefunction(state,cs::LoadBasis(o.remap,o.sites,o.bits,b_comm),b_comm);
          std::ostringstream weights;weights<<std::setprecision(17)<<"remap retained_norm2="<<r.retained_norm2<<" discarded_norm2="<<r.discarded_norm2;
          log_record(rank,weights.str());
          state=std::move(r.state);
          if(o.normalize) {
            if(!(r.retained_norm2>0)) throw std::invalid_argument("cannot normalize zero remapped state");
            for(auto& c:state.coefficients) c/=std::sqrt(r.retained_norm2);
          }
          });
        }
        logged_phase(rank,"extra_basis_read",[&] {extra=cs::LoadBasis(o.extra,o.sites,o.bits,b_comm);});
      }
      sbd::GeneralOp<ElemT> h;bool fermion=false;
      logged_phase(rank,"hamiltonian_read",[&] {sbd::load_GeneralOp_from_file(o.ham,h,fermion,h_comm,b_comm,t_comm);});
      std::vector<sbd::GeneralOp<ElemT>> applied;
      const bool general=o.applied_operator_type=="general";
      auto observables=o.orbitals;
      if(general) {
        observables.resize(o.operator_files.size());std::iota(observables.begin(),observables.end(),0);
        if(h_rank==0 && t_rank==0) {
          logged_phase(rank,"operator_read",[&] {
          applied.resize(o.operator_files.size());
          for(std::size_t v=0;v<applied.size();++v) {
            bool op_fermion=false;
            sbd::load_GeneralOp_from_file(o.operator_files[v],applied[v],op_fermion,MPI_COMM_SELF,b_comm,MPI_COMM_SELF);
            if(op_fermion!=fermion) throw std::invalid_argument("operator/H statistics mismatch");
          }
          });
        }
      }
      auto report=[&](const cs::SpectrumProgress& p) {
        if(rank) return;
        std::ostringstream line;line<<std::setprecision(9);
        if(std::string(p.stage)=="iteration") {
          if(!o.iteration_log_interval || p.iteration.step%o.iteration_log_interval) return;
          line<<"iteration step="<<p.iteration.step<<" rank="<<p.iteration.current_rank
              <<" next_rank="<<p.iteration.next_rank<<" elapsed="<<p.elapsed_seconds<<" sec";
        } else {
          line<<(p.completed?"end ":"start ")<<p.stage;
          if(p.completed) {
            line<<" elapsed="<<p.elapsed_seconds<<" sec";
            if(std::string(p.stage)=="seed_generation")
              line<<" reference_basis="<<p.reference_basis_size
                  <<" generated_basis="<<p.generated_basis_size<<" excitation_basis="<<p.basis_size;
            if(std::string(p.stage)=="seed_qr")
              line<<" seed_count="<<p.seed_count<<" seed_rank="<<p.seed_rank;
          }
        }
        log_record(rank,line.str());
      };
      result=cs::spectrum(state,h,fermion,observables,general||o.channel=="addition",o.sites,o.bits,
            extra,o.energy,h_comm,b_comm,t_comm,o.lanczos,general?&applied:nullptr,o.hamiltonian_mode=="stored",report);
      result.operator_files=o.operator_files;
      MPI_Comm_free(&t_comm);MPI_Comm_free(&b_comm);MPI_Comm_free(&h_comm);
      if(!rank) {
        logged_phase(rank,"coefficient_write",[&] {cs::save_coefficients(o.save,result,o.lanczos);});
        std::ostringstream summary;summary<<"result basis="<<result.basis_size
          <<" seed_rank="<<result.coefficients.ranks.front()<<" steps="<<result.coefficients.A.size()
          <<" stop="<<result.coefficients.stop_reason<<" local_correction="<<result.coefficients.max_local_correction;
        log_record(rank,summary.str());
      }
    }
    if(!rank&&!o.output.empty()) {
      logged_phase(rank,"response_and_csv_write",[&] {
      std::ofstream out(o.output);if(!out) throw std::runtime_error("cannot write spectrum");
      out<<(result.operator_files.empty()?"omega,orbital_a,orbital_b,G_real,G_imag,A_real,A_imag\n":"omega,operator_a,operator_b,G_real,G_imag,A_real,A_imag\n")<<std::setprecision(17);
      for(std::size_t k=0;k<o.points;++k) {
        const double w=o.points==1?o.min:o.min+(o.max-o.min)*double(k)/double(o.points-1);
        const auto g=cs::green_function(result,w,o.eta);const auto a=ss::spectral_matrix(g,result.orbitals.size());
        for(std::size_t i=0;i<result.orbitals.size();++i) for(std::size_t j=0;j<result.orbitals.size();++j)
          out<<w<<','<<result.orbitals[i]<<','<<result.orbitals[j]<<','<<g[i*result.orbitals.size()+j].real()<<','<<g[i*result.orbitals.size()+j].imag()<<','<<a[i*result.orbitals.size()+j].real()<<','<<a[i*result.orbitals.size()+j].imag()<<'\n';
      }
      out.close();if(!out) throw std::runtime_error("spectrum write failed");
      });
    }
}
int main(int argc,char** argv) {
  int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  try {
    if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED required");
    omp_set_dynamic(0); // Stored COO slots require a fixed construction team.
    const auto o=cli::parse(argc,argv);
    if(o.help) {if(!rank) cli::usage();MPI_Finalize();return 0;}
    const double total_start=MPI_Wtime();
    const bool complex=o.read.empty()?o.scalar_type=="complex":cs::complex_coefficient_file(o.read);
    int size=1;MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(!rank) {cli::print_options(std::cout,o,size,complex);std::cout<<"# OpenMP max threads: "<<omp_get_max_threads()<<std::endl;}
    log_record(rank,"start total");
    if(complex) run<std::complex<double>>(o,rank);else run<double>(o,rank);
    const double local_elapsed=MPI_Wtime()-total_start;
    double total_elapsed=0;
    MPI_Reduce(&local_elapsed,&total_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::ostringstream total;total<<std::setprecision(9)<<"end total elapsed="<<total_elapsed<<" sec (max across ranks)";
    log_record(rank,total.str());
    MPI_Finalize();return 0;
  } catch(const std::exception& e) {
    std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
