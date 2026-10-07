#ifndef SBD_CAOP_SPECTRUM_CLI_OPTIONS_H
#define SBD_CAOP_SPECTRUM_CLI_OPTIONS_H
#include "sbd/caop/spectrum/checkpoint_io.h"
#include "sbd/framework/sparse_solver/block_lanczos_coefficients.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbd_caop_spectrum_cli {
struct Options {
  std::string ham,load,read,save,output,channel,hamiltonian_mode="stored",applied_operator_type="single-particle",scalar_type="real",wavefunction_type;
  std::vector<std::string> extra,remap,operator_files;
  std::vector<std::size_t> orbitals;
  std::size_t sites=0,bits=64,shards=0,points=201,h_size=1,t_size=1,iteration_log_interval=0;
  double energy=0,min=-10,max=10,eta=.1;
  bool has_energy=false,normalize=false,help=false;
  sbd::sparse_solver::LanczosOptions lanczos;
};
inline std::size_t integer(const std::string& s) {
  if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("expected nonnegative integer: "+s);
  std::size_t pos=0;auto n=std::stoull(s,&pos);if(pos!=s.size()) throw std::invalid_argument("bad integer");return n;
}
inline double real(const std::string& s) {
  std::size_t pos=0;double x=std::stod(s,&pos);if(pos!=s.size()||!std::isfinite(x)) throw std::invalid_argument("expected finite number");return x;
}
inline void usage() {
  std::cout<<"CAOP matrix spectrum\n"
    "Generate: --hamfile H --loadname PREFIX --wavefunction-shards N --sites N\n"
    "  --orbitals 0,1,... --channel addition|removal --reference-energy E\n"
    "  Or: --applied-operator-type general --operator-file FILE (repeatable), without orbitals/channel\n"
    "  --save-coefficients FILE [--extra-detfile FILE (repeatable)]\n"
    "  [--remap-detfile FILE (repeatable)] [--normalize-remap] [--bit-length 64]\n"
    "  [--hamiltonian-mode stored|on-the-fly] (default stored)\n"
    "  [--iteration-log-interval 0] (0 disables iteration logs)\n"
    "  [--steps 100] [--rank-atol 1e-13] [--rank-rtol 1e-12]\n"
    "  [--h-comm-size 1] [--t-comm-size 1] (b = MPI size / h / t)\n"
    "  [--scalar-type real|complex] (default real); --wavefunction-type defaults to scalar type\n"
    "  [--wavefunction-type real] permits real checkpoints in complex calculations\n"
    "Reevaluate: --read-coefficients FILE (no H/checkpoint required)\n"
    "Either mode: [--output CSV --omega-min -10 --omega-max 10 --points 201 --eta .1]\n"
    "Orbitals are zero-based CAOP mode indices; extra basis is unioned with seed support.\n";
}
inline Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string k=argv[i];
    if(k=="--help") {o.help=true;continue;}
    if(k=="--normalize-remap") {o.normalize=true;continue;}
    if(i+1==argc) throw std::invalid_argument("missing value for "+k);
    std::string v=argv[++i];
    if(k=="--hamfile") o.ham=v;
    else if(k=="--loadname") o.load=v;
    else if(k=="--read-coefficients") o.read=v;
    else if(k=="--save-coefficients") o.save=v;
    else if(k=="--output") o.output=v;
    else if(k=="--hamiltonian-mode") o.hamiltonian_mode=v;
    else if(k=="--scalar-type") o.scalar_type=v;
    else if(k=="--wavefunction-type") o.wavefunction_type=v;
    else if(k=="--applied-operator-type") o.applied_operator_type=v;
    else if(k=="--operator-file") o.operator_files.push_back(v);
    else if(k=="--channel") o.channel=v;
    else if(k=="--extra-detfile") o.extra.push_back(v);
    else if(k=="--remap-detfile") o.remap.push_back(v);
    else if(k=="--h-comm-size"||k=="--h_comm_size") o.h_size=integer(v);
    else if(k=="--t-comm-size"||k=="--t_comm_size") o.t_size=integer(v);
    else if(k=="--sites") o.sites=integer(v);
    else if(k=="--bit-length") o.bits=integer(v);
    else if(k=="--wavefunction-shards") o.shards=integer(v);
    else if(k=="--iteration-log-interval") o.iteration_log_interval=integer(v);
    else if(k=="--steps") o.lanczos.max_steps=integer(v);
    else if(k=="--points") o.points=integer(v);
    else if(k=="--reference-energy") {o.energy=real(v);o.has_energy=true;}
    else if(k=="--eta") o.eta=real(v);
    else if(k=="--omega-min") o.min=real(v);
    else if(k=="--omega-max") o.max=real(v);
    else if(k=="--rank-atol") o.lanczos.rank_atol=real(v);
    else if(k=="--rank-rtol") o.lanczos.rank_rtol=real(v);
    else if(k=="--orbitals") {std::stringstream in(v);std::string x;while(std::getline(in,x,',')) o.orbitals.push_back(integer(x));}
    else throw std::invalid_argument("unknown option: "+k);
  }
  if(o.help) return o;
  if(o.hamiltonian_mode!="stored"&&o.hamiltonian_mode!="on-the-fly") throw std::invalid_argument("hamiltonian-mode must be stored or on-the-fly");
  if(o.scalar_type!="real"&&o.scalar_type!="complex") throw std::invalid_argument("scalar-type must be real or complex");
  if(o.wavefunction_type.empty()) o.wavefunction_type=o.scalar_type;
  if((o.wavefunction_type!="real"&&o.wavefunction_type!="complex")||(o.scalar_type=="real"&&o.wavefunction_type=="complex"))
    throw std::invalid_argument("wavefunction-type must match scalar type or promote real to complex");
  if(!o.h_size||!o.t_size||!o.points||!(o.eta>0)||o.max<o.min||!o.lanczos.max_steps||o.lanczos.rank_atol<0||o.lanczos.rank_rtol<0) throw std::invalid_argument("invalid grid or Lanczos options");
  if(o.applied_operator_type!="single-particle"&&o.applied_operator_type!="general") throw std::invalid_argument("unknown applied-operator-type");
  if(o.read.empty()) {
    if(o.ham.empty()||o.load.empty()||!o.shards||!o.sites||!o.bits||o.bits>64||!o.has_energy||o.save.empty()) throw std::invalid_argument("missing generation arguments");
    if(o.applied_operator_type=="general") {
      if(o.operator_files.empty()||!o.orbitals.empty()||!o.channel.empty()) throw std::invalid_argument("general operators need files and no orbitals/channel");
    } else if(o.orbitals.empty()||(o.channel!="addition"&&o.channel!="removal")||!o.operator_files.empty())
      throw std::invalid_argument("single-particle needs orbitals/channel and no operator files");
  } else if(o.applied_operator_type!="single-particle"||!o.operator_files.empty()||!o.ham.empty()||!o.load.empty()||!o.channel.empty()||!o.orbitals.empty()||o.has_energy||!o.extra.empty()||!o.remap.empty()||!o.save.empty()||o.normalize||o.sites||o.shards) throw std::invalid_argument("do not mix generation and reevaluation arguments");
  if(o.output.empty()&&!o.read.empty()) throw std::invalid_argument("reevaluation needs --output");
  if(o.normalize&&o.remap.empty()) throw std::invalid_argument("--normalize-remap needs target basis");
  std::set<std::string> inputs{o.ham,o.read};
  inputs.insert(o.operator_files.begin(),o.operator_files.end());
  inputs.insert(o.extra.begin(),o.extra.end());inputs.insert(o.remap.begin(),o.remap.end());inputs.erase("");
  for(std::size_t j=0;j<o.shards;++j) inputs.insert(sbd::statefilename(o.load,int(j)));
  if((!o.save.empty()&&inputs.count(o.save))||(!o.output.empty()&&inputs.count(o.output))||(!o.save.empty()&&o.save==o.output)) throw std::invalid_argument("input/output paths collide");
  return o;
}
// Configuration records use #; runtime records are emitted by the CLI driver.
inline void print_options(std::ostream& out,const Options& o,int mpi_size,bool complex) {
  const auto precision=out.precision(17);
  auto files=[&](const char* label,const std::vector<std::string>& paths) {
    out<<"# "<<label<<":";for(const auto& path:paths) out<<' '<<path;out<<'\n';
  };
  out<<"# mode: "<<(o.read.empty()?"generate":"reevaluate")<<'\n'
     <<"# coefficient type: "<<(complex?"complex<double>":"double")<<'\n'
     <<"# MPI size: "<<mpi_size<<'\n'
     <<"# iteration log interval: "<<o.iteration_log_interval<<'\n';
  if(o.read.empty()) {
    out<<"# h size: "<<o.h_size<<"\n# t size: "<<o.t_size<<'\n';
    if(o.h_size<=std::size_t(mpi_size) && o.t_size<=std::size_t(mpi_size)/o.h_size)
      out<<"# b size: "<<mpi_size/o.h_size/o.t_size<<'\n';
#ifdef SBD_THRUST
    out<<"# Hamiltonian application backend: "
       <<(o.hamiltonian_mode=="on-the-fly"?"Thrust GPU":"CPU")<<'\n';
#else
    out<<"# Hamiltonian application backend: CPU\n";
#endif
    out<<"# Hamiltonian: "<<o.ham<<"\n# Hamiltonian mode: "<<o.hamiltonian_mode
       <<"\n# load name: "<<o.load<<"\n# wavefunction shards: "<<o.shards
       <<"\n# wavefunction type: "<<o.wavefunction_type<<"\n# sites: "<<o.sites
       <<"\n# bit length: "<<o.bits<<"\n# reference energy: "<<o.energy
       <<"\n# applied operator type: "<<o.applied_operator_type<<'\n';
    if(o.applied_operator_type=="single-particle") {
      out<<"# channel: "<<o.channel<<"\n# orbitals:";
      for(auto orbital:o.orbitals) out<<' '<<orbital;
      out<<'\n';
    }
    files("operator files",o.operator_files);files("extra determinant files",o.extra);
    files("remap determinant files",o.remap);
    out<<"# normalize remap: "<<o.normalize<<"\n# maximum steps: "<<o.lanczos.max_steps
       <<"\n# rank atol: "<<o.lanczos.rank_atol<<"\n# rank rtol: "<<o.lanczos.rank_rtol
       <<"\n# orthogonality tolerance: "<<o.lanczos.orthogonality_tolerance
       <<"\n# save coefficients: "<<o.save<<'\n';
  } else out<<"# read coefficients: "<<o.read<<'\n';
  out<<"# output CSV: "<<o.output<<'\n';
  if(!o.output.empty()) out<<"# omega min: "<<o.min<<"\n# omega max: "<<o.max
      <<"\n# points: "<<o.points<<"\n# eta: "<<o.eta<<'\n';
  out.precision(precision);out.flush();
}
} // namespace sbd_caop_spectrum_cli
#endif
