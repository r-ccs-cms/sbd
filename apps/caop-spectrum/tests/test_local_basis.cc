#include "sbd/caop/spectrum/spectrum.h"
#include "sbd/caop/basic/arithmetic.h"
#include <array>
#include <iostream>
#include <type_traits>
namespace cs=sbd::caop::spectrum;
namespace ss=sbd::sparse_solver;
static_assert(std::is_same_v<decltype(cs::SeedSpace<>::basis),sbd::det_vector<std::size_t>>);
struct Op {bool create;int q;};
struct Term {double c;std::vector<Op> ops;};
void check(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
void near(double a,double b) {check(std::abs(a-b)<1e-11*(1+std::abs(b)),"numeric mismatch");}
cs::Det det(int d) {return {std::size_t(d&3),std::size_t(d>>2)};}
// Independent Fock action; does not use SBD phase/mask helpers.
std::pair<int,double> action(int d,const Term& t,bool sign) {
  double c=t.c;
  for(auto it=t.ops.rbegin();it!=t.ops.rend();++it) {
    if(bool(d&(1<<it->q))==it->create) return {d,0};
    if(sign) for(int j=0;j<it->q;++j) if(d&(1<<j)) c=-c;
    d^=1<<it->q;
  }
  return {d,c};
}
int main(int argc,char** argv) {
  int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  omp_set_dynamic(0);int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  try {
    const int h_size=argc>1?std::stoi(argv[1]):1,t_size=argc>2?std::stoi(argv[2]):1;
    check(h_size>0&&t_size>0&&size%(h_size*t_size)==0,"test communicator sizes");
    MPI_Comm h_comm,b_comm,t_comm;
    sbd::setup_communicator(MPI_COMM_WORLD,h_size,size/h_size/t_size,t_size,h_comm,b_comm,t_comm);
    int h_rank=0,t_rank=0;
    MPI_Comm_rank(h_comm,&h_rank);MPI_Comm_rank(t_comm,&t_rank);
    MPI_Comm_rank(b_comm,&rank);MPI_Comm_size(b_comm,&size);
    check(provided>=MPI_THREAD_FUNNELED,"MPI thread support");
    cs::Basis::init_elem_size(2);
    cs::Wavefunction<> state;
    for(int i=0;i<2;++i) if((i+1)%size==rank) {
      state.basis.push_back(det(1<<i));state.coefficients.push_back(i?.8:.6);
    }
    cs::Basis extra;
    if(rank==size-1) for(int d:{12,9,10,9}) extra.push_back(det(d));
    // Duplicate extra basis records on another rank must not duplicate ownership.
    if(rank==0) extra.push_back(det(12));
    // Uneven parent shards must be disjoint across the whole h/b/t grid.
    cs::Wavefunction<> input,partition;
    if(h_rank==0 && t_rank==0) for(int d=0;d<13;++d) if(d%size==rank) {
      input.basis.push_back(det(d));input.coefficients.push_back(d+.25);
    }
    if(t_rank==0) partition=cs::detail::scatter_parents(input,h_comm);
    partition=cs::detail::scatter_parents(partition,t_comm);
    std::array<int,16> parent_counts{},global_counts{};
    for(std::size_t i=0;i<partition.basis.size();++i) {
      const auto& d=partition.basis[i];const int value=d[0]+4*d[1];
      ++parent_counts[value];near(partition.coefficients[i],value+.25);
    }
    MPI_Allreduce(parent_counts.data(),global_counts.data(),16,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    for(int d=0;d<16;++d) check(global_counts[d]==(d<13?1:0),"parent partition duplicated or lost a row");
    // Mixed diagonal/off-diagonal contributions from different parent shards
    // must meet in the same column. Thirteen parents exercise all h/t workers.
    sbd::GeneralOp<double> identity(sbd::ProductOp{}),lowering(sbd::ProductOp{});
    lowering*=sbd::An(0);
    auto mixed=identity+2.0*lowering;
    auto assembled=cs::general_operator_seeds(partition,std::vector<sbd::GeneralOp<double>>{identity,mixed},
        false,4,2,{},b_comm,h_comm,t_comm);
    if(h_rank==0 && t_rank==0) {
      check(assembled.generated_basis_size==13,"general parent partition lost generated rows");
      for(std::size_t i=0;i<assembled.basis.size();++i) {
        const auto& d=assembled.basis[i];const int value=d[0]+4*d[1];
        check(value<13,"general parent partition added a row");
        near(assembled.local_seeds[2*i],value+.25);
        near(assembled.local_seeds[2*i+1],value+.25+
            (value%2==0 && value+1<13?2*(value+1.25):0));
      }
    }
    cs::Wavefunction<> local_parents;
    if(t_rank==0) local_parents=cs::detail::scatter_parents(state,h_comm);
    local_parents=cs::detail::scatter_parents(local_parents,t_comm);
    const std::vector<std::size_t> orbitals{0,1,2};
    std::vector<Term> terms{{.3,{}},{.7,{{true,2},{false,2}}},
      {-.6,{{true,0},{false,1}}},{-.6,{{true,1},{false,0}}},
      {.2,{{true,2},{false,3}}},{.2,{{true,3},{false,2}}},
      {.4,{{true,0},{true,2},{false,2},{false,0}}}};
    for(bool sign:{false,true}) {
      auto seeds=cs::single_particle_seeds(state,orbitals,true,sign,4,2,extra,b_comm);
      check(seeds.generated_basis_size==3,"generated union excludes extra rows");
      auto partitioned=cs::single_particle_seeds(local_parents,orbitals,true,sign,4,2,
          h_rank==0&&t_rank==0?extra:cs::Basis{},b_comm,h_comm,t_comm);
      if(t_rank==0) {
        sbd::MpiBcast(partitioned.basis,0,h_comm);sbd::MpiBcast(partitioned.local_seeds,0,h_comm);
      }
      sbd::MpiBcast(partitioned.basis,0,t_comm);sbd::MpiBcast(partitioned.local_seeds,0,t_comm);
      check(partitioned.basis.flat()==seeds.basis.flat(),"parent partition changed final b ownership");
      check(partitioned.local_seeds.size()==seeds.local_seeds.size(),"parent partition changed seed shape");
      for(std::size_t i=0;i<seeds.local_seeds.size();++i) near(partitioned.local_seeds[i],seeds.local_seeds[i]);

      check(seeds.local_seeds.size()==seeds.basis.size()*3,"local seed shape");
      std::array<int,16> owned{},all_owned{};
      for(std::size_t i=0;i<seeds.basis.size();++i) {
        const auto& row=seeds.basis[i];const int d=row[0]+4*row[1];
        check(sbd::murmur_basis::owner_rank(row,size)==rank,"wrong row owner");
        ++owned[d];
        for(int v=0;v<3;++v) {
          double exact=0;
          for(int parent=0;parent<2;++parent) {
            auto result=action(1<<parent,{parent?.8:.6,{{true,v}}},sign);
            if(result.first==d) exact+=result.second;
          }
          near(seeds.local_seeds[i*3+v],exact);
        }
      }
      MPI_Allreduce(owned.data(),all_owned.data(),16,MPI_INT,MPI_SUM,b_comm);
      for(int d=0;d<16;++d) check(all_owned[d]==(__builtin_popcount(unsigned(d))==2?1:0),"global basis union/unique ownership");
      sbd::GeneralOp<double> h;
      for(std::size_t n=h_rank;n<terms.size();n+=h_size) {
        const auto& t=terms[n];
        sbd::GeneralOp<double> term(sbd::ProductOp{});
        for(const auto& op:t.ops) term*=op.create?sbd::Cr(op.q):sbd::An(op.q);
        h+=t.c*term;
      }
      sbd::NormalOrdering(h,sign);sbd::Simplify(h);
      std::vector<int> slide;
      sbd::make_slide(slide,b_comm,t_comm);
      std::vector<double> hii;
      std::vector<std::vector<std::vector<std::size_t>>> ih,jh;
      std::vector<std::vector<std::vector<double>>> hij;
      sbd::makeCAOpHam(seeds.basis,2,slide,h,sign,hii,ih,jh,hij,
          h_comm,b_comm,t_comm);
      // Nonzero output verifies additive H action across h/t replicas.
      for(std::size_t width:{1ul,3ul}) {
        std::vector<std::vector<double>> nx(width,std::vector<double>(seeds.basis.size()));
        auto ny=nx;
        for(std::size_t v=0;v<width;++v) for(std::size_t i=0;i<seeds.basis.size();++i) {
          int d=seeds.basis[i][0]+4*seeds.basis[i][1];
          nx[v][i]=std::sin((d+1)*(v+1));ny[v][i]=.25;
        }
        auto single_mult=[&](const auto& input,auto& output) {
          sbd::mult(hii,ih,jh,hij,input,output,slide,h_comm,b_comm,t_comm);
        };
        auto apply_h=[&](const auto& input,auto& output) {
          for(std::size_t v=0;v<input.size();++v) single_mult(input[v],output[v]);
        };
        apply_h(nx,ny);
        auto onfly=nx;
        for(auto& column:onfly) std::fill(column.begin(),column.end(),.25);
        std::vector<double> diagonal;
        sbd::makeCAOpHamDiagTerms(seeds.basis,2,slide,h,diagonal);
        for(std::size_t v=0;v<width;++v)
          sbd::mult(diagonal,nx[v],onfly[v],seeds.basis,2,slide,h,sign,h_comm,b_comm,t_comm);
        for(std::size_t v=0;v<width;++v) for(std::size_t i=0;i<seeds.basis.size();++i)
          near(onfly[v][i],ny[v][i]);
        for(std::size_t i=0;i<seeds.basis.size();++i) {
          int bra=seeds.basis[i][0]+4*seeds.basis[i][1];
          for(std::size_t v=0;v<width;++v) {
            double exact=.25;
            for(int ket=0;ket<16;++ket) if(__builtin_popcount(unsigned(ket))==2)
              for(const auto& t:terms) {auto r=action(ket,t,sign);if(r.first==bra) exact+=r.second*std::sin((ket+1)*(v+1));}
            near(ny[v][i],exact);
            near(onfly[v][i],exact);
          }
        }
      }
      // One-dimensional removal space ensures empty owners with MPI > 1.
      auto removal=cs::single_particle_seeds(state,orbitals,false,sign,4,2,{},b_comm);
      check(removal.local_seeds.size()==removal.basis.size()*3,"empty-owner seed shape");
      std::vector<cs::SpectrumProgress> events;
      auto result=cs::spectrum<double>(h_rank==0&&t_rank==0?state:cs::Wavefunction<>{},
          h,sign,orbitals,false,4,2,{},0,h_comm,b_comm,t_comm,{},nullptr,true,
          [&](const cs::SpectrumProgress& event){events.push_back(event);});
      check(events.size()==9,"workflow start/end plus one iteration");
      for(const auto& event:events) if(event.completed && std::string(event.stage)=="seed_generation")
        check(event.reference_basis_size==2 && event.generated_basis_size==1 && event.basis_size==1,
            "global counts must not multiply h/t replicas");
      check(result.basis_size==1,"global basis size metadata");
      auto g=cs::green_function(result,.8,.2);
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        const double a=i==0?.6:i==1?.8:0,b=j==0?.6:j==1?.8:0;
        const auto expected=a*b/std::complex<double>(1.1,.2); // +Hvac=.3
        check(std::abs(g[i*3+j]-expected)<1e-11,"empty-rank removal spectrum");
      }
    }
    if(!rank&&!h_rank&&!t_rank) std::cout<<"PASS: local det_vector ownership, multiword seeds, distributed union, ring H, block width and empty ranks\n";
    MPI_Comm_free(&t_comm);MPI_Comm_free(&b_comm);MPI_Comm_free(&h_comm);
    MPI_Finalize();return 0;
  } catch(const std::exception& e) {std::cerr<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
