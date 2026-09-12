#include "support.h"
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  try {
    // Two-word storage exercises cross-word signs and ordering as well.
    const std::size_t bits=2;
    Dets::init_elem_size(2);
    auto det=[](int x) { return std::vector<std::size_t>{std::size_t(x&3),std::size_t(x>>2)}; };
    std::vector<Term> terms = {{0.3,{}}, {0.8,{{true,2},{false,0}}},
      {0.8,{{true,0},{false,2}}}, {-0.25,{{true,2},{false,0},{true,1},{false,1}}},
      {-0.25,{{true,1},{false,1},{true,0},{false,2}}},
      {-0.5,{{true,3},{false,1}}}, {-0.5,{{true,1},{false,3}}},
      {0.7,{{true,2},{false,2}}}, {0.2,{{true,0},{true,3},{false,2},{false,1}}},
      {0.2,{{true,1},{true,2},{false,3},{false,0}}}};
    for(bool sign:{false,true}) {
      const auto h=build(terms,sign);
      const auto flat=sbd::MakeKetSideGeneralOpFlatData<double>(h,2,bits,[](double x){return x;});
      // Exhaustive columns, all allowed two-particle inputs, signed amplitude and z.
      for(int ket=0;ket<16;++ket) if(__builtin_popcount(ket)==2) {
        auto ref=column(ket,terms,sign);
        Dets parent; parent.push_back(det(ket)); Dets child;
        std::vector<cs::Contribution<double,double>> values;
        cs::local_contribution_expansion(parent,{1.0},{3},{1.0},bits,flat,sign,0.0,1,child,values);
        std::array<double,16> got{};
        for(std::size_t k=0;k<child.size();++k) {
          int key=child[k][0]+4*child[k][1]; got[key]+=values[k].weighted_amplitude/3;
          near(values[k].self_product,3*ref[key]*ref[key]);
        }
        near(cs::diagonal_element(det(ket),bits,h),ref[ket]);
        for(int key=0;key<16;++key) if(key!=ket) near(got[key],ref[key]);
      }
      for(double cutoff:{0.0,0.4}) {
        const int keys[2]={3,6}; const double coeff[2]={0.6,-0.8};
        const double p[2]={3.0/7,4.0/7};
        auto c0=column(keys[0],terms,sign), c1=column(keys[1],terms,sign);
        double exact=0,pt2=0;
        for(int a=0;a<16;++a) if(a!=keys[0] && a!=keys[1]) {
          double v0=coeff[0]*c0[a],v1=coeff[1]*c1[a];
          if(std::abs(v0)<cutoff) v0=0;
          if(std::abs(v1)<cutoff) v1=0;
          double sq=(v0+v1)*(v0+v1); exact+=sq;
          pt2+=sq/(-2-column(a,terms,sign)[a]);
        }
        Dets parent; std::vector<double> c,prob;
        for(int i=0;i<2;++i) if(i%size==rank) {parent.push_back(det(keys[i]));c.push_back(coeff[i]);prob.push_back(p[i]);}
        Dets membership=parent; cs::make_hash_owned_membership_basis(membership,MPI_COMM_WORLD);
        double mean=0,meanpt=0;
        for(int n=0;n<=3;++n) {
          std::vector<std::size_t> counts;
          for(int i=0;i<2;++i) if(i%size==rank) counts.push_back(i==0?n:3-n);
          Dets children; std::vector<cs::Contribution<double,double>> values;
          cs::local_contribution_expansion(parent,c,counts,prob,bits,flat,sign,cutoff,2,children,values);
          const auto result=cs::distributed_u_statistic_external_observable(children,values,membership,3,bits,h,-2.0,MPI_COMM_WORLD);
          const double probability=(n==0 || n==3 ? 1:3)*std::pow(p[0],n)*std::pow(p[1],3-n);
          mean+=probability*result.variance; meanpt+=probability*result.pt2;
        }
        near(mean,exact); near(meanpt,pt2);
        cs::StatEvaluatorOptions<double> opt; opt.sites=4; opt.bit_length=bits;
        opt.sample_count=20000; opt.reference_energy=-2; opt.heatbath_cutoff=cutoff;
        const auto a=cs::evaluate_statistical_batch(parent,c,h,sign,opt,{"test",17},MPI_COMM_WORLD);
        const auto b=cs::evaluate_statistical_batch(parent,c,h,sign,opt,{"test",17},MPI_COMM_WORLD);
        near(a.record.variance_estimate,b.record.variance_estimate);
        if(std::abs(a.record.variance_estimate-exact)>0.025) throw std::runtime_error("sampling smoke");
      }
    }
    // Distinct operator products cancel for an occupied spectator: no z may survive.
    {
      std::vector<Term> cancel = {{0.8,{{true,2},{false,0}}},
        {-0.8,{{true,2},{false,0},{true,1},{false,1}}}};
      auto h=build(cancel,true);
      auto f=sbd::MakeKetSideGeneralOpFlatData<double>(h,2,bits,[](double x){return x;});
      Dets p; p.push_back(det(3)); Dets ch;
      std::vector<cs::Contribution<double,double>> v;
      cs::local_contribution_expansion(p,{1.0},{2},{1.0},bits,f,true,0.0,1,ch,v);
      if(!v.empty()) throw std::runtime_error("same-parent cancellation lost");
      // N=2 mixed draws with opposite parent contributions produce a negative batch.
      ch.push_back(det(4)); ch.push_back(det(4)); v={{1,1},{-1,1}};
      Dets empty;
      auto r=cs::u_statistic_external_observable(ch,v,empty,2,bits,h,-2.0);
      near(r.variance,-1.0); near(r.pt2,0.5);
      bool rejected=false;
      try {cs::u_statistic_external_observable(ch,v,empty,1,bits,h,-2.0);}
      catch(const std::invalid_argument&) {rejected=true;}
      if(!rejected) throw std::runtime_error("N=1 accepted");
    }
    // Public CAOP checkpoint writer used by CLI test (single shard, real).
    if(argc==2 && rank==0) {
      Dets p; p.push_back(det(1));p.push_back(det(2));
      sbd::SaveWavefunction(std::string(argv[1])+"/state-",p,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,std::vector<double>{0.6,0.8});
    }
    if(rank==0) std::cout<<"PASS: independent CAOP columns, self-products, exact multinomial means, cutoff, MPI and replay\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
  MPI_Finalize();
}
