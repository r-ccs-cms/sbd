#include "support.h"
#include <complex>

using Z = std::complex<double>;
using ZTerm = BasicTerm<Z>;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    Dets::init_elem_size(2);
    auto det = [](int x) { return std::vector<std::size_t>{std::size_t(x&3), std::size_t(x>>2)}; };
    const std::vector<ZTerm> terms = {
      {0.3, {}}, {{0.8,0.3}, {{true,2},{false,0}}},
      {{0.8,-0.3}, {{true,0},{false,2}}},
      {{-0.25,0.15}, {{true,2},{false,0},{true,1},{false,1}}},
      {{-0.25,-0.15}, {{true,1},{false,1},{true,0},{false,2}}},
      {{-0.5,0.2}, {{true,3},{false,1}}},
      {{-0.5,-0.2}, {{true,1},{false,3}}},
      {0.7, {{true,2},{false,2}}},
      {{0.2,0.4}, {{true,0},{true,3},{false,2},{false,1}}},
      {{0.2,-0.4}, {{true,1},{true,2},{false,3},{false,0}}}
    };
    for(bool sign : {false, true}) {
      const auto full = build(terms, sign);
      const auto flat = sbd::MakeKetSideGeneralOpFlatData<Z>(full,2,2,[](Z x){return x;});
      std::array<std::array<Z,16>,16> columns;
      for(int ket=0; ket<16; ++ket) columns[ket]=column(ket,terms,sign);
      for(int ket=0; ket<16; ++ket) {
        Dets p, ch; p.push_back(det(ket));
        std::vector<cs::Contribution<Z,double>> values;
        cs::local_contribution_expansion(p,{Z(1)},{3},{1.0},2,flat,sign,0.0,1,ch,values);
        std::array<Z,16> got{};
        for(std::size_t i=0;i<ch.size();++i) {
          const int a=ch[i][0]+4*ch[i][1];
          got[a]+=values[i].weighted_amplitude/3.0;
          near(values[i].self_product,3*std::norm(columns[ket][a]));
        }
        near(std::abs(cs::diagonal_element(det(ket),2,full)-columns[ket][ket]),0);
        for(int a=0;a<16;++a) {
          near(std::abs(columns[ket][a]-std::conj(columns[a][ket])),0);
          if(a!=ket) near(std::abs(got[a]-columns[ket][a]),0);
        }
      }
      const int keys[2]={3,6}; const Z coeff[2]={{0.36,0.48},{-0.48,0.64}};
      const double probability[2]={3.0/7,4.0/7};
      for(int hs : {1,2,4}) if(size%hs==0) {
        int bs=size/hs, hr=rank/bs, br=rank%bs;
        MPI_Comm hc,bc;
        MPI_Comm_split(MPI_COMM_WORLD,br,hr,&hc);
        MPI_Comm_split(MPI_COMM_WORLD,hr,br,&bc);
        std::vector<ZTerm> local;
        for(std::size_t i=0;i<terms.size();++i) if(i%hs==std::size_t(hr)) local.push_back(terms[i]);
        const auto shard=build(local,sign);
        const auto f=sbd::MakeKetSideGeneralOpFlatData<Z>(shard,2,2,[](Z x){return x;});
        Dets parents,membership; std::vector<Z> c; std::vector<double> prob;
        for(int i=0;i<2;++i) if(i%bs==br) {
          parents.push_back(det(keys[i])); c.push_back(coeff[i]); prob.push_back(probability[i]);
          if(hr==0) membership.push_back(det(keys[i]));
        }
        cs::make_hash_owned_membership_basis(membership,MPI_COMM_WORLD);
        for(double cutoff : {0.0,0.4}) {
          double exact=0, exact_pt2=0;
          for(int a=0;a<16;++a) if(a!=keys[0] && a!=keys[1]) {
            Z v{};
            for(int i=0;i<2;++i) {
              const Z part=coeff[i]*columns[keys[i]][a];
              if(std::abs(part)>=cutoff) v+=part;
            }
            exact+=std::norm(v);
            exact_pt2+=std::norm(v)/(-2-columns[a][a].real());
          }
          double mean=0, mean_pt2=0;
          for(int n=0;n<=3;++n) {
            std::vector<std::size_t> counts;
            for(int i=0;i<2;++i) if(i%bs==br) counts.push_back(i==0?n:3-n);
            Dets children; std::vector<cs::PartialContribution<Z,double>> partials;
            cs::local_contribution_expansion(parents,c,counts,prob,2,f,sign,cutoff,1,children,partials,br);
            const auto r=cs::distributed_u_statistic_external_observable(children,partials,membership,3,2,shard,-2.0,MPI_COMM_WORLD,0,0.0,true,true,nullptr,nullptr,hc,cutoff);
            const double weight=(n==0 || n==3 ? 1:3)*std::pow(probability[0],n)*std::pow(probability[1],3-n);
            mean+=weight*r.variance; mean_pt2+=weight*r.pt2;
          }
          near(mean,exact); near(mean_pt2,exact_pt2);
          Dets input; std::vector<Z> input_c;
          if(hr==0) {input=parents;input_c=c;}
          cs::StatEvaluatorOptions<double> opt;
          opt.sites=4;opt.bit_length=2;opt.sample_count=20000;opt.reference_energy=-2;opt.heatbath_cutoff=cutoff;
          for(auto obs : {sbd::stat_evaluator::Observable::variance,sbd::stat_evaluator::Observable::pt2,sbd::stat_evaluator::Observable::both}) {
            opt.observable=obs;
            const auto r=cs::evaluate_statistical_batch(input,input_c,shard,sign,opt,{"complex",17},MPI_COMM_WORLD,hc,bc);
            for(auto& x:input_c) x*=Z(0,1);
            const auto rotated=cs::evaluate_statistical_batch(input,input_c,shard,sign,opt,{"complex",17},MPI_COMM_WORLD,hc,bc);
            near(r.record.variance_estimate,rotated.record.variance_estimate);
            near(r.record.pt2_estimate,rotated.record.pt2_estimate);
            double reference[2]={};
            if(hr==0) {
              const auto one=cs::evaluate_statistical_batch(input,input_c,full,sign,opt,{"complex",17},bc);
              reference[0]=one.record.variance_estimate;reference[1]=one.record.pt2_estimate;
            }
            MPI_Bcast(reference,2,MPI_DOUBLE,0,hc);
            near(r.record.variance_estimate,reference[0]);near(r.record.pt2_estimate,reference[1]);
          }
        }
        MPI_Comm_free(&hc);MPI_Comm_free(&bc);
      }
    }
    // Non-real source terms cancel across h shards for an occupied spectator.
    for(int hs : {1,2,4}) if(size%hs==0) {
      const int bs=size/hs,hr=rank/bs,br=rank%bs;
      MPI_Comm hc,bc;MPI_Comm_split(MPI_COMM_WORLD,br,hr,&hc);MPI_Comm_split(MPI_COMM_WORLD,hr,br,&bc);
      const std::vector<ZTerm> cancel={{{0,1},{{true,2},{false,0}}},{{0,-1},{{true,2},{false,0},{true,1},{false,1}}}};
      std::vector<ZTerm> local;
      for(std::size_t i=0;i<cancel.size();++i) if(i%hs==std::size_t(hr)) local.push_back(cancel[i]);
      Dets p;std::vector<Z> c;if(rank==0) {p.push_back(det(3));c.push_back(Z(0,1));}
      cs::StatEvaluatorOptions<double> opt;opt.sites=4;opt.bit_length=2;opt.sample_count=2;opt.reference_energy=-2;
      auto r=cs::evaluate_statistical_batch(p,c,build(local,true),true,opt,{"cancel",1},MPI_COMM_WORLD,hc,bc);
      near(r.record.variance_estimate,0);near(r.record.pt2_estimate,0);
      MPI_Comm_free(&hc);MPI_Comm_free(&bc);
    }
    if(argc==2 && rank==0) {
      Dets p;p.push_back(det(1));p.push_back(det(2));
      sbd::SaveWavefunction(std::string(argv[1])+"/state-",p,MPI_COMM_SELF,MPI_COMM_SELF,MPI_COMM_SELF,std::vector<Z>{0.6,Z(0,0.8)});
    }
    if(rank==0) std::cout<<"PASS: complex independent columns, exact multinomial means, h shards, phase invariance and cancellation\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
  MPI_Finalize();
}
