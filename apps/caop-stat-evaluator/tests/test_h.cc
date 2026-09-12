#include "support.h"
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  try {
    Dets::init_elem_size(2);
    std::vector<Term> terms = {{0.3,{}}, {0.8,{{true,2},{false,0}}},
      {0.8,{{true,0},{false,2}}}, {-0.25,{{true,2},{false,0},{true,1},{false,1}}},
      {-0.25,{{true,1},{false,1},{true,0},{false,2}}},
      {-0.5,{{true,3},{false,1}}}, {-0.5,{{true,1},{false,3}}},
      {0.7,{{true,2},{false,2}}}, {0.2,{{true,0},{true,3},{false,2},{false,1}}},
      {0.2,{{true,1},{true,2},{false,3},{false,0}}}};

    // Additional spectator contribution crosses shards and cancels part of hopping.
    terms.push_back({-0.55,{{true,2},{false,0},{true,1},{false,1}}});
    terms.push_back({-0.55,{{true,1},{false,1},{true,0},{false,2}}});
    for(int hs: {1,2,4}) if(size%hs==0) {
      int bs=size/hs, hr=rank/bs, br=rank%bs;
      MPI_Comm hc,bc;MPI_Comm_split(MPI_COMM_WORLD,br,hr,&hc);MPI_Comm_split(MPI_COMM_WORLD,hr,br,&bc);
      for(bool sign: {false,true}) for(double cutoff:{0.0,0.3}) {
        auto full=build(terms,sign);
        std::vector<Term> local_terms;
        for(std::size_t j=0;j<terms.size();++j) if(j%hs==std::size_t(hr)) local_terms.push_back(terms[j]);
        auto shard=build(local_terms,sign);
        Dets parents;std::vector<double> coefficients;
        const int keys[2]={3,6};const double c[2]={0.6,-0.8};
        if(hr==0) for(int i=0;i<2;++i) if(i%bs==br) {
          parents.push_back(std::vector<std::size_t>{std::size_t(keys[i]&3),std::size_t(keys[i]>>2)});
          coefficients.push_back(c[i]);
        }
        cs::StatEvaluatorOptions<double> opt; opt.sites=4; opt.bit_length=2;
        opt.sample_count=1000;opt.reference_energy=-2;opt.heatbath_cutoff=cutoff;
        std::vector<cs::BatchRequest> batches={{"h",11},{"h",12}};
        for(auto observable:{sbd::stat_evaluator::Observable::both,sbd::stat_evaluator::Observable::variance,sbd::stat_evaluator::Observable::pt2}) {
          opt.observable=observable;
          double ref[4]={};
          if(hr==0) {
            std::size_t k=0;
            cs::evaluate_statistical_batches(parents,coefficients,full,sign,opt,batches,bc,
              [&](const auto& x) {ref[k++]=x.record.variance_estimate;ref[k++]=x.record.pt2_estimate;});
          }
          MPI_Bcast(ref,4,MPI_DOUBLE,0,hc);
          std::size_t k=0;
          cs::evaluate_statistical_batches(parents,coefficients,shard,sign,opt,batches,MPI_COMM_WORLD,
            [&](const auto& x) {
              near(x.record.variance_estimate,ref[k++]);near(x.record.pt2_estimate,ref[k++]);
              std::uint64_t draws=0;
              MPI_Allreduce(&x.profile.draws,&draws,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
              if(draws!=opt.sample_count) throw std::runtime_error("draws duplicated across h");
            },hc,bc);
        }
      }
      // Fewer terms than h ranks, exact complete cross-shard cancellation.
      std::vector<Term> cancelling={{1,{{true,2},{false,0}}},
        {-1,{{true,2},{false,0},{true,1},{false,1}}}};
      std::vector<Term> local;
      for(std::size_t j=0;j<cancelling.size();++j) if(j%hs==std::size_t(hr)) local.push_back(cancelling[j]);
      auto shard=build(local,true);
      Dets parents;std::vector<double> c;
      if(hr==0 && br==0) {parents.push_back(std::vector<std::size_t>{3,0});c.push_back(1);}
      cs::StatEvaluatorOptions<double> opt;opt.sites=4;opt.bit_length=2;opt.sample_count=2;opt.reference_energy=-2;
      auto x=cs::evaluate_statistical_batch(parents,c,shard,true,opt,{"cancel",1},MPI_COMM_WORLD,hc,bc);
      near(x.record.variance_estimate,0);near(x.record.pt2_estimate,0);
      MPI_Comm_free(&hc);MPI_Comm_free(&bc);
    }
    if(rank==0) std::cout<<"PASS: fixed-b h=1/2/4 batch equality, cutoff, cancellation, empty shards, diagonal ring, all observables\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
  MPI_Finalize();
}
