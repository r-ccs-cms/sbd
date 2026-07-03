/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_H
#define SBD_CHEMISTRY_GDB_HELPER_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {

  namespace gdb {

    /**
       DetBasisMapper is an array to find relation between dets.
    */
    struct DetIndexMap {
      std::vector<size_t> storage;
      std::vector<size_t> AdetToDetLen;
      std::vector<size_t> BdetToDetLen;
      std::vector<size_t*> AdetToBdetSM;
      std::vector<size_t*> AdetToDetSM;
      std::vector<size_t*> BdetToAdetSM;
      std::vector<size_t*> BdetToDetSM;
    };
    
    /**
       Labels connected 
    */
    struct ExcitationLookup {
      int slide;
      std::vector<size_t> storage;
      std::vector<int> intstorage;
      std::vector<size_t> SelfFromAdetLen;
      std::vector<size_t> SelfFromBdetLen;
      std::vector<size_t> SinglesFromAdetLen;
      std::vector<size_t> SinglesFromBdetLen;
      std::vector<size_t> DoublesFromAdetLen;
      std::vector<size_t> DoublesFromBdetLen;
      std::vector<size_t*> SelfFromAdetSM;
      std::vector<size_t*> SelfFromBdetSM;
      std::vector<size_t*> SinglesFromAdetSM;
      std::vector<int*> SinglesAdetCrAnSM;
      std::vector<size_t*> SinglesFromBdetSM;
      std::vector<int*> SinglesBdetCrAnSM;
      std::vector<size_t*> DoublesFromAdetSM;
      std::vector<int*> DoublesAdetCrAnSM;
      std::vector<size_t*> DoublesFromBdetSM;
      std::vector<int*> DoublesBdetCrAnSM;
    };
    
    void dumpDetIndexMap(std::string filename,
			 const DetIndexMap & idxmap) {
      std::ofstream ofs(filename);
      ofs << " DetIndexMap: storage size = " << idxmap.storage.size() << std::endl;
      for(size_t i=0; i < idxmap.AdetToDetLen.size(); i++) {
	ofs << " AdetToDetSM[" << i << "]:";
	for(size_t k=0; k < idxmap.AdetToDetLen[i]; k++) {
	  ofs << " " << idxmap.AdetToDetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < idxmap.AdetToDetLen.size(); i++) {
	ofs << " AdetToBdetSM[" << i << "]:";
	for(size_t k=0; k < idxmap.AdetToDetLen[i]; k++) {
	  ofs << " " << idxmap.AdetToBdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < idxmap.BdetToDetLen.size(); i++) {
	ofs << " BdetToDetSM[" << i << "]:";
	for(size_t k=0; k < idxmap.BdetToDetLen[i]; k++) {
	  ofs << " " << idxmap.BdetToDetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < idxmap.BdetToDetLen.size(); i++) {
	ofs << " BdetToAdetSM[" << i << "]:";
	for(size_t k=0; k < idxmap.BdetToDetLen[i]; k++) {
	  ofs << " " << idxmap.BdetToAdetSM[i][k];
	}
	ofs << std::endl;
      }
      ofs.close();
    }

    void dumpExcitationLookup(std::string filename,
			      const ExcitationLookup & exidx) {
      std::ofstream ofs(filename);
      ofs << " ExcitationLookup: storage size = " << exidx.storage.size() << std::endl;
      for(size_t i=0; i < exidx.SelfFromAdetLen.size(); i++) {
	ofs << " SelfFromAdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.SelfFromAdetLen[i]; k++) {
	  ofs << " " << exidx.SelfFromAdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < exidx.SelfFromBdetLen.size(); i++) {
	ofs << " SelfFromBdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.SelfFromBdetLen[i]; k++) {
	  ofs << " " << exidx.SelfFromBdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < exidx.SinglesFromAdetLen.size(); i++) {
	ofs << " SinglesFromAdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.SinglesFromAdetLen[i]; k++) {
	  ofs << " " << exidx.SinglesFromAdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < exidx.SinglesFromBdetLen.size(); i++) {
	ofs << " SinglesFromBdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.SinglesFromBdetLen[i]; k++) {
	  ofs << " " << exidx.SinglesFromBdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < exidx.DoublesFromAdetLen.size(); i++) {
	ofs << " DoublesFromAdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.DoublesFromAdetLen[i]; k++) {
	  ofs << " " << exidx.DoublesFromAdetSM[i][k];
	}
	ofs << std::endl;
      }
      for(size_t i=0; i < exidx.DoublesFromBdetLen.size(); i++) {
	ofs << " DoublesFromBdetSM[" << i << "]:";
	for(size_t k=0; k < exidx.DoublesFromBdetLen[i]; k++) {
	  ofs << " " << exidx.DoublesFromBdetSM[i][k];
	}
	ofs << std::endl;
      }
      ofs.close();
    }

#ifdef SBD_DEBUG_HELPER
    std::string dumpfilename(const std::string & filename,
			     int mpi_rank_h, int mpi_rank_b, int mpi_rank_t, int task) {
      std::ostringstream oss_h;
      oss_h << std::setw(3) << std::setfill('0') << mpi_rank_h;
      std::string tag_h = oss_h.str();
      std::ostringstream oss_b;
      oss_b << std::setw(3) << std::setfill('0') << mpi_rank_b;
      std::string tag_b = oss_b.str();
      std::ostringstream oss_t;
      oss_t << std::setw(3) << std::setfill('0') << mpi_rank_t;
      std::string tag_t = oss_t.str();
      std::ostringstream oss_i;
      oss_i << std::setw(3) << std::setfill('0') << task;
      std::string tag_i = oss_i.str();
      std::string resname = filename
	+ "-" + tag_h + "-" + tag_b + "-" + tag_t
	+ "-" + tag_i + ".txt";
      return resname;
    }
#endif
    
    void getHalfDets(const sbd::det_vector<size_t> & det,
		     size_t bit_length,
		     size_t norb,
		     sbd::det_vector<size_t, sbd::det_kind::half> & adet,
		     sbd::det_vector<size_t, sbd::det_kind::half> & bdet,
		     std::vector<size_t> & adet_count,
		     std::vector<size_t> & bdet_count) {
      const size_t N = det.size();
      const int hdet_size = static_cast<int>((norb + bit_length - 1) / bit_length);
      const int det_size  = static_cast<int>((2 * norb + bit_length - 1) / bit_length);
      const size_t half_bl = bit_length / 2;

      // Flat contiguous storage: row i occupies [i*hdet_size .. (i+1)*hdet_size).
      // Eliminates per-det heap allocation of vector<vector<size_t>>.
      std::vector<size_t> adet_flat(static_cast<size_t>(hdet_size) * N);
      std::vector<size_t> bdet_flat(static_cast<size_t>(hdet_size) * N);

      // Extract half-dets via bit masking rather than per-bit getocc/setocc loops.
      // Alpha bit i is at even det position 2i; beta bit i is at odd position 2i+1.
      // deinterleave(w) packs the 32 even-position bits of w into the lower 32 bits,
      // matching the output of getAdet/getBdet so makeDetIndexMap lookups still work.
      // Each adet/bdet word k receives 32 bits from det word 2k (lower) and 32 from
      // det word 2k+1 (upper, shifted left by half_bl).
      auto deinterleave = [](size_t w) -> size_t {
        w = w & 0x5555555555555555ULL;
        w = (w | (w >>  1)) & 0x3333333333333333ULL;
        w = (w | (w >>  2)) & 0x0f0f0f0f0f0f0f0fULL;
        w = (w | (w >>  4)) & 0x00ff00ff00ff00ffULL;
        w = (w | (w >>  8)) & 0x0000ffff0000ffffULL;
        w = (w | (w >> 16)) & 0x00000000ffffffffULL;
        return w;
      };

      for (size_t i = 0; i < N; i++) {
        size_t* arow = &adet_flat[static_cast<size_t>(hdet_size) * i];
        size_t* brow = &bdet_flat[static_cast<size_t>(hdet_size) * i];
        for (int k = 0; k < hdet_size; k++) {
          size_t aw = 0, bw = 0;
          if (2*k < det_size) {
            aw  = deinterleave(det[i][2*k]);
            bw  = deinterleave(det[i][2*k] >> 1);
          }
          if (2*k+1 < det_size) {
            aw |= deinterleave(det[i][2*k+1]) << half_bl;
            bw |= deinterleave(det[i][2*k+1] >> 1) << half_bl;
          }
          arow[k] = aw;
          brow[k] = bw;
        }
      }

      // Sort via pointer array: swaps 8-byte pointers, not 24-byte vector headers.
      std::vector<size_t*> adet_ptrs(N), bdet_ptrs(N);
      for (size_t i = 0; i < N; i++) {
        adet_ptrs[i] = &adet_flat[static_cast<size_t>(hdet_size) * i];
        bdet_ptrs[i] = &bdet_flat[static_cast<size_t>(hdet_size) * i];
      }
      sbd::sort_from_back_t(adet_ptrs, 0, N, hdet_size - 1);
      sbd::sort_from_back_t(bdet_ptrs, 0, N, hdet_size - 1);

      // Row equality (custom == for ptr rows).
      auto row_eq = [hdet_size](const size_t* x, const size_t* y) -> bool {
        for (int k = 0; k < hdet_size; k++)
          if (x[k] != y[k]) return false;
        return true;
      };

      // Build unique adet + run-length counts in one pass (no copy of full sorted array).
      adet.clear();  adet_count.clear();
      if (N > 0) {
        size_t count = 1;
        for (size_t k = 1; k < N; k++) {
          if (row_eq(adet_ptrs[k], adet_ptrs[k-1])) {
            count++;
          } else {
            adet_count.push_back(count);
            adet.emplace_back(adet_ptrs[k-1], adet_ptrs[k-1] + hdet_size);
            count = 1;
          }
        }
        adet_count.push_back(count);
        adet.emplace_back(adet_ptrs[N-1], adet_ptrs[N-1] + hdet_size);
      }

      bdet.clear();  bdet_count.clear();
      if (N > 0) {
        size_t count = 1;
        for (size_t k = 1; k < N; k++) {
          if (row_eq(bdet_ptrs[k], bdet_ptrs[k-1])) {
            count++;
          } else {
            bdet_count.push_back(count);
            bdet.emplace_back(bdet_ptrs[k-1], bdet_ptrs[k-1] + hdet_size);
            count = 1;
          }
        }
        bdet_count.push_back(count);
        bdet.emplace_back(bdet_ptrs[N-1], bdet_ptrs[N-1] + hdet_size);
      }
    }
    
    void makeDetIndexMap(const sbd::det_vector<size_t> & det,
			 const sbd::det_vector<size_t, sbd::det_kind::half> & adet,
			 const sbd::det_vector<size_t, sbd::det_kind::half> & bdet,
			 const std::vector<size_t> & adet_count,
			 const std::vector<size_t> & bdet_count,
			 size_t bit_length,
			 size_t norb,
			 std::vector<std::vector<size_t>> & adet_to_bdet,
			 std::vector<std::vector<size_t>> & adet_to_det,
			 std::vector<std::vector<size_t>> & bdet_to_adet,
			 std::vector<std::vector<size_t>> & bdet_to_det) {
      adet_to_bdet.resize(adet.size());
      adet_to_det.resize(adet.size());
      bdet_to_adet.resize(bdet.size());
      bdet_to_det.resize(bdet.size());
      for(size_t k=0; k < adet.size(); k++) {
	adet_to_bdet[k].reserve(adet_count[k]);
	adet_to_det[k].reserve(adet_count[k]);
      }
      for(size_t k=0; k < bdet.size(); k++) {
	bdet_to_adet[k].reserve(bdet_count[k]);
	bdet_to_det[k].reserve(bdet_count[k]);
      }
      size_t hdet_size = (norb + bit_length - 1) / bit_length;
      std::vector<size_t> adet_temp(hdet_size,0);
      std::vector<size_t> bdet_temp(hdet_size,0);
      for(size_t i=0; i < det.size(); i++) {
	getAdet(det[i],bit_length,norb,adet_temp);
	getBdet(det[i],bit_length,norb,bdet_temp);
	/*
	auto itia = std::lower_bound(adet.begin(),adet.end(),
				     adet_temp,
				     [](const std::vector<size_t> & lhs,
					const std::vector<size_t> & rhs) {
				       return lhs < rhs;
				     });
	auto itib = std::lower_bound(bdet.begin(),bdet.end(),
				     bdet_temp,
				     [](const std::vector<size_t> & lhs,
					const std::vector<size_t> & rhs) {
				       return lhs < rhs;
				     });
	*/
	auto itia = std::lower_bound(adet.begin(),adet.end(),
				     adet_temp,
				     [](const auto & lhs,
					const auto & rhs) {
				       return sbd::less_from_back(lhs,rhs);
				     });
	auto itib = std::lower_bound(bdet.begin(),bdet.end(),
				     bdet_temp,
				     [](const auto & lhs,
					const auto & rhs) {
				       return sbd::less_from_back(lhs,rhs);
				     });
	if( itia == adet.end() || *itia != adet_temp ) {
	  std::cout << " unexpected situation happened in adet" << std::endl;
	}
	if( itib == bdet.end() || *itib != bdet_temp ) {
	  std::cout << " unexpected situation happened in bdet" << std::endl;
	}
	size_t ia = std::distance(adet.begin(),itia);
	size_t ib = std::distance(bdet.begin(),itib);
	adet_to_bdet[ia].push_back(ib);
	bdet_to_adet[ib].push_back(ia);
	adet_to_det[ia].push_back(i);
	bdet_to_det[ib].push_back(i);
      }
    }
    
    void makeDetIndexMap(const sbd::det_vector<size_t> & det,
			 const sbd::det_vector<size_t, sbd::det_kind::half> & adet,
			 const sbd::det_vector<size_t, sbd::det_kind::half> & bdet,
			 const std::vector<size_t> & adet_count,
			 const std::vector<size_t> & bdet_count,
			 size_t bit_length,
			 size_t norb,
			 DetIndexMap & idxmap) {
      std::vector<std::vector<size_t>> adet_to_bdet;
      std::vector<std::vector<size_t>> adet_to_det;
      std::vector<std::vector<size_t>> bdet_to_adet;
      std::vector<std::vector<size_t>> bdet_to_det;
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,bit_length,norb,
		      adet_to_bdet,adet_to_det,
		      bdet_to_adet,bdet_to_det);
      idxmap.AdetToBdetSM.resize(adet_to_bdet.size());
      idxmap.AdetToDetSM.resize(adet_to_det.size());
      idxmap.BdetToAdetSM.resize(bdet_to_adet.size());
      idxmap.BdetToDetSM.resize(bdet_to_det.size());
      idxmap.AdetToDetLen.resize(adet_to_det.size());
      idxmap.BdetToDetLen.resize(bdet_to_det.size());
      size_t aidx_size = 0;
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToDetLen[k] = adet_to_det[k].size();
	aidx_size += adet_to_det[k].size();
      }
      size_t bidx_size = 0;
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToDetLen[k] = bdet_to_det[k].size();
	bidx_size += bdet_to_det[k].size();
      }
      size_t total_size = 2*aidx_size + 2*bidx_size;
      idxmap.storage.resize(total_size);
      size_t * begin = idxmap.storage.data();
      size_t counter = 0;
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToDetSM[k] = begin + counter;
	counter += adet_to_det[k].size();
      }
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToBdetSM[k] = begin + counter;
	counter += adet_to_det[k].size();
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToDetSM[k] = begin + counter;
	counter += bdet_to_det[k].size();
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToAdetSM[k] = begin + counter;
	counter += bdet_to_det[k].size();
      }
      
      for(size_t k=0; k < adet_to_det.size(); k++) {
	std::memcpy(idxmap.AdetToDetSM[k],
		    adet_to_det[k].data(),
		    idxmap.AdetToDetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < adet_to_det.size(); k++) {
	std::memcpy(idxmap.AdetToBdetSM[k],
		    adet_to_bdet[k].data(),
		    idxmap.AdetToDetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	std::memcpy(idxmap.BdetToDetSM[k],
		    bdet_to_det[k].data(),
		    idxmap.BdetToDetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	std::memcpy(idxmap.BdetToAdetSM[k],
		    bdet_to_adet[k].data(),
		    idxmap.BdetToDetLen[k]*sizeof(size_t));
      }
    }
    
    
    template<typename HDetContainer>
    void makeExcitationLookup(const HDetContainer & hdet_bra,
			      const HDetContainer & hdet_ket,
			      size_t bit_length,
			      size_t norb,
			      std::vector<std::vector<size_t>> & samedet,
			      std::vector<std::vector<size_t>> & singles,
			      std::vector<std::vector<int>> & onecran,
			      std::vector<std::vector<size_t>> & doubles,
			      std::vector<std::vector<int>> & twocran,
			      int spin) {
      samedet.resize(hdet_bra.size());
      singles.resize(hdet_bra.size());
      onecran.resize(hdet_bra.size());
      doubles.resize(hdet_bra.size());
      twocran.resize(hdet_bra.size());
#pragma omp parallel for
      for(size_t i=0; i < hdet_bra.size(); i++) {
	size_t zcount = 0;
	size_t scount = 0;
	size_t dcount = 0;
	for(size_t j=0; j < hdet_ket.size(); j++) {
	  int d = difference(hdet_bra[i],hdet_ket[j],bit_length,norb);
	  if( d == 0 ) zcount++;
	  if( d == 2 ) scount++;
	  if( d == 4 ) dcount++;
	}
	samedet[i].reserve(zcount);
	singles[i].reserve(scount);
	onecran[i].reserve(2*scount);
	doubles[i].reserve(dcount);
	twocran[i].reserve(4*dcount);
	std::vector<int> cr(2);
	std::vector<int> an(2);
	for(size_t j=0; j < hdet_ket.size(); j++) {
	  int d = difference(hdet_bra[i],hdet_ket[j],bit_length,norb);
	  if( d == 0 ) {
	    samedet[i].push_back(j);
	  } else if( d == 2 ) {
	    singles[i].push_back(j);
	    OrbitalDifference(hdet_bra[i],hdet_ket[j],bit_length,norb,cr,an);
	    onecran[i].push_back(2*cr[0]+spin);
	    onecran[i].push_back(2*an[0]+spin);
	  } else if ( d == 4 ) {
	    doubles[i].push_back(j);
	    OrbitalDifference(hdet_bra[i],hdet_ket[j],bit_length,norb,cr,an);
	    twocran[i].push_back(2*cr[0]+spin);
	    twocran[i].push_back(2*cr[1]+spin);
	    twocran[i].push_back(2*an[0]+spin);
	    twocran[i].push_back(2*an[1]+spin);
	  }
	}
      }
    }
    
    void makeExcitationLookup(const sbd::det_vector<size_t, sbd::det_kind::half> & adet_bra,
			      const sbd::det_vector<size_t, sbd::det_kind::half> & bdet_bra,
			      const sbd::det_vector<size_t, sbd::det_kind::half> & adet_ket,
			      const sbd::det_vector<size_t, sbd::det_kind::half> & bdet_ket,
			      size_t bit_length,
			      size_t norb,
			      ExcitationLookup & exidx) {
      std::vector<std::vector<size_t>> selfdet_a;
      std::vector<std::vector<size_t>> singles_a;
      std::vector<std::vector<int>>    onecran_a;
      std::vector<std::vector<size_t>> doubles_a;
      std::vector<std::vector<int>>    twocran_a;
      std::vector<std::vector<size_t>> selfdet_b;
      std::vector<std::vector<size_t>> singles_b;
      std::vector<std::vector<int>>    onecran_b;
      std::vector<std::vector<size_t>> doubles_b;
      std::vector<std::vector<int>>    twocran_b;
      makeExcitationLookup(adet_bra,adet_ket,
			   bit_length,norb,
			   selfdet_a, singles_a, onecran_a,
			   doubles_a, twocran_a, 0);
      makeExcitationLookup(bdet_bra,bdet_ket,
			   bit_length,norb,
			   selfdet_b, singles_b, onecran_b,
			   doubles_b, twocran_b, 1);
      exidx.SelfFromAdetLen.resize(selfdet_a.size());
      exidx.SinglesFromAdetLen.resize(singles_a.size());
      exidx.DoublesFromAdetLen.resize(doubles_a.size());
      exidx.SelfFromBdetLen.resize(selfdet_b.size());
      exidx.SinglesFromBdetLen.resize(singles_b.size());
      exidx.DoublesFromBdetLen.resize(doubles_b.size());
      size_t total_size = 0;
      size_t total_int  = 0;
      for(size_t k=0; k < selfdet_a.size(); k++) {
	exidx.SelfFromAdetLen[k] = selfdet_a[k].size();
	total_size += selfdet_a[k].size();
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	exidx.SinglesFromAdetLen[k] = singles_a[k].size();
	total_size += singles_a[k].size();
	total_int  += 2 * singles_a[k].size();
      }
      for(size_t k=0; k < doubles_a.size(); k++) {
	exidx.DoublesFromAdetLen[k] = doubles_a[k].size();
	total_size += doubles_a[k].size();
	total_int  += 4 * doubles_a[k].size();
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	exidx.SelfFromBdetLen[k] = selfdet_b[k].size();
	total_size += selfdet_b[k].size();
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	exidx.SinglesFromBdetLen[k] = singles_b[k].size();
	total_size += singles_b[k].size();
	total_int  += 2 * singles_b[k].size();
      }
      for(size_t k=0; k < doubles_b.size(); k++) {
	exidx.DoublesFromBdetLen[k] = doubles_b[k].size();
	total_size += doubles_b[k].size();
	total_int  += 4 * doubles_b[k].size();
      }
      exidx.storage.resize(total_size);
      size_t * begin = exidx.storage.data();
      size_t counter = 0;
      exidx.SelfFromAdetSM.resize(selfdet_a.size());
      exidx.SinglesFromAdetSM.resize(singles_a.size());
      exidx.DoublesFromAdetSM.resize(doubles_a.size());
      exidx.SelfFromBdetSM.resize(selfdet_b.size());
      exidx.SinglesFromBdetSM.resize(singles_b.size());
      exidx.DoublesFromBdetSM.resize(doubles_b.size());
      for(size_t k=0; k < selfdet_a.size(); k++) {
	exidx.SelfFromAdetSM[k] = begin + counter;
	counter += exidx.SelfFromAdetLen[k];
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	exidx.SinglesFromAdetSM[k] = begin + counter;
	counter += exidx.SinglesFromAdetLen[k];
      }
      for(size_t k=0; k < doubles_a.size(); k++) {
	exidx.DoublesFromAdetSM[k] = begin + counter;
	counter += exidx.DoublesFromAdetLen[k];
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	exidx.SelfFromBdetSM[k] = begin + counter;
	counter += exidx.SelfFromBdetLen[k];
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	exidx.SinglesFromBdetSM[k] = begin + counter;
	counter += exidx.SinglesFromBdetLen[k];
      }
      for(size_t k=0; k < doubles_b.size(); k++) {
	exidx.DoublesFromBdetSM[k] = begin + counter;
	counter += exidx.DoublesFromBdetLen[k];
      }

      exidx.intstorage.resize(total_int);
      int * begin_int = exidx.intstorage.data();
      size_t counter_int = 0;
      exidx.SinglesAdetCrAnSM.resize(singles_a.size());
      exidx.DoublesAdetCrAnSM.resize(doubles_a.size());
      exidx.SinglesBdetCrAnSM.resize(singles_b.size());
      exidx.DoublesBdetCrAnSM.resize(doubles_b.size());
      for(size_t k=0; k < singles_a.size(); k++) {
	exidx.SinglesAdetCrAnSM[k] = begin_int + counter_int;
	counter_int += 2*exidx.SinglesFromAdetLen[k];
      }
      for(size_t k=0; k < doubles_a.size(); k++) {
	exidx.DoublesAdetCrAnSM[k] = begin_int + counter_int;
	counter_int += 4*exidx.DoublesFromAdetLen[k];
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	exidx.SinglesBdetCrAnSM[k] = begin_int + counter_int;
	counter_int += 2*exidx.SinglesFromBdetLen[k];
      }
      for(size_t k=0; k < doubles_b.size(); k++) {
	exidx.DoublesBdetCrAnSM[k] = begin_int + counter_int;
	counter_int += 4*exidx.DoublesFromBdetLen[k];
      }
      
      for(size_t k=0; k < selfdet_a.size(); k++) {
	std::memcpy(exidx.SelfFromAdetSM[k],
		    selfdet_a[k].data(),
		    exidx.SelfFromAdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	std::memcpy(exidx.SinglesFromAdetSM[k],
		    singles_a[k].data(),
		    exidx.SinglesFromAdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < doubles_a.size(); k++) {
	std::memcpy(exidx.DoublesFromAdetSM[k],
		    doubles_a[k].data(),
		    exidx.DoublesFromAdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	std::memcpy(exidx.SelfFromBdetSM[k],
		    selfdet_b[k].data(),
		    exidx.SelfFromBdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	std::memcpy(exidx.SinglesFromBdetSM[k],
		    singles_b[k].data(),
		    exidx.SinglesFromBdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < doubles_b.size(); k++) {
	std::memcpy(exidx.DoublesFromBdetSM[k],
		    doubles_b[k].data(),
		    exidx.DoublesFromBdetLen[k]*sizeof(size_t));
      }

      for(size_t k=0; k < singles_a.size(); k++) {
	std::memcpy(exidx.SinglesAdetCrAnSM[k],
		    onecran_a[k].data(),
		    2*exidx.SinglesFromAdetLen[k]*sizeof(int));
      }
      for(size_t k=0; k < doubles_a.size(); k++) {
	std::memcpy(exidx.DoublesAdetCrAnSM[k],
		    twocran_a[k].data(),
		    4*exidx.DoublesFromAdetLen[k]*sizeof(int));
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	std::memcpy(exidx.SinglesBdetCrAnSM[k],
		    onecran_b[k].data(),
		    2*exidx.SinglesFromBdetLen[k]*sizeof(int));
      }
      for(size_t k=0; k < doubles_b.size(); k++) {
	std::memcpy(exidx.DoublesBdetCrAnSM[k],
		    twocran_b[k].data(),
		    4*exidx.DoublesFromBdetLen[k]*sizeof(int));
      }
    }
    
    void DetIndexMapCopy(const DetIndexMap & idxmap,
			 DetIndexMap & new_idxmap) {
      new_idxmap.storage = idxmap.storage; // hard copy
      new_idxmap.AdetToDetLen = idxmap.AdetToDetLen;
      new_idxmap.BdetToDetLen = idxmap.BdetToDetLen;
      new_idxmap.AdetToDetSM.resize(idxmap.AdetToDetLen.size());
      new_idxmap.AdetToBdetSM.resize(idxmap.AdetToDetLen.size());
      new_idxmap.BdetToDetSM.resize(idxmap.BdetToDetLen.size());
      new_idxmap.BdetToAdetSM.resize(idxmap.BdetToDetLen.size());
      size_t * begin = new_idxmap.storage.data();
      size_t counter = 0;
      for(size_t i=0; i < new_idxmap.AdetToDetLen.size(); i++) {
	new_idxmap.AdetToDetSM[i] = begin + counter;
	counter += new_idxmap.AdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.AdetToDetLen.size(); i++) {
	new_idxmap.AdetToBdetSM[i] = begin + counter;
	counter += new_idxmap.AdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.BdetToDetLen.size(); i++) {
	new_idxmap.BdetToDetSM[i] = begin + counter;
	counter += new_idxmap.BdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.BdetToDetLen.size(); i++) {
	new_idxmap.BdetToAdetSM[i] = begin + counter;
	counter += new_idxmap.BdetToDetLen[i];
      }
    }

    void MpiSlide(const DetIndexMap & send_map,
		  DetIndexMap & recv_map,
		  int slide,
		  MPI_Comm comm) {
      sbd::MpiSlide(send_map.AdetToDetLen,recv_map.AdetToDetLen,slide,comm);
      sbd::MpiSlide(send_map.BdetToDetLen,recv_map.BdetToDetLen,slide,comm);
      sbd::MpiSlide(send_map.storage,recv_map.storage,slide,comm);
      recv_map.AdetToDetSM.resize(recv_map.AdetToDetLen.size());
      recv_map.AdetToBdetSM.resize(recv_map.AdetToDetLen.size());
      recv_map.BdetToDetSM.resize(recv_map.BdetToDetLen.size());
      recv_map.BdetToAdetSM.resize(recv_map.BdetToDetLen.size());
      size_t * begin = recv_map.storage.data();
      size_t counter = 0;
      for(size_t i=0; i < recv_map.AdetToDetLen.size(); i++) {
	recv_map.AdetToDetSM[i] = begin + counter;
	counter += recv_map.AdetToDetLen[i];
      }
      for(size_t i=0; i < recv_map.AdetToDetLen.size(); i++) {
	recv_map.AdetToBdetSM[i] = begin + counter;
	counter += recv_map.AdetToDetLen[i];
      }
      for(size_t i=0; i < recv_map.BdetToDetLen.size(); i++) {
	recv_map.BdetToDetSM[i] = begin + counter;
	counter += recv_map.BdetToDetLen[i];
      }
      for(size_t i=0; i < recv_map.BdetToDetLen.size(); i++) {
	recv_map.BdetToAdetSM[i] = begin + counter;
	counter += recv_map.BdetToDetLen[i];
      }
    }
    
    void DetBasisCommunicator(MPI_Comm comm,
			      int h_comm_size,
			      int b_comm_size,
			      int t_comm_size,
			      MPI_Comm & h_comm,
			      MPI_Comm & b_comm,
			      MPI_Comm & t_comm) {
      int mpi_size; MPI_Comm_size(comm,&mpi_size);
      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int a_comm_size = b_comm_size * t_comm_size;
      MPI_Comm a_comm;
      int a_comm_color = mpi_rank / a_comm_size;
      int h_comm_color = mpi_rank % a_comm_size;
      MPI_Comm_split(comm,a_comm_color,mpi_rank,&a_comm);
      MPI_Comm_split(comm,h_comm_color,mpi_rank,&h_comm);
      
      int mpi_size_a; MPI_Comm_size(a_comm,&mpi_size_a);
      int mpi_rank_a; MPI_Comm_rank(a_comm,&mpi_rank_a);
      
      int t_comm_color = mpi_rank_a % b_comm_size;
      int b_comm_color = mpi_rank_a / b_comm_size;
      MPI_Comm_split(a_comm,t_comm_color,mpi_rank,&t_comm);
      MPI_Comm_split(a_comm,b_comm_color,mpi_rank,&b_comm);
    }
    
    void MakeHelpers(const sbd::det_vector<size_t> & det,
		     size_t bit_length,
		     size_t norb,
		     DetIndexMap & idxmap,
		     std::vector<ExcitationLookup> & exidx,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm) {
      
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      
      size_t task_begin = 0;
      size_t task_end   = static_cast<size_t>(mpi_size_b);
      get_mpi_range(mpi_size_t,mpi_rank_t,task_begin,task_end);
      size_t task_size = task_end-task_begin;
      exidx.resize(task_size);
      for(size_t task=task_begin; task < task_end; task++) {
	exidx[task-task_begin].slide = static_cast<int>(task);
      }
      
      sbd::det_vector<size_t, sbd::det_kind::half> adet;
      sbd::det_vector<size_t, sbd::det_kind::half> bdet;
      std::vector<size_t> adet_count;
      std::vector<size_t> bdet_count;
      getHalfDets(det,bit_length,norb,adet,bdet,adet_count,bdet_count);
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,
		      bit_length,norb,idxmap);

      // Persistent scratch buffers for MpiSlide<det> flat packing.
      // Reusing across the initial slide and 3 task-loop rounds avoids
      // repeated zero-initialization of the 60.8 MB send/recv buffers.
      std::vector<size_t> det_scratch_send, det_scratch_recv;

      sbd::det_vector<size_t> ket_det;
      sbd::det_vector<size_t, sbd::det_kind::half> ket_adet;
      sbd::det_vector<size_t, sbd::det_kind::half> ket_bdet;
      if ( task_begin != static_cast<size_t>(0) ) {
	int slide = - exidx[0].slide;
	sbd::MpiSlideWithScratch(det,ket_det,slide,b_comm,det_scratch_send,det_scratch_recv);
	sbd::MpiSlide(adet,ket_adet,slide,b_comm);
	sbd::MpiSlide(bdet,ket_bdet,slide,b_comm);
      } else {
	ket_det = det;
	ket_adet = adet;
	ket_bdet = bdet;
      }

#ifdef SBD_DEBUG_HELPER
      DetIndexMap ket_idxmap;
      if ( task_begin != static_cast<size_t>(0) ) {
	int slide = - exidx[0].slide;
	sbd::gdb::MpiSlide(idxmap,ket_idxmap,slide,b_comm);
      } else {
	DetIndexMapCopy(idxmap,ket_idxmap);
      }
#endif

      for(size_t task=task_begin; task < task_end; task++) {
#ifdef SBD_DEBUG_HELPER
	std::string idxmap_task_file = dumpfilename(std::string("idxmap"),
						    mpi_rank_h,mpi_rank_b,mpi_rank_t,task);
	dumpDetIndexMap(idxmap_task_file,
			ket_idxmap);
#endif
	makeExcitationLookup(adet,bdet,ket_adet,ket_bdet,
			     bit_length,norb,
			     exidx[task-task_begin]);
#ifdef SBD_DEBUG_HELPER
	std::string exidx_task_file = dumpfilename(std::string("exidx"),
						   mpi_rank_h,mpi_rank_b,mpi_rank_t,task);
	dumpExcitationLookup(exidx_task_file,exidx[task-task_begin]);
#endif
	if( task != task_end-1 ) {
	  int slide = exidx[task-task_begin].slide - exidx[task+1-task_begin].slide;
	  // In-place slide: pass ket_det/ket_adet/ket_bdet as both A and B.
	  // Safe because MpiSlideWithScratch packs A into scratch before B.resize(),
	  // so the existing capacity survives intact until pack is done.
	  // When ranks hold equal det counts (balanced), B.resize() is a no-op.
	  sbd::MpiSlideWithScratch(ket_det,ket_det,slide,b_comm,det_scratch_send,det_scratch_recv);
	  sbd::MpiSlide(ket_adet,ket_adet,slide,b_comm);
	  sbd::MpiSlide(ket_bdet,ket_bdet,slide,b_comm);
#ifdef SBD_DEBUG_HELPER
	  DetIndexMap send_idxmap;
	  sbd::gdb::MpiSlide(send_idxmap,ket_idxmap,slide,b_comm);
#endif
	}
      }
    }

    
  } // end namespace gdb
  
} // end namespace sbd

#endif
