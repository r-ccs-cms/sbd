/**
@file sbd/chemistry/gdb/expansion.h
@brief Utility functions for basis expansion in the generalized determinant basis.
**/
#ifndef SBD_CHEMISTRY_GDB_EXPANSION_H
#define SBD_CHEMISTRY_GDB_EXPANSION_H

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif

namespace sbd {

  template<typename DetsContainer>
  void append_sorted_candidates_unique(
    DetsContainer& dets,
    DetsContainer& candidates) {

    if (candidates.empty()) {
      return;
    }

    if (dets.empty()) {
      dets = std::move(candidates);
      return;
    }

    auto comp = [](const auto& a, const auto& b) {
      return less_from_back(a, b);
    };
    DetsContainer merged;
    merged.resize(dets.size() + candidates.size());
    std::merge(
        std::make_move_iterator(dets.begin()),
        std::make_move_iterator(dets.end()),
        std::make_move_iterator(candidates.begin()),
        std::make_move_iterator(candidates.end()),
        merged.begin(),
        comp);
    merged.resize(std::unique(merged.begin(), merged.end()) - merged.begin());
    dets = std::move(merged);
  }

  template<typename DetsContainer>
  void append_candidates_unique(
    DetsContainer& dets,
    DetsContainer& candidates) {

    if (candidates.empty()) {
      return;
    }
    sort_unique_local_bitarray(candidates);
    sort_unique_local_bitarray(dets);
    append_sorted_candidates_unique(dets,candidates);
  }

  void append_candidates_unique(
    std::vector<std::vector<size_t>>& dets,
    std::vector<std::vector<std::vector<size_t>>>& candidates_per_thread) {

    std::vector<std::vector<size_t>> candidates_all;
    size_t total_size = 0;
    for (const auto& cand : candidates_per_thread) {
      total_size += cand.size();
    }
    candidates_all.reserve(total_size);
    for (auto& cand : candidates_per_thread) {
      candidates_all.insert(
          candidates_all.end(),
          std::make_move_iterator(cand.begin()),
          std::make_move_iterator(cand.end()));
      cand.clear();
    }
    append_candidates_unique(dets, candidates_all);
  }

  namespace gdb {

    namespace detail {

      template <typename RealT>
      struct IntegralDoubleExcitation {
        int created_first;
        int created_second;
        RealT abs_integral;
      };

      template <typename ElemT, typename RealT>
      class IntegralDoubleExcitationLookup {
      public:
        using entry_type = IntegralDoubleExcitation<RealT>;
        using const_iterator = typename std::vector<entry_type>::const_iterator;

        IntegralDoubleExcitationLookup(size_t norb,
                                       const twoInt<ElemT> & I2,
                                       RealT cutoff,
                                       RealT max_abs_coefficient)
          : norb_(norb) {
          if( norb_ == 0 ) {
            throw std::invalid_argument("number of orbitals must be positive");
          }
          if( cutoff < RealT(0) ) {
            throw std::invalid_argument("heatbath cutoff must be non-negative");
          }

          const size_t nso = 2 * norb_;
          const size_t num_pairs = nso * (nso - 1) / 2;
          pair_offsets_.assign(num_pairs + 1, 0);
          if( max_abs_coefficient == RealT(0) ) return;

          for(size_t j=1; j < nso; ++j) {
            for(size_t i=0; i < j; ++i) {
              size_t count = 0;
              for(size_t b=1; b < nso; ++b) {
                for(size_t a=0; a < b; ++a) {
                  if( a == i || a == j || b == i || b == j ) continue;
                  const RealT abs_integral = std::abs(
                    I2.Value(static_cast<int>(a),static_cast<int>(i),
                             static_cast<int>(b),static_cast<int>(j))
                    - I2.Value(static_cast<int>(a),static_cast<int>(j),
                               static_cast<int>(b),static_cast<int>(i)));
                  if( abs_integral * max_abs_coefficient > cutoff ) ++count;
                }
              }
              const size_t pair = pair_index(i,j);
              pair_offsets_[pair+1] = count;
            }
          }
          for(size_t pair=0; pair < num_pairs; ++pair) {
            pair_offsets_[pair+1] += pair_offsets_[pair];
          }

          entries_.resize(pair_offsets_.back());
          for(size_t j=1; j < nso; ++j) {
            for(size_t i=0; i < j; ++i) {
              const size_t pair = pair_index(i,j);
              size_t position = pair_offsets_[pair];
              for(size_t b=1; b < nso; ++b) {
                for(size_t a=0; a < b; ++a) {
                  if( a == i || a == j || b == i || b == j ) continue;
                  const RealT abs_integral = std::abs(
                    I2.Value(static_cast<int>(a),static_cast<int>(i),
                             static_cast<int>(b),static_cast<int>(j))
                    - I2.Value(static_cast<int>(a),static_cast<int>(j),
                               static_cast<int>(b),static_cast<int>(i)));
                  if( abs_integral * max_abs_coefficient > cutoff ) {
                    entries_[position++] = {
                      static_cast<int>(a),static_cast<int>(b),abs_integral};
                  }
                }
              }
              std::sort(entries_.begin() + pair_offsets_[pair],
                        entries_.begin() + pair_offsets_[pair+1],
                        [](const entry_type & lhs, const entry_type & rhs) {
                          return lhs.abs_integral > rhs.abs_integral;
                        });
            }
          }
        }

        const_iterator begin(int first, int second) const {
          const size_t pair = checked_pair_index(first,second);
          return entries_.begin() + pair_offsets_[pair];
        }

        const_iterator end(int first, int second) const {
          const size_t pair = checked_pair_index(first,second);
          return entries_.begin() + pair_offsets_[pair+1];
        }

        size_t entry_count() const noexcept { return entries_.size(); }
        size_t storage_bytes() const noexcept {
          return pair_offsets_.size() * sizeof(size_t)
            + entries_.size() * sizeof(entry_type);
        }

      private:
        static size_t pair_index(size_t first, size_t second) noexcept {
          return second * (second - 1) / 2 + first;
        }

        size_t checked_pair_index(int first, int second) const {
          const int nso = static_cast<int>(2*norb_);
          if( first < 0 || second < 0 || first == second
              || first >= nso || second >= nso ) {
            throw std::out_of_range("invalid annihilation pair");
          }
          if( first > second ) std::swap(first,second);
          return pair_index(static_cast<size_t>(first),
                            static_cast<size_t>(second));
        }

        size_t norb_;
        std::vector<size_t> pair_offsets_;
        std::vector<entry_type> entries_;
      };

      template <typename ElemT, typename RealT>
      void local_heatbath_expansion_integral(
          const sbd::det_vector<size_t> & det,
          const std::vector<ElemT> & w,
          size_t bit_length,
          size_t norb,
          const oneInt<ElemT> & I1,
          const twoInt<ElemT> & I2,
          RealT cutoff,
          size_t max_batch_size,
          sbd::det_vector<size_t> & edet) {
        if( det.size() != w.size() ) {
          throw std::invalid_argument(
            "determinant and coefficient counts must match");
        }
        if( cutoff < RealT(0) ) {
          throw std::invalid_argument("heatbath cutoff must be non-negative");
        }
        edet = det;
        sort_unique_local_bitarray(edet);
        if( det.empty() ) return;

        RealT max_abs_coefficient = RealT(0);
        for(const auto & coefficient : w) {
          max_abs_coefficient = std::max(max_abs_coefficient,
                                         std::abs(coefficient));
        }
        IntegralDoubleExcitationLookup<ElemT,RealT> lookup(
          norb,I2,cutoff,max_abs_coefficient);

        constexpr size_t automatic_batch_size = 1000000;
        const size_t aggregate_batch_size = (max_batch_size == 0)
          ? automatic_batch_size : max_batch_size;
        const size_t nso = 2*norb;

#pragma omp parallel
        {
          const size_t num_threads =
            static_cast<size_t>(omp_get_num_threads());
          const size_t local_batch_size =
            std::max<size_t>(1,aggregate_batch_size/num_threads);
          sbd::det_vector<size_t> candidates;
          candidates.resize(local_batch_size);
          size_t candidate_count = 0;
          std::vector<size_t> candidate(det[0].begin(), det[0].end());
          std::vector<int> occupied(nso);
          std::vector<int> unoccupied(nso);

          auto flush_candidates = [&]() {
            if( candidate_count == 0 ) return;
            candidates.resize(candidate_count);
            sort_unique_local_bitarray(candidates);
#pragma omp critical(sbd_gdb_heatbath_append_candidates)
            { append_sorted_candidates_unique(edet,candidates); }
            candidates.resize(local_batch_size);
            candidate_count = 0;
          };
          auto store_candidate = [&]() {
            candidates[candidate_count++] = candidate;
            if( candidate_count == local_batch_size ) flush_candidates();
          };

#pragma omp for schedule(static)
          for(size_t idet=0; idet < det.size(); ++idet) {
            const RealT abs_coefficient = std::abs(w[idet]);
            if( abs_coefficient == RealT(0) ) continue;
            size_t occupied_count = 0;
            size_t unoccupied_count = 0;
            for(size_t orbital=0; orbital < nso; ++orbital) {
              if( getocc(det[idet],bit_length,static_cast<int>(orbital)) ) {
                occupied[occupied_count++] = static_cast<int>(orbital);
              } else {
                unoccupied[unoccupied_count++] = static_cast<int>(orbital);
              }
            }

            for(size_t oi=0; oi < occupied_count; ++oi) {
              int annihilated = occupied[oi];
              for(size_t ua=0; ua < unoccupied_count; ++ua) {
                int created = unoccupied[ua];
                if( (annihilated % 2) != (created % 2) ) continue;
                const ElemT hij = OneExcite(det[idet],bit_length,
                                            annihilated,created,I1,I2);
                if( std::abs(hij) * abs_coefficient > cutoff ) {
                  assign_det(candidate, det[idet]);
                  setocc(candidate,bit_length,annihilated,false);
                  setocc(candidate,bit_length,created,true);
                  store_candidate();
                }
              }
            }

            for(size_t oi=0; oi < occupied_count; ++oi) {
              for(size_t oj=oi+1; oj < occupied_count; ++oj) {
                const int first = occupied[oi];
                const int second = occupied[oj];
                const auto entries_begin = lookup.begin(first,second);
                const auto entries_end = lookup.end(first,second);
                for(auto entry=entries_begin; entry != entries_end; ++entry) {
                  if( entry->abs_integral * abs_coefficient <= cutoff ) break;
                  if( getocc(det[idet],bit_length,entry->created_first)
                      || getocc(det[idet],bit_length,entry->created_second) ) {
                    continue;
                  }
                  assign_det(candidate, det[idet]);
                  setocc(candidate,bit_length,first,false);
                  setocc(candidate,bit_length,second,false);
                  setocc(candidate,bit_length,entry->created_first,true);
                  setocc(candidate,bit_length,entry->created_second,true);
                  store_candidate();
                }
              }
            }
          }
          flush_candidates();
        }
      }

    } // end namespace detail

    template<typename HDetT>
    void singles_from_hdet(const HDetT& hdet,
			  size_t bit_length,
			  size_t norb,
			  size_t num_one,
			  int spin,
			  const std::vector<int> & open,
			  const std::vector<int> & closed,
			  std::vector<std::vector<size_t>> & edet,
			  std::vector<int> & cran) {
      size_t num_ex = num_one * (norb - num_one);
      edet.resize(num_ex);
      cran.resize(2*num_ex);
      std::vector<size_t> base(hdet.begin(), hdet.end());
      size_t ex_count = 0;
      for(size_t i=0; i < num_one; i++) {
	setocc(base,bit_length,closed[i],false);
	for(size_t j=0; j < norb-num_one; j++) {
	  setocc(base,bit_length,open[j],true);
	  cran[2*ex_count+0] = 2*closed[i]+spin;
	  cran[2*ex_count+1] = 2*open[j]+spin;
	  edet[ex_count++] = base;
	  setocc(base,bit_length,open[j],false);
	}
	setocc(base,bit_length,closed[i],true);
      }
    }

    template<typename HDetT>
    void doubles_from_hdet(const HDetT& hdet,
			  size_t bit_length,
			  size_t norb,
			  size_t num_one,
			  int spin,
			  const std::vector<int> & open,
			  const std::vector<int> & closed,
			  std::vector<std::vector<size_t>> & edet,
			  std::vector<int> & cran) {
      size_t num_ex = num_one * (num_one-1) * (norb - num_one) * (norb - num_one - 1) / 4;
      edet.resize(num_ex);
      cran.resize(4*num_ex);
      std::vector<size_t> base(hdet.begin(), hdet.end());
      size_t ex_count = 0;
      for(size_t i=0; i < num_one; i++) {
	setocc(base,bit_length,closed[i],false);
	for(size_t j=i+1; j < num_one; j++) {
	  setocc(base,bit_length,closed[j],false);
	  for(size_t k=0; k < norb-num_one; k++) {
	    setocc(base,bit_length,open[k],true);
	    for(size_t l=k+1; l < norb-num_one; l++) {
	      setocc(base,bit_length,open[l],true);
	      cran[4*ex_count+0] = 2*closed[i]+spin;
	      cran[4*ex_count+1] = 2*closed[j]+spin;
	      cran[4*ex_count+2] = 2*open[k]+spin;
	      cran[4*ex_count+3] = 2*open[l]+spin;
	      edet[ex_count++] = base;
	      setocc(base,bit_length,open[l],false);
	    }
	    setocc(base,bit_length,open[k],false);
	  }
	  setocc(base,bit_length,closed[j],true);
	}
	setocc(base,bit_length,closed[i],true);
      }
    }

    void single_from_hdet(const std::vector<int> & open,
			  const std::vector<int> & closed,
			  size_t norb,
			  size_t num_one,
			  int spin,
			  std::vector<int> & cran) {
      size_t num_ex = num_one * (norb - num_one);
      cran.resize(2*num_ex);
      size_t ex_count = 0;
      for(size_t i=0; i < num_one; i++) {
	for(size_t j=0; j < norb-num_one; j++) {
	  cran[2*ex_count+0] = 2*closed[i]+spin;
	  cran[2*ex_count+1] = 2*open[j]+spin;
	  ex_count++;
	}
      }
    }

    void double_from_hdet(const std::vector<int> & open,
			  const std::vector<int> & closed,
			  size_t norb,
			  size_t num_one,
			  int spin,
			  std::vector<int> & cran) {
      size_t num_ex = num_one * (num_one-1) * (norb - num_one) * (norb - num_one - 1) / 4;
      cran.resize(4*num_ex);
      size_t ex_count = 0;
      for(size_t i=0; i < num_one; i++) {
	for(size_t j=i+1; j < num_one; j++) {
	  for(size_t k=0; k < norb-num_one; k++) {
	    for(size_t l=k+1; l < norb-num_one; l++) {
	      cran[4*ex_count+0] = 2*closed[i]+spin;
	      cran[4*ex_count+1] = 2*closed[j]+spin;
	      cran[4*ex_count+2] = 2*open[k]+spin;
	      cran[4*ex_count+3] = 2*open[l]+spin;
	      ex_count++;
	    }
	  }
	}
      }
    }
    
    void makeHeatbathLookup(const sbd::det_vector<size_t, sbd::det_kind::half> & hdet,
			    size_t bit_length,
			    size_t norb,
			    int spin,
			    std::vector<std::vector<std::vector<size_t>>> & edet_single,
			    std::vector<std::vector<int>> & cran_single,
			    std::vector<std::vector<std::vector<size_t>>> & edet_double,
			    std::vector<std::vector<int>> & cran_double) {
      size_t num_one = static_cast<size_t>(bitcount(hdet[0],bit_length,norb));
      size_t num_ex_single = (norb - num_one) * num_one;
      size_t num_ex_double = (norb - num_one) * (norb - num_one - 1) * num_one * (num_one - 1) / 4;
      edet_single.resize(hdet.size());
      cran_single.resize(hdet.size());
      edet_double.resize(hdet.size());
      cran_double.resize(hdet.size());
#pragma omp parallel
      {
	size_t thread_num  = omp_get_thread_num();
	size_t num_threads = omp_get_num_threads();
	std::vector<int> open(norb-num_one);
	std::vector<int> closed(num_one);
	for(size_t ih=thread_num; ih < hdet.size(); ih=ih+num_threads) {
	  int nc = getOpenClosed(hdet[ih],bit_length,norb,open,closed);
	  sbd::gdb::singles_from_hdet(hdet[ih],bit_length,norb,num_one,spin,open,closed,
				      edet_single[ih],cran_single[ih]);
	  sbd::gdb::doubles_from_hdet(hdet[ih],bit_length,norb,num_one,spin,open,closed,
				      edet_double[ih],cran_double[ih]);
	}
      }
    }

    /**
       @brief Local version for heatbath expansion
       @note results includes the original determinants
     */
    template <typename ElemT, typename RealT>
    void local_heatbath_expansion_lookup(const sbd::det_vector<size_t> & det,
					 const sbd::det_vector<size_t, sbd::det_kind::half> & adet,
					 const sbd::det_vector<size_t, sbd::det_kind::half> & bdet,
					 const std::vector<size_t> & adet_count,
					 const std::vector<size_t> & bdet_count,
					 const std::vector<ElemT> & w,
					 size_t bit_length,
					 size_t norb,
					 const ElemT & I0,
					 const oneInt<ElemT> & I1,
					 const twoInt<ElemT> & I2,
					 RealT cutoff,
					 size_t max_batch_size,
					 sbd::det_vector<size_t> & edet) {

      edet.clear();
      if( det.empty() ) return;

      DetIndexMap idxmap;
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,
		      bit_length,norb,idxmap);

      std::vector<std::vector<std::vector<size_t>>> adet_single;
      std::vector<std::vector<int>> aorb_single;
      std::vector<std::vector<std::vector<size_t>>> adet_double;
      std::vector<std::vector<int>> aorb_double;
      std::vector<std::vector<std::vector<size_t>>> bdet_single;
      std::vector<std::vector<int>> borb_single;
      std::vector<std::vector<std::vector<size_t>>> bdet_double;
      std::vector<std::vector<int>> borb_double;
      makeHeatbathLookup(adet,bit_length,norb,0,
			 adet_single,aorb_single,
			 adet_double,aorb_double);
      makeHeatbathLookup(bdet,bit_length,norb,1,
			 bdet_single,borb_single,
			 bdet_double,borb_double);

      size_t num_one_a = static_cast<size_t>(bitcount(adet[0],bit_length,norb));
      size_t num_one_b = static_cast<size_t>(bitcount(bdet[0],bit_length,norb));
      size_t num_ex_single_a = (norb - num_one_a) * num_one_a;
      size_t num_ex_single_b = (norb - num_one_b) * num_one_b;
      size_t num_ex_double_aa = (norb - num_one_a)*(norb - num_one_a - 1)
	* num_one_a * (num_one_a - 1) / 4;
      size_t num_ex_double_bb = (norb - num_one_b)*(norb - num_one_b - 1)
	* num_one_b * (num_one_b - 1) / 4;
      size_t num_ex_double_ab = (norb - num_one_a) * (norb - num_one_b)
	* num_one_a * num_one_b;

      size_t max_candidates_per_det =
          num_ex_single_a + num_ex_single_b
        + num_ex_double_aa + num_ex_double_bb + num_ex_double_ab;

      const size_t effective_max_batch_size =
          (max_batch_size == 0) ? det.size() * max_candidates_per_det
                                : max_batch_size;
      edet = det;
      sort_unique_local_bitarray(edet);

      // Flatten the ragged (ia, ib) space using prefix sums of AdetToDetLen.
      // Each OpenMP thread owns a contiguous range in this flattened space.
      std::vector<size_t> adet_to_det_offset(idxmap.AdetToDetLen.size() + 1, 0);
      for(size_t ia = 0; ia < idxmap.AdetToDetLen.size(); ++ia) {
	adet_to_det_offset[ia + 1] = adet_to_det_offset[ia] + idxmap.AdetToDetLen[ia];
      }
      const size_t num_adet_det_pairs = adet_to_det_offset.back();

      // make LookUps
#pragma omp parallel
      {
	size_t thread_id = 0;
	size_t num_threads_in_parallel = 1;
#ifdef _OPENMP
	thread_id = omp_get_thread_num();
	num_threads_in_parallel = omp_get_num_threads();
#endif
	const size_t local_batch_size =
	  std::max<size_t>(1, effective_max_batch_size / num_threads_in_parallel);

	const size_t pair_begin = num_adet_det_pairs * thread_id / num_threads_in_parallel;
	const size_t pair_end   = num_adet_det_pairs * (thread_id + 1) / num_threads_in_parallel;

	sbd::det_vector<size_t> candidates;
	candidates.reserve(local_batch_size);

	auto flush_candidates = [&]() {
	  if (candidates.empty()) {
	    return;
	  }
	  sort_unique_local_bitarray(candidates);
#pragma omp critical(sbd_gdb_heatbath_append_candidates)
	  {
	    append_sorted_candidates_unique(edet, candidates);
	  }
	  candidates.clear();
	  candidates.reserve(local_batch_size);
	};

	auto push_candidate = [&](const std::vector<size_t>& candidate) {
	  candidates.push_back(candidate);
	  if (candidates.size() >= local_batch_size) {
	    flush_candidates();
	  }
	};

	std::vector<size_t> cdet(det[0].begin(), det[0].end());

	if(pair_begin < pair_end) {
	  size_t ia = static_cast<size_t>(
	    std::upper_bound(adet_to_det_offset.begin(),
	                     adet_to_det_offset.end(),
	                     pair_begin) - adet_to_det_offset.begin() - 1);
	  size_t ib = pair_begin - adet_to_det_offset[ia];

	  for(size_t ipair = pair_begin; ipair < pair_end; ++ipair) {
	    size_t iast = ia;
	    size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	    size_t idet = idxmap.AdetToDetSM[ia][ib];

	    // single alpha excitations
	    for(size_t ja=0; ja < adet_single[iast].size(); ja++) {
	      ElemT hij = OneExcite(det[idet],bit_length,
				    aorb_single[iast][2*ja+0],
				    aorb_single[iast][2*ja+1],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		DetFromAlphaBeta(adet_single[iast][ja],bdet[ibst],bit_length,norb,cdet);
		push_candidate(cdet);
	      }
	    }

	    // double alpha excitations
	    for(size_t ja=0; ja < adet_double[iast].size(); ja++) {
	      ElemT hij = TwoExcite(det[idet],bit_length,
				    aorb_double[iast][4*ja+0],
				    aorb_double[iast][4*ja+1],
				    aorb_double[iast][4*ja+2],
				    aorb_double[iast][4*ja+3],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		DetFromAlphaBeta(adet_double[iast][ja],bdet[ibst],bit_length,norb,cdet);
		push_candidate(cdet);
	      }
	    }

	    // single-alpha * single-beta two-particle excitations excitations
	    for(size_t ja=0; ja < adet_single[ia].size(); ja++) {
	      for(size_t jb=0; jb < bdet_single[ibst].size(); jb++) {
		ElemT hij = TwoExcite(det[idet],bit_length,
				      aorb_single[iast][2*ja+0],
				      borb_single[ibst][2*jb+0],
				      aorb_single[iast][2*ja+1],
				      borb_single[ibst][2*jb+1],
				      I1,I2);
		RealT hijc = std::abs( hij * w[idet] );
		if( hijc > cutoff ) {
		  DetFromAlphaBeta(adet_single[iast][ja],bdet_single[ibst][jb],bit_length,norb,cdet);
		  push_candidate(cdet);
		}
	      }
	    }

	    // single beta excitations
	    for(size_t jb=0; jb < bdet_single[ibst].size(); jb++) {
	      ElemT hij = OneExcite(det[idet],bit_length,
				    borb_single[ibst][2*jb+0],
				    borb_single[ibst][2*jb+1],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		DetFromAlphaBeta(adet[ia],bdet_single[ibst][jb],bit_length,norb,cdet);
		push_candidate(cdet);
	      }
	    }

	    // double beta excitations
	    for(size_t jb=0; jb < bdet_double[ibst].size(); jb++) {
	      ElemT hij = TwoExcite(det[idet],bit_length,
				    borb_double[ibst][4*jb+0],
				    borb_double[ibst][4*jb+1],
				    borb_double[ibst][4*jb+2],
				    borb_double[ibst][4*jb+3],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		DetFromAlphaBeta(adet[iast],bdet_double[ibst][jb],bit_length,norb,cdet);
		push_candidate(cdet);
	      }
	    }

	    ++ib;
	    if(ib == idxmap.AdetToDetLen[ia]) {
	      ++ia;
	      ib = 0;
	      while(ia < idxmap.AdetToDetLen.size() && idxmap.AdetToDetLen[ia] == 0) {
		++ia;
	      }
	    }
	  }
	}

	flush_candidates();
      }
    }

    /**
    @brief 
    */
    template <typename ElemT, typename RealT>
    void local_heatbath_expansion(const sbd::det_vector<size_t> & det,
				  const std::vector<ElemT> & w,
				  size_t bit_length,
				  size_t norb,
				  const ElemT & I0,
				  const oneInt<ElemT> & I1,
				  const twoInt<ElemT> & I2,
				  RealT cutoff,
				  size_t max_batch_size,
				  sbd::det_vector<size_t> & edet) {

      edet.clear();
      if( det.empty() ) return;

      sbd::det_vector<size_t, sbd::det_kind::half> adet;
      sbd::det_vector<size_t, sbd::det_kind::half> bdet;
      std::vector<size_t> adet_count;
      std::vector<size_t> bdet_count;
      getHalfDets(det,bit_length,norb,adet,bdet,adet_count,bdet_count);
      DetIndexMap idxmap;
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,bit_length,norb,idxmap);

      size_t num_one_a = static_cast<size_t>(bitcount(adet[0],bit_length,norb));
      size_t num_one_b = static_cast<size_t>(bitcount(bdet[0],bit_length,norb));
      size_t num_ex_single_a = (norb - num_one_a) * num_one_a;
      size_t num_ex_single_b = (norb - num_one_b) * num_one_b;
      size_t num_ex_double_aa = (norb - num_one_a)*(norb - num_one_a - 1)
	* num_one_a * (num_one_a - 1) / 4;
      size_t num_ex_double_bb = (norb - num_one_b)*(norb - num_one_b - 1)
	* num_one_b * (num_one_b - 1) / 4;
      size_t num_ex_double_ab = (norb - num_one_a) * (norb - num_one_b)
	* num_one_a * num_one_b;

      size_t max_candidates_per_det =
          num_ex_single_a + num_ex_single_b
        + num_ex_double_aa + num_ex_double_bb + num_ex_double_ab;

      const size_t effective_max_batch_size =
          (max_batch_size == 0) ? det.size() * max_candidates_per_det
                                : max_batch_size;
      edet = det;
      sort_unique_local_bitarray(edet);
      std::vector<size_t> adet_to_det_offset(idxmap.AdetToDetLen.size() + 1, 0);
      for(size_t ia = 0; ia < idxmap.AdetToDetLen.size(); ++ia) {
	adet_to_det_offset[ia + 1] = adet_to_det_offset[ia] + idxmap.AdetToDetLen[ia];
      }
      const size_t num_adet_det_pairs = adet_to_det_offset.back();

      // make LookUps
#pragma omp parallel
      {
	size_t thread_id = 0;
	size_t num_threads_in_parallel = 1;
#ifdef _OPENMP
	thread_id = omp_get_thread_num();
	num_threads_in_parallel = omp_get_num_threads();
#endif
	const size_t local_batch_size =
	  std::max<size_t>(1, effective_max_batch_size / num_threads_in_parallel);

	const size_t pair_begin = num_adet_det_pairs * thread_id / num_threads_in_parallel;
	const size_t pair_end   = num_adet_det_pairs * (thread_id + 1) / num_threads_in_parallel;

	sbd::det_vector<size_t> candidates;
	candidates.reserve(local_batch_size);

	auto flush_candidates = [&]() {
	  if (candidates.empty()) {
	    return;
	  }
	  sort_unique_local_bitarray(candidates);
#pragma omp critical(sbd_gdb_heatbath_append_candidates)
	  {
	    append_sorted_candidates_unique(edet, candidates);
	  }
	  candidates.clear();
	  candidates.reserve(local_batch_size);
	};

	auto push_candidate = [&](const std::vector<size_t>& candidate) {
	  candidates.push_back(candidate);
	  if (candidates.size() >= local_batch_size) {
	    flush_candidates();
	  }
	};

	std::vector<size_t> cdet(det[0].begin(), det[0].end());

	if(pair_begin < pair_end) {
	  size_t ia = static_cast<size_t>(
	    std::upper_bound(adet_to_det_offset.begin(),
	                     adet_to_det_offset.end(),
	                     pair_begin) - adet_to_det_offset.begin() - 1);
	  size_t ib = pair_begin - adet_to_det_offset[ia];

	  size_t iast_prev = ia;
	  size_t ibst_prev = idxmap.AdetToBdetSM[ia][ib];

	  std::vector<int> aorb_single(2*num_ex_single_a);
	  std::vector<int> borb_single(2*num_ex_single_b);
	  std::vector<int> aorb_double(2*num_ex_double_aa);
	  std::vector<int> borb_double(2*num_ex_double_bb);
	  std::vector<int> adet_open(norb-num_one_a);
	  std::vector<int> adet_closed(num_one_a);
	  std::vector<int> bdet_open(norb-num_one_b);
	  std::vector<int> bdet_closed(num_one_b);
	  
	  int nc_a = getOpenClosed(adet[iast_prev],bit_length,norb,adet_open,adet_closed);
	  int nc_b = getOpenClosed(bdet[ibst_prev],bit_length,norb,bdet_open,bdet_closed);
	  single_from_hdet(adet_open,adet_closed,norb,num_one_a,0,aorb_single);
	  single_from_hdet(bdet_open,bdet_closed,norb,num_one_b,1,borb_single);
	  double_from_hdet(adet_open,adet_closed,norb,num_one_a,0,aorb_double);
	  double_from_hdet(bdet_open,bdet_closed,norb,num_one_b,1,borb_double);

	  for(size_t ipair = pair_begin; ipair < pair_end; ++ipair) {
	    size_t iast = ia;
	    size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	    size_t idet = idxmap.AdetToDetSM[ia][ib];

	    if( iast != iast_prev ) {
	      int nc = getOpenClosed(adet[iast],bit_length,norb,adet_open,adet_closed);
	      single_from_hdet(adet_open,adet_closed,norb,num_one_a,0,aorb_single);
	      double_from_hdet(adet_open,adet_closed,norb,num_one_a,0,aorb_double);
	    }

	    if( ibst != ibst_prev ) {
	      int nc = getOpenClosed(bdet[ibst],bit_length,norb,bdet_open,bdet_closed);
	      single_from_hdet(bdet_open,bdet_closed,norb,num_one_b,1,borb_single);
	      double_from_hdet(bdet_open,bdet_closed,norb,num_one_b,1,borb_double);
	    }

	    // single alpha excitations
	    for(size_t ja=0; ja < num_ex_single_a; ja++) {
	      ElemT hij = OneExcite(det[idet],bit_length,
				    aorb_single[2*ja+0],
				    aorb_single[2*ja+1],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		assign_det(cdet, det[idet]);
		setocc(cdet,bit_length,aorb_single[2*ja+1],true);
		setocc(cdet,bit_length,aorb_single[2*ja+0],false);
		push_candidate(cdet);
	      }
	    }

	    // double alpha excitations
	    for(size_t ja=0; ja < num_ex_double_aa; ja++) {
	      ElemT hij = TwoExcite(det[idet],bit_length,
				    aorb_double[4*ja+0],
				    aorb_double[4*ja+1],
				    aorb_double[4*ja+2],
				    aorb_double[4*ja+3],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		assign_det(cdet, det[idet]);
		setocc(cdet,bit_length,aorb_double[4*ja+3],true);
		setocc(cdet,bit_length,aorb_double[4*ja+1],false);
		setocc(cdet,bit_length,aorb_double[4*ja+2],true);
		setocc(cdet,bit_length,aorb_double[4*ja+0],false);
		push_candidate(cdet);
	      }
	    }

	    // single-alpha * single-beta two-particle excitations excitations
	    for(size_t ja=0; ja < num_ex_single_a; ja++) {
	      for(size_t jb=0; jb < num_ex_single_b; jb++) {
		ElemT hij = TwoExcite(det[idet],bit_length,
				      aorb_single[2*ja+0],
				      borb_single[2*jb+0],
				      aorb_single[2*ja+1],
				      borb_single[2*jb+1],
				      I1,I2);
		RealT hijc = std::abs( hij * w[idet] );
		if( hijc > cutoff ) {
		  assign_det(cdet, det[idet]);
		  setocc(cdet,bit_length,aorb_single[2*ja+1],true);
		  setocc(cdet,bit_length,aorb_single[2*ja+0],false);
		  setocc(cdet,bit_length,borb_single[2*jb+1],true);
		  setocc(cdet,bit_length,borb_single[2*jb+0],false);
		  push_candidate(cdet);
		}
	      }
	    }

	    // single beta excitations
	    for(size_t jb=0; jb < num_ex_single_b; jb++) {
	      ElemT hij = OneExcite(det[idet],bit_length,
				    borb_single[2*jb+0],
				    borb_single[2*jb+1],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		assign_det(cdet, det[idet]);
		setocc(cdet,bit_length,borb_single[2*jb+1],true);
		setocc(cdet,bit_length,borb_single[2*jb+0],false);
		push_candidate(cdet);
	      }
	    }

	    // double beta excitations
	    for(size_t jb=0; jb < num_ex_double_bb; jb++) {
	      ElemT hij = TwoExcite(det[idet],bit_length,
				    borb_double[4*jb+0],
				    borb_double[4*jb+1],
				    borb_double[4*jb+2],
				    borb_double[4*jb+3],
				    I1,I2);
	      RealT hijc = std::abs( hij * w[idet] );
	      if( hijc > cutoff ) {
		assign_det(cdet, det[idet]);
		setocc(cdet,bit_length,borb_double[4*jb+3],true);
		setocc(cdet,bit_length,borb_double[4*jb+1],false);
		setocc(cdet,bit_length,borb_double[4*jb+2],true);
		setocc(cdet,bit_length,borb_double[4*jb+0],false);
		push_candidate(cdet);
	      }
	    }

	    iast_prev = iast;
	    ibst_prev = ibst;

	    ++ib;
	    if(ib == idxmap.AdetToDetLen[ia]) {
	      ++ia;
	      ib = 0;
	      while(ia < idxmap.AdetToDetLen.size() && idxmap.AdetToDetLen[ia] == 0) {
		++ia;
	      }
	    }
	  }
	}

	flush_candidates();
      }
    }

    /**
       @brief Global heatbah expansion
     */
    template <typename ElemT, typename RealT, typename DetsContainer>
    void HeatbathExpansion(const DetsContainer & det,
			   const std::vector<ElemT> & w,
			   size_t bit_length,
			   size_t norb,
			   const ElemT & I0,
			   const oneInt<ElemT> & I1,
			   const twoInt<ElemT> & I2,
			   int type,
			   RealT cutoff,
			   size_t max_batch_size,
			   sbd::det_vector<size_t> & edet,
			   MPI_Comm b_comm,
			   MPI_Comm comm) {

      if( det.size() != w.size() ) {
	throw std::invalid_argument(
	  "determinant and coefficient counts must match");
      }
      if( type != 0 && type != 1 ) {
	throw std::invalid_argument("heatbath method must be 0 or 1");
      }

      int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
      int mpi_size; MPI_Comm_size(comm,&mpi_size);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      MPI_Comm x_comm;
      int mpi_color_x = mpi_rank_b;
      MPI_Comm_split(comm,mpi_color_x,mpi_rank,&x_comm);
      int mpi_rank_x; MPI_Comm_rank(x_comm,&mpi_rank_x);
      int mpi_size_x; MPI_Comm_size(x_comm,&mpi_size_x);

      size_t i_begin = 0;
      size_t i_end   = det.size();
      get_mpi_range(mpi_size_x,mpi_rank_x,i_begin,i_end);

      sbd::det_vector<size_t> xdet(det.begin() + i_begin, det.begin() + i_end);
      std::vector<ElemT> xw(w.begin() + i_begin, w.begin() + i_end);
      edet.clear();

      if( type == 0 ) {
	local_heatbath_expansion(xdet,xw,bit_length,norb,I0,I1,I2,cutoff,max_batch_size,edet);
      } else if( type == 1 ) {
	detail::local_heatbath_expansion_integral(
	  xdet,xw,bit_length,norb,I1,I2,cutoff,max_batch_size,edet);
      }
      
      sort_global_bitarray(edet,comm);
      redistribution_bitarray(edet,comm);

      if( x_comm != MPI_COMM_NULL ) {
	MPI_Comm_free(&x_comm);
      }
      
    }
    
    
  } // end namespace gdb
} // end namespace sbd

#endif
