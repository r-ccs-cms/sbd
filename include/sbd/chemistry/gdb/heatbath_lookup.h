/**
 * @file sbd/chemistry/gdb/heatbath_lookup.h
 * @brief Integral-magnitude lookup for GDB heatbath candidate generation.
 */
#ifndef SBD_CHEMISTRY_GDB_HEATBATH_LOOKUP_H
#define SBD_CHEMISTRY_GDB_HEATBATH_LOOKUP_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "sbd/chemistry/basic/integrals.h"

namespace sbd {
  namespace gdb {
    namespace detail {

      template <typename RealT>
      struct HeatbathLookupEntry {
        int created_first;
        int created_second;
        RealT abs_integral;
      };

      template <typename ElemT, typename RealT>
      class HeatbathLookup {
      public:
        using entry_type = HeatbathLookupEntry<RealT>;
        using const_iterator = typename std::vector<entry_type>::const_iterator;

        HeatbathLookup(std::size_t norb,
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

          const std::size_t nso = 2 * norb_;
          const std::size_t num_pairs = nso * (nso - 1) / 2;
          pair_offsets_.assign(num_pairs + 1, 0);
          if( max_abs_coefficient == RealT(0) ) return;

          for(std::size_t j=1; j < nso; ++j) {
            for(std::size_t i=0; i < j; ++i) {
              std::size_t count = 0;
              for(std::size_t b=1; b < nso; ++b) {
                for(std::size_t a=0; a < b; ++a) {
                  if( a == i || a == j || b == i || b == j ) continue;
                  const RealT abs_integral = std::abs(
                    I2.Value(static_cast<int>(a),static_cast<int>(i),
                             static_cast<int>(b),static_cast<int>(j))
                    - I2.Value(static_cast<int>(a),static_cast<int>(j),
                               static_cast<int>(b),static_cast<int>(i)));
                  if( abs_integral * max_abs_coefficient > cutoff ) ++count;
                }
              }
              const std::size_t pair = pair_index(i,j);
              pair_offsets_[pair+1] = count;
            }
          }
          for(std::size_t pair=0; pair < num_pairs; ++pair) {
            pair_offsets_[pair+1] += pair_offsets_[pair];
          }

          entries_.resize(pair_offsets_.back());
          for(std::size_t j=1; j < nso; ++j) {
            for(std::size_t i=0; i < j; ++i) {
              const std::size_t pair = pair_index(i,j);
              std::size_t position = pair_offsets_[pair];
              for(std::size_t b=1; b < nso; ++b) {
                for(std::size_t a=0; a < b; ++a) {
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
          const std::size_t pair = checked_pair_index(first,second);
          return entries_.begin() + pair_offsets_[pair];
        }

        const_iterator end(int first, int second) const {
          const std::size_t pair = checked_pair_index(first,second);
          return entries_.begin() + pair_offsets_[pair+1];
        }

        std::size_t entry_count() const noexcept { return entries_.size(); }
        std::size_t storage_bytes() const noexcept {
          return pair_offsets_.size() * sizeof(std::size_t)
            + entries_.size() * sizeof(entry_type);
        }

      private:
        static std::size_t pair_index(std::size_t first, std::size_t second) noexcept {
          return second * (second - 1) / 2 + first;
        }

        std::size_t checked_pair_index(int first, int second) const {
          const int nso = static_cast<int>(2*norb_);
          if( first < 0 || second < 0 || first == second
              || first >= nso || second >= nso ) {
            throw std::out_of_range("invalid annihilation pair");
          }
          if( first > second ) std::swap(first,second);
          return pair_index(static_cast<std::size_t>(first),
                            static_cast<std::size_t>(second));
        }

        std::size_t norb_;
        std::vector<std::size_t> pair_offsets_;
        std::vector<entry_type> entries_;
      };

    } // namespace detail
  } // namespace gdb
} // namespace sbd

#endif
