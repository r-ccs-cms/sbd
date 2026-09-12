#ifndef SBD_CAOP_STAT_CONTRIBUTION_EXPANSION_H
#define SBD_CAOP_STAT_CONTRIBUTION_EXPANSION_H

#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/framework/bit_manipulation.h"
#include "sbd/caop/basic/generalop.h"
#include <mpi.h>
#include <cstddef>
#include <type_traits>
#include <vector>
#include "sbd/caop/basic/generalop_flatdata.h"
#include "sbd/framework/murmurhash_basis_distribution.h"
#include <map>
#include <cmath>
#include <stdexcept>

namespace sbd::caop::stat {

template<class ElemT, class RealT> struct Contribution {
  ElemT weighted_amplitude{};
  RealT self_product{};
};

// Parent identity is (sampling b-rank, index in that batch's sampled list).
// x is partial across Hamiltonian shards; never square it before owner reduction.
template<class ElemT, class RealT> struct PartialContribution {
  ElemT weighted_amplitude{};
  std::size_t parent_rank{}, parent_index{}, count{};
  RealT probability{};
};

// Match SBD's diagonal number-operator convention after NormalOrdering.
template<class Det, class ElemT>
ElemT diagonal_element(const Det& child, std::size_t bits,
                       const sbd::GeneralOp<ElemT>& h) {
  ElemT value{};
  for(std::size_t i=0; i<h.NumNcTerms(); ++i) {
    const auto term = h.NcTerm(i);
    bool occupied = true;
    for(int k=0; k<term.n_dag(); ++k) {
      const auto q = static_cast<std::size_t>(term.FOp(k).q());
      if(!(child[q/bits] & (std::size_t{1} << (q%bits)))) {
        occupied = false; break;
      }
    }
    if(occupied) value += h.NcCoef(i);
  }
  return value;
}

template<class ElemT>
std::size_t flat_storage_bytes(const sbd::GeneralOpFlatData<ElemT>& f) {
  return f.coefficients.capacity()*sizeof(ElemT) +
      f.orbitals.capacity()*sizeof(int) +
      (f.offsets.capacity()+f.creation_counts.capacity())*sizeof(std::size_t) +
      (f.creation_masks.size()+f.annihilation_masks.size())*
      f.creation_masks.elem_size()*sizeof(std::size_t);
}

template<class ElemT>
void validate_wavefunction(const sbd::det_vector<std::size_t>& parents,
                          const std::vector<ElemT>& coefficients, MPI_Comm comm) {
  int bad = parents.size() != coefficients.size();
  double norm = 0;
  for(const auto& c: coefficients) norm += sbd::SquaredNorm(c);
  if(!std::isfinite(norm)) bad = 1;
  int any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, comm);
  if(any) throw std::invalid_argument("invalid wavefunction coefficients/size");
  double global = 0;
  MPI_Allreduce(&norm, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  if(!std::isfinite(global) || std::abs(global-1.0)>1e-8)
    throw std::invalid_argument("wavefunction must be normalized (tolerance 1e-8)");
  auto unique = parents;
  sbd::murmur_basis::redistribute_unique_determinants_by_hash(unique, comm);
  std::size_t local[2] = {parents.size(), unique.size()}, total[2] = {};
  MPI_Allreduce(local, total, 2, SBD_MPI_SIZE_T, MPI_SUM, comm);
  if(total[0] != total[1]) throw std::invalid_argument("duplicate variational determinant");
}

// Signed ket action is right-to-left. All terms for (parent,child) must be
// summed BEFORE cutoff and self-product; termwise squares lose cross terms.
template<class ElemT, class RealT, class RecordT>
void local_contribution_expansion(
    const sbd::det_vector<std::size_t>& parents,
    const std::vector<ElemT>& coefficients,
    const std::vector<std::size_t>& counts,
    const std::vector<RealT>& probabilities,
    std::size_t bits, const sbd::GeneralOpFlatData<ElemT>& flat, bool sign,
    RealT cutoff, std::size_t batch_size,
    sbd::det_vector<std::size_t>& children,
    std::vector<RecordT>& contributions, std::size_t parent_rank = 0) {
  if(parents.size()!=coefficients.size() || parents.size()!=counts.size() ||
     parents.size()!=probabilities.size() || batch_size==0 ||
     !std::isfinite(cutoff) || cutoff<0)
    throw std::invalid_argument("invalid CAOP expansion arguments");
  for(std::size_t i=0; i<parents.size(); ++i)
    if(counts[i] && (!(probabilities[i]>0) || !std::isfinite(probabilities[i])))
      throw std::invalid_argument("sampled parent has invalid probability");
  children.clear(); contributions.clear();
#pragma omp parallel
  {
    sbd::det_vector<std::size_t> buffer;
    std::vector<RecordT> values;
    auto flush = [&]() {
#pragma omp critical(caop_stat_append)
      {
        for(std::size_t k=0; k<buffer.size(); ++k) {
          children.push_back(buffer[k]); contributions.push_back(values[k]);
        }
      }
      buffer.clear(); values.clear();
    };
#pragma omp for schedule(static)
    for(std::size_t i=0; i<parents.size(); ++i) {
      if(!counts[i] || std::abs(coefficients[i])==0) continue;
      std::map<std::vector<std::size_t>,ElemT> amplitudes;
      std::vector<std::size_t> bra, work;
      for(std::size_t n=0; n<flat.coefficients.size(); ++n) {
        if(!sbd::TryGenerateBraDetFromGeneralOpFlatTerm(parents[i], flat, n, bra)) continue;
        int phase = 1;
        if(sign) {
          work.assign(parents[i].begin(), parents[i].end());
          for(std::size_t k=flat.offsets[n+1]; k>flat.offsets[n]; ) {
            --k;
            const auto q = static_cast<std::size_t>(flat.orbitals[k]);
            work[q/bits] ^= std::size_t{1} << (q%bits);
            phase *= sbd::bit_string_sign_factor(work, bits, q%bits, q/bits);
          }
        }
        amplitudes[bra] += flat.coefficients[n]*static_cast<ElemT>(phase);
      }
      for(const auto& entry: amplitudes) {
        const ElemT v = coefficients[i]*entry.second;
        if(std::abs(v)==0) continue;
        if constexpr(std::is_same_v<RecordT, Contribution<ElemT,RealT>>)
          if(std::abs(v)<cutoff) continue;
        const RealT p = probabilities[i];
        const RealT w = static_cast<RealT>(counts[i]);
        buffer.push_back(entry.first);
        if constexpr(std::is_same_v<RecordT, Contribution<ElemT,RealT>>)
          values.push_back({w*v/p, w*static_cast<RealT>(sbd::SquaredNorm(v))/(p*p)});
        else
          values.push_back({w*v/p, parent_rank, i, counts[i], p});
        if(buffer.size()>=batch_size) flush();
      }
    }
    flush();
  }
}
} // namespace sbd::caop::stat
#endif
