#ifndef SBD_GDB_HASH_OWNED_EXTERNAL_OBSERVABLE_H
#define SBD_GDB_HASH_OWNED_EXTERNAL_OBSERVABLE_H

#include "sbd/chemistry/gdb/u_statistic_external_observable.h"
#include "sbd/framework/murmurhash_basis_distribution.h"
#include "sbd/framework/type_def.h"
#include "sbd/framework/det_vector.h"
#include "sbd/chemistry/basic/integrals.h"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sbd {
namespace gdb {

struct HashObservableCounts {
  std::size_t received_records = 0;
  std::size_t unique_children = 0;
  std::size_t external_children = 0;
};

struct HashObservableTiming {
  murmur_basis::RedistributionTiming redistribution;
  LocalObservableTiming local_observable;
  double denominator_allreduce_seconds = 0.0;
  double final_allreduce_seconds = 0.0;
};

inline void make_hash_owned_membership_basis(
    sbd::det_vector<std::size_t>& variational_basis,
    MPI_Comm communicator,
    std::uint32_t hash_seed = 0) {
  murmur_basis::redistribute_unique_determinants_by_hash(
      variational_basis, communicator, hash_seed);
}

// Move sampled child records to the same MurmurHash3 owners as the
// variational basis, reduce/exclude locally, then sum the observable globally.
template <typename ElemT, typename RealT>
UStatisticExternalObservable<RealT>
distributed_u_statistic_external_observable(
    sbd::det_vector<std::size_t>& children,
    std::vector<Contribution<ElemT, RealT>>& contributions,
    const sbd::det_vector<std::size_t>& hash_owned_variational_basis,
    std::size_t sample_count,
    std::size_t bit_length,
    std::size_t norb,
    const ElemT& scalar_integral,
    const sbd::oneInt<ElemT>& one_integrals,
    const sbd::twoInt<ElemT>& two_integrals,
    RealT reference_energy,
    MPI_Comm communicator,
    std::uint32_t hash_seed = 0,
    RealT minimum_abs_denominator = RealT(0),
    bool compute_variance = true,
    bool compute_pt2 = true,
    HashObservableTiming* timing = nullptr,
    HashObservableCounts* counts = nullptr) {
  if(counts != nullptr) *counts = HashObservableCounts{};
  if(timing != nullptr) *timing = HashObservableTiming{};
  murmur_basis::redistribute_determinant_values_by_hash(
      children, contributions, communicator, hash_seed,
      timing == nullptr ? nullptr : &timing->redistribution);
  bool local_denominator_error = false;
  const auto local =
      u_statistic_external_observable_sorted_basis<ElemT, RealT>(
      children, contributions, hash_owned_variational_basis,
      sample_count, bit_length, norb, scalar_integral,
      one_integrals, two_integrals, reference_energy,
      minimum_abs_denominator, compute_variance, compute_pt2,
      &local_denominator_error,
      timing == nullptr ? nullptr : &timing->local_observable);

  if(counts != nullptr) {
    counts->received_records = children.size();
    counts->unique_children = local.unique_children;
    counts->external_children = local.external_determinants;
  }

  int local_error = local_denominator_error ? 1 : 0;
  int global_error = 0;
  double phase_start = MPI_Wtime();
  MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX,
                communicator);
  if(timing != nullptr)
    timing->denominator_allreduce_seconds = MPI_Wtime() - phase_start;
  if(global_error != 0)
    throw std::domain_error("external PT2 denominator is too small");

  UStatisticExternalObservable<RealT> global;
  phase_start = MPI_Wtime();
  MPI_Allreduce(&local.variance, &global.variance, 1,
                sbd::GetMpiType<RealT>::MpiT, MPI_SUM, communicator);
  MPI_Allreduce(&local.pt2, &global.pt2, 1,
                sbd::GetMpiType<RealT>::MpiT, MPI_SUM, communicator);
  MPI_Allreduce(&local.external_determinants,
                &global.external_determinants, 1,
                SBD_MPI_SIZE_T, MPI_SUM, communicator);
  MPI_Allreduce(&local.unique_children, &global.unique_children, 1,
                SBD_MPI_SIZE_T, MPI_SUM, communicator);
  if(timing != nullptr)
    timing->final_allreduce_seconds = MPI_Wtime() - phase_start;
  return global;
}

}  // namespace gdb
}  // namespace sbd

#endif
