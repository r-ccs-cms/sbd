#ifndef SBD_GDB_PARENT_SAMPLING_H
#define SBD_GDB_PARENT_SAMPLING_H

#include "sbd/framework/stat/parent_sampling.h"

// Preserve the original GDB API while sharing coefficient-only sampling.
namespace sbd {
namespace gdb {
using stat_evaluator::ParentSample;
using stat_evaluator::PreparedParentSampling;
using stat_evaluator::coefficient_sampling_probabilities;
using stat_evaluator::prepare_parent_sampling_by_coefficient_magnitude;
using stat_evaluator::sample_prepared_parents;
using stat_evaluator::sample_parents_by_coefficient_magnitude;
}  // namespace gdb
}  // namespace sbd

#endif
