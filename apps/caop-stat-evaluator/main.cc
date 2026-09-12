// CLI adapted from public SBD apps/gdb-stat-evaluator at 93ebabec6c84.
#include "cli_options.h"
#include "sbd/sbd.h"
#include "sbd/caop/stat/stat_evaluator.h"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void free_communicator(MPI_Comm& communicator) {
  if(communicator != MPI_COMM_NULL) MPI_Comm_free(&communicator);
}

}  // namespace

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm world = MPI_COMM_WORLD;
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  MPI_Comm load_b_comm = MPI_COMM_NULL;
  MPI_Comm h_comm = MPI_COMM_NULL, b_comm = MPI_COMM_NULL;
  int exit_code = 0;
  try {
    const auto cli = caop_stat_evaluator::parse_options(argc, argv);
    if(cli.help) {
      if(rank == 0) caop_stat_evaluator::print_usage(std::cout);
    } else {
      if(cli.h_comm_size>static_cast<std::size_t>(size) || size%cli.h_comm_size!=0)
        throw std::invalid_argument("--h-comm-size must divide MPI size");
      const int b_size = size/static_cast<int>(cli.h_comm_size);
      const int h_rank = rank/b_size;
      MPI_Comm_split(world,rank%b_size,h_rank,&h_comm);
      MPI_Comm_split(world,h_rank,rank%b_size,&b_comm);
      const std::size_t load_b_size = cli.wavefunction_shards == 0
          ? static_cast<std::size_t>(b_size) : cli.wavefunction_shards;
      if(load_b_size == 0 || load_b_size > static_cast<std::size_t>(b_size))
        throw std::invalid_argument(
            "--wavefunction-shards must be in [1,b size]");
      const bool load_rank = static_cast<std::size_t>(rank) < load_b_size;
      MPI_Comm_split(world, load_rank ? 0 : MPI_UNDEFINED, rank,
                     &load_b_comm);

      int local_checkpoint_missing = 0;
      if(load_rank) {
        const std::string shard = sbd::statefilename(
            cli.load_name, rank);
        std::ifstream input(shard, std::ios::binary);
        local_checkpoint_missing = input ? 0 : 1;
      }
      int checkpoint_missing = 0;
      MPI_Allreduce(&local_checkpoint_missing, &checkpoint_missing, 1,
                    MPI_INT, MPI_MAX, world);
      if(checkpoint_missing != 0)
        throw std::runtime_error(
            "one or more requested wavefunction shards are missing");

      sbd::GeneralOp<double> hamiltonian;
      bool sign = false;
      sbd::load_GeneralOp_from_file(cli.hamiltonian_path, hamiltonian, sign,
                                  h_comm, b_comm, MPI_COMM_SELF);
      if(static_cast<std::size_t>(hamiltonian.max_index()) >= cli.sites)
        throw std::invalid_argument("Hamiltonian site exceeds --sites");
      const std::size_t sites = cli.sites;
      sbd::det_vector<std::size_t>::init_elem_size(
          (sites + cli.bit_length - 1) / cli.bit_length);
      sbd::det_vector<std::size_t> parents;
      std::vector<double> coefficients;
      if(load_rank) {
        sbd::load_basis_from_files(cli.determinant_files, parents,
                                   cli.bit_length, sites, load_b_comm);
        sbd::LoadWavefunction(cli.load_name, parents, MPI_COMM_SELF,
                              load_b_comm, MPI_COMM_SELF, coefficients);
      }
      free_communicator(load_b_comm);

      std::string calculation_id = cli.calculation_id;
      if(calculation_id.empty()) {
        calculation_id =
            sbd::stat_evaluator::make_default_calculation_id(world);
      }

      int output_ok = 1;
      bool write_header = false;
      std::ofstream csv;
      if(rank == 0) {
        std::ifstream existing(cli.output_path,
                               std::ios::binary | std::ios::ate);
        write_header = !existing || existing.tellg() == std::streampos(0);
        csv.open(cli.output_path, std::ios::app);
        output_ok = csv ? 1 : 0;
      }
      MPI_Bcast(&output_ok, 1, MPI_INT, 0, world);
      if(!output_ok)
        throw std::runtime_error("failed to open output CSV: " +
                                 cli.output_path);
      if(rank == 0 && write_header)
        sbd::stat_evaluator::write_csv_header(csv);

      sbd::ca_stat::StatEvaluatorOptions<double> options;
      options.observable = cli.observable;
      options.bit_length = cli.bit_length;
      options.sites = sites;
      options.sample_count = cli.sample_count;
      options.heatbath_cutoff = cli.heatbath_cutoff;
      options.reference_energy = cli.reference_energy;
      options.base_seed = cli.base_seed;
      options.expansion_batch_size = cli.expansion_batch_size;
      options.minimum_abs_denominator = cli.minimum_abs_denominator;

      if(rank == 0)
        caop_stat_evaluator::print_options(std::cout, cli, calculation_id, load_b_size, size);

      std::vector<sbd::ca_stat::BatchRequest> batches(cli.batch_count);
      for(std::size_t offset = 0; offset < cli.batch_count; ++offset) {
        batches[offset].calculation_id = calculation_id;
        batches[offset].batch_id = cli.first_batch_id + offset;
      }
      sbd::ca_stat::evaluate_statistical_batches<double, double>(
          parents, coefficients, hamiltonian, sign, options, batches, world,
          [&](const sbd::ca_stat::EvaluatedBatch& evaluated) {
            sbd::stat_evaluator::write_stats_profiles(
                std::cout, evaluated.record.batch_id, evaluated.profile, world);
            sbd::stat_evaluator::write_stats_timing(
                std::cout, evaluated.record.batch_id, evaluated.timing, world);
            if(rank != 0) return;
            sbd::stat_evaluator::write_csv_record(csv, evaluated.record);
          }, h_comm, b_comm);
    }
  } catch(const std::exception& error) {
    if(rank == 0) {
      std::cerr << "error: " << error.what() << '\n';
      caop_stat_evaluator::print_usage(std::cerr);
    } else {
      std::cerr << "error on rank " << rank << ": " << error.what() << '\n';
    }
    exit_code = 1;
    MPI_Abort(world, exit_code);
  }

  free_communicator(load_b_comm);
  free_communicator(h_comm);
  free_communicator(b_comm);
  int global_exit_code = 0;
  MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, world);
  MPI_Finalize();
  return global_exit_code;
}
