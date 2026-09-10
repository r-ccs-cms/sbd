/** CLI options and argument handling for the gdb-stat-evaluator application. */
#ifndef SBD_APPS_GDB_STAT_EVALUATOR_CLI_OPTIONS_H
#define SBD_APPS_GDB_STAT_EVALUATOR_CLI_OPTIONS_H

#include "sbd/framework/stat/stat_evaluator_batch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gdb_stat_evaluator {

struct CliOptions {
  std::string fcidump_path;
  std::vector<std::string> determinant_files;
  std::string load_name;
  std::string output_path;
  std::string calculation_id;
  sbd::stat_evaluator::Observable observable =
      sbd::stat_evaluator::Observable::both;
  std::size_t bit_length = 64;
  std::size_t sample_count = 20000;
  std::size_t batch_count = 1;
  std::size_t wavefunction_shards = 0;
  std::uint64_t first_batch_id = 0;
  std::uint64_t base_seed = 0;
  double heatbath_cutoff = 0.0;
  double reference_energy = std::numeric_limits<double>::quiet_NaN();
  std::size_t expansion_batch_size = 1000000;
  double minimum_abs_denominator = 0.0;
  bool help = false;
};

inline std::string require_value(int argc, char** argv, int& index) {
  if(index + 1 >= argc)
    throw std::invalid_argument(std::string("missing value after ") + argv[index]);
  return argv[++index];
}

inline void append_comma_separated(const std::string& text,
                            std::vector<std::string>& output) {
  std::stringstream stream(text);
  std::string value;
  while(std::getline(stream, value, ',')) {
    if(!value.empty()) output.push_back(value);
  }
}

inline CliOptions parse_options(int argc, char** argv) {
  CliOptions options;
  for(int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if(argument == "--help" || argument == "-h") {
      options.help = true;
    } else if(argument == "--fcidump") {
      options.fcidump_path = require_value(argc, argv, index);
    } else if(argument == "--detfiles") {
      append_comma_separated(require_value(argc, argv, index),
                             options.determinant_files);
    } else if(argument == "--detfile") {
      options.determinant_files.push_back(require_value(argc, argv, index));
    } else if(argument == "--loadname") {
      options.load_name = require_value(argc, argv, index);
    } else if(argument == "--output" || argument == "--output-csv") {
      options.output_path = require_value(argc, argv, index);
    } else if(argument == "--calculation-id") {
      options.calculation_id = require_value(argc, argv, index);
    } else if(argument == "--observable") {
      options.observable = sbd::stat_evaluator::parse_observable(
          require_value(argc, argv, index));
    } else if(argument == "--bit-length" || argument == "--bit_length") {
      options.bit_length = std::stoull(require_value(argc, argv, index));
    } else if(argument == "--samples" || argument == "--sample-count") {
      options.sample_count = std::stoull(require_value(argc, argv, index));
    } else if(argument == "--batches" || argument == "--batch-count") {
      options.batch_count = std::stoull(require_value(argc, argv, index));
    } else if(argument == "--batch-id") {
      options.first_batch_id = std::stoull(require_value(argc, argv, index));
    } else if(argument == "--wavefunction-shards") {
      options.wavefunction_shards =
          std::stoull(require_value(argc, argv, index));
    } else if(argument == "--seed") {
      options.base_seed = std::stoull(require_value(argc, argv, index));
    } else if(argument == "--heatbath-cutoff" ||
              argument == "--heatbath_cutoff") {
      options.heatbath_cutoff = std::stod(require_value(argc, argv, index));
    } else if(argument == "--reference-energy") {
      options.reference_energy = std::stod(require_value(argc, argv, index));
    } else if(argument == "--expansion-batch-size") {
      options.expansion_batch_size =
          std::stoull(require_value(argc, argv, index));
    } else if(argument == "--minimum-abs-denominator") {
      options.minimum_abs_denominator =
          std::stod(require_value(argc, argv, index));
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if(options.help) return options;
  if(options.fcidump_path.empty())
    throw std::invalid_argument("--fcidump is required");
  if(options.determinant_files.empty())
    throw std::invalid_argument("--detfiles is required");
  if(options.load_name.empty())
    throw std::invalid_argument("--loadname is required");
  if(options.output_path.empty())
    throw std::invalid_argument("--output is required");
  if(options.calculation_id.find_first_of(",\r\n") != std::string::npos)
    throw std::invalid_argument("--calculation-id cannot contain comma or newline");
  if(options.bit_length == 0 || options.bit_length > 64)
    throw std::invalid_argument("--bit-length must be in [1,64]");
  if(options.sample_count < 2)
    throw std::invalid_argument("--samples must be at least 2");
  if(options.batch_count == 0)
    throw std::invalid_argument("--batches must be positive");
  if(options.expansion_batch_size == 0)
    throw std::invalid_argument("--expansion-batch-size must be positive");
  if(!std::isfinite(options.heatbath_cutoff) || options.heatbath_cutoff < 0.0)
    throw std::invalid_argument("--heatbath-cutoff must be finite and non-negative");
  if(!std::isfinite(options.minimum_abs_denominator) ||
     options.minimum_abs_denominator < 0.0)
    throw std::invalid_argument(
        "--minimum-abs-denominator must be finite and non-negative");
  if(options.observable != sbd::stat_evaluator::Observable::variance &&
     !std::isfinite(options.reference_energy))
    throw std::invalid_argument(
        "--reference-energy is required for both and pt2");
  if(options.first_batch_id >
     std::numeric_limits<std::uint64_t>::max() - (options.batch_count - 1))
    throw std::invalid_argument("batch ID range overflows uint64_t");
  return options;
}

inline void print_usage(std::ostream& output) {
  output
      << "usage: gdb-stat-evaluator --fcidump FILE --detfiles FILE[,FILE...]\n"
         "       --loadname PREFIX --output FILE [options]\n"
         "options:\n"
         "  --observable both|variance|pt2  (default: both)\n"
         "  --reference-energy E0           (required for both/pt2)\n"
         "  --samples N                     (default: 20000, minimum: 2)\n"
         "  --batches N                     (default: 1)\n"
         "  --batch-id N                    (first batch ID, default: 0)\n"
         "  --wavefunction-shards N         (saved b size; default: MPI size)\n"
         "  --seed N                        (default: 0)\n"
         "  --heatbath-cutoff EPS           (default: 0)\n"
         "  --bit-length N                  (default: 64)\n"
         "  --calculation-id ID             (default: start timestamp)\n"
         "  --expansion-batch-size N        (default: 1000000)\n"
         "  --minimum-abs-denominator EPS   (default: 0)\n";
}

inline void print_options(std::ostream& output,
                   const CliOptions& options,
                   const std::string& calculation_id,
                   std::size_t wavefunction_shards,
                   int mpi_size) {
  const std::streamsize previous_precision = output.precision(16);
  output << "# FCIDUMP: " << options.fcidump_path << '\n';
  output << "# determinant files: ";
  for(std::size_t index = 0; index < options.determinant_files.size(); ++index) {
    if(index != 0) output << ',';
    output << options.determinant_files[index];
  }
  output << '\n';
  output << "# load name: " << options.load_name << '\n';
  output << "# output CSV: " << options.output_path << '\n';
  output << "# calculation ID: " << calculation_id << '\n';
  output << "# observable: "
         << sbd::stat_evaluator::observable_name(options.observable) << '\n';
  output << "# reference energy: " << options.reference_energy << '\n';
  output << "# samples per batch: " << options.sample_count << '\n';
  output << "# batches: " << options.batch_count << '\n';
  output << "# first batch ID: " << options.first_batch_id << '\n';
  output << "# seed: " << options.base_seed << '\n';
  output << "# heatbath cutoff: " << options.heatbath_cutoff << '\n';
  output << "# bit length: " << options.bit_length << '\n';
  output << "# expansion batch size: " << options.expansion_batch_size << '\n';
  output << "# minimum absolute PT2 denominator: "
         << options.minimum_abs_denominator << '\n';
  output << "# wavefunction shards: " << wavefunction_shards << '\n';
  output << "# MPI size: " << mpi_size << '\n';
  output.precision(previous_precision);
}

}  // namespace gdb_stat_evaluator

#endif
