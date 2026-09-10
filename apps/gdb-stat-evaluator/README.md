# GDB statistical variance / PT2 evaluator

Evaluate the external-space residual norm (reported as `variance`) and/or
Epstein–Nesbet PT2 from a fixed, normalized GDB wavefunction. This CPU/MPI
application reads the determinant files and wavefunction shards written by SBD;
it does not optimize the wavefunction.

## Build

From this directory, with an MPI C++ compiler, OpenMP, and BLAS/LAPACK:

```sh
make -j2
make uhf
```

The default `SBD_PATH` is `../..`. Compiler and library flags are overridable.
For Homebrew on macOS:

```sh
make -j2 all uhf \
  CCFLAGS='-std=c++17 -Xpreprocessor -fopenmp -O3 -I/opt/homebrew/opt/libomp/include' \
  SYSLIB='-L/opt/homebrew/opt/libomp/lib -lomp -L/opt/homebrew/opt/openblas/lib -llapack -lblas'
```

From the repository root, CMake also builds both applications:

```sh
cmake -S . -B build -DSBD_GPU_BACKEND=none
cmake --build build --target gdb-stat-evaluator gdb-stat-evaluator-uhf
```

Executables are under `build/apps/gdb-stat-evaluator/`. Both are real-valued CPU
programs: the diagonalization application's `SBD_COMPLEX` and GPU backend options
do not switch this evaluator to complex arithmetic or GPU execution.
Use the regular executable for restricted FCIDUMP and the `-uhf` executable for
SBD's interleaved-spin UHF format. The file format is not detected automatically.

## Run

```sh
OMP_NUM_THREADS=2 mpirun -np 4 ./gdb-stat-evaluator \
  --fcidump FCIDUMP \
  --detfiles det-000000.txt,det-000001.txt,det-000002.txt,det-000003.txt \
  --loadname checkpoint/state- \
  --wavefunction-shards 4 \
  --output batches.csv \
  --observable both \
  --reference-energy -6.7 \
  --samples 20000 --batches 10 --seed 1729
```

The determinant files must describe the saved wavefunction's variational basis.
`load_basis_from_files` loads that basis, then `LoadWavefunction` matches the
saved coefficients to it and normalizes them.

`--wavefunction-shards` is the saved `b_comm` size, not the current MPI size.
The current MPI size must be at least this number. Only the first shard-count
ranks load the checkpoint; afterwards the parents and coefficients are spread
across all evaluator ranks. For example, 2 saved shards can be evaluated with
4 MPI ranks. Loading fewer MPI ranks than saved shards is not supported.

### Options

| Option | Default / meaning |
|---|---|
| `--fcidump FILE` | Required integral file |
| `--detfiles A,B,...` | Required determinant files; `--detfile FILE` can also be repeated |
| `--loadname PREFIX` | Required saved wavefunction prefix |
| `--output FILE` | Required batch CSV path; alias `--output-csv` |
| `--observable both\|variance\|pt2` | `both` |
| `--reference-energy E0` | Required for `both` and `pt2`; unused for variance-only |
| `--samples N` | `20000`, minimum 2; alias `--sample-count` |
| `--batches N` | `1`, positive; alias `--batch-count` |
| `--batch-id N` | `0`, first ID; subsequent batches use consecutive IDs |
| `--wavefunction-shards N` | Current MPI size |
| `--seed N` | `0` |
| `--heatbath-cutoff EPS` | `0`, nonnegative; alias `--heatbath_cutoff` |
| `--bit-length N` | `64`, range 1–64; alias `--bit_length` |
| `--calculation-id ID` | Shared start timestamp; no commas or newlines |
| `--expansion-batch-size N` | `1000000`, positive; controls thread-local append buffers |
| `--minimum-abs-denominator EPS` | `0`, nonnegative; reject PT2 denominators at or below this magnitude |
| `--help`, `-h` | Print usage |

The expansion buffer option does not bound the total child records retained by
a batch. All generated contribution records are collected before hash redistribution.

## Estimator

Parents are sampled with replacement with probability `p_i` proportional to
`abs(c_i)`. For each parent selected `w_i` times, single/double excitations passing
`abs(H_ai c_i) > cutoff` produce

```text
x_ai = w_i c_i H_ai / p_i
z_ai = w_i |c_i H_ai|^2 / p_i^2
```

Duplicate children are retained. Child records and variational membership use
the same MurmurHash owner. Each owner reduces contributions for a child, excludes
children in the variational basis, and evaluates

```text
U_a      = (|sum_i x_ai|^2 - sum_i z_ai) / (N (N-1))
variance = sum_a U_a
PT2      = sum_a U_a / (E0 - H_aa)
```

The variance column estimates the external-space residual norm. It equals the
full energy variance when the input is an eigenstate within its variational
space; an internal residual from an unconverged input is not included. A nonzero
cutoff truncates the parent-child contributions. Individual batch estimates can
be negative; they are not clamped.

The shared heatbath lookup stores integral magnitudes. The statistical expansion
still calls `TwoExcite` for the signed matrix element. Membership distribution,
coefficient-weight round-robin distribution, probabilities, Alias tables, and
lookup construction are performed once per multiple-batch call.

PT2 diagonals use `ZeroExcite`. Nonfinite denominators and magnitudes at or below
the configured threshold are rejected collectively; there is no skipping,
clamping, or level shift. The default accepts finite nonzero denominators.

The sampling counts are reproducible for the same weights, rank arrangement,
seed, and batch ID. Changing the MPI decomposition changes the streams. OpenMP
contribution append/reduction order can change floating-point rounding.

## Output and aggregation

CSV is appended, with a header for a new or empty file. Each completed batch is
flushed immediately. Unrequested observables are blank. CLI configuration and
per-batch timing go to standard output. Setup time is charged only to the first
batch. All timing fields retain each rank's local elapsed seconds. At the output
boundary, `write_stats_timing` gathers them and root writes one `timing` line per
rank and one `timing_summary` line per metric (minimum, mean, maximum). The CSV
elapsed field remains rank 0's local measurement. The host corresponding to a
rank is available in the profile line for the same batch. Independent phase
maxima need not sum to the maximum total time.

```sh
perl ./summarize-batches.pl --output summary.csv batches-a.csv batches-b.csv
```

The script groups by calculation ID and reports means, sample standard deviations,
standard errors, and summed elapsed time. It rejects duplicate calculation/batch
IDs and mismatched observable, sample count, cutoff, or reference energy.
Use a calculation ID only for compatible runs of the same wavefunction; input
identity is not checked by the summary script.

### Aggregate profile and timing logs

Capture standard output when running the evaluator, then pass one or more logs
alongside the result CSVs:

```sh
perl ./summarize-batches.pl --output summary.csv \
  --log run-a.log --log run-b.log \
  --profile-output diagnostics.csv --rank-output rank-values.csv \
  batches-a.csv batches-b.csv
```

`summary.csv` keeps the existing numerical result format. The additional outputs
use long-form CSV rows:

- `diagnostics.csv`: one row per calculation, batch, kind (`profile`/`timing`),
  and metric, with rank count, minimum, mean, maximum, and `max_over_mean`.
  The ratio is blank when every value is zero. These statistics compare ranks
  within each batch; they do not average different batches together.
- Optional `rank-values.csv`: the original per-rank metric values, with
  calculation ID, batch ID, rank, and host. It supports subsequent grouping by
  host or inspection of individual ranks.

The parser uses rank rows, ignoring precomputed `profile_summary` and
`timing_summary` lines. Keep the `# calculation ID:` and `# MPI size:` metadata
in each log. Both profile and timing must have all ranks for every included
batch; duplicate rows, incomplete batches, invalid numbers, and inconsistent
counts are rejected. Older logs without rank rows are not supported by this mode.

When CSVs are supplied, every log batch must match a CSV row, its MPI size, and
its total number of draws. CSV batches without a corresponding log are allowed.
The CSV's elapsed total still sums the root times; rank maxima are available in
the diagnostic `timing,total` rows. Log-only analysis is also supported:

```sh
perl ./summarize-batches.pl --log run.log --profile-output diagnostics.csv
```

`--profile-output` is required with `--log`; `--rank-output` is optional. Each
output must use a distinct path. No changes to C++ log generation are required.

## Rank profiles

Each batch also writes `sbd::stats: profile` lines to standard output, one per
MPI rank, with `batch_id`, `rank`, and a quoted `host` (MPI processor name).
The CLI gathers these small records and writes them in rank order on rank 0.
`profile_summary` lines give the minimum, arithmetic mean, and maximum across
ranks for each metric, including empty ranks. These are rank statistics, not
aggregates per physical node; the host field allows downstream grouping.

| Field | Rank-local meaning |
|---|---|
| `parents` | Fixed parent count after coefficient-weight round-robin distribution |
| `membership` | Variational basis count after hash distribution and deduplication |
| `lookup_entries` | Integral-magnitude lookup entry count |
| `lookup_bytes` | Lookup offset/entry payload bytes, from `storage_bytes()` |
| `sampled_parents` | Number of parents selected at least once in this batch |
| `draws` | Sum of local sampling multiplicities; sum over ranks equals `--samples` |
| `generated_records` | Generated child/contribution records, including duplicate children |
| `received_records` | Records after hash-owner redistribution, including duplicates |
| `unique_children` | Distinct received children before variational membership exclusion |
| `external_children` | Distinct children outside the variational basis |

The last two counts describe the children visited during aggregation; the
implementation does not construct a separate compacted array. Children whose
contributions cancel still count as children. Globally, generated and received
record counts agree. Hash balance should be judged using both received records
(memory/aggregation work) and distinct children (owner assignment).

`lookup_bytes` uses vector sizes and does not include spare capacity, allocator
overhead, other arrays, MPI buffers, or process RSS. This profile does not report
peak process memory. The CSV schema and phase timing names remain unchanged.
The timing text format now has rank rows and summary rows instead of one timing
row per batch. Profile and timing gathering/output occur in the CLI callback,
outside the driver's recorded batch elapsed time.

Library callbacks receive rank-local `EvaluatedBatch::profile` on every rank.
Both `write_stats_profiles` and `write_stats_timing` are collective and must be
called by all ranks in the communicator, even though they write only on the
specified root. `write_stats_timing` now requires a communicator argument.
CSV writers remain local functions called only by the output rank.

## Library entry points

Include `sbd/chemistry/gdb/stat_evaluator.h` for
`sbd::gdb::evaluate_statistical_batches` or the one-batch wrapper
`evaluate_statistical_batch`. The driver accepts a fixed wavefunction, integrals,
`StatEvaluatorOptions`, batch requests, an MPI communicator, and a callback for
completed results. The callback executes on every participating rank; the CLI
writes only on rank 0. All ranks must supply consistent collective options and
batch requests.

Common components are separately available:

- `sbd/framework/distributed_alias_sampling.h` (`sbd::distributed_sampling`)
- `sbd/framework/murmurhash_basis_distribution.h` (`sbd::murmur_basis`)
- `sbd/framework/stat/stat_evaluator_batch.h` (`sbd::stat_evaluator`)

The exact external observable header provides a one-rank reference for testing.
The CLI supports real restricted/UHF inputs. General complex statistical
contribution validation and GPU evaluation are not included in this integration.

## Tests

From this directory, passing the same compiler/library flags used for building:

```sh
OMP_NUM_THREADS=2 make check check-cli check-cli-uhf check-summary
```

The tests cover contribution generation, finite-multinomial U-statistic expectation,
exact references, distributed Alias sampling, hash-owner reduction, round-robin
pair preservation, multiple-batch evaluation, CSV aggregation, regular/UHF CLI
fixtures, and expansion from 2 saved shards to 4 MPI ranks. MPI tests use up to
4 ranks. `MPIEXEC` can select the MPI launcher used by the CLI test script.

CMake/CTest registration is optional:

```sh
cmake -S . -B build -DSBD_BUILD_STAT_TESTS=ON
cmake --build build
OMP_NUM_THREADS=2 ctest --test-dir build -R '^stat_' --output-on-failure
```

The CMake commands in this section are run from the repository root. The test
fixtures are generated in temporary directories and removed by the scripts.
