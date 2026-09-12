# CAOP stat evaluator

A first CPU/MPI CLI for stochastic external variance and Epstein–Nesbet PT2 of
a fixed, normalized real CAOP wavefunction. It follows the GDB statistical CLI and shares SBD framework helpers directly.

## Build and tests

Requires C++17, MPI, OpenMP and BLAS/LAPACK. From this directory:

```sh
make
make check
```

`SBD_PATH` defaults to `../..`; `CCCOM`, `CCFLAGS`, `SYSLIB` and `MPIEXEC`
are overridable. For macOS Homebrew:

```sh
make CCFLAGS='-std=c++17 -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include' \
  SYSLIB='-L/opt/homebrew/opt/libomp/lib -lomp -L/opt/homebrew/opt/openblas/lib -llapack -lblas'
```

From the repository root, CMake also builds the CLI:

```sh
cmake -S . -B build -DSBD_GPU_BACKEND=none -DSBD_BUILD_STAT_TESTS=ON
cmake --build build -j 2
ctest --test-dir build -R '^stat_' --output-on-failure
```

The evaluator targets always use real CPU/MPI code even when diagonalization
applications select complex values or a GPU backend. `SBD_TRADMODE` is supported.
There is no separate UHF executable for the general CAOP Hamiltonian.
The CLI regression requires Python 3 and Perl, accepts evaluator and fixture
executable paths, and creates temporary inputs outside the source tree.
CTest applies a 120-second timeout, including to the MPI error cases.

For Fugaku CPU, the expected overrides (not yet tested for this CLI) are:

```sh
make CCCOM=mpiFCCpx \
  CCFLAGS='-Nclang -std=c++17 -stdlib=libc++ -Kfast,openmp -Xpreprocessor -fopenmp' \
  SYSLIB=-SSL2
```

## Usage

```sh
OMP_NUM_THREADS=1 mpirun -np 4 ./caop-stat-evaluator \
  --hamfile hamiltonian.txt --sites 16 --h-comm-size 1 \
  --basisfiles basis.txt --loadname state- --wavefunction-shards 1 \
  --observable both --reference-energy -4.2 \
  --samples 20000 --batches 30 --seed 1729 --heatbath-cutoff 0 \
  --calculation-id my-state --output batches.csv
```

`--help` lists all options. `--sites` is the total number of occupation bits,
not spatial orbitals. `--bit-length` defaults to 64 and must match the checkpoint.
`--basisfiles` and `--detfiles` accept comma-separated sorted text basis shards.
Hamiltonian files use the public CAOP format: first non-comment line `-1` for
fermions or `1` for hard-core bosons/spins, then real coefficients and zero-based
operators (`cdag`, `c`, `bdag`, `b`, `s+`, `s-`, `sx`, `sz`).

Read real states saved with SBD's public `SaveWavefunction`; do not pass the
ExtSBD Heatbath-CI custom checkpoint format directly. The prefix is used literally:
`state-` means `state-000000.bin`, etc. Saved b-shard count defaults to evaluator
b size (`MPI size / h size`); explicitly set it when increasing b ranks. All saved shards must be
present. Global coefficient norm must equal one within 1e-8; duplicate parents
are rejected. Input basis and checkpoint must describe the same state.

`--h-comm-size N` (alias `--h_comm_size`) defaults to 1 and must divide MPI size.
For example, 8 MPI ranks with `--h-comm-size 2` use `(h,b,t)=(2,4,1)`.
The saved shard count must not exceed current b size. Parents are loaded and
sampled only at h-rank 0; only sampled parents/counts/coefficients/probabilities
are broadcast within h. Hamiltonian file terms are distributed by the public
SBD loader across h, with each shard replicated across b.

All h*b ranks own part of the membership and child vector by MurmurHash3.
For h>1 each record retains a parent ID, allowing the child owner to merge that
parent's contributions across operator shards **before** cutoff and self-product.
PT2 circulates external-child queries through h shards and returns the summed
diagonal to their owner; neither the full child vector nor full Hamiltonian is
replicated for this step. h=1 retains the compact original contribution path.
Profile parent/sample/draw counts are unique (reported on h-rank 0 only);
generated/received record counts include shard partials. Layout is printed in
stdout; the shared CSV schema is unchanged, so keep logs and use separate
calculation IDs when comparing different layouts.

`variance` needs no reference energy and skips diagonal work; `pt2` and `both`
require `--reference-energy`. Results append one flushed row per completed batch.
Use a new output file for a new configuration; appending does not validate prior
configuration or reject duplicate batch IDs. The GDB summarizer is shared:

```sh
perl /path/to/sbd/apps/gdb-stat-evaluator/summarize-batches.pl batches.csv
```

## Meaning and implementation

This is external variance, `||Q H psi||^2`, which equals full variance when the
input is converged within its variational space. PT2 uses `E0-H_aa`. It is not a
guarantee of improved basis efficiency or more accurate corrected energy.

The signed CAOP contributions to the **same parent/child pair** are summed
before applying the cutoff or squaring the self term. Cutoff zero is the exact
Hamiltonian target; positive cutoff screens the aggregated `|c_i H_ai|` and
therefore differs from CAOP HCI's term-level selection threshold.

Round-robin distribution, prepared Alias sampling, MurmurHash3 ownership and
CSV/profile writers come from SBD framework helpers.
CAOP calculation headers live in `include/sbd/caop/stat/`, under `sbd::ca_stat`,
and are also available through `sbd/sbd.h`. CLI options remain application-local
in `caop_stat_evaluator`. Coefficient-only parent sampling lives in
`sbd/framework/stat/parent_sampling.h` under `sbd::stat_evaluator`; the original
GDB include path and names remain available for compatibility.

Rank-local profile/timing writers and the CSV schema are shared with GDB.
`lookup_entries` counts local off-diagonal flat terms; `lookup_bytes` estimates
flat-array payload storage (capacities for scalar arrays and size for masks),
not RSS or peak memory. For h>1, parent merging and the diagonal query ring are
included in `local_observable_seconds`. The first batch also includes preparation.
CSV elapsed time is root-local; rank timing summaries report min/mean/max.
Profile/timing gathering and printing are outside the driver batch timer.

The same summarizer can process CAOP diagnostics:

```sh
perl ../gdb-stat-evaluator/summarize-batches.pl --output summary.csv \
  --log run.log --profile-output diagnostics.csv --rank-output ranks.csv batches.csv
```

## Limits

Real Hermitian models only; t=1, h configurable with default 1. Full-batch child buffers and per-parent aggregation maps are not memory
capped by `--expansion-batch-size`; that option only bounds each thread's append
buffer. The first version scans all off-diagonal terms for each sampled parent
and local diagonal terms per queried external child. PT2 uses an h-step query ring. Large-system scaling is unmeasured.
No semistochastic correction or internal residual evaluation. Singular PT2
denominators fail collectively; negative stochastic batch estimates are allowed.

## Validation coverage

`tests/test_stat.cc` uses an independent Fock-space oracle and exact multinomial
means, with fermionic and bosonic terms, multiword determinants, zero/positive
cutoff, same-parent cancellation, N=2 negative batches, N=1 rejection, empty ranks
and seed replay. `tests/test_h.cc` compares identical batches at fixed b with
h=1/2/4, including cross-shard cancellation, empty operator shards, all three
observables and preservation of global draw counts.

`tests/test_cli.sh` generates a public `SaveWavefunction` checkpoint, checks
1/2/4-rank evaluation and fixed-b h=1/2/4 equality, variance-only output and
collective singular-denominator rejection. The fixture's exact external variance
is 0.04 and PT2 is -0.016; each batch satisfies PT2 = -variance/2.5.
It also checks numeric and rank-diagnostic parsing with the shared summarizer.

Large-system scaling and GPU execution are not covered by these tests.
