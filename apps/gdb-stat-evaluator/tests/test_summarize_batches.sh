#!/bin/sh
set -eu

fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/gdb-stat-summary.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT HUP INT TERM
header='format_version,calculation_id,observable,base_seed,batch_id,sample_count,heatbath_cutoff,reference_energy,variance_estimate,pt2_estimate,mpi_size,omp_threads,elapsed_seconds'
{
  echo "$header"
  echo '1,test,both,7,0,100,0,-2,1,-1,1,2,0.2'
  echo '1,test,both,7,1,100,0,-2,3,-3,1,2,0.4'
} >"$fixture_dir/a.csv"
{
  echo "$header"
  echo '1,test,both,7,2,100,0,-2,5,-5,1,2,0.6'
} >"$fixture_dir/b.csv"

perl ./summarize-batches.pl --output "$fixture_dir/summary.csv" \
  "$fixture_dir/a.csv" "$fixture_dir/b.csv"
grep -q '^1,test,both,3,100,0,-2,3,2,1.1547005383792517,-3,2,1.1547005383792517,1.2' \
  "$fixture_dir/summary.csv"
echo 'stat evaluator batch summarizer: PASS'
