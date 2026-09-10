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

perl -e '
  print "# calculation ID: diagnostic-test\n# MPI size: 2\n";
  for my $rank (0, 1) {
    print qq( 123 sbd::stats: profile batch_id=7 rank=$rank host="node, \\"A\\"" parents=1 membership=1 lookup_entries=0 lookup_bytes=56 sampled_parents=1 draws=50 generated_records=2 received_records=2 unique_children=1 external_children=1\n);
    my $time = 1 + 2 * $rank;
    print " 123 sbd::stats: timing batch_id=7 rank=$rank ",
      join(" ", map { "$_=$time" } qw(distribution sampling expansion hash_observable hash hash_pack hash_mpi hash_receive_sort child_sort local_observable observable_allreduce total)), "\n";
  }
  print " 123 sbd::stats: timing_summary batch_id=7 metric=total min=99 mean=99 max=99\n";
' >"$fixture_dir/run.log"
{
  echo "$header"
  echo '1,diagnostic-test,both,7,7,100,0,-2,1,-1,2,2,1'
} >"$fixture_dir/diagnostic.csv"
perl ./summarize-batches.pl --output "$fixture_dir/result-summary.csv" \
  --log "$fixture_dir/run.log" --profile-output "$fixture_dir/profile.csv" \
  --rank-output "$fixture_dir/ranks.csv" "$fixture_dir/diagnostic.csv"
grep -q '^diagnostic-test,7,timing,total,2,1,2,3,1.5$' "$fixture_dir/profile.csv"
grep -q '^diagnostic-test,7,profile,lookup_entries,2,0,0,0,$' "$fixture_dir/profile.csv"
grep -q '^diagnostic-test,7,0,"node, ""A""",profile,parents,1$' "$fixture_dir/ranks.csv"
test "$(wc -l < "$fixture_dir/profile.csv" | tr -d ' ')" = 23
test "$(wc -l < "$fixture_dir/ranks.csv" | tr -d ' ')" = 45
perl ./summarize-batches.pl --log "$fixture_dir/run.log" \
  --profile-output "$fixture_dir/log-only.csv"
cmp "$fixture_dir/profile.csv" "$fixture_dir/log-only.csv"
if perl ./summarize-batches.pl --log "$fixture_dir/run.log" --log "$fixture_dir/run.log" \
  --profile-output "$fixture_dir/bad.csv" >"$fixture_dir/error" 2>&1; then
  echo 'duplicate log accepted' >&2; exit 1
fi
grep -q 'duplicate calculation/batch/kind/rank' "$fixture_dir/error"
sed '/timing batch_id=7 rank=1/d' "$fixture_dir/run.log" >"$fixture_dir/incomplete.log"
if perl ./summarize-batches.pl --log "$fixture_dir/incomplete.log" \
  --profile-output "$fixture_dir/bad.csv" >"$fixture_dir/error" 2>&1; then
  echo 'incomplete log accepted' >&2; exit 1
fi
grep -q 'incomplete log' "$fixture_dir/error"
sed 's/total=3/total=NaN/' "$fixture_dir/run.log" >"$fixture_dir/nan.log"
if perl ./summarize-batches.pl --log "$fixture_dir/nan.log" \
  --profile-output "$fixture_dir/bad.csv" >"$fixture_dir/error" 2>&1; then
  echo 'nonfinite timing accepted' >&2; exit 1
fi
grep -q 'invalid total' "$fixture_dir/error"
sed 's/,100,/,99,/' "$fixture_dir/diagnostic.csv" >"$fixture_dir/mismatch.csv"
if perl ./summarize-batches.pl --log "$fixture_dir/run.log" \
  --profile-output "$fixture_dir/bad.csv" "$fixture_dir/mismatch.csv" \
  >"$fixture_dir/error" 2>&1; then
  echo 'CSV/log mismatch accepted' >&2; exit 1
fi
grep -q 'disagrees with CSV' "$fixture_dir/error"
echo 'stat evaluator diagnostic summarizer: PASS'
