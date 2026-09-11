#!/bin/sh
set -eu
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

ranks=${1:-1}
evaluator=${2:-./gdb-stat-evaluator}
fixture_maker=${3:-tests/make_cli_fixture}
saved_shards=${4:-$ranks}
fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/gdb-stat-evaluator.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT HUP INT TERM

"${MPIEXEC:-mpirun}" -np "$saved_shards" "$fixture_maker" "$fixture_dir"

detfiles=
rank=0
while [ "$rank" -lt "$saved_shards" ]; do
  file="$fixture_dir/det-$rank.txt"
  if [ -z "$detfiles" ]; then
    detfiles=$file
  else
    detfiles="$detfiles,$file"
  fi
  rank=$((rank + 1))
done

"${MPIEXEC:-mpirun}" -np "$ranks" "$evaluator" \
  --fcidump "$fixture_dir/FCIDUMP" \
  --detfiles "$detfiles" \
  --loadname "$fixture_dir/state-" \
  --wavefunction-shards "$saved_shards" \
  --output "$fixture_dir/result.csv" \
  --observable both \
  --reference-energy -2.0 \
  --samples 20 \
  --batches 2 \
  --batch-id 7 \
  --seed 19 \
  --calculation-id cli-fixture \
  >"$fixture_dir/stdout.txt"

test "$(wc -l < "$fixture_dir/result.csv" | tr -d ' ')" = 3
grep -q '^format_version,calculation_id,observable,' "$fixture_dir/result.csv"
grep -q '^1,cli-fixture,both,19,7,20,' "$fixture_dir/result.csv"
grep -q '^1,cli-fixture,both,19,8,20,' "$fixture_dir/result.csv"
test "$(grep -c 'sbd::stats: timing batch_id=' "$fixture_dir/stdout.txt")" = "$((2 * ranks))"
grep -q ' hash=.* hash_pack=.* hash_mpi=.* hash_receive_sort=.* child_sort=.* local_observable=.* observable_allreduce=' "$fixture_dir/stdout.txt"
grep -q '^# FCIDUMP: ' "$fixture_dir/stdout.txt"
grep -q '^# determinant files: ' "$fixture_dir/stdout.txt"
grep -q '^# observable: both$' "$fixture_dir/stdout.txt"
grep -q '^# samples per batch: 20$' "$fixture_dir/stdout.txt"
grep -q '^# first batch ID: 7$' "$fixture_dir/stdout.txt"
grep -q '^# wavefunction shards: ' "$fixture_dir/stdout.txt"
echo "gdb stat evaluator CLI ($ranks ranks): PASS"

test "$(grep -c 'sbd::stats: profile batch_id=' "$fixture_dir/stdout.txt")" = "$((2 * ranks))"
test "$(grep -c 'sbd::stats: profile_summary batch_id=' "$fixture_dir/stdout.txt")" = 20
awk '
/sbd::stats: profile batch_id=/ {
  for(i = 1; i <= NF; ++i) {
    split($i, pair, "="); value[pair[1]] = pair[2];
  }
  batch = value["batch_id"];
  draws[batch] += value["draws"];
  generated[batch] += value["generated_records"];
  received[batch] += value["received_records"];
  if(value["external_children"] + 0 > value["unique_children"] + 0 ||
     value["unique_children"] + 0 > value["received_records"] + 0) bad = 1;
}
END {
  for(batch in draws)
    if(draws[batch] != 20 || generated[batch] != received[batch]) bad = 1;
  exit bad;
}' "$fixture_dir/stdout.txt"
echo "gdb stat evaluator rank profiles ($ranks ranks): PASS"

test "$(grep -c 'sbd::stats: timing_summary batch_id=' "$fixture_dir/stdout.txt")" = 24

perl "$script_dir/../summarize-batches.pl" \
  --output "$fixture_dir/summary.csv" \
  --log "$fixture_dir/stdout.txt" \
  --profile-output "$fixture_dir/diagnostics.csv" \
  --rank-output "$fixture_dir/rank-values.csv" "$fixture_dir/result.csv"
test "$(wc -l < "$fixture_dir/diagnostics.csv" | tr -d ' ')" = 45
test "$(wc -l < "$fixture_dir/rank-values.csv" | tr -d ' ')" = "$((1 + 44 * ranks))"
grep -q '^cli-fixture,7,timing,total,' "$fixture_dir/diagnostics.csv"
echo "gdb stat evaluator log aggregation ($ranks ranks): PASS"
