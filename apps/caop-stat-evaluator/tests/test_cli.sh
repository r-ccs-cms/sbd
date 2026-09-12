#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
evaluator=${1:-"$script_dir/../caop-stat-evaluator"}
fixture=${2:-"$script_dir/test_stat"}
kind=${3:-real}
summary="$script_dir/../../gdb-stat-evaluator/summarize-batches.pl"
MPIEXEC=${MPIEXEC:-mpirun}
MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG:--np}
work=$(mktemp -d "${TMPDIR:-/tmp}/sbd-caop-stat-XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
export OMP_NUM_THREADS=1
"$MPIEXEC" "$MPIEXEC_NUMPROC_FLAG" 1 "$fixture" "$work" > "$work/fixture.log"
cat > "$work/ham.txt" <<'HAM'
-1
1 cdag 2 c 0
1 cdag 0 c 2
-0.5 cdag 2 c 1
-0.5 cdag 1 c 2
0.5 cdag 2 c 2
HAM
if [ "$kind" = complex ]; then
  cat > "$work/ham.txt" <<'HAM'
-1
(0,1) cdag 2 c 0
(0,-1) cdag 0 c 2
(-0.5,0) cdag 2 c 1
(-0.5,0) cdag 1 c 2
(0.5,0) cdag 2 c 2
HAM
fi
cat > "$work/basis.txt" <<'BASIS'
001
010
BASIS
for ranks in 1 2 4; do
  "$MPIEXEC" "$MPIEXEC_NUMPROC_FLAG" "$ranks" "$evaluator" --hamfile "$work/ham.txt" --sites 3 --bit-length 2 \
    --detfiles "$work/basis.txt" --loadname "$work/state-" --wavefunction-shards 1 \
    --samples 10000 --batches 3 --seed 1729 --reference-energy -2 \
    --calculation-id "cli-$ranks" --output "$work/both-$ranks.csv" > "$work/cli-$ranks.log"
done
"$MPIEXEC" "$MPIEXEC_NUMPROC_FLAG" 2 "$evaluator" --hamfile "$work/ham.txt" --sites 3 --bit-length 2 \
  --detfiles "$work/basis.txt" --loadname "$work/state-" --wavefunction-shards 1 \
  --observable variance --samples 10000 --output "$work/variance.csv" > "$work/variance.log"
for h in 1 2 4; do
  "$MPIEXEC" "$MPIEXEC_NUMPROC_FLAG" "$h" "$evaluator" --hamfile "$work/ham.txt" --sites 3 --bit-length 2 \
    --h-comm-size "$h" --detfiles "$work/basis.txt" --loadname "$work/state-" --wavefunction-shards 1 \
    --samples 10000 --batches 3 --seed 1729 --reference-energy -2 \
    --calculation-id "h-$h" --output "$work/h-$h.csv" > "$work/h-$h.log"
done
python3 - "$work" <<'PY'
import csv,sys,pathlib
root=pathlib.Path(sys.argv[1])
for ranks in (1,2,4):
    rows=list(csv.DictReader((root/f'both-{ranks}.csv').open()))
    assert len(rows)==3
    for row in rows:
        v=float(row['variance_estimate']);p=float(row['pt2_estimate'])
        assert abs(v-.04)<.015, row
        assert abs(p+v/2.5)<1e-12, row
row=next(csv.DictReader((root/'variance.csv').open()))
assert row['pt2_estimate']=='' and row['reference_energy']==''
reference=list(csv.DictReader((root/'h-1.csv').open()))
for h in (2,4):
    rows=list(csv.DictReader((root/f'h-{h}.csv').open()))
    assert len(rows)==len(reference)
    for a,b in zip(rows,reference):
        for key in ('variance_estimate','pt2_estimate'):
            assert abs(float(a[key])-float(b[key]))<1e-12, (h,key,a,b)
print('PASS: public checkpoint -> CLI, 1/2/4 ranks, variance-only, CSV, fixed-b h=1/2/4 equality')
PY
# Deliberate singular denominator must terminate all ranks.
if "$MPIEXEC" "$MPIEXEC_NUMPROC_FLAG" 2 "$evaluator" --hamfile "$work/ham.txt" --sites 3 --bit-length 2 \
    --detfiles "$work/basis.txt" --loadname "$work/state-" --wavefunction-shards 1 \
    --h-comm-size 2 --samples 10 --reference-energy 0.5 --output "$work/reject.csv" > "$work/reject.log" 2>&1; then
  echo 'FAIL: singular denominator accepted'; exit 1
fi
# Both numeric batches and shared rank diagnostics must remain parseable.
for ranks in 1 2 4; do
  perl "$summary" --output "$work/summary-$ranks.csv" \
    --log "$work/cli-$ranks.log" --profile-output "$work/diagnostics-$ranks.csv" \
    --rank-output "$work/ranks-$ranks.csv" "$work/both-$ranks.csv"
done
for h in 1 2 4; do
  perl "$summary" --output "$work/h-summary-$h.csv" \
    --log "$work/h-$h.log" --profile-output "$work/h-diagnostics-$h.csv" "$work/h-$h.csv"
done
printf 'PASS: collective rejection and shared batch/diagnostic summaries\n'
