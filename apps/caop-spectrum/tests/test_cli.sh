#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exe=$1
fixture=$2
python=${PYTHON:-python3}
work=$(mktemp -d "${TMPDIR:-/tmp}/sbd-spectrum-XXXXXX")
cleanup() {
  status=$?
  if [ "$status" -eq 0 ] && [ "${SBD_KEEP_TEST_OUTPUT:-0}" = 0 ]; then
    rm -rf "$work"
  else
    printf 'CLI test evidence retained: %s\n' "$work"
  fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
export CAOP_SPECTRUM_EXE="$exe"
mpi_run() { "$python" "$script_dir/mpi_command.py" "$@"; }
export OMP_NUM_THREADS=1
mpi_run 1 "$fixture" "$work" > "$work/fixture.log"
cat > "$work/ham.txt" <<'HAM'
-1
0.92 cdag 0 c 0
0.08 cdag 1 c 1
-1.44 cdag 0 c 1
-1.44 cdag 1 c 0
0.7 cdag 2 c 2
1.3 cdag 3 c 3
0.2 cdag 2 c 3
0.2 cdag 3 c 2
0.4 cdag 0 cdag 2 c 2 c 0
HAM
cat > "$work/extra.txt" <<'BASIS'
0011
0101
0110
1001
1010
1100
BASIS
cat > "$work/remap.txt" <<'BASIS'
0001
0010
0100
BASIS
for ranks in 1 2 4; do
  for channel in addition removal; do
    for shards in 1 2; do
      prefix=state-
      if [ "$shards" = 2 ]; then prefix=two-; fi
      extra=""
      if [ "$channel" = addition ]; then extra="--extra-detfile $work/extra.txt"; fi
      mpi_run "$ranks" "$exe" --hamfile "$work/ham.txt" \
        --loadname "$work/$prefix" --wavefunction-shards "$shards" --sites 4 \
        --orbitals 0,1,2 --channel "$channel" --reference-energy -1 --steps 12 \
        --remap-detfile "$work/remap.txt" $extra \
        --save-coefficients "$work/$channel-$ranks-$shards.coeff" \
        --output "$work/$channel-$ranks-$shards.csv" --omega-min -3 --omega-max 3 --points 13 --eta .13 \
        > "$work/$channel-$ranks-$shards.log"
    done
  done
done
# h/b/t decompositions: actual sharded loader, root-plane checkpoint/remap,
# extra basis and seed distribution. b=2,t=2 exercises nonlocal first slides;
# b=1,t=4 includes ranks with empty slide lists.
for ht in '2 1' '1 2' '2 2' '1 4' '4 1'; do
  set -- $ht
  hs=$1; ts=$2
  for channel in addition removal; do
    extra=""
    if [ "$channel" = addition ]; then extra="--extra-detfile $work/extra.txt"; fi
    OMP_NUM_THREADS=3 mpi_run 4 "$exe" --hamfile "$work/ham.txt" \
      --h-comm-size "$hs" --t-comm-size "$ts" \
      --loadname "$work/two-" --wavefunction-shards 2 --sites 4 \
      --orbitals 0,1,2 --channel "$channel" --reference-energy -1 --steps 12 \
      --remap-detfile "$work/remap.txt" $extra \
      --save-coefficients "$work/$channel-h$hs-t$ts.coeff" \
      --output "$work/$channel-h$hs-t$ts.csv" --omega-min -3 --omega-max 3 --points 13 --eta .13 \
      > "$work/$channel-h$hs-t$ts.log"
  done
done
"$python" - "$work" <<'PYHT'
import csv, pathlib, sys
root=pathlib.Path(sys.argv[1])
for channel in ('addition','removal'):
    baseline=list(csv.DictReader((root/f'{channel}-1-1.csv').open()))
    for path in root.glob(f'{channel}-h*-t*.csv'):
        rows=list(csv.DictReader(path.open()))
        assert len(rows)==len(baseline)
        for actual,expected in zip(rows,baseline):
            for key in expected:
                a,e=float(actual[key]),float(expected[key])
                assert abs(a-e)<1e-11*(1+abs(e)),(path,key,a,e)
print('PASS: h/b/t CLI decomposition invariance, checkpoint shards, Remap and extra basis')
PYHT
# A single identity term leaves three of four h shards empty.
printf '%s\n' '-1' '0.3' > "$work/sparse-ham.txt"
OMP_NUM_THREADS=2 mpi_run 4 "$exe" --hamfile "$work/sparse-ham.txt" \
  --h-comm-size 4 --loadname "$work/two-" --wavefunction-shards 2 --sites 4 \
  --orbitals 0,1 --channel removal --reference-energy 0 \
  --save-coefficients "$work/empty-h.coeff" --output "$work/empty-h.csv" \
  --omega-min .8 --omega-max .8 --points 1 --eta .2 > "$work/empty-h.log"
"$python" - "$work" <<'PYEMPTY'
import csv, pathlib, sys
for row in csv.DictReader((pathlib.Path(sys.argv[1])/'empty-h.csv').open()):
    a,b=(int(row[k]) for k in ('orbital_a','orbital_b'))
    expected=(.6,.8)[a]*(.6,.8)[b]/complex(1.1,.2)
    actual=complex(float(row['G_real']),float(row['G_imag']))
    assert abs(actual-expected)<1e-12
print('PASS: empty Hamiltonian shards with h=4')
PYEMPTY
"$exe" --read-coefficients "$work/addition-1-1.coeff" --output "$work/replay.csv" \
  --omega-min -3 --omega-max 3 --points 13 --eta .13
"$exe" --read-coefficients "$work/addition-1-1.coeff" --output "$work/new-eta.csv" \
  --omega-min -3 --omega-max 3 --points 13 --eta .21
sed '1s/-1/1/' "$work/ham.txt" > "$work/boson.txt"
mpi_run 2 "$exe" --hamfile "$work/boson.txt" --loadname "$work/state-" \
  --wavefunction-shards 1 --sites 4 --orbitals 0,1,2 --channel addition --reference-energy -1 \
  --extra-detfile "$work/extra.txt" --steps 12 --save-coefficients "$work/boson.coeff" \
  --output "$work/boson.csv" --omega-min -3 --omega-max 3 --points 13 --eta .13 > "$work/boson.log"
mpi_run 2 "$exe" --hamfile "$work/ham.txt" --loadname "$work/state-" \
  --wavefunction-shards 1 --sites 4 --orbitals 3 --channel removal --reference-energy -1 \
  --save-coefficients "$work/zero.coeff" --output "$work/zero.csv" > "$work/zero.log"
"$exe" --read-coefficients "$work/zero.coeff" --output "$work/zero-replay.csv"
"$python" "$script_dir/verify_cli.py" "$work"
# Truncating the reference remains explicit: reject without normalization,
# then check the normalized one-mode removal response through the full CLI.
printf '0010\n0100\n' > "$work/truncated-basis.txt"
if mpi_run 2 "$exe" --hamfile "$work/ham.txt" --loadname "$work/state-" \
  --wavefunction-shards 1 --sites 4 --orbitals 1 --channel removal --reference-energy .08 \
  --remap-detfile "$work/truncated-basis.txt" --save-coefficients "$work/rejected-remap.coeff" \
  > "$work/reject-remap.log" 2>&1; then
  echo 'FAIL non-unit remapped state accepted';exit 1
fi
mpi_run 4 "$exe" --hamfile "$work/ham.txt" --loadname "$work/state-" \
  --h-comm-size 2 --t-comm-size 2 \
  --wavefunction-shards 1 --sites 4 --orbitals 1 --channel removal --reference-energy .08 \
  --remap-detfile "$work/truncated-basis.txt" --normalize-remap \
  --save-coefficients "$work/normalized-remap.coeff" --output "$work/normalized-remap.csv" \
  --omega-min 0 --omega-max 0 --points 1 --eta .2 > "$work/normalized-remap.log"
"$python" - "$work" <<'PY'
import csv, pathlib, re, sys
root=pathlib.Path(sys.argv[1])
row=next(csv.DictReader((root/'normalized-remap.csv').open()))
actual=complex(float(row['G_real']),float(row['G_imag']))
assert abs(actual-1/complex(-.08,.2))<1e-12
log=(root/'normalized-remap.log').read_text()
for label,expected in [('retained_norm2',.64),('discarded_norm2',.36)]:
    assert abs(float(re.search(label+r'=([^\s]+)',log).group(1))-expected)<1e-14
print('PASS: framework remap CLI weight accounting and explicit normalization')
PY
# Malformed checkpoint and invalid eta must fail rather than hang/normalize.
printf 'broken' > "$work/bad-000000.bin"
if mpi_run 2 "$exe" --hamfile "$work/ham.txt" --loadname "$work/bad-" \
  --wavefunction-shards 1 --sites 4 --orbitals 0 --channel addition --reference-energy -1 \
  --save-coefficients "$work/bad.coeff" > "$work/reject-checkpoint.log" 2>&1; then
  echo 'FAIL invalid checkpoint accepted';exit 1
fi
if "$exe" --read-coefficients "$work/addition-1-1.coeff" --output "$work/bad.csv" --eta 0 \
  > "$work/reject-eta.log" 2>&1; then
  echo 'FAIL eta=0 accepted';exit 1
fi
for threads in 1 2 3; do
  OMP_NUM_THREADS=$threads "$python" "$script_dir/test_general_operators.py" "$work"
done
"$python" "$script_dir/test_complex_cli.py" "$work"
# Logging is optional per iteration, but workflow/timing remains visible.
for interval in 0 1 2; do
  mpi_run 2 "$exe" --hamfile "$work/ham.txt" --loadname "$work/state-" \
    --wavefunction-shards 1 --sites 4 --orbitals 0,1,2 --channel addition \
    --reference-energy -1 --steps 12 --extra-detfile "$work/extra.txt" \
    --iteration-log-interval "$interval" --save-coefficients "$work/log-$interval.coeff" \
    --output "$work/log-$interval.csv" > "$work/log-$interval.log"
done
"$python" - "$work" <<'PYLOG'
import pathlib,re,sys
root=pathlib.Path(sys.argv[1])
for interval in (0,1,2):
    lines=(root/f'log-{interval}.log').read_text().splitlines()
    assert lines and all(x.startswith('# ') or re.match(r' \d+ sbd::spectrum: ',x) for x in lines)
    steps=int(re.search(r'result .*steps=(\d+)', '\n'.join(lines))[1])
    reported=[int(re.search(r'iteration step=(\d+)',x)[1]) for x in lines if ': iteration ' in x]
    assert reported==(list(range(interval,steps+1,interval)) if interval else []), reported
    for stage in ('checkpoint_read','hamiltonian_read','seed_generation','seed_qr','hamiltonian_preparation','lanczos','coefficient_write','response_and_csv_write','total'):
        assert sum(': start '+stage==x.split(' sbd::spectrum')[-1] for x in lines)==1, stage
        assert sum(': end '+stage+' ' in x for x in lines)==1, stage
    assert 'reference_basis=2 generated_basis=3 excitation_basis=6' in '\n'.join(lines)
    assert (root/f'log-{interval}.coeff').read_bytes()==(root/'log-0.coeff').read_bytes()
    assert (root/f'log-{interval}.csv').read_bytes()==(root/'log-0.csv').read_bytes()
print('PASS: timestamped workflow, iteration intervals, global counts and unchanged outputs')
PYLOG
# Check the saved format across real/complex and single/general fixtures.
"$python" - "$work" <<'PYFORMAT'
import pathlib, sys
root=pathlib.Path(sys.argv[1]); kinds=set()
for path in root.rglob('*.coeff'):
    if not path.stat().st_size:
        continue
    lines=path.read_text().splitlines()
    header=lines[0].split()
    assert len(header)==2 and header[0]=='EXTSBD_CAOP_SPECTRUM', path
    assert header[1] in ('real','complex'), path
    count=int(lines[4]); labels=int(lines[6])
    assert labels in (0,count), path
    kinds.add((header[1],bool(labels)))
assert kinds=={('real',False),('real',True),('complex',False),('complex',True)}, kinds
print('PASS: unversioned coefficient format for all scalar/operator combinations')
PYFORMAT
printf 'PASS: CLI logs %s\n' "$work"
