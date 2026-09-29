#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
example="$PWD/examples/spin_dimer"
work=$(mktemp -d "${TMPDIR:-/tmp}/sbd-spin-dimer-XXXXXX")
python=${PYTHON:-python3}
mpi_run() { "$python" "$PWD/tests/mpi_command.py" "$@"; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mpi_run 1 "$example/write_state" "$work/state-"
mpi_run "${NP:-2}" ./caop-spectrum \
  --scalar-type complex --wavefunction-type real \
  --hamfile "$example/ham.txt" --sites 2 --loadname "$work/state-" \
  --wavefunction-shards 1 --reference-energy -0.75 \
  --h-comm-size "${H_SIZE:-1}" --t-comm-size "${T_SIZE:-1}" \
  --applied-operator-type general \
  --operator-file "$example/sz0.txt" --operator-file "$example/sz1.txt" \
  --operator-file "$example/sy0.txt" --operator-file "$example/sy1.txt" \
  --steps 8 --save-coefficients "$work/lanczos.coeff" \
  --omega-min 0 --omega-max 2 --points 101 --eta 0.05 \
  --output "$work/spectrum.csv" > "$work/calculation.log"
./caop-spectrum --read-coefficients "$work/lanczos.coeff" \
  --omega-min 0 --omega-max 2 --points 101 --eta 0.2 \
  --output "$work/broader.csv"
"$python" "$example/verify.py" "$work"
printf 'Spin dimer results: %s\n' "$work"
