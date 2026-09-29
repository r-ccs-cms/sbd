# Two-site spin spectrum

Small runnable example for the spectrum CLI, using public SBD CAOP files.
The Hamiltonian is H = S0·S1 with J=1. The supplied exact singlet has E0=-3/4;
the triplet energy is +1/4, so the excitation gap is 1. This example creates the
known state directly; it does not add a ground-state eigensolver to the CLI.

From `apps/caop-spectrum`:

```sh
make example-spin-dimer
```

This builds an exact-state writer, saves a real singlet using SBD SaveWavefunction,
and runs the complex spectrum with real checkpoint promotion. Default execution
uses MPI 2 and OpenMP 2. The run script creates a fresh output directory under `TMPDIR` (`/tmp` by default).
For a different supported MPI decomposition, for example:

```sh
NP=4 H_SIZE=2 T_SIZE=2 OMP_NUM_THREADS=2 make example-spin-dimer
```

The four files sz0.txt, sz1.txt, sy0.txt, sy1.txt are the four applied operators
in that order. No extra determinant file is needed: their combined support spans
the four-dimensional basis. The CSV labels operator_a/operator_b are 0,1,2,3.

Outputs in the printed directory:

- state-000000.bin: native real-double singlet checkpoint.
- lanczos.coeff: complex Lanczos coefficients and response metadata.
- spectrum.csv: omega=0..2, eta=.05.
- broader.csv: same saved coefficients reevaluated with eta=.2.
- calculation.log: basis size, seed rank and iteration diagnostics.

For each same-component pair (Sz/Sz or Sy/Sy), G_aa=1/[4(omega-1+i eta)]
and the opposite-site element is its negative. Sz/Sy cross elements vanish.
verify.py checks the full response and spectral matrices against this analytic
answer. These checks deliberately fail if you edit the physical model/operators;
use the commands inside run.sh directly when exploring a changed model, and
supply a compatible reference state and reference energy.

A straightforward first experiment is reevaluating lanczos.coeff with a new eta
or frequency grid, leaving H and the singlet unchanged. Larger examples should
also test excitation-basis/step convergence; this four-state check does not
establish long-iteration numerical stability or large-system performance.
