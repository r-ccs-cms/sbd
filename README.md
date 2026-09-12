# Library for selected basis diagonalization

This is a header-only library for diagonalizing quantum systems in a selected basis, with a focus on handling wavefunction vectors that are too large to fit in the memory of a single node.
The library leverages MPI-based parallelization to distribute the wavefunction across multiple nodes.
Command-Line Interface (CLI) applications are provided in the `/apps` directory.

## Author

- Tomonori Shirakawa, RIKEN Center for Computational Science

## Versions

- **v1.0.0**: Initial public release corresponding to the arXiv submission.
- **v1.1.0**: Feature additions, refactoring, and bug fixes
- **v1.2.0**: Feature additions (sbd for general determinant), and bug fixes
- **v1.3.0**: GPU implementation and refactoring.

## Requirement

- Message Passing Interface (MPI)
- OpenMP
- BLAS and LAPACK

## Install

- This code is provided as a header-only llibrary, so no installation is required.

## How to build the Command-Line Interface application codes

- The CLI Application code for parallelized selected basis diagonalization for tensor-product basis is located in `apps/chemistry_tpb_selected_basis_diagonalization`.
- Edit the configuration file to suit your environment and build it with the make command.
- For more information and options for the executable, see README.md in the same directory.
- From v1.1.0, the sample code for parallelized selected basis diagonalization for general Hamiltonian written by creation/annihilation operator is added in `apps/caop_selected_basis_diagonalization`.
- From v1.2.0, the sample code for parallelized selected basis diagonalization for quantum chemistry Hamiltonian in the general determinant basis is added in `apps/chemistry_gdb_selected_basis_diagonalization`.
- From v1.3.0, the GPU implementation for quantum chemistry Hamiltonian in the tensor-product basis is added.

- CPU/MPI statistical external-variance and PT2 evaluators are available for
  [GDB](apps/gdb-stat-evaluator/README.md) and
  [CAOP](apps/caop-stat-evaluator/README.md) wavefunctions. Both are built by CMake;
  enable `SBD_BUILD_STAT_TESTS` to register their regression tests.

## Documentation

For more details on the input file formats and internal structure, see the [User Manual](https://www.doxygen.nl/manual/doxygen_usage.html).
You can generate the documentation by running:
```
doxygen ./doc/Doxyfile
```

---
## Note
This repository contains research code related to the following paper:

- **Title:** Closed-loop calculations of electronic structure on a quantum processor and a classical supercomputer at full scale
- **arXiv:** https://arxiv.org/abs/2511.00224

Version **v1.0.0** corresponds to the code used for the above arXiv submission and represents the initial public release associated with that paper.

Subsequent versions (v1.1.0 and later) include additional features, refactoring, and bug fixes, and may go beyond the exact implementation described in the paper.

The code is shared publicly to support transparency and academic collaboration.  
If you use this repository in your research, please cite the corresponding arXiv paper.

---
This repository also contains research code related to the following paper:

- **Title:** GPU-Accelerated Selected Basis Diagonalization with Thrust for SQD-based Algorithms
- **arXiv:** https://arxiv.org/abs/2601.16637

The GPU implementation of code is mainly developed by IBM Collaborators.
Version **v1.3.0** corresponds to the code used for the above arXiv submission and represents the public release associated with the above paper.

---

## Licence

[Apache License 2.0](https://github.com/r-ccs-cms/sbd/blob/main/LICENSE.txt)
