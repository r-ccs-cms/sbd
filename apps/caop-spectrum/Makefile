SBD_PATH ?= ../..
CCCOM ?= mpicxx
CCFLAGS ?= -std=c++17 -O3 -fopenmp
SYSLIB ?= -llapack -lblas
MPIEXEC ?= mpirun
MPIEXEC_NUMPROC_FLAG ?= -np
PYTHON ?= python3
export MPIEXEC MPIEXEC_NUMPROC_FLAG PYTHON
INCLUDES := -I$(SBD_PATH)/include
DEPFLAGS = -MMD -MP
TESTS := tests/test_checkpoint_io tests/test_local_basis tests/test_remap_wavefunction tests/test_block_lanczos
.PHONY: all check clean example-spin-dimer
all:caop-spectrum
caop-spectrum: main.cc cli_options.h
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ $(SYSLIB)
tests/test_remap_wavefunction: $(SBD_PATH)/tests/functionality/test_remap_wavefunction.cc
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ $(SYSLIB)
tests/test_block_lanczos: $(SBD_PATH)/tests/functionality/test_block_lanczos.cc
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ $(SYSLIB)
tests/%: tests/%.cc
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ $(SYSLIB)
check: all $(TESTS)
	@set -e; for test in $(TESTS); do for n in 1 2 4; do \
	  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 $(PYTHON) tests/mpi_command.py $$n ./$$test; \
	done; done
	@set -e; for ht in '2 1' '1 2' '2 2' '1 4' '4 1'; do \
	  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 $(PYTHON) tests/mpi_command.py 4 ./tests/test_local_basis $$ht; \
	done
	OPENBLAS_NUM_THREADS=1 sh tests/test_cli.sh "$(CURDIR)/caop-spectrum" "$(CURDIR)/tests/test_checkpoint_io"
examples/spin_dimer/write_state: examples/spin_dimer/write_state.cc
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@ $(SYSLIB)
example-spin-dimer: all examples/spin_dimer/write_state
	sh examples/spin_dimer/run.sh
-include *.d tests/*.d examples/spin_dimer/*.d
clean:
	rm -f caop-spectrum $(TESTS) *.d tests/*.d examples/spin_dimer/write_state examples/spin_dimer/*.d
