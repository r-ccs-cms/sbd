SBD_PATH ?= ../..
CCCOM ?= mpicxx
CCFLAGS ?= -std=c++17 -O3 -fopenmp
SYSLIB ?= -llapack -lblas
MPIEXEC ?= mpirun
INCLUDES := -I$(SBD_PATH)/include
DEPFLAGS = -MMD -MP
TESTS := tests/test_stat tests/test_h

.PHONY: all complex check check-cli check-complex clean
all: caop-stat-evaluator $(TESTS)
caop-stat-evaluator: main.cc cli_options.h
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)
complex: caop-stat-evaluator-complex
caop-stat-evaluator-complex: main.cc cli_options.h
	$(CCCOM) $(CCFLAGS) -D_COMPLEX $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)
tests/%: tests/%.cc tests/support.h
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)
check: all
	OMP_NUM_THREADS=1 $(MPIEXEC) -np 1 tests/test_stat
	OMP_NUM_THREADS=2 $(MPIEXEC) -np 2 tests/test_stat
	OMP_NUM_THREADS=1 $(MPIEXEC) -np 4 tests/test_stat
	OMP_NUM_THREADS=2 $(MPIEXEC) -np 4 tests/test_h
	$(MAKE) check-cli
check-cli: caop-stat-evaluator tests/test_stat
	MPIEXEC='$(MPIEXEC)' sh tests/test_cli.sh ./caop-stat-evaluator ./tests/test_stat
check-complex: caop-stat-evaluator-complex tests/test_complex
	OMP_NUM_THREADS=1 $(MPIEXEC) -np 1 tests/test_complex
	OMP_NUM_THREADS=2 $(MPIEXEC) -np 2 tests/test_complex
	OMP_NUM_THREADS=2 $(MPIEXEC) -np 4 tests/test_complex
	MPIEXEC='$(MPIEXEC)' sh tests/test_cli.sh ./caop-stat-evaluator-complex ./tests/test_complex complex
-include $(wildcard *.d tests/*.d)
clean:
	rm -f caop-stat-evaluator $(TESTS) caop-stat-evaluator-complex tests/test_complex *.d tests/*.d
