SBD_PATH ?= ../..
CCCOM ?= mpicxx
CCFLAGS ?= -std=c++17 -O3 -fopenmp
SYSLIB ?= -llapack -lblas
INCLUDES := -I$(SBD_PATH)/include
DEPFLAGS = -MMD -MP

TESTS := tests/test_gdb_statistical_contribution_expansion \
	tests/test_gdb_contribution_expansion_extended \
	tests/test_distributed_alias_sampling \
	tests/test_gdb_exact_external_observable \
	tests/test_gdb_hash_owned_external_observable \
	tests/test_gdb_sampled_pipeline \
	tests/test_gdb_u_statistic_external_observable \
	tests/test_generator_round_robin

TESTS += tests/test_stat_evaluator_batch
TESTS += tests/test_gdb_stat_evaluator

.PHONY: all uhf check check-cli check-cli-uhf check-summary clean
all: gdb-stat-evaluator $(TESTS)
uhf: gdb-stat-evaluator-uhf

gdb-stat-evaluator: main.cc cli_options.h
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)
gdb-stat-evaluator-uhf: main.cc cli_options.h
	$(CCCOM) $(CCFLAGS) -D_UHF $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)

check: $(TESTS)
	mpirun -np 1 tests/test_gdb_statistical_contribution_expansion
	mpirun -np 1 tests/test_gdb_contribution_expansion_extended
	mpirun -np 1 tests/test_distributed_alias_sampling
	mpirun -np 2 tests/test_distributed_alias_sampling
	mpirun -np 4 tests/test_distributed_alias_sampling
	mpirun -np 1 tests/test_gdb_exact_external_observable
	mpirun -np 1 tests/test_gdb_hash_owned_external_observable
	mpirun -np 2 tests/test_gdb_hash_owned_external_observable
	mpirun -np 4 tests/test_gdb_hash_owned_external_observable
	mpirun -np 1 tests/test_gdb_sampled_pipeline
	mpirun -np 2 tests/test_gdb_sampled_pipeline
	mpirun -np 4 tests/test_gdb_sampled_pipeline
	mpirun -np 1 tests/test_gdb_u_statistic_external_observable
	mpirun -np 1 tests/test_generator_round_robin
	mpirun -np 2 tests/test_generator_round_robin
	mpirun -np 4 tests/test_generator_round_robin
	mpirun -np 1 tests/test_stat_evaluator_batch
	mpirun -np 4 tests/test_stat_evaluator_batch
	mpirun -np 1 tests/test_gdb_stat_evaluator
	mpirun -np 4 tests/test_gdb_stat_evaluator

check-cli: gdb-stat-evaluator tests/make_cli_fixture
	sh tests/test_cli.sh 1
	sh tests/test_cli.sh 4
	sh tests/test_cli.sh 4 ./gdb-stat-evaluator tests/make_cli_fixture 2

check-cli-uhf: gdb-stat-evaluator-uhf tests/make_cli_fixture_uhf
	sh tests/test_cli.sh 1 ./gdb-stat-evaluator-uhf tests/make_cli_fixture_uhf
	sh tests/test_cli.sh 4 ./gdb-stat-evaluator-uhf tests/make_cli_fixture_uhf

check-summary:
	sh tests/test_summarize_batches.sh


tests/make_cli_fixture_uhf: tests/make_cli_fixture.cc
	$(CCCOM) $(CCFLAGS) -D_UHF $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)
tests/%: tests/%.cc
	$(CCCOM) $(CCFLAGS) $(DEPFLAGS) -MF $@.d $(INCLUDES) -o $@ $< $(SYSLIB)

-include $(wildcard *.d tests/*.d)
clean:
	rm -f gdb-stat-evaluator gdb-stat-evaluator-uhf $(TESTS) tests/make_cli_fixture tests/make_cli_fixture_uhf tests/check_checkpoint_against_exact *.d tests/*.d
