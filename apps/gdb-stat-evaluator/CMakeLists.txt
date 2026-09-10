# The statistical evaluator is a real-valued CPU/MPI application, including
# when the diagonalization applications use a GPU backend or complex values.
function(sbd_configure_stat target)
  target_link_libraries(${target} PRIVATE sbd_headers MPI::MPI_CXX
    OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK)
  target_compile_definitions(${target} PRIVATE
    $<$<BOOL:${SBD_TRADMODE}>:SBD_TRADMODE>)
endfunction()

add_executable(gdb-stat-evaluator main.cc cli_options.h)
sbd_configure_stat(gdb-stat-evaluator)
add_executable(gdb-stat-evaluator-uhf main.cc cli_options.h)
target_compile_definitions(gdb-stat-evaluator-uhf PRIVATE _UHF)
sbd_configure_stat(gdb-stat-evaluator-uhf)

if(SBD_BUILD_STAT_TESTS)
  set(stat_tests
    gdb_statistical_contribution_expansion gdb_contribution_expansion_extended
    distributed_alias_sampling gdb_exact_external_observable
    gdb_hash_owned_external_observable gdb_sampled_pipeline
    gdb_u_statistic_external_observable generator_round_robin
    stat_evaluator_batch gdb_stat_evaluator)
  foreach(name IN LISTS stat_tests)
    add_executable(stat_test_${name} tests/test_${name}.cc)
    sbd_configure_stat(stat_test_${name})
    set(ranks 1)
    if(name MATCHES "^(distributed_alias_sampling|gdb_hash_owned_external_observable|gdb_sampled_pipeline|generator_round_robin)$")
      set(ranks 1 2 4)
    elseif(name MATCHES "^(stat_evaluator_batch|gdb_stat_evaluator)$")
      set(ranks 1 4)
    endif()
    foreach(n IN LISTS ranks)
      add_test(NAME stat_${name}_${n}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stat_test_${name}> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stat_${name}_${n} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2" PROCESSORS ${n} TIMEOUT 120)
    endforeach()
  endforeach()
  foreach(kind IN ITEMS regular uhf)
    add_executable(stat_fixture_${kind} tests/make_cli_fixture.cc)
    sbd_configure_stat(stat_fixture_${kind})
    if(kind STREQUAL "uhf")
      target_compile_definitions(stat_fixture_${kind} PRIVATE _UHF)
      set(evaluator gdb-stat-evaluator-uhf)
    else()
      set(evaluator gdb-stat-evaluator)
    endif()
    foreach(n IN ITEMS 1 4)
      add_test(NAME stat_cli_${kind}_${n}
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh ${n}
          $<TARGET_FILE:${evaluator}> $<TARGET_FILE:stat_fixture_${kind}>)
      set_tests_properties(stat_cli_${kind}_${n} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;MPIEXEC=${MPIEXEC_EXECUTABLE}" PROCESSORS ${n} TIMEOUT 120)
    endforeach()
  endforeach()
  add_test(NAME stat_cli_expand_shards
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh 4
      $<TARGET_FILE:gdb-stat-evaluator> $<TARGET_FILE:stat_fixture_regular> 2)
  set_tests_properties(stat_cli_expand_shards PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;MPIEXEC=${MPIEXEC_EXECUTABLE}" PROCESSORS 4 TIMEOUT 120)
  add_test(NAME stat_summary COMMAND sh tests/test_summarize_batches.sh)
  set_tests_properties(stat_summary PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} TIMEOUT 30)
endif()
