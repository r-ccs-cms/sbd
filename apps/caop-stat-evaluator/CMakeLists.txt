add_executable(caop-stat-evaluator main.cc cli_options.h)
sbd_configure_stat(caop-stat-evaluator)

if(SBD_BUILD_STAT_TESTS)
  foreach(name IN ITEMS stat h)
    add_executable(stat_test_caop_${name} tests/test_${name}.cc tests/support.h)
    sbd_configure_stat(stat_test_caop_${name})
    if(name STREQUAL "stat")
      set(ranks 1 2 4)
    else()
      set(ranks 4)
    endif()
    foreach(n IN LISTS ranks)
      add_test(NAME stat_caop_${name}_${n}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stat_test_caop_${name}> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(stat_caop_${name}_${n} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2" PROCESSORS ${n} TIMEOUT 120)
    endforeach()
  endforeach()
  add_test(NAME stat_caop_cli
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh
      $<TARGET_FILE:caop-stat-evaluator> $<TARGET_FILE:stat_test_caop_stat>)
  set_tests_properties(stat_caop_cli PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;MPIEXEC=${MPIEXEC_EXECUTABLE};MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    PROCESSORS 4 TIMEOUT 120)
endif()
