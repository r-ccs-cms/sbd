add_executable(caop-stat-evaluator main.cc cli_options.h)
sbd_configure_stat(caop-stat-evaluator)
target_compile_definitions(caop-stat-evaluator PRIVATE $<$<BOOL:${SBD_COMPLEX}>:_COMPLEX>)

if(SBD_BUILD_STAT_TESTS)
  foreach(name IN ITEMS stat h complex)
    add_executable(stat_test_caop_${name} tests/test_${name}.cc tests/support.h)
    sbd_configure_stat(stat_test_caop_${name})
    if(NOT name STREQUAL "h")
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
  # Test both checkpoint scalar types, independently of the selected CLI type.
  if(SBD_COMPLEX)
    add_executable(stat_caop_cli_real main.cc cli_options.h)
    sbd_configure_stat(stat_caop_cli_real)
    set(real_cli stat_caop_cli_real)
    set(complex_cli caop-stat-evaluator)
  else()
    add_executable(stat_caop_cli_complex main.cc cli_options.h)
    sbd_configure_stat(stat_caop_cli_complex)
    target_compile_definitions(stat_caop_cli_complex PRIVATE _COMPLEX)
    set(real_cli caop-stat-evaluator)
    set(complex_cli stat_caop_cli_complex)
  endif()
  foreach(kind IN ITEMS real complex)
    if(kind STREQUAL "real")
      set(fixture stat_test_caop_stat)
    else()
      set(fixture stat_test_caop_complex)
    endif()
    add_test(NAME stat_caop_cli_${kind}
      COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh
        $<TARGET_FILE:${${kind}_cli}> $<TARGET_FILE:${fixture}> ${kind})
    set_tests_properties(stat_caop_cli_${kind} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;MPIEXEC=${MPIEXEC_EXECUTABLE};MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      PROCESSORS 4 TIMEOUT 120)
  endforeach()
endif()
