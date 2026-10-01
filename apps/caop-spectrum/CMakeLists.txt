add_executable(caop-spectrum main.cc cli_options.h)
if(SBD_GPU_BACKEND STREQUAL "thrust")
  sbd_configure_diag(caop-spectrum "${SBD_CAOP_THRUST_MEMORY_MODE}")
else()
  sbd_configure_cpu(caop-spectrum)
endif()

if(SBD_BUILD_SPECTRUM_TESTS)
  foreach(name IN ITEMS checkpoint_io local_basis)
    add_executable(spectrum_test_${name} tests/test_${name}.cc)
    sbd_configure_cpu(spectrum_test_${name})
    foreach(n IN ITEMS 1 2 4)
      add_test(NAME spectrum_${name}_${n}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:spectrum_test_${name}> ${MPIEXEC_POSTFLAGS})
      math(EXPR slots "${n} * 2")
      set_tests_properties(spectrum_${name}_${n} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1"
        PROCESSORS ${slots} TIMEOUT 120)
    endforeach()
  endforeach()
  foreach(ht IN ITEMS "2,1" "1,2" "2,2" "1,4" "4,1")
    string(REPLACE "," ";" geometry "${ht}")
    string(REPLACE "," "_" label "${ht}")
    add_test(NAME spectrum_geometry_${label}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:spectrum_test_local_basis> ${MPIEXEC_POSTFLAGS} ${geometry})
    set_tests_properties(spectrum_geometry_${label} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=3;OPENBLAS_NUM_THREADS=1" PROCESSORS 12 TIMEOUT 120)
  endforeach()
  add_test(NAME spectrum_geometry_2_2_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:spectrum_test_local_basis> ${MPIEXEC_POSTFLAGS} 2 2)
  set_tests_properties(spectrum_geometry_2_2_2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1" PROCESSORS 16 TIMEOUT 120)
  add_test(NAME spectrum_cli
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.sh
      $<TARGET_FILE:caop-spectrum> $<TARGET_FILE:spectrum_test_checkpoint_io>)
  string(REPLACE ";" "\\;" spectrum_mpi_pre "${MPIEXEC_PREFLAGS}")
  string(REPLACE ";" "\\;" spectrum_mpi_post "${MPIEXEC_POSTFLAGS}")
  set_tests_properties(spectrum_cli PROPERTIES
    ENVIRONMENT "PYTHON=${Python3_EXECUTABLE};MPIEXEC=${MPIEXEC_EXECUTABLE};MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG};MPIEXEC_PREFLAGS=${spectrum_mpi_pre};MPIEXEC_POSTFLAGS=${spectrum_mpi_post};OPENBLAS_NUM_THREADS=1"
    PROCESSORS 12 TIMEOUT 300)
endif()
