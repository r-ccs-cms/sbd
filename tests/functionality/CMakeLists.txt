# Only the new CPU spectrum prerequisites are registered here. The existing
# functionality suite remains available through its Makefile.
foreach(name IN ITEMS remap_wavefunction block_lanczos)
  add_executable(spectrum_test_${name} test_${name}.cc)
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
