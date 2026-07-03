add_executable(gen_dets_gdet main.cc)

set_target_properties(gen_dets_gdet PROPERTIES
  OUTPUT_NAME gdet
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/gen_dets"
)

target_link_libraries(gen_dets_gdet PRIVATE
  sbd_headers
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  BLAS::BLAS
  LAPACK::LAPACK
)

target_compile_definitions(gen_dets_gdet PRIVATE
  $<$<BOOL:${SBD_SBDIAG_DEBUG}>:SBD_SBDIAG_DEBUG>
)

