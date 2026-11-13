add_executable(tpb_diag main.cc)

set_target_properties(tpb_diag PROPERTIES
  OUTPUT_NAME diag
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/chemistry_tpb_selected_basis_diagonalization"
)

sbd_configure_diag(tpb_diag)
# Unrestricted integrals change only the chemistry FCIDUMP reader (sbd/chemistry/basic), not CAOP.
target_compile_definitions(tpb_diag PRIVATE $<$<BOOL:${SBD_UHF}>:_UHF>)
