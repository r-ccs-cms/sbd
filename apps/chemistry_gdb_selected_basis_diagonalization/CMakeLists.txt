add_executable(gdb_diag main.cc)

set_target_properties(gdb_diag PROPERTIES
  OUTPUT_NAME diag
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/chemistry_gdb_selected_basis_diagonalization"
)

sbd_configure_diag(gdb_diag)
