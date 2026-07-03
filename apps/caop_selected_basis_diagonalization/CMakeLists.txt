add_executable(caop_diag main.cc)

set_target_properties(caop_diag PROPERTIES
  OUTPUT_NAME diag
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps/caop_selected_basis_diagonalization"
)

sbd_configure_diag(caop_diag)
