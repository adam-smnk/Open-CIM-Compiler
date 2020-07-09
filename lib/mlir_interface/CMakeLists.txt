set(MLIR_INTERFACE_SOURCES
  cim/cim_ops.cpp
)

add_library(mlirInterface SHARED
  ${MLIR_INTERFACE_SOURCES}
)

target_link_libraries(mlirInterface PUBLIC
  cimSimMock
)
