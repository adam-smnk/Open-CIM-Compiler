set(MLIR_INTERFACE_SOURCES
  cim/cim_ops.cpp
)

add_library(mlirInterface STATIC
  ${MLIR_INTERFACE_SOURCES}
  ${SIM_OBJ}
)

# target_link_libraries(mlirInterface PUBLIC
#   cimSimMock
# )
