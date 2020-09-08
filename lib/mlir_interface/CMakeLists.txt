set(MLIR_INTERFACE_SOURCES
  cim/cim_ops.cpp
)

set(SIM_OBJ
  ${SIM_ROOT}/libs/cim.o
  ${SIM_ROOT}/libs/syscalls.o
  ${SIM_ROOT}/libs/gic.o
  ${SIM_ROOT}/libs/startup.o
)

add_library(mlirInterface STATIC
  ${MLIR_INTERFACE_SOURCES}
  ${SIM_OBJ}
)

# target_link_libraries(mlirInterface PUBLIC
#   cimSimMock
# )
