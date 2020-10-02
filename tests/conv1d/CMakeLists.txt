#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/conv1d.tc
)

tc_to_mlir(
  MLIR_FILES
  "${TC_SOURCES}"
)

#################################################
# (DEBUG) Lower MLIR LINALG to CIM/STD
#################################################
mlir_cim_lowering(
  CIM_FILES
  "${MLIR_FILES}"
)

# Generate lowered MLIR for debugging
add_custom_target(conv1d_mlir_debug ALL
  DEPENDS ${CIM_FILES}
)

#################################################
# Compile MLIR to .obj
#################################################
mlir_to_obj(
  OBJ_FILES
  "${MLIR_FILES}"
)

#################################################
# Compile test application
#################################################
set(SOURCES
  conv1d.cpp
)

add_executable(test-cim-conv1d
  ${SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-conv1d
  mlirInterface
)
