#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/tc_4x5x3.tc
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
add_custom_target(tc_4x5x3_mlir_debug ALL
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
  tc_4x5x3.cpp
)

add_executable(test-cim-tc-4x5x3
  ${SOURCES}
  ${OBJ_FILES}
  ${SIM_OBJ}
)

target_link_libraries(test-cim-tc-4x5x3
  mlirInterface
)
