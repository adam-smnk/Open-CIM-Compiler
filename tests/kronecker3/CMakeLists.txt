#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/kronecker3.tc
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
add_custom_target(kronecker3_mlir_debug ALL
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
  kronecker3.cpp
)

add_executable(test-cim-kronecker3
  ${SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-kronecker3
  mlirInterface
)
