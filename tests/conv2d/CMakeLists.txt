#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/conv2d.tc
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
add_custom_target(conv2d_mlir_debug ALL
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
  conv2d.cpp
)

add_executable(test-cim-conv2d
  ${SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-conv2d
  mlirInterface
)
