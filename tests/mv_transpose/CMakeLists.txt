#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/mv_transpose.tc
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

add_custom_target(mv_transpose_mlir_debug ALL
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
  mv_transpose.cpp
)

add_executable(test-cim-mv-transpose
  ${SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-mv-transpose
  mlirInterface
)
