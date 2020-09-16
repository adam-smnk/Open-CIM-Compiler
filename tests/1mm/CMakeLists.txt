#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/1mm.tc
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
add_custom_target(1mm_mlir_debug ALL
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
  1mm.cpp
)

add_executable(test-cim-1mm
  ${SOURCES}
  ${OBJ_FILES}
  ${SIM_OBJ}
)

target_link_libraries(test-cim-1mm
  mlirInterface
)
