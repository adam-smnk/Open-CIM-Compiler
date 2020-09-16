#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/2mm.tc
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
add_custom_target(2mm_mlir_debug ALL
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
  2mm.cpp
)

add_executable(test-cim-2mm
  ${SOURCES}
  ${OBJ_FILES}
  ${SIM_OBJ}
)

target_link_libraries(test-cim-2mm
  mlirInterface
)
