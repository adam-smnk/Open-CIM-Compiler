#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/mm.tc
)

set(MLIR_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_gen)
set(MLIR_FILES)

tc_to_cim(
  MLIR_FILES
  "${TC_SOURCES}"
  "${MLIR_GEN_DIR}"
  "${MLIR_GEN_EXTENSION}"
)

#################################################
# (DEBUG) Lower MLIR LINALG to CIM/STD
#################################################
set(CIM_FILES)
set(MLIR_CIM_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_gen)

mlir_opt(
  CIM_FILES
  "${MLIR_FILES}"
  "${MLIR_CIM_DIR}"
  "${MLIR_CIM_EXTENSION}"
  "${MLIR_OPT_CIM_FLAGS}"
)

# Generate lowered MLIR for debugging
add_custom_target(mm_mlir_debug ALL
  DEPENDS ${CIM_FILES}
)

#################################################
# Compile MLIR to .obj
#################################################
set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_gen)
set(OBJ_FILES)

compile_mlir(
  OBJ_FILES
  "${MLIR_FILES}"
  "${OBJ_GEN_DIR}"
  "${OBJ_GEN_EXTENSION}"
  "${MLIR_OPT_FLAGS}"
  "${MLIR_TRANSLATE_FLAGS}"
  "${LLC_FLAGS}"
)

#################################################
# Compile test application
#################################################
set(SOURCES
  mm.cpp
)

add_executable(test-cim-mm
  ${SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-mm
  mlirInterface
)
