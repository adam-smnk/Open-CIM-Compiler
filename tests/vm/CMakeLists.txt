#################################################
# Transform TC to MLIR
#################################################
set(TC_SOURCES
  tc/vm.tc
)

set(MLIR_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_gen)
set(MLIR_GEN_EXTENSION ".mlir")
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
set(MLIR_CIM_EXTENSION "_cim.mlir")

separate_arguments(MLIR_OPT_CIM_FLAGS UNIX_COMMAND "--convert-linalg-to-cim --convert-cim-to-std --cse")

mlir_opt(
  CIM_FILES
  "${MLIR_FILES}"
  "${MLIR_CIM_DIR}"
  "${MLIR_CIM_EXTENSION}"
  "${MLIR_OPT_CIM_FLAGS}"
)

#################################################
# Compile MLIR to .obj
#################################################
set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_gen)
set(OBJ_GEN_EXTENSION ".o")
set(OBJ_FILES)

separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "--convert-linalg-to-cim --convert-cim-to-std --convert-std-to-llvm=\'emit-c-wrappers=1\' --cse")
separate_arguments(MLIR_TRANSLATE_FLAGS UNIX_COMMAND "--mlir-to-llvmir")
separate_arguments(LLC_FLAGS UNIX_COMMAND "--filetype=obj")

compile_mlir(
  OBJ_FILES
  "${MLIR_FILES}"
  "${OBJ_GEN_DIR}"
  "${OBJ_GEN_EXTENSION}"
  "${MLIR_OPT_FLAGS}"
  "${MLIR_TRANSLATE_FLAGS}"
  "${LLC_FLAGS}"
)

add_custom_target(vm_mlir_debug ALL
  DEPENDS ${CIM_FILES}
)

#################################################
# Compile test application
#################################################
set(VM_SOURCES
  vm.cpp
)

add_executable(test-cim-vm
  ${VM_SOURCES}
  ${OBJ_FILES}
)

target_link_libraries(test-cim-vm
  mlirInterface
)
