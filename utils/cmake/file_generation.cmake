#################################################
# TC to MLIR generation using Teckyl frontend
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
# @param[in] GEN_DIR_PATH path to output directory
# @param[in] GEN_ENTENSION extension for generated
#            files (should contain dot)
# @param[in] TECKYL_FLAGS Teckyl options
#################################################
function(teckyl
    OUT_FILE_LIST
    INPUT_FILES
    GEN_DIR_PATH
    GEN_EXTENSION
    TECKYL_FLAGS
  )
  set(GENERATED_FILES)
  file(MAKE_DIRECTORY ${GEN_DIR_PATH})
  
  foreach(INPUT_FILE ${INPUT_FILES})
    get_filename_component(FILE_NAME "${INPUT_FILE}" NAME_WE)
    set(GEN_FILE "${GEN_DIR_PATH}/${FILE_NAME}${GEN_EXTENSION}")

    add_custom_command(
      OUTPUT ${GEN_FILE}
      COMMAND ${TECKYL_BIN} ${TECKYL_FLAGS} ${INPUT_FILE} 2>/dev/null 1> ${GEN_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${INPUT_FILE}
    )

    list(APPEND GENERATED_FILES ${GEN_FILE})
  endforeach()

  set(${OUT_FILE_LIST} ${GENERATED_FILES} PARENT_SCOPE)
endfunction()

#################################################
# Compile MLIR to object file via LLVM IR
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
# @param[in] GEN_DIR_PATH path to output directory
# @param[in] GEN_ENTENSION extension for generated
#            files (should contain dot)
# @param[in] MLIR_OPT_FLAGS MLIR passes
# @param[in] MLIR_TRANSLATE_FLAGS MLIR translation options
# @param[in] LLC_FLAGS LLVM compiler flags
#################################################
function(compile_mlir
    OUT_FILE_LIST
    INPUT_FILES
    GEN_DIR_PATH
    GEN_EXTENSION
    MLIR_OPT_FLAGS
    MLIR_TRANSLATE_FLAGS
    LLC_FLAGS
  )
  set(GENERATED_FILES)
  file(MAKE_DIRECTORY ${GEN_DIR_PATH})
  
  foreach(INPUT_FILE ${INPUT_FILES})
    get_filename_component(FILE_NAME "${INPUT_FILE}" NAME_WE)
    set(GEN_FILE "${GEN_DIR_PATH}/${FILE_NAME}${GEN_EXTENSION}")

    add_custom_command(
      OUTPUT ${GEN_FILE}
      COMMAND ${MLIR_OPT_BIN} ${MLIR_OPT_FLAGS} ${INPUT_FILE} | ${MLIR_TRANSLATE_BIN} ${MLIR_TRANSLATE_FLAGS} | ${LLC_BIN} ${LLC_FLAGS} -o ${GEN_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${INPUT_FILE}
    )

    list(APPEND GENERATED_FILES ${GEN_FILE})
  endforeach()

  set(${OUT_FILE_LIST} ${GENERATED_FILES} PARENT_SCOPE)
endfunction()

#################################################
# Compile MLIR to object file via LLVM IR
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
# @param[in] GEN_DIR_PATH path to output directory
# @param[in] GEN_ENTENSION extension for generated
#            files (should contain dot)
# @param[in] MLIR_OPT_FLAGS MLIR passes
#################################################
function(mlir_opt
    OUT_FILE_LIST
    INPUT_FILES
    GEN_DIR_PATH
    GEN_EXTENSION
    MLIR_OPT_FLAGS
  )
  set(GENERATED_FILES)
  file(MAKE_DIRECTORY ${GEN_DIR_PATH})
  
  foreach(INPUT_FILE ${INPUT_FILES})
    get_filename_component(FILE_NAME "${INPUT_FILE}" NAME_WE)
    set(GEN_FILE "${GEN_DIR_PATH}/${FILE_NAME}${GEN_EXTENSION}")

    add_custom_command(
      OUTPUT ${GEN_FILE}
      COMMAND ${MLIR_OPT_BIN} ${MLIR_OPT_FLAGS} ${INPUT_FILE} > ${GEN_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS ${INPUT_FILE}
    )

    list(APPEND GENERATED_FILES ${GEN_FILE})
  endforeach()

  set(${OUT_FILE_LIST} ${GENERATED_FILES} PARENT_SCOPE)
endfunction()

#################################################
# Shortcut function for TC to MLIR generation
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
#################################################
function(tc_to_mlir
    OUT_FILE_LIST
    INPUT_FILES
  )
  set(MLIR_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_gen)

  teckyl(
    MLIR_FILES
    "${INPUT_FILES}"
    "${MLIR_GEN_DIR}"
    "${MLIR_GEN_EXTENSION}"
    "${TECKYL_FLAGS}"
  )

  set(${OUT_FILE_LIST} ${MLIR_FILES} PARENT_SCOPE)
endfunction()

#################################################
# Shortcut function to lower MLIR to CIM dialect
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
#################################################
function(mlir_cim_lowering
    OUT_FILE_LIST
    INPUT_FILES
  )
  set(MLIR_CIM_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_gen)

  mlir_opt(
    CIM_FILES
    "${MLIR_FILES}"
    "${MLIR_CIM_DIR}"
    "${MLIR_CIM_EXTENSION}"
    "${MLIR_OPT_CIM_FLAGS}"
  )

  set(${OUT_FILE_LIST} ${CIM_FILES} PARENT_SCOPE)
endfunction()

#################################################
# Shortcut function to compile MLIR to object files
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
#################################################
function(mlir_to_obj
    OUT_FILE_LIST
    INPUT_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_gen)

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  set(${OUT_FILE_LIST} ${OBJ_FILES} PARENT_SCOPE)
endfunction()
