#################################################
# TC to MLIR generation using Teckyl frontend
# @param[out] OUT_FILE_LIST list of generated files
# @param[in] INPUT_FILES list of source files
# @param[in] GEN_DIR_PATH path to output directory
# @param[in] GEN_ENTENSION extension for generated
#            files (should contain dot)
#################################################
function(tc_to_cim 
    OUT_FILE_LIST
    INPUT_FILES
    GEN_DIR_PATH
    GEN_EXTENSION
  )
  set(GENERATED_FILES)
  file(MAKE_DIRECTORY ${GEN_DIR_PATH})
  
  foreach(INPUT_FILE ${INPUT_FILES})
    get_filename_component(FILE_NAME "${INPUT_FILE}" NAME_WE)
    set(GEN_FILE "${GEN_DIR_PATH}/${FILE_NAME}${GEN_EXTENSION}")

    add_custom_command(
      OUTPUT ${GEN_FILE}
      COMMAND ${TECKYL_BIN} --emit=mlir ${INPUT_FILE} 2> ${GEN_FILE}
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
