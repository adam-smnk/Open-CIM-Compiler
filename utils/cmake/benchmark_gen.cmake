#################################################
# Utility compilation function
#################################################
function(compile_benchmark
    TARGET
    CPP_FILES
    OBJ_FILES
  )
  add_executable(bench-${TARGET}
    ${CPP_FILES}
    ${OBJ_FILES}
    ${SIM_OBJ}
  )

  target_link_libraries(bench-${TARGET}
    mlirInterface
  )
endfunction()

#################################################
# Builds with various optimizations
#################################################
# Execute on host without CIM
function(bench_arm
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_arm)

  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")
  
  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-arm
    ${CPP_FILES}
    ${OBJ_FILES}
  )

  target_compile_definitions(bench-${TARGET}-arm PUBLIC -DBENCH_BUILD_ARM)
endfunction()

# Run on CIM without any additional optimizations
# Assumes workloads can fit crossbar
function(bench_cim
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_cim)

  set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=1 --cim-tile-size=0 --cim-min-writes=0 --convert-cim-to-std")
  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-cim
    ${CPP_FILES}
    ${OBJ_FILES}
  )
endfunction()

# CIM - tiling enabled
function(bench_cim_tiled
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_cim_tiled)

  set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=1 --cim-tile-size=${CIM_TILE_SIZE} --cim-min-writes=0 --convert-cim-to-std")
  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-cim-tiled
    ${CPP_FILES}
    ${OBJ_FILES}
  )
endfunction()

# CIM - tiling and loop interchange to reduce number of xbar writes
function(bench_cim_min_writes
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_cim_min_writes)

  set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=1 --cim-tile-size=${CIM_TILE_SIZE} --cim-min-writes=1 --convert-cim-to-std")
  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-cim-min-writes
    ${CPP_FILES}
    ${OBJ_FILES}
  )
endfunction()

# CIM - tiling and unrolling to parallelize across available tiles
function(bench_cim_unroll
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_cim_unroll)

  set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=${CIM_NUM_TILES} --cim-tile-size=${CIM_TILE_SIZE} --cim-min-writes=0 --convert-cim-to-std")
  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-cim-unroll
    ${CPP_FILES}
    ${OBJ_FILES}
  )
endfunction()

# CIM - all optimizations enabled
function(bench_cim_opt
    TARGET
    CPP_FILES
    MLIR_FILES
  )
  set(OBJ_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/obj_cim_opt)

  set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=${CIM_NUM_TILES} --cim-tile-size=${CIM_TILE_SIZE} --cim-min-writes=1 --convert-cim-to-std")
  separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

  compile_mlir(
    OBJ_FILES
    "${MLIR_FILES}"
    "${OBJ_GEN_DIR}"
    "${OBJ_GEN_EXTENSION}"
    "${MLIR_OPT_FLAGS}"
    "${MLIR_TRANSLATE_FLAGS}"
    "${LLC_FLAGS}"
  )

  compile_benchmark(${TARGET}-cim-opt
    ${CPP_FILES}
    ${OBJ_FILES}
  )
endfunction()

function(make_all_benchmarks
    TARGET
    CPP_FILES
    MLIR_FILES
  )

  bench_arm(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
  bench_cim(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
  bench_cim_tiled(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
  bench_cim_min_writes(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
  bench_cim_unroll(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
  bench_cim_opt(
    ${TARGET}
    ${SOURCES}
    ${MLIR_FILES}
  )
endfunction()
