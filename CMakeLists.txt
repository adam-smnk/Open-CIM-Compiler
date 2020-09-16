cmake_minimum_required(VERSION 3.4.3)

#################################################
# CIM Settings
#################################################
# Number of available CIM tile units (int)
set(CIM_NUM_TILES 0)

# CIM tile size (int)
set(CIM_TILE_SIZE 0)

# Minimize writes to crossbar (bool)
set(CIM_MIN_WRITES 0)

#################################################
# Architecture Settings
#################################################
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CODE_TYPE "arm")
set(CPU "-mcpu=cortex-a9")
set(ARCH_FLAGS "${CPU} -mfloat-abi=soft -m${CODE_TYPE}")

if(DEFINED ENV{SIM_ROOT})
  set(SIM_ROOT "$ENV{SIM_ROOT}")
else()
  set(SIM_ROOT "$ENV{HOME}/EMSim")
  message("Using the default SIM_ROOT path: ${SIM_ROOT}")
endif()

set(SIM_INCLUDES "-I${SIM_ROOT}/libs -I${SIM_ROOT}")
set(SIM_LINKMAP "${SIM_ROOT}/benchmarks/linkmap.txt")

set(LD_SCRIPT "${SIM_ROOT}/libs/gcc.ld")
set(LD_FLAGS "-T${LD_SCRIPT} -Wl,--build-id=none,-Map=${SIM_LINKMAP} --specs=nosys.specs -lc -lnosys ${SIM_INCLUDES}")

# Do not run linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Simulator libraries
set(SIM_OBJ
  ${SIM_ROOT}/libs/cim.o
  ${SIM_ROOT}/libs/syscalls.o
  ${SIM_ROOT}/libs/gic.o
  ${SIM_ROOT}/libs/startup.o
)

#################################################
# Compilation Settings
#################################################
# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")

# set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O2 -fexceptions -fno-rtti -fPIC")
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2 -fexceptions -fno-rtti -fPIC")
# SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")
set(CMAKE_LINKER "arm-none-eabi-ld")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3 ${ARCH_FLAGS} -fno-exceptions -fPIC")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 ${ARCH_FLAGS} -fno-exceptions -fno-rtti -fPIC")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${ARCH_FLAGS} ${LD_FLAGS}" CACHE INTERNAL "")

set(CMAKE_BUILD_TYPE "")

project("GemmInMemory")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${SIM_ROOT})
include_directories(${SIM_ROOT}/libs)
link_directories(${SIM_ROOT}/libs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#################################################
# Teckyl Settings
#################################################
set(TECKYL_BIN "$ENV{HOME}/llvm/teckyl/build/bin/teckyl")

separate_arguments(TECKYL_FLAGS UNIX_COMMAND "--emit=mlir")

#################################################
# MLIR Settings
#################################################
set(MLIR_OPT_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-opt")
set(MLIR_TRANSLATE_BIN "$ENV{HOME}/llvm/llvm-project/build/bin/mlir-translate")
set(LLC_BIN "llc")

# Intermediate file extensions
set(MLIR_GEN_EXTENSION ".mlir")
set(MLIR_CIM_EXTENSION "_cim.mlir")
set(OBJ_GEN_EXTENSION ".o")

# CIM compilation flags
set(MLIR_CIM_FLAGS "--convert-linalg-to-cim --cim-num-tiles=${CIM_NUM_TILES} --cim-tile-size=${CIM_TILE_SIZE} --cim-min-writes=${CIM_MIN_WRITES} --convert-cim-to-std")
set(MLIR_CONV_FLAGS "--convert-linalg-to-loops --convert-loop-to-std --convert-std-to-llvm=\'emit-c-wrappers=1\'")
set(MLIR_EXTRA_FLAGS "--expensive-combines --loop-coalescing")
set(MLIR_COMMON_FLAGS "--canonicalize --cse --symbol-dce")

# Linalg to CIM lowering flags
separate_arguments(MLIR_OPT_CIM_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")

# MLIR to .obj compilation flags
separate_arguments(MLIR_OPT_FLAGS UNIX_COMMAND "${MLIR_CIM_FLAGS} ${MLIR_CONV_FLAGS} ${MLIR_EXTRA_FLAGS} ${MLIR_COMMON_FLAGS}")
separate_arguments(MLIR_TRANSLATE_FLAGS UNIX_COMMAND "--mlir-to-llvmir")
separate_arguments(LLC_FLAGS UNIX_COMMAND "--march=arm --filetype=obj --relocation-model=pic")

#################################################
# Cmake includes
#################################################
include(${CMAKE_SOURCE_DIR}/utils/cmake/file_generation.cmake)

#################################################
# Subdirectories
#################################################
add_subdirectory(lib)
add_subdirectory(tests)